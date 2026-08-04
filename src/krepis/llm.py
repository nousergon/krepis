"""Provider-agnostic LLM client — the fleet's plug-and-play chokepoint.

Generalization of the Think Tank's ratified pattern
(``crucible-research/thinktank/client.py``) into a library-grade adapter:
one call surface over two transports —

- **anthropic** — the native Anthropic SDK. Keeps every
  Anthropic-specific capability the fleet relies on: the server-side
  ``web_search`` tool, forced server-tool ``tool_choice``, ephemeral
  ``cache_control`` prompt caching, and forced-tool structured outputs.
  Payloads are built through :mod:`krepis.anthropic_payload`, so its
  invariants (server-tool ⊥ assistant-prefill) are inherited.
- **openai** — the OpenAI SDK pointed at any OpenAI-compatible
  ``base_url``: OpenAI itself, OpenRouter (the fleet's open-source-model
  aggregator), or a self-hosted vLLM endpoint. Structured outputs via
  strict ``response_format=json_schema`` where supported, with a
  JSON-instruction + tolerant-extraction fallback where not.

Which transport runs is pure configuration — a :class:`~krepis.llm_config.ModelSpec`
resolved from SSM/env via :func:`krepis.llm_config.resolve_model_spec` —
so flipping a product between Anthropic, OpenAI, and open-source models
is an ``aws ssm put-parameter``, never a code change.

**No silent provider fallback.** A failed call on the configured
provider raises (:exc:`LLMError`); a capability the configured transport
cannot provide raises (:exc:`~krepis.llm_config.LLMConfigError`).
Rollback is an operator flipping the config back — not the library
guessing (``feedback_no_silent_fails``).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random as _random
import re
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from krepis.anthropic_payload import (
    build_messages_payload,
    build_web_search_tool,
)
from krepis.llm_config import (
    TRANSPORT_ANTHROPIC,
    TRANSPORT_LITELLM,
    LLMConfigError,
    ModelSpec,
)
from krepis.llm_search import (
    Citation,
    SearchEvent,
    extract_anthropic_citations,
    extract_anthropic_search_events,
    extract_openrouter_citations,
    final_text_after_last_tool,
)

logger = logging.getLogger(__name__)

# ── LiteLLM Router (lazy singleton) ────────────────────────────────────────
# When provider=litellm, calls route through a litellm.Router configured with
# the model groups defined in LLM_MODEL_REGISTRY.yaml. The Router handles
# fallback chains transparently — a call to model "low" tries the primary
# in the low group, then falls back through the ordered chain on failure.
#
# The Router config is the SINGLE source of truth, derived from
# LLM_MODEL_REGISTRY.yaml at init time (with a hardcoded fallback if the
# registry file can't be found). See krepis.router for the registry loader
# and CLI (python3 -m krepis.router resolve <group>).
#
# Model groups (no Anthropic — per Brian's 2026-07-24 ruling):
#   low:  deepseek-v4-flash → gemini-2.5-flash → gpt-oss-120b → gemini-2.5-pro
#   med:  deepseek-v4-flash (reasoning=max) → same via OpenRouter → v4-pro
#   high: deepseek-v4-pro (reasoning=max) → same via OpenRouter
#   ultra: glm-5.2 → kimi-k3 → deepseek-v4-pro (reasoning=max)
#
# Initialized on first use so importing krepis.llm doesn't pay the Router
# construction cost until a caller actually uses the litellm transport.


def _get_router() -> Any:
    """Return the module-level LiteLLM Router singleton.

    Delegates to :func:`krepis.router.get_router` which builds from
    LLM_MODEL_REGISTRY.yaml (preferred) or a hardcoded fallback.
    """
    from krepis.router import get_router as _router_get

    return _router_get()


# Per-group fallback-state tracker so callers can detect sign-on / sign-off
# transitions.  Keyed by group name ("low", "med", "high", "ultra"); True
# means the last call for that group was served by a fallback model.
_fallback_state: dict[str, bool] = {}


def _check_fallback_transition(group: str, fallback_used: bool, served_model: str, primary: str) -> None:
    """Log a warning/info when a group enters or exits fallback.

    Called after every LiteLLM Router completion so the operator sees
    exactly when a backup model signs on and when the primary recovers.
    """
    was_in_fallback = _fallback_state.get(group, False)
    if fallback_used and not was_in_fallback:
        logger.warning(
            "🔄 group %r FALLBACK ENGAGED — primary %s failed, now served by %s",
            group, primary, served_model,
        )
    elif not fallback_used and was_in_fallback:
        logger.info(
            "✅ group %r PRIMARY RESTORED — %s is healthy again",
            group, primary,
        )
    _fallback_state[group] = fallback_used


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Special/control-token leakage: some open-weight models (confirmed live
# 2026-07-14 with moonshotai/kimi-k2.6 via OpenRouter) emit their OWN
# native function-calling token dialect (e.g. Kimi's
# ``<|tool_calls_section_begin|>...<|tool_call_begin|>...<|tool_call_end|>``)
# directly into ``message.content`` instead of a structured ``tool_calls``
# field — even when the tool is declared as an OpenRouter SERVER-SIDE tool
# that is supposed to be resolved before the response reaches us. The
# gateway does not always intercept/execute it, so the raw protocol text
# leaks through as if it were the final answer. This pattern is
# model-agnostic on purpose (``<|...|>``) rather than matching Kimi's exact
# tokens, since any vendor's internal control-token dialect leaking into
# content is the same failure mode.
_CONTROL_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]{1,60}\|>")

_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a single JSON object matching this JSON Schema — "
    "no prose, no markdown fences:\n{schema}"
)

# Bounded jittered backoff between attempts of the retry loops below. The
# SDK's own ``max_retries`` backs off for status/connection failures, but it
# never sees the BODY-level failures this module retries (a non-JSON body, a
# null-choices body, an unresolved tool call) — those were retried in a tight
# loop, which is the worst possible cadence against a gateway that is briefly
# unhealthy. Full jitter, same shape as ``krepis.http_retry.backoff_delay``.
_RETRY_BASE_DELAY_S = 2.0
_RETRY_DELAY_CAP_S = 30.0


def _retry_backoff_sleep(attempt: int) -> None:
    """Sleep before re-issuing attempt ``attempt`` (0-indexed, already failed)."""
    delay = min(_RETRY_BASE_DELAY_S * (2**attempt), _RETRY_DELAY_CAP_S)
    _time.sleep(_random.uniform(0, delay))


class NullChoicesError(Exception):
    """A 200 response whose ``choices`` is null or empty.

    OpenRouter reports an upstream provider failure in the BODY of a 200 —
    ``choices`` null (or empty) with an ``error`` object beside it — rather
    than as an HTTP error. The SDK builds that object without complaint, so it
    raises nothing, and the natural ``resp.choices[0]`` then raises
    ``TypeError: 'NoneType' object is not subscriptable``: a bare exception
    that escapes every bounded-retry loop and discards the provider's own
    error message unread.

    Modelling it as a typed exception lets it share the retry path with the
    other body-level transport failures. Lifted from
    ``crucible-research/thinktank/client.py::_NullChoicesError``, which had to
    solve this locally because this chokepoint did not
    (alpha-engine-config#5223 / crucible-research#530).
    """


def _first_choice(resp: Any) -> Any:
    """Return ``resp.choices[0]`` or raise :class:`NullChoicesError`.

    Every ``choices[0]`` read on an OpenAI-shaped response goes through here —
    a guard applied at four of five sites is not a guard.
    """
    choices = getattr(resp, "choices", None)
    if not choices:
        raise NullChoicesError(
            f"provider returned no choices "
            f"(error={getattr(resp, 'error', None)!r}, "
            f"id={getattr(resp, 'id', None)!r}, "
            f"model={getattr(resp, 'model', None)!r})"
        )
    return choices[0]


def _empty_content_diagnostics(resp: Any, choice: Any) -> str:
    """Why ``message.content`` came back empty, in one line.

    An empty content on an otherwise successful response has several causes
    that look identical from the outside, and the SDK surfaces none of them:

    - a **reasoning model that spent the whole output budget on its trace**.
      ``max_tokens`` bounds reasoning + content together, so the response is
      large, ``choices`` is present, and ``content`` is ``''``. This is what
      the Director hit on 2026-08-04 (alpha-engine-config#6396): two ~100s
      completions, both fully billed, nginx logging 30 KB responses, and the
      only signal was ``no JSON object found in response: ''``;
    - a refusal or a content filter, which lands in a sibling field;
    - a genuinely truncated body (``finish_reason='length'``);
    - a provider whose content sits somewhere other than ``message.content``.

    ``finish_reason`` alone does not separate them — a budget consumed by
    reasoning reports ``length`` on some routes and ``stop`` on others. The
    reasoning-token count and the sibling field names do.
    """
    # The ONE swallow in this module's fail-loud contract, and it is bounded to
    # instrumentation: (a) the failure mode swallowed is this function's own
    # introspection raising — an SDK response object whose attribute access
    # has side effects, which `getattr(x, y, None)` does NOT protect against,
    # because a raising `__getattr__` propagates past the default; (b) the
    # primary deliverable survives untouched — the caller still receives the
    # content it read and classifies it exactly as before, since a diagnostic
    # that can break the call it describes is worse than no diagnostic;
    # (c) the recording surface is the same ERROR line, which still emits and
    # names the introspection failure instead of the fields.
    try:
        msg = getattr(choice, "message", None)
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(msg, "reasoning_content", None) or getattr(
            msg, "reasoning", None
        )
        # `vars()` alone is not enough: on a pydantic-modelled SDK response
        # the provider's non-standard fields live in `__pydantic_extra__`, and
        # `reasoning_content` — the single most diagnostic name here — is
        # exactly one of those. Measured 2026-08-04 against the live edge:
        # `vars()` reported `['role']` on a message carrying 29,877 chars of
        # reasoning, i.e. it named none of what the operator needs.
        attrs = dict(vars(msg)) if msg is not None and hasattr(msg, "__dict__") else {}
        attrs.update(getattr(msg, "model_extra", None) or {})
        fields = sorted(k for k, v in attrs.items() if v not in (None, "", [], {}))
        return (
            f"finish_reason={getattr(choice, 'finish_reason', None)!r} "
            f"native_finish_reason="
            f"{getattr(choice, 'native_finish_reason', None)!r} "
            f"completion_tokens={getattr(usage, 'completion_tokens', None)!r} "
            f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)!r} "
            f"reasoning_chars="
            f"{len(reasoning) if isinstance(reasoning, str) else None} "
            f"populated_message_fields={fields} "
            f"response_type={type(resp).__name__} "
            f"id={getattr(resp, 'id', None)!r} "
            f"model={getattr(resp, 'model', None)!r}"
        )
    except Exception as exc:  # noqa: BLE001 — see the three-part rationale above
        return (
            f"diagnostics unavailable: introspecting the response raised "
            f"{exc.__class__.__name__}: {exc} "
            f"(response_type={type(resp).__name__})"
        )


def _choice_text(resp: Any) -> str:
    """First choice's message content, stripped. Raises on null choices.

    Logs at ERROR when the content is empty. The emptiness itself is not an
    error here — callers classify it — but it is invisible without this line,
    and the caller-facing symptom actively misdirects: a structured caller
    reports ``no JSON object found in response: ''``, which reads as a model
    that answered in prose. Instrumented at THIS chokepoint rather than at the
    structured paths, for the same reason ``_first_choice`` is: a guard
    applied at four of five call sites is not a guard.
    """
    choice = _first_choice(resp)
    text = (getattr(choice.message, "content", None) or "").strip()
    if not text:
        logger.error(
            "llm: EMPTY message.content on a successful response — %s",
            _empty_content_diagnostics(resp, choice),
        )
    return text


class LLMError(RuntimeError):
    """A call failed after its bounded corrective retries — fail loud.

    Carries ``usage`` (the :class:`LLMUsage` accumulated across the failed
    attempts) so callers can still record the spend of a failed call —
    tokens were consumed even though no valid output was produced.
    """

    def __init__(self, message: str, *, usage: "Optional[LLMUsage]" = None):
        super().__init__(message)
        self.usage = usage


class BudgetExhaustedError(LLMError):
    """The completion budget ran out before any content was produced.

    A distinct type because it has a distinct fix. ``no JSON object found in
    response: ''`` — what this used to surface as — says *the model returned
    something unparseable*, which is a prompt or model problem. The actual
    fault is *``max_tokens`` was too small for this ask*, a one-line change to
    the registry row. Three wrong hypotheses were chased against a live paid
    endpoint before anyone looked at ``finish_reason``
    (alpha-engine-config#6391).

    **Never retried.** The second attempt cannot succeed under the same
    budget: it re-issues the identical ask against the identical ceiling.
    Measured on the Director's weekly call — two attempts, ~100s of generation
    each, both fully billed, both guaranteed to fail before the first one
    returned. Retrying does not merely fail to inform, it doubles the cost of
    a certain failure.

    Empty content with ``finish_reason='stop'`` is a DIFFERENT fault and keeps
    the ordinary corrective-retry path — that one really is a model returning
    nothing useful, and a retry can fix it.
    """


def _budget_exhausted_error(
    *,
    spec: "ModelSpec",
    max_tokens: int,
    stop_signal: str,
    reasoning_tokens: Any = None,
    usage: "Optional[LLMUsage]" = None,
) -> "BudgetExhaustedError":
    """The one message, built the same way for every transport."""
    return BudgetExhaustedError(
        f"provider={spec.provider} model={spec.model}: the completion budget "
        f"was exhausted before any content was produced — max_tokens="
        f"{max_tokens}, {stop_signal}, reasoning_tokens={reasoning_tokens!r}. "
        f"On a reasoning model max_tokens bounds reasoning AND content "
        f"together, so the trace consumed the whole budget and nothing was "
        f"left to answer with. Raise the budget for this ask — not the "
        f"prompt, and not the schema.",
        usage=usage,
    )


def _reject_budget_exhausted(
    resp: Any,
    text: str,
    *,
    spec: "ModelSpec",
    max_tokens: int,
    usage: "Optional[LLMUsage]" = None,
) -> None:
    """Raise :exc:`BudgetExhaustedError` for an empty, length-capped response.

    No-op unless the content is empty AND the provider says it stopped because
    it hit the ceiling. Both conditions matter: a length-capped response WITH
    content is ordinary truncation the caller may still parse, and an empty
    response that stopped naturally is a different fault entirely
    (a model that answered with nothing, which a retry can fix).
    """
    if text:
        return
    choice = _first_choice(resp)
    if getattr(choice, "finish_reason", None) != "length":
        return
    details = getattr(getattr(resp, "usage", None), "completion_tokens_details", None)
    raise _budget_exhausted_error(
        spec=spec,
        max_tokens=max_tokens,
        stop_signal="finish_reason='length'",
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
        usage=usage,
    )


# ── Result types ──────────────────────────────────────────────────────────


@dataclass
class LLMUsage:
    """Normalized token/fee usage for one logical call (all attempts)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    cache_create_1h_tokens: int = 0
    # prompt_cache_miss_tokens — input tokens that missed the cache on
    # providers that report it (DeepSeek's prompt_cache_miss_tokens).
    # Together with cache_read_tokens this gives a cache-hit ratio:
    #   hit_rate = cache_read / (cache_read + prompt_cache_miss)
    # when both fields are populated (0 = provider didn't report it).
    prompt_cache_miss_tokens: int = 0
    web_search_requests: int = 0
    web_fetch_requests: int = 0
    # Provider-reported USD cost when available (OpenRouter returns it in
    # ``usage.cost`` when the request opts in). Preferred over card-priced
    # recompute by :func:`krepis.cost.record_llm_call` — the aggregator
    # knows the actually-routed backend's price; our cards are ceilings.
    provider_cost_usd: Optional[float] = None


@dataclass
class LLMResult:
    """Outcome of :meth:`LLMClient.complete`."""

    text: str
    model: str  # model the provider reports (may differ from spec.model)
    provider: str
    usage: LLMUsage
    # The exact request kwargs and raw provider response — exposed so
    # product-side capture (SFT traces, debugging) needs no adapter
    # changes. Never mutated by the adapter after the call.
    raw_request: dict
    raw_response: Any = None
    # The upstream inference backend that actually served this request —
    # DISTINCT from ``provider`` (the static transport name, e.g.
    # "openrouter"). OpenRouter's OpenAI-compatible response carries a
    # non-standard top-level ``provider`` field (e.g. "DeepInfra",
    # "SiliconFlow") naming the routed backend; verified live 2026-07-22
    # via ``resp.provider`` on a real ``ChatCompletion`` (pydantic
    # extra="allow" exposes it as a real attribute). ``None`` on the
    # anthropic transport (single-backend, no routing ambiguity) and on
    # any openai-compatible provider that doesn't emit the field.
    # Consumers needing jurisdiction/compliance checks (config#3006) read
    # this instead of parsing ``raw_response`` themselves.
    served_provider: Optional[str] = None
    # True when a fallback model in the group's chain served this request
    # (the primary failed and LiteLLM's Router transparently tried the
    # next model).  Always False on non-litellm transports.
    fallback_used: bool = False
    # The group name (or model id) that was requested — preserved so
    # callers can tell WHAT was asked for separately from WHAT served it.
    model_requested: str = ""
    # Optional parameters the route could not honor, which the caller
    # explicitly allowed to be dropped (on_unsupported="drop"). Empty on a
    # fully-honored call. Present on the RESULT so a degraded call is visible
    # in the artifact, not only in a log line.
    dropped_params: list[str] = field(default_factory=list)
    # Set when cost-record emission failed for this call (pricing lookup
    # miss, sink write error). ``None`` on success and on clients with no
    # ``cost_sink``. Emission never raises — see
    # ``LLMClient._emit_cost_record`` — so this field is how a telemetry
    # loss stays visible in the artifact rather than only in a log line.
    cost_emission_error: Optional[str] = None


@dataclass
class StructuredResult(LLMResult):
    """Outcome of :meth:`LLMClient.structured` — validated payload."""

    data: dict = field(default_factory=dict)
    # Pydantic instance when ``schema`` was a BaseModel subclass.
    parsed: Any = None


@dataclass
class SearchOptions:
    """Grounding options for :meth:`LLMClient.complete_grounded`.

    ``force_first`` deterministically forces a web search before any text
    (Anthropic forced server-tool ``tool_choice`` — verified live
    2026-06-29). The OpenRouter server tool cannot be forced, so
    ``force_first=True`` on the openai transport raises
    :exc:`~krepis.llm_config.LLMConfigError` rather than silently degrading
    to a prose request.
    """

    max_uses: int = 20
    force_first: bool = False
    # OpenRouter engine choice ("exa", "parallel", ...); None = provider auto.
    engine: Optional[str] = None
    # OpenRouter per-search result cap; None = provider default.
    max_results: Optional[int] = None


@dataclass
class GroundedResult(LLMResult):
    """Outcome of :meth:`LLMClient.complete_grounded`.

    ``text`` is the post-final-tool text (anthropic) or the message content
    (openai) — the answer, without inter-search narration. ``searches``
    carries per-query events (anthropic only — OpenRouter exposes citations,
    not queries); ``citations`` is populated on both transports.
    """

    searches: List[SearchEvent] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)


# ── Client ────────────────────────────────────────────────────────────────


def _emits_cost(method):
    """Emit a priced cost record for whatever result *method* returns.

    Applied at the PUBLIC method boundary rather than at each ``return``
    site, deliberately. ``complete`` / ``structured`` / ``complete_grounded``
    construct results at seven different points today; decorating the method
    means a future return path cannot silently escape telemetry. A missed
    return path would be invisible — no error, no log, just a call site that
    quietly stops being accounted for — which is precisely the failure class
    this arc exists to close (alpha-engine-config-I5206).
    """

    @functools.wraps(method)
    def _wrapped(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self._emit_cost_record(result)
        return result

    return _wrapped


class LLMClient:
    """One (provider, model) client with a normalized call surface.

    Construct with a resolved :class:`~krepis.llm_config.ModelSpec`.
    Cheap to construct per call — SDK clients are created lazily and this
    object holds no other state — so consumers that re-resolve their spec
    per request (picking up SSM flips) can build a fresh ``LLMClient``
    each time.

    ``client_factory`` is the test seam (Think Tank pattern): a callable
    ``(spec, api_key) -> transport_client`` returning an object exposing
    ``messages.create`` (anthropic transport) or ``chat.completions.create``
    (openai transport).

    **Cost attribution.** ``callsite_id`` is REQUIRED. Every call this
    client makes is billed to it, and it is the join key between spend and
    the call site that caused it. It is required rather than optional
    because an optional attribution field is one nobody fills: the fleet
    ran 17 days with no per-call cost telemetry at all, and the recovery
    plan depends on attribution being impossible to omit rather than merely
    encouraged (alpha-engine-config-I5206).

    krepis validates only that it is a non-empty string — this is a public,
    pip-installable library and cannot read anyone's private call-site
    registry. Consumers that keep one (the Nous Ergon fleet keeps
    ``LLM_CALLSITE_REGISTRY.yaml``) validate membership in their own CI.

    ``cost_sink`` is an optional ``callable(record: dict) -> None``. When
    set, each completed call builds a priced record via
    :func:`krepis.cost.record_llm_call` — carrying token counts, cache
    read/write splits and USD — and hands it to the sink. Default ``None``
    means no emission, so public consumers pay nothing for a feature they
    have not asked for.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        callsite_id: str,
        api_key: Optional[str] = None,
        client_factory: Optional[Callable[[ModelSpec, str], Any]] = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        cost_sink: Optional[Callable[[dict], None]] = None,
    ):
        if not isinstance(callsite_id, str) or not callsite_id.strip():
            raise ValueError(
                "LLMClient requires a non-empty callsite_id — it is the join "
                "key between spend and the call site that caused it. Pass the "
                "identifier this call site is registered under."
            )
        self.spec = spec
        self.callsite_id = callsite_id
        self._api_key = api_key
        self._client_factory = client_factory
        self._timeout = timeout
        self._max_retries = max_retries
        self._cost_sink = cost_sink
        self._client: Any = None
        if spec.supports_automatic_prefix_caching:
            logger.info(
                "LLMClient(%s/%s): automatic prefix caching is active for this model "
                "(server-side, no client-side cache_control markers needed)",
                spec.provider, spec.model,
            )
        # Parameters the route could not honor and the caller allowed us to
        # drop. Surfaced on LLMResult so a degraded call is visible in the
        # artifact rather than only in a log line.
        self.dropped_params: list[str] = []

    # ── cost emission ─────────────────────────────────────────────────

    def _emit_cost_record(self, result: Any) -> None:
        """Build a priced cost record for *result* and hand it to the sink.

        No-op when no ``cost_sink`` is configured. **Never raises.**
        """
        if self._cost_sink is None:
            return
        try:
            from krepis.cost import record_llm_call

            record = record_llm_call(
                result, extra_fields={"callsite_id": self.callsite_id}
            )
            self._cost_sink(record)
        except Exception as exc:  # noqa: BLE001
            # DELIBERATE non-raising degradation. Written rationale per the
            # fail-loud rule, which forbids silent swallows by default:
            #
            # (a) Failure mode swallowed: cost-record construction or sink
            #     write failed — an unpriced model card, an S3 error, a full
            #     disk.
            # (b) Why the primary deliverable survives: this runs AFTER the
            #     LLM call returned successfully. The caller already holds
            #     its answer. Raising here would convert a telemetry problem
            #     into a production outage, which is a strictly worse trade
            #     than losing one cost row.
            # (c) Recording surface, three layers so no single miss hides it:
            #     this ERROR log; ``result.cost_emission_error`` on the
            #     artifact itself; and the ARTIFACT_REGISTRY freshness row on
            #     ``decision_artifacts/_cost/``, which catches SUSTAINED loss
            #     even if every individual log goes unread. That third layer
            #     is the one that was missing for 17 days (I5206).
            logger.error(
                "cost emission failed for callsite_id=%s model=%s: %s",
                self.callsite_id,
                getattr(result, "model", "?"),
                exc,
            )
            try:
                result.cost_emission_error = str(exc)
            except Exception:  # noqa: BLE001 — frozen/exotic result type
                pass

    # ── transport plumbing ────────────────────────────────────────────

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        env_name = self.spec.resolved_api_key_env()
        key = os.environ.get(env_name)
        if not key:
            raise LLMConfigError(
                f"no API key for provider {self.spec.provider!r}: pass "
                f"api_key= or set the {env_name} environment variable"
            )
        return key

    def _transport_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._resolve_api_key()
        if self._client_factory is not None:
            self._client = self._client_factory(self.spec, api_key)
        elif self.spec.transport == TRANSPORT_ANTHROPIC:
            # Lazy import — anthropic is an optional extra (krepis[anthropic]).
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        else:
            # Lazy import — openai is an optional extra (krepis[openai]).
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.spec.resolved_base_url(),
                api_key=api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._client

    def _is_openrouter(self) -> bool:
        if self.spec.provider == "openrouter":
            return True
        base_url = self.spec.base_url or ""
        return "openrouter.ai" in base_url

    def _capability_gate(
        self,
        param: str,
        supported: bool,
        *,
        on_unsupported: str,
        detail: str = "",
    ) -> bool:
        """Decide what to do with *param* when the target route can't honor it.

        Returns True when the caller should still send it.

        Three outcomes, chosen explicitly rather than by omission
        (``model-portability-policy`` §7 / I9): honor, degrade-and-record, or
        raise. Default is ``raise`` — a config knob that quietly does nothing
        is exactly the failure ``feedback_no_silent_fails`` forbids, and it is
        indistinguishable from working.

        Exists because the same argument to the same method used to mean three
        different things across three transports: ``cache_system`` was honored
        on anthropic, deliberately (and correctly) ignored on openai, and
        SILENTLY dropped on litellm with no comment or signal
        (alpha-engine-config-I4469). Routing every optional parameter through
        one gate makes the next one impossible to forget.
        """
        if supported:
            return True
        if on_unsupported == "drop":
            self.dropped_params.append(param)
            logger.info(
                "dropping %s: not supported by %s%s",
                param, self.spec.model, f" ({detail})" if detail else "",
            )
            return False
        raise LLMConfigError(
            f"{param} is not supported by model {self.spec.model!r} "
            f"(transport={self.spec.transport}){f' — {detail}' if detail else ''}. "
            f"Pass on_unsupported='drop' to send the request without it and "
            f"record the drop on the result."
        )

    def _reject_reasoning_on_anthropic(self) -> None:
        """``ModelSpec.reasoning`` has no anthropic-transport equivalent.

        Fail loud rather than silently dropping it — a config-only knob
        that quietly does nothing on the wrong transport is exactly the
        failure mode ``feedback_no_silent_fails`` forbids.
        """
        if self.spec.transport == TRANSPORT_ANTHROPIC and self.spec.reasoning is not None:
            raise LLMConfigError(
                "ModelSpec.reasoning has no anthropic-transport equivalent "
                "— set it only on an openai/openrouter ModelSpec."
            )

    # ── usage extraction ──────────────────────────────────────────────

    @staticmethod
    def _usage_from_anthropic(msg: Any, into: Optional[LLMUsage] = None) -> LLMUsage:
        usage = into or LLMUsage()
        u = getattr(msg, "usage", None)
        if u is None:
            return usage
        usage.input_tokens += int(getattr(u, "input_tokens", 0) or 0)
        usage.output_tokens += int(getattr(u, "output_tokens", 0) or 0)
        usage.cache_read_tokens += int(
            getattr(u, "cache_read_input_tokens", None) or 0
        )
        cache_create_total = int(getattr(u, "cache_creation_input_tokens", None) or 0)
        cache_creation = getattr(u, "cache_creation", None)
        cache_1h = (
            int(getattr(cache_creation, "ephemeral_1h_input_tokens", 0) or 0)
            if cache_creation is not None
            else 0
        )
        usage.cache_create_1h_tokens += cache_1h
        usage.cache_create_tokens += max(cache_create_total - cache_1h, 0)
        # DeepSeek's Anthropic-compatible endpoint returns
        # prompt_cache_hit_tokens / prompt_cache_miss_tokens instead of
        # Anthropic's cache_read_input_tokens / cache_creation_input_tokens.
        # The Anthropic SDK stores unrecognized usage fields as extra
        # attributes; read them via getattr so the standard Anthropic path
        # (cache_read_input_tokens above) takes precedence when both exist.
        ds_hit = int(getattr(u, "prompt_cache_hit_tokens", None) or 0)
        if ds_hit:
            usage.cache_read_tokens += ds_hit
        ds_miss = int(getattr(u, "prompt_cache_miss_tokens", None) or 0)
        if ds_miss:
            usage.prompt_cache_miss_tokens += ds_miss
        stu = getattr(u, "server_tool_use", None)
        if stu is not None:
            usage.web_search_requests += int(
                getattr(stu, "web_search_requests", 0) or 0
            )
            usage.web_fetch_requests += int(
                getattr(stu, "web_fetch_requests", 0) or 0
            )
        return usage

    @staticmethod
    def _usage_from_openai(resp: Any, into: Optional[LLMUsage] = None) -> LLMUsage:
        usage = into or LLMUsage()
        u = getattr(resp, "usage", None)
        if u is None:
            return usage
        usage.input_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
        usage.output_tokens += int(getattr(u, "completion_tokens", 0) or 0)
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            usage.cache_read_tokens += int(getattr(details, "cached_tokens", 0) or 0)
        # DeepSeek-native cache fields — may appear at the usage top level
        # (prompt_cache_hit_tokens / prompt_cache_miss_tokens) rather than
        # inside prompt_tokens_details.cached_tokens.  The standard OpenAI
        # path above takes precedence; these are fallbacks for providers that
        # use the DeepSeek field names on their OpenAI-compatible endpoint.
        usage.cache_read_tokens += int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
        usage.prompt_cache_miss_tokens += int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
        # OpenRouter nests the server-tool search count under
        # ``server_tool_use_details`` (mirroring Anthropic's
        # ``server_tool_use`` shape) rather than a flat ``web_search_requests``
        # field on ``usage``. Confirmed live 2026-07-06 (config#1659): the
        # flat read below always silently returned 0 despite real grounding
        # (55-75 citations per call) — which would have permanently broken
        # the ``min_web_searches`` production incident-guard floor on this
        # transport.
        #
        # ``server_tool_use_details`` is NOT a field the openai SDK's
        # ``CompletionUsage`` model declares — it's an unrecognized/"extra"
        # field, and Pydantic v2 stores those verbatim as the raw decoded
        # JSON value (a plain ``dict``), not as a nested attribute-bearing
        # object the way Anthropic's SDK properly types ``server_tool_use``.
        # An initial fix here (krepis 0.11.1) used ``getattr(stu, ...)``,
        # which silently returns the default on a ``dict`` (dicts have no
        # attributes for their keys) — confirmed live 2026-07-06: it found
        # the right field NAME but still always read 0. Handle both shapes.
        stu = getattr(u, "server_tool_use_details", None)
        if isinstance(stu, dict):
            usage.web_search_requests += int(stu.get("web_search_requests", 0) or 0)
        elif stu is not None:
            usage.web_search_requests += int(
                getattr(stu, "web_search_requests", 0) or 0
            )
        else:
            usage.web_search_requests += int(getattr(u, "web_search_requests", 0) or 0)
        cost = getattr(u, "cost", None)
        if cost is not None:
            usage.provider_cost_usd = (usage.provider_cost_usd or 0.0) + float(cost)
        return usage

    def _effective_max_tokens(self, max_tokens: Optional[int]) -> int:
        """The budget actually sent, warning when a caller shrinks the row's.

        ``max_tokens`` is a registry-owned parameter, and a caller-supplied
        value wins over :attr:`ModelSpec.max_tokens` outright. Raising it is
        ordinary — a caller that knows its own ask is larger than the row's
        default. LOWERING it silently reverses the registry, and on a
        reasoning model that is not a smaller answer, it is NO answer:
        ``max_tokens`` bounds reasoning + content together, so the trace
        consumes the budget and ``content`` comes back ``''``.

        Live 2026-08-04 (alpha-engine-config#6396): the Director passed a
        literal 8000 against a row carrying 65536. Two ~100s completions, both
        fully billed, both empty — and raising the ROW from 16384 to 65536 as
        the remediation changed nothing, because the literal was what the
        request carried. Nothing anywhere logged the number on the wire.

        This is a warning rather than a refusal: shrinking the budget is a
        legitimate cost control, and this library does not get to overrule a
        caller. It only has to stop the override being invisible.
        """
        if max_tokens is None:
            return self.spec.max_tokens
        if max_tokens < self.spec.max_tokens:
            logger.warning(
                "llm: caller max_tokens=%d OVERRIDES the registry's %d for "
                "provider=%s model=%s — the wire carries %d. A registry-side "
                "budget change cannot reach this call while the override "
                "stands, and on a reasoning model max_tokens bounds reasoning "
                "+ content together (alpha-engine-config#6396).",
                max_tokens, self.spec.max_tokens, self.spec.provider,
                self.spec.model, max_tokens,
            )
        return max_tokens

    def _openai_extra_body(self) -> Optional[dict]:
        body: dict = {}
        if self._is_openrouter():
            # OpenRouter reports the actually-billed USD cost in usage when
            # the request opts in — the canonical cost source for :floor
            # routing, where the routed backend's price varies below our
            # card ceilings.
            body["usage"] = {"include": True}
        if self.spec.reasoning is not None:
            # OpenRouter's unified reasoning-control object (e.g.
            # {"effort": "low"}, {"exclude": True}). Without an explicit
            # override, a reasoning-capable model can spend its entire
            # output budget on chain-of-thought and return an empty
            # message.content even at a generous max_tokens — reproduced
            # live 2026-07-06 (config#1659) with Kimi K2.6 against a long
            # production prompt: finish_reason="stop", ~15K reasoning
            # chars, ~1 char of actual content. See ModelSpec.reasoning.
            body["reasoning"] = self.spec.reasoning
        return body or None

    def _choice_text_or_llm_error(self, resp: Any) -> str:
        """``_choice_text`` for the SINGLE-SHOT paths, as a documented error.

        :meth:`complete` has no retry loop of its own, so there is nothing to
        classify a null-choices body INTO. The module contract is that a failed
        call on the configured provider raises :exc:`LLMError` — so convert,
        rather than letting the caller catch a bare ``NullChoicesError`` (or,
        before the guard existed, a ``TypeError`` naming nothing at all).
        """
        try:
            return _choice_text(resp)
        except NullChoicesError as exc:
            raise LLMError(
                f"provider={self.spec.provider} model={self.spec.model}: {exc}",
                usage=self._usage_from_openai(resp),
            ) from exc

    # ── plain completion ──────────────────────────────────────────────

    @_emits_cost
    def complete(
        self,
        *,
        system: str,
        user_content: str,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
        extra: Optional[dict] = None,
        on_unsupported: str = "raise",
    ) -> LLMResult:
        """One plain text generation. Returns normalized :class:`LLMResult`.

        ``cache_system`` attaches Anthropic ephemeral ``cache_control`` to
        the system block. It is a cost-optimization *hint*, not a semantic
        guarantee: on the openai transport it is a no-op because
        OpenAI-compatible providers cache prompt prefixes implicitly (the
        discount shows up in ``usage.prompt_tokens_details.cached_tokens``)
        — there is nothing to forward and nothing is lost.
        """
        self._reject_reasoning_on_anthropic()
        limit = self._effective_max_tokens(max_tokens)

        if self.spec.transport == TRANSPORT_ANTHROPIC:
            payload = build_messages_payload(
                model=self.spec.model,
                system_prompt=system,
                user_content=user_content,
                max_tokens=limit,
                cache_system=cache_system,
                extra=extra,
            )
            msg = self._transport_client().messages.create(**payload)
            text = "\n\n".join(
                getattr(b, "text", "")
                for b in getattr(msg, "content", []) or []
                if getattr(b, "type", None) == "text"
            ).strip()
            return LLMResult(
                text=text,
                model=getattr(msg, "model", self.spec.model),
                provider=self.spec.provider,
                usage=self._usage_from_anthropic(msg),
                raw_request=payload,
                raw_response=msg,
            )

        if self.spec.transport == TRANSPORT_LITELLM:
            # LiteLLM Router handles fallback chains — model is the group name.
            from krepis.router import get_group_primary as _get_primary
            from krepis.router import group_supports_explicit_cache_breakpoints as _grp_pc

            # cache_system asks for EXPLICIT cache_control breakpoints. Whether
            # the served model honors them is a per-model fact resolved from
            # the registry via the group's primary — never assumed, and never
            # silently dropped, which is what this branch used to do.
            if cache_system:
                self._capability_gate(
                    "cache_system",
                    _grp_pc(self.spec.model),
                    on_unsupported=on_unsupported,
                    detail="the model serving this group uses automatic prefix "
                           "caching (or none), which takes no client markers",
                )

            router = _get_router()
            resp = router.completion(
                model=self.spec.model,  # "low", "med", "high", "ultra"
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=limit,
            )
            served_model = getattr(resp, "model", "")
            primary = _get_primary(self.spec.model)
            fallback_used = bool(primary and served_model and served_model != primary)
            _check_fallback_transition(self.spec.model, fallback_used, served_model, primary or "")
            text = self._choice_text_or_llm_error(resp)
            return LLMResult(
                text=text,
                model=served_model or self.spec.model,
                provider=self.spec.provider,
                served_provider=getattr(resp, "_hidden_params", {}).get("model_id", None),
                usage=self._usage_from_openai(resp),
                raw_request={"model": self.spec.model, "system": system, "user_content": user_content},
                raw_response=resp,
                fallback_used=fallback_used,
                model_requested=self.spec.model,
                dropped_params=list(self.dropped_params),
            )

        kwargs: dict = {
            "model": self.spec.model,
            "max_tokens": limit,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        extra_body = self._openai_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        if extra:
            kwargs.update(extra)
        resp = self._transport_client().chat.completions.create(**kwargs)
        text = self._choice_text_or_llm_error(resp)
        return LLMResult(
            text=text,
            model=getattr(resp, "model", self.spec.model),
            provider=self.spec.provider,
            served_provider=getattr(resp, "provider", None),
            usage=self._usage_from_openai(resp),
            raw_request=kwargs,
            raw_response=resp,
        )

    # ── structured completion ─────────────────────────────────────────

    @_emits_cost
    def structured(
        self,
        *,
        system: str,
        user_content: str,
        schema: Any,
        schema_name: str,
        validate: Optional[Callable[[Any], None]] = None,
        attempts: int = 2,
        max_tokens: Optional[int] = None,
    ) -> StructuredResult:
        """One schema-constrained call. Validates or raises :exc:`LLMError`.

        ``schema`` is either a Pydantic ``BaseModel`` subclass (its
        ``model_json_schema()`` is used and the payload is validated back
        into an instance on ``StructuredResult.parsed``) or a raw
        JSON-schema dict (``parsed`` stays ``None``; ``data`` carries the
        dict).

        ``validate`` is the domain-validation hook: called with the parsed
        object (Pydantic instance when available, else the dict); raise
        ``ValueError`` to reject and trigger a bounded corrective retry
        with the error text fed back to the model — the same loop shape
        as schema-validation failure. This is how consumer-side grounding
        checks (e.g. vires' program-spec grounding) plug into the retry.

        ``attempts`` bounds TOTAL model calls (initial + corrective
        retries). Exhaustion raises :exc:`LLMError` carrying the
        accumulated usage so the failed spend can still be recorded.

        Transport mapping: anthropic = forced ``tool_choice`` on a tool
        whose ``input_schema`` is the schema (the fleet's existing
        structured-output idiom); openai = strict
        ``response_format=json_schema`` when ``spec.structured_outputs``,
        else a JSON-instruction suffix + fence/preamble-tolerant extraction
        (Think Tank pattern).
        """
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self._reject_reasoning_on_anthropic()

        is_pydantic = hasattr(schema, "model_json_schema")
        schema_dict = schema.model_json_schema() if is_pydantic else dict(schema)
        limit = self._effective_max_tokens(max_tokens)

        def _parse_and_validate(raw_data: Any):
            if is_pydantic:
                parsed = schema.model_validate(raw_data)
            else:
                if not isinstance(raw_data, dict):
                    raise ValueError(
                        f"structured output is not a JSON object: "
                        f"{type(raw_data).__name__}"
                    )
                parsed = raw_data
            if validate is not None:
                validate(parsed)
            return parsed

        if self.spec.transport == TRANSPORT_ANTHROPIC:
            return self._structured_anthropic(
                system=system,
                user_content=user_content,
                schema_dict=schema_dict,
                schema_name=schema_name,
                parse_and_validate=_parse_and_validate,
                is_pydantic=is_pydantic,
                attempts=attempts,
                max_tokens=limit,
            )
        if self.spec.transport == TRANSPORT_LITELLM:
            return self._structured_litellm(
                system=system,
                user_content=user_content,
                schema_dict=schema_dict,
                schema_name=schema_name,
                parse_and_validate=_parse_and_validate,
                is_pydantic=is_pydantic,
                attempts=attempts,
                max_tokens=limit,
            )
        return self._structured_openai(
            system=system,
            user_content=user_content,
            schema_dict=schema_dict,
            schema_name=schema_name,
            parse_and_validate=_parse_and_validate,
            is_pydantic=is_pydantic,
            attempts=attempts,
            max_tokens=limit,
        )

    def _structured_anthropic(
        self,
        *,
        system: str,
        user_content: str,
        schema_dict: dict,
        schema_name: str,
        parse_and_validate: Callable[[Any], Any],
        is_pydantic: bool,
        attempts: int,
        max_tokens: int,
    ) -> StructuredResult:
        tool = {
            "name": schema_name,
            "description": f"Emit the {schema_name} payload.",
            "input_schema": schema_dict,
        }
        base_payload = build_messages_payload(
            model=self.spec.model,
            system_prompt=system,
            user_content=user_content,
            max_tokens=max_tokens,
            cache_system=True,
            extra={
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": schema_name},
            },
        )
        messages = list(base_payload["messages"])
        usage = LLMUsage()
        last_error: Optional[Exception] = None
        client = self._transport_client()

        for attempt in range(attempts):
            payload = dict(base_payload)
            payload["messages"] = messages
            msg = client.messages.create(**payload)
            self._usage_from_anthropic(msg, into=usage)
            tool_input = self._extract_tool_input(msg, schema_name)
            # Same fault, Anthropic's spelling of it: the forced tool never
            # got emitted because the budget ran out first. Raised outside the
            # retry classification for the same reason as the openai path —
            # the next attempt re-issues the identical ask under the identical
            # ceiling.
            if tool_input is None and getattr(msg, "stop_reason", None) == "max_tokens":
                raise _budget_exhausted_error(
                    spec=self.spec,
                    max_tokens=max_tokens,
                    stop_signal="stop_reason='max_tokens'",
                    usage=usage,
                )
            try:
                if tool_input is None:
                    raise ValueError(
                        f"response contained no {schema_name!r} tool_use block"
                    )
                parsed = parse_and_validate(tool_input)
                return StructuredResult(
                    text="",
                    model=getattr(msg, "model", self.spec.model),
                    provider=self.spec.provider,
                    usage=usage,
                    raw_request=payload,
                    raw_response=msg,
                    data=parsed.model_dump() if is_pydantic else parsed,
                    parsed=parsed if is_pydantic else None,
                )
            except Exception as exc:  # noqa: BLE001 — re-raised loud on exhaustion
                last_error = exc
                logger.warning(
                    "llm structured provider=%s model=%s attempt=%d failed "
                    "validation: %s",
                    self.spec.provider,
                    self.spec.model,
                    attempt + 1,
                    exc,
                )
                messages = messages + [
                    {"role": "assistant", "content": msg.content},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed validation with: "
                            f"{exc}\nCall the {schema_name} tool again with a "
                            f"corrected payload."
                        ),
                    },
                ]

        raise LLMError(
            f"provider={self.spec.provider} model={self.spec.model}: "
            f"structured output failed validation after {attempts} "
            f"attempt(s): {last_error}",
            usage=usage,
        )

    def _structured_openai(
        self,
        *,
        system: str,
        user_content: str,
        schema_dict: dict,
        schema_name: str,
        parse_and_validate: Callable[[Any], Any],
        is_pydantic: bool,
        attempts: int,
        max_tokens: int,
    ) -> StructuredResult:
        messages: List[dict] = [{"role": "system", "content": system}]
        kwargs: dict = {"model": self.spec.model, "max_tokens": max_tokens}
        if self.spec.structured_outputs:
            messages.append({"role": "user", "content": user_content})
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema_dict,
                },
            }
        else:
            messages.append(
                {
                    "role": "user",
                    "content": user_content
                    + _JSON_INSTRUCTION.format(schema=json.dumps(schema_dict)),
                }
            )
        extra_body = self._openai_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body

        usage = LLMUsage()
        last_error: Any = None  # Exception (validation) or str (transport decode)
        raw_text = ""
        client = self._transport_client()

        for attempt in range(attempts):
            try:
                resp = client.chat.completions.create(messages=messages, **kwargs)
                self._usage_from_openai(resp, into=usage)
                raw_text = _choice_text(resp)
                # Deliberately OUTSIDE the retry classification below: a
                # budget exhausted before any content is not an attempt
                # failure, it is a certainty about every remaining attempt.
                _reject_budget_exhausted(
                    resp, raw_text, spec=self.spec,
                    max_tokens=max_tokens, usage=usage,
                )
            except (json.JSONDecodeError, NullChoicesError) as exc:
                # Two body-level transport failures on what the SDK treated as
                # a SUCCESSFUL transaction, both invisible to its own
                # ``max_retries`` (status/connection-based) because the
                # response was already considered final:
                #
                # - a non-JSON body (live incident 2026-07-20, krepis#38);
                # - a null/empty ``choices`` array carrying the provider's
                #   error in the body (OpenRouter's shape for an upstream
                #   failure) — previously escaped as a bare
                #   ``TypeError: 'NoneType' object is not subscriptable``.
                #
                # Both are ordinary bounded-retry attempt failures, not caller
                # crashes.
                # Cause-specific wording: "which body-level failure" is the
                # first thing an operator needs off the log line.
                kind = (
                    "a null-choices response body"
                    if isinstance(exc, NullChoicesError)
                    else "a non-JSON response body"
                )
                last_error = (
                    f"transport returned {kind} "
                    f"({exc.__class__.__name__}: {exc})"
                )
                logger.warning(
                    "llm structured provider=%s model=%s attempt=%d/%d: %s",
                    self.spec.provider,
                    self.spec.model,
                    attempt + 1,
                    attempts,
                    last_error,
                )
                if attempt < attempts - 1:
                    _retry_backoff_sleep(attempt)
                continue
            try:
                parsed = parse_and_validate(_extract_json(raw_text))
                return StructuredResult(
                    text=raw_text,
                    model=getattr(resp, "model", self.spec.model),
                    provider=self.spec.provider,
                    served_provider=getattr(resp, "provider", None),
                    usage=usage,
                    raw_request={"messages": messages, **kwargs},
                    raw_response=resp,
                    data=parsed.model_dump() if is_pydantic else parsed,
                    parsed=parsed if is_pydantic else None,
                )
            except Exception as exc:  # noqa: BLE001 — re-raised loud on exhaustion
                last_error = exc
                logger.warning(
                    "llm structured provider=%s model=%s attempt=%d failed "
                    "validation: %s",
                    self.spec.provider,
                    self.spec.model,
                    attempt + 1,
                    exc,
                )
                messages = messages + [
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed validation with: "
                            f"{exc}\nReturn ONLY the corrected JSON object."
                        ),
                    },
                ]

        raise LLMError(
            f"provider={self.spec.provider} model={self.spec.model}: "
            f"structured output failed validation after {attempts} "
            f"attempt(s): {last_error}",
            usage=usage,
        )

    def _structured_litellm(
        self,
        *,
        system: str,
        user_content: str,
        schema_dict: dict,
        schema_name: str,
        parse_and_validate: Callable[[Any], Any],
        is_pydantic: bool,
        attempts: int,
        max_tokens: int,
    ) -> StructuredResult:
        """Structured completion via LiteLLM Router with fallback chains."""
        from krepis.router import get_group_primary as _get_primary

        json_instruction = _JSON_INSTRUCTION.format(
            schema=json.dumps(schema_dict, indent=2)
        )
        usage = LLMUsage()
        last_error: Optional[str] = None
        fallback_used = False

        for attempt in range(1, attempts + 1):
            router = _get_router()
            prompt = (
                user_content + json_instruction
                if attempt == 1
                else user_content
                + f"\n\nPrevious attempt failed: {last_error}\n"
                + json_instruction
            )
            try:
                resp = router.completion(
                    model=self.spec.model,  # "low", "med", "high", "ultra"
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt == attempts:
                    raise LLMError(
                        f"litellm Router call failed after {attempts} "
                        f"attempt(s) — all models in group "
                        f"{self.spec.model!r} exhausted (primary → "
                        f"fallbacks all failed): {last_error}",
                        usage=usage,
                    ) from exc
                continue

            served_model = getattr(resp, "model", "")
            primary = _get_primary(self.spec.model)
            if primary and served_model and served_model != primary:
                fallback_used = True
            _check_fallback_transition(self.spec.model, fallback_used, served_model, primary or "")

            self._usage_from_openai(resp, into=usage)
            try:
                # _choice_text is inside the guarded block: a null/empty
                # ``choices`` body is a retryable provider failure here too,
                # not a bare TypeError escaping the loop. No explicit backoff
                # on this branch — the Router's fallback chain means the next
                # attempt goes to a DIFFERENT model, so sleeping first would
                # delay a call that is not hitting the unhealthy endpoint.
                raw_text = _choice_text(resp)
                # Raised INSIDE the guarded block here, unlike the openai
                # path: the Router's next attempt goes to a DIFFERENT model in
                # the group, which may well answer within the same budget, so
                # a budget exhaustion is a genuine attempt failure rather than
                # a certainty about the rest. What it must not be is
                # anonymous — caught below, its message becomes ``last_error``
                # and names the budget in the final LLMError.
                _reject_budget_exhausted(
                    resp, raw_text, spec=self.spec,
                    max_tokens=max_tokens, usage=usage,
                )
                parsed = self._extract_json(raw_text)
                validated = parse_and_validate(parsed)
            except Exception as exc:
                last_error = str(exc)
                continue
            return StructuredResult(
                text=raw_text,
                parsed=validated if is_pydantic else None,
                data=validated if not is_pydantic else None,
                model=served_model or self.spec.model,
                provider=self.spec.provider,
                usage=usage,
                raw_request={"model": self.spec.model, "system": system, "user_content": user_content},
                raw_response=resp,
                fallback_used=fallback_used,
                model_requested=self.spec.model,
            )

        raise LLMError(
            f"structured output failed validation after {attempts} "
            f"attempt(s): {last_error}",
            usage=usage,
        )

    @staticmethod
    def _extract_tool_input(msg: Any, tool_name: str) -> Optional[dict]:
        for block in getattr(msg, "content", None) or []:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == tool_name
            ):
                tool_input = getattr(block, "input", None)
                if isinstance(tool_input, dict):
                    return tool_input
        return None

    # ── grounded completion ───────────────────────────────────────────

    @_emits_cost
    def complete_grounded(
        self,
        *,
        system: str,
        user_content: str,
        search: SearchOptions,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
        attempts: int = 2,
    ) -> GroundedResult:
        """One web-search-grounded generation.

        Transport mapping:

        - **anthropic** — declares the server-side ``web_search`` tool
          (``max_uses`` capped per ``search.max_uses``);
          ``search.force_first`` forces the tool via ``tool_choice``.
          ``text`` is the post-final-tool text; ``searches`` carries one
          event per issued query; ``citations`` carries every returned URL.
        - **openrouter** — declares the ``openrouter:web_search`` server
          tool. ``citations`` comes from ``url_citation`` annotations;
          ``searches`` is EMPTY (the response does not expose queries) and
          ``usage.web_search_requests`` carries the billed search count.
          ``force_first`` is not supported and raises
          :exc:`~krepis.llm_config.LLMConfigError`. ``spec.reasoning``, if
          set, is forwarded into ``extra_body["reasoning"]`` — strongly
          recommended for reasoning-capable models (see
          :attr:`~krepis.llm_config.ModelSpec.reasoning`'s docstring for
          why: an unset default can return an empty ``text`` even at a
          generous ``max_tokens``).

          On this transport, each of the two known-transient OpenRouter
          failure classes below is retried up to ``attempts`` times
          (same provider/model, no caller involvement) before raising —
          live incidents on 2026-07-14, -16, and -20 each confirmed a
          bare retry of the SAME call succeeds immediately (the failure
          is stochastic sampling/gateway noise, not a persistent
          condition), so escalating straight to a caller's cross-provider
          fallback on the first occurrence wastes that fallback tier on
          what a retry would have resolved for free:

          1. **Unresolved tool call.** The model returned structured
             ``tool_calls``, ``finish_reason="tool_calls"``, or its own
             native tool-call token dialect leaked into ``content`` (e.g.
             Kimi K2's ``<|tool_calls_section_begin|>...``) instead of a
             final answer — the declared server-side tool was not
             honored for this model on this transport.
          2. **Non-JSON transport response.** The gateway returned a
             malformed/non-JSON body on what the SDK treated as a
             successful transaction — invisible to the SDK's own
             ``max_retries`` (status/connection-based) since parsing
             only fails after the response is already considered final.

          Raises :exc:`LLMError` — carrying the accumulated usage across
          all attempts — only once ``attempts`` is exhausted; per the
          class's own contract (see :class:`LLMError`), this signals a
          PERSISTENT failure and is the caller's cue to escalate to its
          own cross-provider fallback, not a first-occurrence signal.

        Any other openai-transport provider raises
        :exc:`~krepis.llm_config.LLMConfigError` — plain OpenAI-compatible
        endpoints have no server-side search; grounding there is the
        caller's job (fetch + inject context).
        """
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self._reject_reasoning_on_anthropic()
        limit = self._effective_max_tokens(max_tokens)

        if self.spec.transport == TRANSPORT_ANTHROPIC:
            extra: dict = {
                "tools": [build_web_search_tool(max_uses=search.max_uses)],
            }
            if search.force_first:
                extra["tool_choice"] = {"type": "tool", "name": "web_search"}
            payload = build_messages_payload(
                model=self.spec.model,
                system_prompt=system,
                user_content=user_content,
                max_tokens=limit,
                cache_system=cache_system,
                extra=extra,
            )
            msg = self._transport_client().messages.create(**payload)
            return GroundedResult(
                text=final_text_after_last_tool(getattr(msg, "content", [])),
                model=getattr(msg, "model", self.spec.model),
                provider=self.spec.provider,
                usage=self._usage_from_anthropic(msg),
                raw_request=payload,
                raw_response=msg,
                searches=extract_anthropic_search_events(msg),
                citations=extract_anthropic_citations(msg),
            )

        if not self._is_openrouter():
            raise LLMConfigError(
                f"complete_grounded is only supported on the anthropic "
                f"provider (server-side web_search tool) or openrouter "
                f"(openrouter:web_search server tool); provider "
                f"{self.spec.provider!r} has neither. Ground the call "
                f"yourself (fetch + inject) or flip the model spec."
            )
        if search.force_first:
            raise LLMConfigError(
                "SearchOptions.force_first is not supported on the "
                "openrouter transport — the openrouter:web_search server "
                "tool cannot be forced via tool_choice. Use a prose "
                "directive plus a citation-count floor instead."
            )

        tool_params: dict = {}
        if search.engine:
            tool_params["engine"] = search.engine
        if search.max_results is not None:
            tool_params["max_results"] = search.max_results
        web_tool: dict = {"type": "openrouter:web_search"}
        if tool_params:
            web_tool["parameters"] = tool_params

        extra_body: dict = {"usage": {"include": True}, "tools": [web_tool]}
        if self.spec.reasoning is not None:
            extra_body["reasoning"] = self.spec.reasoning
        kwargs: dict = {
            "model": self.spec.model,
            "max_tokens": limit,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "extra_body": extra_body,
        }
        usage = LLMUsage()
        last_error: Optional[str] = None
        for attempt in range(attempts):
            try:
                resp = self._transport_client().chat.completions.create(**kwargs)
                self._usage_from_openai(resp, into=usage)
                choice = _first_choice(resp)
            except (json.JSONDecodeError, NullChoicesError) as exc:
                # Two body-level failures on what the SDK treated as a
                # successful transaction, both invisible to its own
                # ``max_retries`` because the response was already considered
                # final: a non-JSON body (live incident 2026-07-20, krepis#38)
                # and a null/empty ``choices`` carrying the provider's error in
                # the body. Neither is a ``RuntimeError``/``LLMError``
                # subclass, so unclassified they crash the caller instead of
                # engaging its cross-provider fallback.
                # Cause-specific wording: "which body-level failure" is the
                # first thing an operator needs off the log line.
                kind = (
                    "a null-choices response body"
                    if isinstance(exc, NullChoicesError)
                    else "a non-JSON response body"
                )
                last_error = (
                    f"transport returned {kind} "
                    f"({exc.__class__.__name__}: {exc})"
                )
                logger.warning(
                    "llm complete_grounded provider=%s model=%s "
                    "attempt=%d/%d: %s",
                    self.spec.provider,
                    self.spec.model,
                    attempt + 1,
                    attempts,
                    last_error,
                )
                if attempt < attempts - 1:
                    _retry_backoff_sleep(attempt)
                continue

            text = (choice.message.content or "").strip()

            # A declared server-side tool (``openrouter:web_search``) is
            # meant to be resolved by the gateway before the response
            # reaches us — if the model instead requested it as a
            # client-side tool call that never got executed (structured
            # ``tool_calls`` present, or a ``finish_reason`` of
            # ``"tool_calls"``), OR its own native tool-call token dialect
            # leaked as literal text into ``content``, this is NOT a
            # usable grounded answer. Retry the same call (live incidents
            # 2026-07-14/-16/-20 each confirmed this resolves on a bare
            # retry — stochastic sampling noise, not a persistent
            # condition) before raising loud so the caller's
            # cross-provider fallback engages only on a genuinely
            # persistent failure — live incident 2026-07-14: a 283-char
            # "script" consisting of ``<|tool_calls_section_begin|>...``
            # shipped as a live episode.
            unresolved_tool_call = getattr(choice.message, "tool_calls", None)
            finish_reason = getattr(choice, "finish_reason", None)
            leak_match = _CONTROL_TOKEN_RE.search(text)
            if unresolved_tool_call or finish_reason == "tool_calls" or leak_match:
                if leak_match:
                    last_error = (
                        f"grounded response leaked raw control-token syntax "
                        f"into content ({leak_match.group()!r}) — almost "
                        f"certainly an unresolved/malformed tool call, not "
                        f"usable text."
                    )
                else:
                    last_error = (
                        f"grounded call returned an unresolved tool call "
                        f"instead of a final answer "
                        f"(finish_reason={finish_reason!r}) — the "
                        f"server-side web_search tool was not honored for "
                        f"this model on this transport."
                    )
                logger.warning(
                    "llm complete_grounded provider=%s model=%s "
                    "attempt=%d/%d: %s",
                    self.spec.provider,
                    self.spec.model,
                    attempt + 1,
                    attempts,
                    last_error,
                )
                continue

            return GroundedResult(
                text=text,
                model=getattr(resp, "model", self.spec.model),
                provider=self.spec.provider,
                usage=usage,
                raw_request=kwargs,
                raw_response=resp,
                searches=[],
                citations=extract_openrouter_citations(resp),
            )

        raise LLMError(
            f"provider={self.spec.provider} model={self.spec.model}: "
            f"{last_error} Exhausted {attempts} attempt(s) — this is a "
            f"bounded same-provider retry for known-transient OpenRouter "
            f"failure classes (control-token leak / unresolved tool call "
            f"/ malformed transport response); a persistent failure here "
            f"is the caller's cue to escalate to its own cross-provider "
            f"fallback.",
            usage=usage,
        )


# ── JSON extraction (Think Tank lift) ─────────────────────────────────────


def _extract_json(text: str) -> Any:
    """Parse a JSON object out of model text (tolerates markdown fences).

    Lifted from ``thinktank/client.py`` — the fallback parser for models
    without strict structured-output support that add fences or a preamble
    sentence around the JSON body.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
