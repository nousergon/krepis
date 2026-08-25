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
import queue
import random as _random
import re
import threading
import time as _time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

from krepis.anthropic_payload import (
    build_messages_payload,
    build_web_search_tool,
)
from krepis.llm_config import (
    ROUTER_EDGE_PROVIDER,
    TRANSPORT_ANTHROPIC,
    LLMConfigError,
    ModelSpec,
)
from krepis.router import served_model_for_deployment
# Only the prefix constant, so the router-edge failure message can name the
# SSM parameter an operator has to look at. `krepis.secrets` imports boto3
# lazily, so this costs nothing at import time.
from krepis.secrets import SSM_PREFIX as _SSM_PREFIX
from krepis.llm_search import (
    Citation,
    SearchEvent,
    extract_anthropic_citations,
    extract_anthropic_search_events,
    extract_openrouter_citations,
    final_text_after_last_tool,
)
from krepis.session_dlp import (
    check_request,
    dlp_enabled,
)

logger = logging.getLogger(__name__)

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


def _resolve_group_served_model(resp: Any, *, spec: Any) -> str:
    """Return the real model that served a group-addressed call, or raise.

    ``spec.model`` on a group-addressed spec is a synthetic alias ("low" /
    "med" / "high" / "ultra") that never appears in ``krepis.cost``'s price
    cards and is never itself a billable model. When the router response's
    ``model`` field comes back empty, or (for reasons not yet root-caused —
    alpha-engine-config-I6543) equal to the alias itself, the previous code
    silently substituted the alias as the served model. That value then
    flowed into ``krepis.cost.load_default_pricing().get(served_model, ...)``
    and, for THIS consumer, failed there — but the defect is here: an alias
    is not a served model, in a Lambda whose cost lookup happens to succeed
    to find a coincidentally-matching card, or logged to a manifest as the
    served model, either would be silently wrong rather than loudly wrong.

    Raising here surfaces the failure at its source with the diagnostic
    payload (``_hidden_params``, when the transport is litellm's own Router
    object) needed to root-cause why the field was unusable, instead of at
    a downstream consumer with none of that context.

    A served model of the qualified ``{group}-{mid}`` form — the deployment
    naming the registry derivation produces as of 0.39.0 — is resolved
    through the registry to the entry's upstream model identifier, so what
    this returns is always a real, priceable model id, never a derived
    deployment name. An unresolvable ``{group}-``-prefixed name raises:
    it means the router served a deployment this consumer's registry does
    not know (registry drift), and letting it flow would only move the
    failure into the price-card lookup with less context.

    **A response echoing a QUALIFIED DEPLOYMENT name is not masquerade.**
    Since ``router.py``'s litellm_proxy route began emitting
    ``deployment_id = _qualified_primary`` (config-I6543), a proxy-routed
    consumer addresses ``med-deepseek-v4-flash-max`` on the wire, not the
    bare group — and LiteLLM stamps the model AS ADDRESSED back onto the
    response, so ``served_model == spec.model`` on every healthy call. The
    addressing half of that fix shipped without the accounting half: this
    function still assumed ``spec.model`` was always a bare group, so it
    rejected every successful call. Measured cost: the Think Tank
    challenger arm aborted with ``theses_written: 0`` and wrote no
    challenger selection (2026-08-10 run ``b150c317eeef``; the arm's last
    valid selection is 2026-07-31).

    So when the echoed name resolves through the registry as a derived
    deployment, this returns its upstream model — a real, priceable id, and
    exactly what the caller asked to be served. Genuine masquerade still
    raises: a bare group name ("med") is not a derived deployment name, so
    it does not resolve and falls through to the error below.
    """
    served_model = getattr(resp, "model", "") or ""
    if served_model and served_model == spec.model and "-" in served_model:
        # Deployment-addressed call: the caller named a concrete deployment
        # and the router served it. Resolve to the billable upstream id.
        # The REGISTRY licenses the pass — an echoed name that does not
        # resolve is indistinguishable from the alias masquerade this
        # function exists to reject, and still falls through to the raise.
        # The ``"-" in`` precondition is what keeps a bare group name off
        # the registry path entirely: groups are "low"/"med"/"high"/"ultra",
        # and asking the registry about one turned the precise masquerade
        # error into a FileNotFoundError wherever no registry is on disk
        # (caught by this repo's own CI, which has none). An unreadable
        # registry still propagates for a ``{group}-{mid}``-shaped name,
        # exactly as the != branch below already lets it: a consumer
        # holding such a name resolved it through a registry to begin with.
        upstream = served_model_for_deployment(served_model)
        if upstream:
            return upstream
    if served_model and served_model != spec.model:
        if served_model.startswith(f"{spec.model}-"):
            upstream = served_model_for_deployment(served_model)
            if upstream:
                return upstream
            raise LLMConfigError(
                f"provider={spec.provider!r} group={spec.model!r}: the router "
                f"reported served model {served_model!r}, which is shaped like "
                f"a derived deployment name for this group but does not "
                f"resolve through the local LLM_MODEL_REGISTRY.yaml. The "
                f"router and this consumer are reading different registries "
                f"(alpha-engine-config-I6543)."
            )
        return served_model
    hidden = getattr(resp, "_hidden_params", None)
    raise LLMConfigError(
        f"provider={spec.provider!r} group={spec.model!r}: the router "
        f"response did not report a served model distinct from the group "
        f"alias (model field was {served_model!r}). Refusing to bill or "
        f"record the call under the alias — it is not a real model and "
        f"carries no price card (alpha-engine-config-I6543). "
        f"resp._hidden_params={hidden!r}"
    )


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

# ── structured-output degradation ladder (model-portability-policy §7) ────
#
# The policy declares one ladder for structured output:
#
#   native (strict ``response_format=json_schema``)
#     → tool_emulation (forced single-tool call whose input IS the object)
#     → prompt_only (instruct + tolerant extraction + bounded repair)
#
# "The caller always receives a validated object or an exception. Which rung
# ran is recorded on the result, because rung affects reliability and belongs
# in the artifact." ``StructuredResult.structured_output_rung`` is that record,
# and it is populated on EVERY structured call — including the undegraded ones,
# so a healthy call publishes ``native``/``tool_emulation`` rather than nothing
# (``principles.md`` §2.7: a component emitting nothing is not healthy, it is
# unobserved).
_STRUCTURED_RUNG_NATIVE = "native"
_STRUCTURED_RUNG_TOOL_EMULATION = "tool_emulation"
_STRUCTURED_RUNG_PROMPT_ONLY = "prompt_only"

# A provider REFUSING ``response_format`` at request time — distinct from a
# registry model that never declared support (I4 handles that declaratively,
# via ``spec.structured_outputs``). This is the case the declaration cannot
# cover: the registry says the deployment can serve it, or a caller overrode
# ``params.structured_outputs``, and the endpoint answers 400.
#
# Live incident (alpha-engine-config-I7232): the Think Tank's ``sweep`` tier
# overrides ``structured_outputs=True`` onto a DeepSeek deployment whose
# registry params default it False; DeepSeek answered
# ``400 This response_format type is unavailable now``. The exception escaped
# ``_structured_openai`` entirely — it is neither a decode failure nor a
# validation failure — and aborted every Think Tank run from 2026-08-11.
# Descending one rung is the DECLARED path for exactly this, and it is what
# turns a whole-run abort into a recorded degradation.
#
# Deliberately matched on the refusal WORDING and not on provider identity or
# an HTTP status: I4 forbids branching request construction on provider name,
# and the status alone (400) cannot distinguish "this parameter is unavailable"
# from "your schema is malformed" — descending on the latter would hide a real
# caller bug behind a weaker rung.
_RESPONSE_FORMAT_REFUSAL_RE = re.compile(
    r"response[_ ]?format",
    re.IGNORECASE,
)
_RESPONSE_FORMAT_REFUSAL_CAUSE_RE = re.compile(
    r"unavailable|unsupported|not supported|not available|not enabled",
    re.IGNORECASE,
)


def _is_response_format_refusal(exc: BaseException) -> bool:
    """True when *exc* is a provider refusing ``response_format`` itself.

    Both halves must match: the message has to name the parameter AND say it
    cannot be served. A 400 naming ``response_format`` because the SCHEMA was
    rejected is a caller defect and must keep raising — degrading it would
    swap a loud, correct failure for a quietly weaker rung.
    """
    text = str(exc)
    return bool(
        _RESPONSE_FORMAT_REFUSAL_RE.search(text)
        and _RESPONSE_FORMAT_REFUSAL_CAUSE_RE.search(text)
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


def _choice_text(resp: Any, *, caller_raises_on_empty: bool = False) -> str:
    """First choice's message content, stripped. Raises on null choices.

    Logs the diagnostics when the content is empty. The emptiness itself is not
    an error here — callers classify it — but it is invisible without this
    line, and the caller-facing symptom actively misdirects: a structured
    caller reports ``no JSON object found in response: ''``, which reads as a
    model that answered in prose. Instrumented at THIS chokepoint rather than
    at the structured paths, for the same reason ``_first_choice`` is: a guard
    applied at four of five call sites is not a guard.

    ``caller_raises_on_empty`` sets the LEVEL, and only the level — the line is
    emitted either way. This function's own docstring says the emptiness is not
    an error and that callers classify it, so logging it at ERROR
    unconditionally contradicts that: on the structured path the caller raises
    with the SAME diagnostics microseconds later, so an ERROR here is a second
    report of one event. Alert handlers attach at ERROR
    (``krepis.logging.setup_logging``), so that duplication reached the on-call
    human: one Think Tank abort on 2026-08-11 produced three separate ERROR
    dispatches for a single failed call (alpha-engine-config-I6921 D3).

    The default stays ERROR, deliberately. On the plain-completion path
    (:meth:`LLMClient.complete`) an empty string is RETURNED to the caller and
    nothing raises — there this line is the only signal that anything happened,
    and demoting it fleet-wide to buy quiet on the structured path would trade
    a duplicate alert for a missing one.
    """
    choice = _first_choice(resp)
    text = (getattr(choice.message, "content", None) or "").strip()
    if not text:
        logger.log(
            logging.WARNING if caller_raises_on_empty else logging.ERROR,
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

    **Never retried under the SAME budget.** A second attempt on the identical
    ceiling cannot succeed: it re-issues the identical ask against the
    identical bound. Measured on the Director's weekly call — two attempts,
    ~100s of generation each, both fully billed, both guaranteed to fail
    before the first one returned. Retrying *there* does not merely fail to
    inform, it doubles the cost of a certain failure.

    A retry under an ESCALATED budget is a different request, and it is the
    one thing that can succeed. :meth:`LLMClient.structured` performs exactly
    one, doubling the ceiling — see
    :meth:`LLMClient._escalated_budget` for why a larger constant cannot
    replace it.

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


#: Factor by which an exhausted ``max_tokens`` ceiling is raised for the ONE
#: escalated retry :meth:`LLMClient.structured` performs. 2.0 — a doubling.
#:
#: WHY A FACTOR AND NOT A BIGGER CONSTANT. The reasoning draw of a
#: reasoning-effort model is free-running: measured on `glm-5.2` against an
#: identical trivial prompt it ranged 11..163 tokens across eight calls, a 15x
#: spread (alpha-engine-config-I6858). No single ceiling is therefore "large
#: enough" — it is only large enough *so far*, which is precisely how six
#: instances of this class reached production between 2026-08-04 and
#: 2026-08-25, each one remediated by raising a per-caller literal that the
#: next outlier then cleared.
#:
#: Worse, the distribution cannot be measured out of the problem. The recorded
#: ``reasoning_tokens`` on a SUCCESSFUL call is censored at the ceiling by
#: construction: the draws that would set the tail are exactly the ones the
#: ceiling truncates, and truncated calls return no usable sample. Measured
#: 2026-08-25 over 72 successful `thinktank-thesis` calls: p50 2213, p95 6033,
#: max 7514 — and that same day the tier drew >=16000 and aborted the run. A
#: p95-derived floor would have sat at 6033 and prevented nothing.
#:
#: So the ceiling stops being a cliff instead of being guessed higher: when
#: the budget is provably the binding constraint (empty content,
#: ``finish_reason='length'``), re-issue once with twice the room. Bounded at
#: ONE escalation — a second exhaustion at double the budget is a pathological
#: ask, and paging on it is correct.
_BUDGET_ESCALATION_FACTOR = 2.0


def _absorb_usage(into: "LLMUsage", other: "Optional[LLMUsage]") -> None:
    """Add *other*'s counters into *into*, in place.

    The spend of an attempt that RAISED is real spend. ``_emit_cost_record``
    runs off the returned result only, so without this the tokens burned by an
    exhausted attempt are absent from the cost record entirely and every
    budget guard reading it sits low by that much.
    """
    if other is None:
        return
    for name in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_create_tokens",
        "cache_create_1h_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
        "web_search_requests",
        "web_fetch_requests",
        "budget_escalations",
        "attempts",
    ):
        setattr(into, name, getattr(into, name) + getattr(other, name, 0))
    # A MAX, never a sum — see LLMUsage.reasoning_tokens_max_attempt. Adding
    # it here would make an escalated call report a per-attempt draw no single
    # attempt made, which is precisely the confusion this field exists to end.
    into.reasoning_tokens_max_attempt = max(
        into.reasoning_tokens_max_attempt,
        getattr(other, "reasoning_tokens_max_attempt", 0),
    )
    if other.provider_cost_usd is not None:
        into.provider_cost_usd = (into.provider_cost_usd or 0.0) + other.provider_cost_usd
    # OR'd, not summed — it is a boolean and it is STICKY. An exhausted
    # attempt whose usage never arrived (a streamed call on a route that
    # dropped its usage chunk) is absorbed into the escalated attempt's
    # counters; if the flag did not travel with them, the merged total would
    # be priced as though every token in it had been reported, and the
    # understatement would arrive wearing a complete-looking cost row
    # (alpha-engine-config-I8164).
    if getattr(other, "usage_unknown", False):
        into.usage_unknown = True


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


# ── streaming ─────────────────────────────────────────────────────────────

#: Default inter-chunk idle budget for a streamed call, in seconds.
#:
#: A streamed call replaces "how long may the whole generation take" with "how
#: long may the model be SILENT". The first is a wall the generation runs into
#: — every Director plan failure this month was one, and each discarded a
#: partial generation that was probably still in progress
#: (alpha-engine-config-I8164). The second bounds the only condition a client
#: can actually diagnose from the outside: a route that has stopped producing.
#: A genuinely hung call is therefore detected FASTER than a merely slow one is
#: under a total-duration deadline, which is the inversion this exists for.
DEFAULT_STREAM_IDLE_TIMEOUT_S = 90.0


class StreamingUnsupportedError(LLMConfigError):
    """``stream=True`` on a route whose registry entry does not declare it.

    A loud, named condition — never a quiet non-streaming call. Silently
    honouring the request without the streaming semantics would hand back the
    exact failure mode streaming was asked for to remove, wearing a result
    shape that says it worked (``model-portability-policy`` §I9).

    The fix is a registry declaration (``capabilities.streaming: true`` on the
    entry, which is a MEASURED claim about the deployment), not a client flag.
    """


class StreamIdleTimeoutError(LLMError):
    """A stream went silent for longer than its inter-chunk idle budget.

    Carries the evidence the non-streaming deadline could never carry: the
    partial generation, how many chunks arrived, the last ``finish_reason``
    seen (usually ``None`` — that is the point), and the usage accumulated so
    far. A failure that says "600 seconds elapsed" is indistinguishable from a
    hundred other faults; one that says "1,842 characters arrived across 96
    chunks and then nothing for 90 seconds" names its own cause.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: "Optional[LLMUsage]" = None,
        partial_text: str = "",
        chunks: int = 0,
        idle_timeout: float = 0.0,
        elapsed: float = 0.0,
        finish_reason: Optional[str] = None,
    ):
        super().__init__(message, usage=usage)
        self.partial_text = partial_text
        self.chunks = chunks
        self.idle_timeout = idle_timeout
        self.elapsed = elapsed
        self.finish_reason = finish_reason


_STREAM_ITEM = "item"
_STREAM_ERROR = "error"
_STREAM_DONE = "done"


class _StreamIdle(Exception):
    """Internal signal: the pump produced nothing within the idle budget."""


def _close_stream(stream: Any) -> None:
    """Best-effort release of an abandoned stream's underlying connection.

    Bounded swallow, per the fail-loud rule: (a) the failure mode swallowed is
    a transport ``close()`` raising while we are already unwinding a stream
    abort; (b) the primary deliverable — the abort itself, with its partial
    generation — is unaffected, and letting a close error replace the idle
    timeout would hide the diagnosis behind the cleanup; (c) recorded on this
    DEBUG line and, if the socket really leaked, in the pump thread's own
    eventual read failure.
    """
    closer = getattr(stream, "close", None)
    if closer is None:
        return
    try:
        closer()
    except Exception as exc:  # noqa: BLE001 — see the three-part rationale above
        logger.debug("stream close raised while aborting: %s", exc)


def _iter_with_idle_timeout(stream: Any, *, idle_timeout: float):
    """Yield items from *stream*, bounding the SILENCE BETWEEN them.

    A blocking ``next()`` on a synchronous provider stream cannot be
    interrupted from the outside, so the iteration runs on a daemon worker and
    the consumer waits on a queue with a deadline. That makes the idle budget a
    property of THIS client rather than of whichever SDK, HTTP library or proxy
    happens to sit underneath — the transport's own read timeout still applies
    beneath it, and whichever is smaller binds first.

    The worker is a daemon and is not joined: after an idle abort the caller
    closes the stream (see :func:`_close_stream`), the pending read fails, and
    the thread exits on its own. Joining it would reintroduce exactly the
    unbounded wait this function exists to bound.
    """
    q: "queue.Queue" = queue.Queue()

    def _pump() -> None:
        try:
            for item in stream:
                q.put((_STREAM_ITEM, item))
        except BaseException as exc:  # noqa: BLE001 — forwarded verbatim
            q.put((_STREAM_ERROR, exc))
        finally:
            q.put((_STREAM_DONE, None))

    threading.Thread(
        target=_pump, name="krepis-llm-stream", daemon=True
    ).start()

    while True:
        try:
            kind, payload = q.get(timeout=idle_timeout)
        except queue.Empty:
            raise _StreamIdle() from None
        if kind == _STREAM_ITEM:
            yield payload
        elif kind == _STREAM_ERROR:
            raise payload
        else:
            return


@dataclass
class _StreamedChoice:
    """``ChatCompletion.choices[0]``-shaped view of an accumulated stream."""

    message: Any
    finish_reason: Optional[str] = None
    index: int = 0


class _StreamedChatCompletion:
    """An OpenAI-wire stream, accumulated back into the non-streaming shape.

    The whole point: every helper downstream of the transport call —
    :func:`_choice_text`, :func:`_reject_budget_exhausted`,
    :func:`_resolve_group_served_model`, :meth:`LLMClient._usage_from_openai`
    — reads a ``ChatCompletion``. Re-synthesizing that object means a streamed
    call takes the identical path afterwards, so ``stream=True`` cannot quietly
    return a different result shape, a different degradation ladder, or a
    different set of guards. The alternative — a parallel post-processing path
    for streamed responses — is two implementations of one contract, and the
    second one is the one nobody re-reads.

    ``krepis_*`` attributes are the streaming-only facts that have no place on
    a ``ChatCompletion``; :func:`_finalize_result` stamps them onto the result.
    """

    def __init__(
        self,
        *,
        text: str,
        finish_reason: Optional[str],
        model: Optional[str],
        usage: Any,
        served_provider: Optional[str],
        chunks: int,
        usage_reported: bool,
    ) -> None:
        self.choices = [
            _StreamedChoice(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ]
        self.model = model
        self.usage = usage
        if served_provider is not None:
            self.provider = served_provider
        self.krepis_streamed = True
        self.krepis_stream_chunks = chunks
        self.krepis_usage_reported = usage_reported


class _StreamedAnthropicUsage:
    """``message_start`` usage with ``message_delta``'s final output count.

    Anthropic reports the two halves in two events, and the second is
    CUMULATIVE rather than incremental — summing them double-counts the
    output. Merging by delegation rather than by copying fields keeps every
    provider extension ``_usage_from_anthropic`` reads (DeepSeek's
    Anthropic-compatible cache fields, ``server_tool_use``) reachable without
    this class having to enumerate them.
    """

    def __init__(self, base: Any, *, output_tokens: int) -> None:
        self._base = base
        self.output_tokens = output_tokens

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("_base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)


class _StreamedMessage:
    """An Anthropic event stream accumulated back into ``Message`` shape.

    Same contract as :class:`_StreamedChatCompletion` on the other wire: the
    text-block join in :meth:`LLMClient.complete` and
    :meth:`LLMClient._extract_tool_input` on the structured path both read a
    ``Message``, and a streamed call must not get its own copy of either.
    """

    def __init__(
        self,
        *,
        content: List[Any],
        model: Optional[str],
        usage: Any,
        stop_reason: Optional[str],
        chunks: int,
        usage_reported: bool,
    ) -> None:
        self.content = content
        self.model = model
        self.usage = usage
        self.stop_reason = stop_reason
        self.krepis_streamed = True
        self.krepis_stream_chunks = chunks
        self.krepis_usage_reported = usage_reported


def _accumulate_openai_stream(
    stream: Any, *, idle_timeout: float, spec: "ModelSpec"
) -> _StreamedChatCompletion:
    """Drain an OpenAI-wire stream into a ``ChatCompletion``-shaped object."""
    parts: List[str] = []
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    served_provider: Optional[str] = None
    usage_obj: Any = None
    chunks = 0
    started = _time.monotonic()
    try:
        for chunk in _iter_with_idle_timeout(stream, idle_timeout=idle_timeout):
            chunks += 1
            model = getattr(chunk, "model", None) or model
            served_provider = getattr(chunk, "provider", None) or served_provider
            # The usage-bearing final chunk carries an EMPTY ``choices`` on
            # every OpenAI-compatible route, so this must precede any choice
            # read rather than living inside one.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage_obj = chunk_usage
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    parts.append(piece)
                reason = getattr(choice, "finish_reason", None)
                if reason:
                    finish_reason = reason
    except _StreamIdle:
        _close_stream(stream)
        elapsed = _time.monotonic() - started
        partial = "".join(parts)
        raise StreamIdleTimeoutError(
            f"provider={spec.provider} model={spec.model}: the stream produced "
            f"no chunk for {idle_timeout:.0f}s after {chunks} chunk(s) and "
            f"{len(partial)} character(s) over {elapsed:.0f}s — aborting on the "
            f"inter-chunk idle budget, not on total duration. The partial "
            f"generation is on this exception as ``partial_text``.",
            partial_text=partial,
            chunks=chunks,
            idle_timeout=idle_timeout,
            elapsed=elapsed,
        ) from None
    return _StreamedChatCompletion(
        text="".join(parts),
        finish_reason=finish_reason,
        model=model or spec.model,
        usage=usage_obj,
        served_provider=served_provider,
        chunks=chunks,
        usage_reported=usage_obj is not None,
    )


def _accumulate_anthropic_stream(
    stream: Any, *, idle_timeout: float, spec: "ModelSpec"
) -> _StreamedMessage:
    """Drain an Anthropic event stream into a ``Message``-shaped object.

    Text arrives as ``text_delta``; a forced-tool structured call arrives as
    ``input_json_delta`` fragments that only become a JSON object once the
    stream ends — which is why the block is assembled here and parsed at the
    end rather than incrementally.
    """
    blocks: dict[int, dict] = {}
    order: List[int] = []
    model: Optional[str] = None
    stop_reason: Optional[str] = None
    start_usage: Any = None
    output_tokens: Optional[int] = None
    chunks = 0
    started = _time.monotonic()

    def _partial_text() -> str:
        return "".join(
            "".join(blocks[i]["parts"]) for i in order if blocks[i]["type"] == "text"
        )

    try:
        for event in _iter_with_idle_timeout(stream, idle_timeout=idle_timeout):
            chunks += 1
            etype = getattr(event, "type", None)
            if etype == "message_start":
                message = getattr(event, "message", None)
                model = getattr(message, "model", None) or model
                start_usage = getattr(message, "usage", None)
            elif etype == "content_block_start":
                index = int(getattr(event, "index", 0) or 0)
                block = getattr(event, "content_block", None)
                if index not in blocks:
                    order.append(index)
                blocks[index] = {
                    "type": getattr(block, "type", "text"),
                    "name": getattr(block, "name", None),
                    "id": getattr(block, "id", None),
                    "parts": [],
                }
            elif etype == "content_block_delta":
                index = int(getattr(event, "index", 0) or 0)
                if index not in blocks:
                    order.append(index)
                    blocks[index] = {
                        "type": "text", "name": None, "id": None, "parts": [],
                    }
                delta = getattr(event, "delta", None)
                piece = (
                    getattr(delta, "text", None)
                    or getattr(delta, "partial_json", None)
                )
                if piece:
                    blocks[index]["parts"].append(piece)
            elif etype == "message_delta":
                stop_reason = (
                    getattr(getattr(event, "delta", None), "stop_reason", None)
                    or stop_reason
                )
                delta_usage = getattr(event, "usage", None)
                if delta_usage is not None:
                    reported = getattr(delta_usage, "output_tokens", None)
                    if reported is not None:
                        output_tokens = int(reported)
    except _StreamIdle:
        _close_stream(stream)
        elapsed = _time.monotonic() - started
        partial = _partial_text()
        raise StreamIdleTimeoutError(
            f"provider={spec.provider} model={spec.model}: the stream produced "
            f"no event for {idle_timeout:.0f}s after {chunks} event(s) and "
            f"{len(partial)} character(s) over {elapsed:.0f}s — aborting on the "
            f"inter-chunk idle budget, not on total duration. The partial "
            f"generation is on this exception as ``partial_text``.",
            partial_text=partial,
            chunks=chunks,
            idle_timeout=idle_timeout,
            elapsed=elapsed,
        ) from None

    content: List[Any] = []
    for index in order:
        block = blocks[index]
        joined = "".join(block["parts"])
        if block["type"] == "tool_use":
            try:
                tool_input = json.loads(joined) if joined.strip() else {}
            except json.JSONDecodeError as exc:
                # NOT swallowed into an empty result: the caller's structured
                # path classifies a missing tool input and retries, and a
                # silently-empty dict would read as "the model returned {}".
                logger.error(
                    "llm stream: tool_use block %r did not assemble into JSON "
                    "(%s) — %d character(s) accumulated",
                    block["name"], exc, len(joined),
                )
                tool_input = None
            content.append(
                SimpleNamespace(
                    type="tool_use",
                    name=block["name"],
                    id=block["id"],
                    input=tool_input,
                )
            )
        else:
            content.append(SimpleNamespace(type=block["type"], text=joined))

    usage_reported = start_usage is not None and output_tokens is not None
    usage = (
        _StreamedAnthropicUsage(start_usage, output_tokens=output_tokens or 0)
        if start_usage is not None
        else None
    )
    return _StreamedMessage(
        content=content,
        model=model or spec.model,
        usage=usage,
        stop_reason=stop_reason,
        chunks=chunks,
        usage_reported=usage_reported,
    )


def _finish_reason_of(resp: Any) -> Optional[str]:
    """The provider's stop signal, on either wire, or ``None``.

    Surfaced on every :class:`LLMResult` — streamed or not. It is the field
    that separates "the model finished" from "the budget ran out" from "the
    stream died", and until now it was read only inside the error paths, so a
    call that succeeded left no record of how it ended.
    """
    stop_reason = getattr(resp, "stop_reason", None)
    if stop_reason:
        return str(stop_reason)
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return None
    reason = getattr(choices[0], "finish_reason", None)
    return str(reason) if reason else None

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
    # reasoning_tokens — the share of ``output_tokens`` the model spent on its
    # chain of thought, where the provider reports it (0 = not reported, which
    # on a non-reasoning model is also the true value).
    #
    # WHY THIS FIELD EXISTS. On a reasoning model ``max_tokens`` bounds
    # reasoning AND content together, so a budget sized to the expected ANSWER
    # yields no answer at all — a fully-billed response with
    # ``finish_reason=length`` and ``content=''``. That failure has now
    # occurred three times in eight days (alpha-engine-config#6396 the
    # Director, I6893 Think Tank's ``pillar`` tier aborting a daily run with
    # zero theses, I6858 ``router-canary`` paging intermittently), and every
    # remediation so far has been a GUESS, because the quantity a budget must
    # clear was recorded nowhere.
    #
    # It was visible only in the error path: ``_budget_exhausted_error`` reads
    # ``reasoning_tokens`` off the response when a call comes back empty. So
    # the draw was observable exactly once per outage and never on a healthy
    # call — an unobserved quantity, not a healthy one (principles.md §2.7).
    # Recording it on every call is what makes a measured floor possible at
    # all; sizing rules are alpha-engine-config-I6901 and are deliberately NOT
    # in this change, because the two candidate rules both fail against
    # measurement today (see that issue).
    reasoning_tokens: int = 0
    # budget_escalations — how many times this logical call had its
    # ``max_tokens`` ceiling raised and re-issued after the budget was
    # exhausted before any content was produced. 0 on every healthy call.
    #
    # WHY THIS FIELD EXISTS. The escalation is what stops the exhaustion class
    # being an outage, and an escalation nobody can count is indistinguishable
    # from a healthy call: the run succeeds, the spend is ordinary, and the
    # only trace is a WARNING in a log that dies with the box. Counted here it
    # reaches the persisted cost record, so a call site whose base ceiling is
    # chronically undersized shows up as a NUMBER before it shows up as the
    # next aborted run (alpha-engine-config-I6917 deliverable 3).
    budget_escalations: int = 0
    # attempts — transport calls this logical call made (initial + corrective
    # retries + a rung descent + an escalated re-issue). 1 on a clean call.
    #
    # WHY THIS FIELD EXISTS. Every other counter here is a SUM over attempts,
    # and one number is emitted per logical call — so without an attempt count
    # nothing downstream can tell a 1-attempt row from a 4-attempt one, and
    # every per-call figure is uninterpretable. It was read as a per-attempt
    # draw on 2026-08-25 and used to size a ceiling; the resulting claim had
    # to be corrected the same day (alpha-engine-config-I8334).
    attempts: int = 0
    # reasoning_tokens_max_attempt — the largest reasoning draw of any SINGLE
    # attempt, as distinct from ``reasoning_tokens``, which is their sum.
    #
    # These answer different questions and both are wanted. ``max_tokens``
    # bounds ONE attempt, so the MAX is the quantity a ceiling must clear; the
    # SUM is the quantity that gets billed and is what the budget guard reads.
    # Recording only the sum meant the record could not express the thing the
    # ceiling was being sized against — a second and independent reason a
    # measured floor cannot be the guard, on top of the sample being censored
    # at the ceiling (see ``budget_escalations``).
    reasoning_tokens_max_attempt: int = 0
    web_search_requests: int = 0
    web_fetch_requests: int = 0
    # Provider-reported USD cost when available (OpenRouter returns it in
    # ``usage.cost`` when the request opts in). Preferred over card-priced
    # recompute by :func:`krepis.cost.record_llm_call` — the aggregator
    # knows the actually-routed backend's price; our cards are ceilings.
    provider_cost_usd: Optional[float] = None
    # True when the provider returned NO usage block for at least one attempt
    # of this call — the token counts here are therefore incomplete and any
    # cost derived from them would be an understatement dressed as a number.
    #
    # It exists because streaming makes the absence possible at all. A
    # non-streaming response always carries usage; a streamed one carries it
    # only in a final chunk the client has to ask for
    # (``stream_options.include_usage`` on the OpenAI wire, ``message_delta``
    # on the Anthropic wire), and a route that drops that chunk would
    # otherwise price the call at zero. ``krepis.cost.record_llm_call`` reads
    # this flag and refuses to price such a call: ``cost_usd`` is ``None`` and
    # ``cost_source`` is ``"usage_unreported"``. UNKNOWN, never 0
    # (alpha-engine-config-I8164).
    usage_unknown: bool = False


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
    # The registry entry this call ADDRESSED, carried through from
    # :attr:`ModelSpec.registry_id`. Distinct from ``model``, which is the
    # upstream name the provider reports: three registry entries can share one
    # upstream model string while declaring three different reasoning configs,
    # so ``model`` alone cannot say which was addressed
    # (alpha-engine-config-I6908). ``None`` for a hand-built spec.
    registry_id: Optional[str] = None
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
    # The provider's stop signal — ``stop``/``length``/``tool_calls`` on the
    # OpenAI wire, ``end_turn``/``max_tokens``/``tool_use`` on the Anthropic
    # one. Stamped on EVERY result, streamed or not: it separates "the model
    # finished" from "the budget ran out", and it was previously read only
    # inside the failure paths, so a successful call recorded nothing about
    # how it ended. ``None`` when the route reported none.
    finish_reason: Optional[str] = None
    # True when this generation arrived as a stream. With it, the binding
    # constraint on the call was inter-chunk silence rather than total
    # duration — a materially different failure envelope, and one a reader of
    # the artifact cannot infer from anything else on it.
    streamed: bool = False
    # Chunks (OpenAI wire) or events (Anthropic wire) received. Zero on a
    # non-streamed call. Together with the duration this is the tokens-per-
    # second signal that a non-streamed call can only ever reconstruct.
    stream_chunks: int = 0


@dataclass
class StructuredResult(LLMResult):
    """Outcome of :meth:`LLMClient.structured` — validated payload."""

    data: dict = field(default_factory=dict)
    # Pydantic instance when ``schema`` was a BaseModel subclass.
    parsed: Any = None
    # Which rung of the model-portability-policy §7 structured-output ladder
    # actually produced this payload: "native" (strict response_format=
    # json_schema), "tool_emulation" (forced tool call — the anthropic
    # transport's idiom), or "prompt_only" (JSON instruction + tolerant
    # extraction). Always populated, including on undegraded calls, so the
    # rung is a value in the artifact rather than an inference from silence.
    structured_output_rung: str = ""


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


def _finalize_result(method):
    """Stamp the call's degradation record onto the result, then emit a priced
    cost record for it.

    Applied at the PUBLIC method boundary rather than at each ``return``
    site, deliberately. ``complete`` / ``structured`` / ``complete_grounded``
    construct results at seven different points today; decorating the method
    means a future return path cannot silently escape telemetry. A missed
    return path would be invisible — no error, no log, just a call site that
    quietly stops being accounted for — which is precisely the failure class
    this arc exists to close (alpha-engine-config-I5206).

    ``dropped_params`` is stamped for the same reason and was missing for the
    same reason: the field existed on :class:`LLMResult` and documented itself
    as the surface making a degraded call visible in the artifact, and NO
    return site ever assigned it — so every degraded call read as fully
    honored to every consumer of the artifact (alpha-engine-config-I7232).
    Snapshot-and-clear, not just snapshot: ``self.dropped_params`` lives on the
    client, so without the reset a drop on one call would ride along on every
    later result from the same client and misreport it as degraded.
    """

    @functools.wraps(method)
    def _wrapped(self, *args, **kwargs):
        self.dropped_params = []
        try:
            result = method(self, *args, **kwargs)
        finally:
            # Cleared even when the call RAISES: a drop recorded on a failed
            # call must not attach itself to the next successful one.
            dropped = self.dropped_params
            self.dropped_params = []
        result.dropped_params = dropped
        _stamp_response_facts(result)
        self._emit_cost_record(result)
        return result

    return _wrapped


def _stamp_response_facts(result: Any) -> None:
    """Copy the transport-level facts off the raw response onto the result.

    Stamped in :func:`_finalize_result`, at the ONE public-method boundary,
    for the reason that decorator already exists: ``complete`` /``structured``
    / ``complete_grounded`` construct results at seven different points, and a
    field assigned at six of them is the ``dropped_params`` defect again
    (alpha-engine-config-I7232) — a result that reads as undegraded because no
    return site said otherwise.

    ``usage_unknown`` is only ever set, never cleared: on the structured path
    ``raw_response`` is the LAST attempt's, and an earlier attempt that
    reported no usage must not be forgotten because a later one did.
    """
    resp = getattr(result, "raw_response", None)
    if resp is None:
        return
    reason = _finish_reason_of(resp)
    if reason is not None:
        result.finish_reason = reason
    if getattr(resp, "krepis_streamed", False):
        result.streamed = True
        result.stream_chunks = getattr(resp, "krepis_stream_chunks", 0)
        if not getattr(resp, "krepis_usage_reported", True):
            result.usage.usage_unknown = True


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
    read/write splits and USD — and hands it to the sink.

    **When it is NOT set, the sink is resolved from the environment**
    (:func:`krepis.cost_sink.default_sink_from_env`): if
    ``KREPIS_COST_SINK_BUCKET`` and ``KREPIS_COST_SINK_PREFIX`` are both
    exported, every client built in that process emits, whether or not
    its author thought about cost telemetry. With neither set the default
    is ``None`` and a public consumer pays nothing for a feature it has
    not asked for.

    That inversion is the point. While emission required a constructor
    argument, coverage equalled *the set of authors who remembered*, and
    on 2026-08-13 that set was one process: every per-call cost record in
    the Alpha Engine research bucket came from the Think Tank, while the
    weekly pipeline's own LLM stages — each holding a correct
    ``callsite_id`` and passing no sink — were attributed to nothing
    (``alpha-engine-config-I7179``). Emission is now a property of the
    execution environment, which is where a fleet operator can actually
    set it once, rather than a property of each call site.
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
        stream_idle_timeout: float = DEFAULT_STREAM_IDLE_TIMEOUT_S,
        budget_escalation_factor: float = _BUDGET_ESCALATION_FACTOR,
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
        if stream_idle_timeout <= 0:
            raise ValueError(
                "stream_idle_timeout must be > 0 — it is the inter-chunk "
                "silence a streamed call is allowed before it is declared "
                "hung; there is no 'unbounded' setting, that is the failure "
                "mode streaming exists to remove."
            )
        self._stream_idle_timeout = float(stream_idle_timeout)
        if budget_escalation_factor < 1.0:
            raise ValueError(
                "budget_escalation_factor must be >= 1.0 "
                "(1.0 disables escalation; below 1.0 would SHRINK the ceiling "
                "on the one failure that proves it was already too small)"
            )
        self._budget_escalation_factor = float(budget_escalation_factor)
        if cost_sink is None:
            # Deliberately NOT wrapped in try/except. A half-configured
            # sink environment raises CostSinkConfigError here, before the
            # first billable call — see default_sink_from_env for why
            # falling through to silence is the worse failure.
            from krepis.cost_sink import default_sink_from_env

            cost_sink = default_sink_from_env()
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

    def _dlp_scan_request(self, payload: dict, *, context: str = "") -> None:
        """Scan *payload* (the exact dict about to be sent) for secrets.

        Raises :exc:`LLMError` on a block or scan failure (fail-closed).
        No-op when DLP is administratively disabled
        (``KREPIS_DLP_DISABLED=1``).

        Called by every public method before forwarding to the transport —
        one chokepoint, called at each of the three methods rather than via
        a reusable decorator, because the payload dict is constructed
        differently per transport (Anthropic vs OpenAI-kwargs vs LiteLLM).
        A future consolidation may unify the payload shape; until then this
        explicit call at each site is the unambiguous single-point hook the
        Lambda-path DLP gap requires (``alpha-engine-config-I4927``).
        """
        if not dlp_enabled():
            return
        try:
            body = json.dumps(payload, default=str).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            logger.warning(
                "dlp: could not serialize request payload for scanning (%s) — "
                "failing closed (the request will NOT be forwarded)",
                exc,
            )
            raise LLMError(
                f"DLP scan could not serialize request payload: {exc}"
            ) from exc
        try:
            verdict = check_request(body)
        except Exception as exc:
            logger.error(
                "dlp: scan raised for %s request: %s", context, exc
            )
            raise LLMError(
                f"DLP scan failed (fail-closed): {exc}"
            ) from exc
        if verdict.should_block:
            logger.warning(
                "dlp: BLOCKED %s request — %s (scan=%.0fms cache=%.0f%%)",
                context,
                verdict.reason,
                verdict.scan_ms,
                verdict.cache_ratio,
            )
            raise LLMError(
                f"DLP scan blocked outbound request: {verdict.reason}"
            )
        logger.debug(
            "dlp: ok %s request (scan=%.0fms cache=%.0f%%)",
            context,
            verdict.scan_ms,
            verdict.cache_ratio,
        )

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
        if not key and self.spec.provider == ROUTER_EDGE_PROVIDER:
            # The ROUTER EDGE resolves on the full credential chain, not the
            # environment alone (alpha-engine-config-I6373).
            #
            # Every other provider here authenticates from a key that is
            # deliberately in the environment, and that stays true. The edge is
            # different in kind: `resolve_group_spec` names a PER-CONSUMER
            # credential (`$KREPIS_ROUTER_CREDENTIAL_SECRET`), the edge
            # identifies the consumer BY that credential's value, and the
            # supported home for it is SSM under `/alpha-engine/<name>` —
            # precisely so the secret never enters an environment, a Lambda
            # config, a CloudWatch log, or an SSM command string on the way to
            # the box.
            #
            # Route admission already resolved it that way. This leg did not,
            # so a consumer configured exactly as intended passed admission and
            # then failed the call it had just been admitted for. Measured
            # 2026-08-04: the Think Tank spot box aborted 5s into its daily run
            # having written 0 theses, with all six KREPIS_* variables set and
            # the SSM parameter present and readable; `alpha-engine-research-
            # runner` failed identically. Both halves had tests; neither test
            # could see the other half.
            #
            # Lazy import: `krepis.router` imports `krepis.llm_config` and is
            # the heavier module, so it is reached the same way every other
            # router call in this file is.
            from krepis.router import resolve_router_credential

            key = resolve_router_credential(env_name)
        if not key:
            # `EGRESS_PROXY_PLACEHOLDER` is exported into the LOCAL egress
            # proxy process's own environment (nous-ergon-ops
            # litellm-proxy-shim.sh), not into every consumer's — the proxy
            # injects the real upstream key server-side and ignores whatever
            # this client sends, so the client needs any non-empty string,
            # not a shared secret. `krepis.model_registry.api_key_for()`
            # already encodes that ("unset placeholder env -> literal
            # default") for the config-generation path; this mirrors it for
            # the runtime call path so a consumer with no reason to export
            # the proxy's own placeholder variable is not blocked by its
            # absence (alpha-engine-config-I7031).
            from krepis import model_registry as _mr

            if env_name == _mr.EGRESS_PLACEHOLDER_ENV:
                key = _mr.EGRESS_PLACEHOLDER_DEFAULT
        if not key:
            if self.spec.provider == ROUTER_EDGE_PROVIDER:
                # Naming only the environment variable sends an operator to the
                # wrong place on the path whose supported source is SSM.
                raise LLMConfigError(
                    f"no router-edge credential {env_name!r}: pass api_key=, "
                    f"set the {env_name} environment variable, or put the "
                    f"value in SSM at {_SSM_PREFIX}{env_name} "
                    "(this consumer's identity at the edge is its credential "
                    "VALUE — do not point it at a shared key)"
                )
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

    def _call_transport(self, fn, /, **kwargs):
        """Issue one provider call, classifying its failure before anyone retries.

        A 4xx that refuses the REQUEST is permanent: no attempt count and no
        fallback chain can satisfy it, and letting one travel as an ordinary
        transport error is how the eval judge spent three attempts and a
        fallback on a rejected `tool_choice`, then reported a rate limit on a
        model that was never the problem (alpha-engine-config-I7904).

        Availability failures pass through untouched — this may only ever
        remove a retry, never add one. See :mod:`krepis.llm_errors` for the
        calibration, including which 4xx codes stay in the availability class.
        """
        from .llm_errors import raise_if_permanent_contract_error

        try:
            return fn(**kwargs)
        except Exception as exc:
            raise_if_permanent_contract_error(
                exc,
                deployment=self.spec.model,
                model_group=self.spec.registry_id,
            )
            raise

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

    # ── streaming gates ───────────────────────────────────────────────

    def _require_streaming_route(self) -> None:
        """Refuse ``stream=True`` on a route that does not declare streaming.

        Deliberately NOT routed through :meth:`_capability_gate`: that gate
        offers ``on_unsupported='drop'``, and dropping ``stream`` is precisely
        the silent fall back to a non-streaming call this must never do. There
        is one legal outcome here.

        A route resolved from the registry carries the declaration; a
        hand-built :class:`~krepis.llm_config.ModelSpec` defaults to ``True``,
        which is the caller asserting it about their own endpoint — the same
        split ``structured_outputs`` already has.
        """
        if not getattr(self.spec, "supports_streaming", True):
            raise StreamingUnsupportedError(
                f"stream=True was requested for model {self.spec.model!r} "
                f"(provider={self.spec.provider}, "
                f"registry_id={self.spec.registry_id!r}) but the resolved "
                "route does not declare capabilities.streaming. An undeclared "
                "capability is not a capability — nobody measured it. Declare "
                "it on the registry entry (and address the group with "
                "resolve_group_spec(..., requires=('streaming',)) so the chain "
                "is filtered to members that can serve this call shape), or "
                "call without stream=True. This will NOT silently fall back "
                "to a non-streaming request: that is the request-deadline "
                "failure mode streaming exists to remove."
            )

    def _effective_idle_timeout(self, idle_timeout: Optional[float]) -> float:
        """The inter-chunk budget actually enforced, warning when it is moot.

        The transport's own read timeout (``LLMClient(timeout=...)``, handed to
        the SDK) also bounds a silent socket. Whichever is smaller binds first,
        so an idle budget at or above it can never fire and the caller would be
        reasoning about a number that does nothing — the config knob that
        quietly has no effect (``feedback_no_silent_fails``).
        """
        value = self._stream_idle_timeout if idle_timeout is None else float(idle_timeout)
        if value <= 0:
            raise ValueError("idle_timeout must be > 0")
        if value >= self._timeout:
            logger.warning(
                "llm stream: idle_timeout=%.0fs is at or above the transport "
                "timeout=%.0fs, so the transport's read deadline binds first "
                "and the inter-chunk budget can never fire. Lower "
                "idle_timeout, or raise LLMClient(timeout=...).",
                value, self._timeout,
            )
        return value

    @staticmethod
    def _openai_stream_kwargs(kwargs: dict) -> dict:
        """Add ``stream`` plus the usage opt-in the OpenAI wire requires.

        ``stream_options.include_usage`` is not optional for us: without it a
        streamed OpenAI-compatible response carries no usage at all and the
        call becomes unattributable. Requesting it makes an absent usage block
        a provider-contract violation we can name, rather than an expected
        condition we would have to price at zero.
        """
        kw = dict(kwargs)
        kw["stream"] = True
        options = dict(kw.get("stream_options") or {})
        options.setdefault("include_usage", True)
        kw["stream_options"] = options
        return kw

    # The wire payload is passed as a DICT rather than **kwargs: it legitimately
    # contains a key named ``stream``, and splatting it alongside a control
    # parameter of the same name is a TypeError at the one call shape this
    # method exists for.
    def _openai_completion(
        self, create: Any, kwargs: dict, *, stream: bool, idle_timeout: float
    ) -> Any:
        """One OpenAI-wire completion, streamed or not, same return shape."""
        raw = self._call_transport(create, **kwargs)
        if not stream:
            return raw
        return _accumulate_openai_stream(
            raw, idle_timeout=idle_timeout, spec=self.spec
        )

    def _anthropic_message(
        self, create: Any, payload: dict, *, stream: bool, idle_timeout: float
    ) -> Any:
        """One Anthropic-wire message, streamed or not, same return shape."""
        raw = self._call_transport(create, **payload)
        if not stream:
            return raw
        return _accumulate_anthropic_stream(
            raw, idle_timeout=idle_timeout, spec=self.spec
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
        # One response processed == one ATTEMPT on the wire. Every call site
        # of this accumulator handles exactly one response, so counting here
        # is exact (alpha-engine-config-I8334).
        usage.attempts += 1
        _reasoning_before = usage.reasoning_tokens
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
        # Anthropic's own API does NOT break out a reasoning share — extended
        # thinking is counted inside ``output_tokens``, so this stays 0 on the
        # real Anthropic transport and that zero is truthful. Read anyway,
        # because DeepSeek's Anthropic-compatible endpoint already returns
        # OpenAI-shaped extras here (see the cache fields above) and a
        # provider that does report it should not be silently dropped.
        usage.reasoning_tokens += int(getattr(u, "reasoning_tokens", None) or 0)
        usage.reasoning_tokens_max_attempt = max(
            usage.reasoning_tokens_max_attempt,
            usage.reasoning_tokens - _reasoning_before,
        )
        return usage

    @staticmethod
    def _usage_from_openai(resp: Any, into: Optional[LLMUsage] = None) -> LLMUsage:
        usage = into or LLMUsage()
        # One response processed == one ATTEMPT on the wire. Every call site
        # of this accumulator handles exactly one response, so counting here
        # is exact (alpha-engine-config-I8334).
        usage.attempts += 1
        _reasoning_before = usage.reasoning_tokens
        u = getattr(resp, "usage", None)
        if u is None:
            return usage
        usage.input_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
        usage.output_tokens += int(getattr(u, "completion_tokens", 0) or 0)
        # OpenAI-shape providers report the reasoning share under
        # completion_tokens_details.reasoning_tokens. Absent on non-reasoning
        # models and on providers that do not break it out.
        # Handle both shapes deliberately: the openai SDK types this field, but
        # a proxied or non-conforming provider can deliver it as a raw dict,
        # and ``getattr`` on a dict silently returns the default — the exact
        # way ``server_tool_use_details`` read 0 for weeks below (config#1659).
        completion_details = getattr(u, "completion_tokens_details", None)
        if isinstance(completion_details, dict):
            usage.reasoning_tokens += int(
                completion_details.get("reasoning_tokens", 0) or 0
            )
        elif completion_details is not None:
            usage.reasoning_tokens += int(
                getattr(completion_details, "reasoning_tokens", 0) or 0
            )
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
        usage.reasoning_tokens_max_attempt = max(
            usage.reasoning_tokens_max_attempt,
            usage.reasoning_tokens - _reasoning_before,
        )
        return usage

    def _escalated_budget(self, exhausted: int) -> Optional[int]:
        """The ceiling for the ONE escalated retry, or ``None`` to give up.

        ``None`` when escalation is disabled (factor 1.0) or the arithmetic
        would not actually raise the ceiling — re-issuing at the same bound is
        the doubled-cost certain failure ``BudgetExhaustedError`` documents.
        """
        if self._budget_escalation_factor <= 1.0:
            return None
        escalated = int(exhausted * self._budget_escalation_factor)
        return escalated if escalated > exhausted else None

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

    @_finalize_result
    def complete(
        self,
        *,
        system: str,
        user_content: str,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
        extra: Optional[dict] = None,
        on_unsupported: str = "raise",
        stream: bool = False,
        idle_timeout: Optional[float] = None,
    ) -> LLMResult:
        """One plain text generation. Returns normalized :class:`LLMResult`.

        ``cache_system`` attaches Anthropic ephemeral ``cache_control`` to
        the system block. It is a cost-optimization *hint*, not a semantic
        guarantee: on the openai transport it is a no-op because
        OpenAI-compatible providers cache prompt prefixes implicitly (the
        discount shows up in ``usage.prompt_tokens_details.cached_tokens``)
        — there is nothing to forward and nothing is lost.

        ``stream=True`` accumulates the generation from the wire and returns
        the SAME :class:`LLMResult` a non-streamed call returns. What changes
        is which condition can fail the call: the bound becomes ``idle_timeout``
        seconds of inter-chunk SILENCE (default
        :data:`DEFAULT_STREAM_IDLE_TIMEOUT_S`, or the client's
        ``stream_idle_timeout``) rather than the total duration of the request,
        so a slow generation completes and a dead one raises
        :exc:`StreamIdleTimeoutError` carrying the partial text. A route that
        does not declare ``capabilities.streaming`` raises
        :exc:`StreamingUnsupportedError`; it never quietly becomes a
        non-streaming call.
        """
        self._reject_reasoning_on_anthropic()
        limit = self._effective_max_tokens(max_tokens)
        if stream:
            self._require_streaming_route()
        idle = self._effective_idle_timeout(idle_timeout) if stream else 0.0

        if self.spec.transport == TRANSPORT_ANTHROPIC:
            payload = build_messages_payload(
                model=self.spec.model,
                system_prompt=system,
                user_content=user_content,
                max_tokens=limit,
                cache_system=cache_system,
                extra=extra,
            )
            if stream:
                payload["stream"] = True
            self._dlp_scan_request(payload, context=f"complete anthropic model={self.spec.model}")
            msg = self._anthropic_message(
                self._transport_client().messages.create,
                payload,
                stream=stream,
                idle_timeout=idle,
            )
            text = "\n\n".join(
                getattr(b, "text", "")
                for b in getattr(msg, "content", []) or []
                if getattr(b, "type", None) == "text"
            ).strip()
            return LLMResult(
                text=text,
                model=getattr(msg, "model", self.spec.model),
                provider=self.spec.provider,
                registry_id=self.spec.registry_id,
                usage=self._usage_from_anthropic(msg),
                raw_request=payload,
                raw_response=msg,
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
        if stream:
            kwargs = self._openai_stream_kwargs(kwargs)
        # Scanned AFTER the streaming flags are applied so the DLP chokepoint
        # sees the exact payload that goes on the wire, not a near-copy.
        self._dlp_scan_request(kwargs, context=f"complete openai model={self.spec.model}")
        resp = self._openai_completion(
            self._transport_client().chat.completions.create,
            kwargs,
            stream=stream,
            idle_timeout=idle,
        )
        text = self._choice_text_or_llm_error(resp)
        served_model = (
            _resolve_group_served_model(resp, spec=self.spec)
            if self.spec.provider == ROUTER_EDGE_PROVIDER
            else getattr(resp, "model", self.spec.model)
        )
        return LLMResult(
            text=text,
            model=served_model,
            provider=self.spec.provider,
            registry_id=self.spec.registry_id,
            served_provider=getattr(resp, "provider", None),
            usage=self._usage_from_openai(resp),
            raw_request=kwargs,
            raw_response=resp,
        )

    # ── structured completion ─────────────────────────────────────────

    @_finalize_result
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
        stream: bool = False,
        idle_timeout: Optional[float] = None,
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

        **Degradation ladder** (``model-portability-policy`` §7). Which rung
        served is on the result as ``structured_output_rung``, always — the
        undegraded calls publish theirs too. On the openai transport, an
        endpoint that REFUSES ``response_format`` (as opposed to a registry
        entry that never claimed it) drops one rung to ``prompt_only`` for that
        call, records ``response_format`` in ``dropped_params``, and logs at
        ERROR. It is a recorded degradation, never a silent one, and it is
        bounded: one descent per call, and the descent does not spend an
        ``attempts`` retry. A refusal naming the SCHEMA rather than the
        parameter's availability still raises — see
        :func:`_is_response_format_refusal`.

        **Streaming** (``stream=True``). The schema constraint and the stream
        are not in tension: chunks are accumulated and the assembled body is
        parsed against the schema at stream END, exactly as the non-streaming
        path parses the assembled body it was handed. What changes is the
        failure envelope — the bound becomes ``idle_timeout`` seconds of
        inter-chunk silence rather than the request's total duration, so a
        generation longer than any request deadline completes, and a stream
        that dies raises :exc:`StreamIdleTimeoutError` with the partial body on
        it rather than discarding it. Every attempt of the corrective-retry
        loop is streamed; the descent ladder, the budget-exhaustion guard and
        the served-model resolution are unchanged, because a streamed response
        is re-assembled into the same object shape they already read.
        """
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self._reject_reasoning_on_anthropic()
        if stream:
            self._require_streaming_route()
        idle = self._effective_idle_timeout(idle_timeout) if stream else 0.0

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

        def _dispatch(budget: int) -> StructuredResult:
            if self.spec.transport == TRANSPORT_ANTHROPIC:
                return self._structured_anthropic(
                    system=system,
                    user_content=user_content,
                    schema_dict=schema_dict,
                    schema_name=schema_name,
                    parse_and_validate=_parse_and_validate,
                    is_pydantic=is_pydantic,
                    attempts=attempts,
                    max_tokens=budget,
                    stream=stream,
                    idle_timeout=idle,
                )
            return self._structured_openai(
                system=system,
                user_content=user_content,
                schema_dict=schema_dict,
                schema_name=schema_name,
                parse_and_validate=_parse_and_validate,
                is_pydantic=is_pydantic,
                attempts=attempts,
                max_tokens=budget,
                stream=stream,
                idle_timeout=idle,
            )

        try:
            return _dispatch(limit)
        except BudgetExhaustedError as exhausted:
            escalated = self._escalated_budget(limit)
            if escalated is None:
                raise
            # Deliberately outside ``attempts``: the caller's retry budget
            # bounds CORRECTIVE retries against the same request, and this is
            # a different request. The escalation is bounded by being outside
            # the loop — one, and only one, per logical call.
            logger.warning(
                "llm structured provider=%s model=%s callsite=%s: the "
                "completion budget was exhausted before any content "
                "(max_tokens=%d) — re-issuing ONCE at max_tokens=%d. A "
                "call site that escalates routinely has an undersized base "
                "ceiling; the count is on the cost record as "
                "budget_escalations (alpha-engine-config-I6917).",
                self.spec.provider, self.spec.model, self.callsite_id,
                limit, escalated,
            )
            try:
                result = _dispatch(escalated)
            except LLMError as exc:
                # The exhausted attempt's spend is real and must not vanish
                # because the escalated one also failed.
                if exc.usage is not None:
                    _absorb_usage(exc.usage, exhausted.usage)
                raise
            result.usage.budget_escalations += 1
            _absorb_usage(result.usage, exhausted.usage)
            return result

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
        stream: bool = False,
        idle_timeout: float = 0.0,
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
            if stream:
                payload["stream"] = True
            if attempt == 0:
                self._dlp_scan_request(payload, context=f"structured anthropic model={self.spec.model}")
            msg = self._anthropic_message(
                client.messages.create,
                payload,
                stream=stream,
                idle_timeout=idle_timeout,
            )
            self._usage_from_anthropic(msg, into=usage)
            # See the openai path: per ATTEMPT, because ``raw_response`` on the
            # result is only the last attempt's.
            if not getattr(msg, "krepis_usage_reported", True):
                usage.usage_unknown = True
                logger.error(
                    "llm stream: no complete usage arrived for a streamed call "
                    "on provider=%s model=%s — this call's spend is UNKNOWN "
                    "and will not be priced.",
                    self.spec.provider, self.spec.model,
                )
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
                    registry_id=self.spec.registry_id,
                    # The anthropic transport's structured-output idiom IS the
                    # §7 ladder's tool_emulation rung (forced tool_choice on a
                    # tool whose input_schema is the schema). Stamped so the
                    # rung is present on every transport's result, not only the
                    # one that can degrade.
                    structured_output_rung=_STRUCTURED_RUNG_TOOL_EMULATION,
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
        stream: bool = False,
        idle_timeout: float = 0.0,
    ) -> StructuredResult:
        extra_body = self._openai_extra_body()

        def _build(rung: str) -> tuple[List[dict], dict]:
            """Request for one rung of the §7 ladder. The openai transport has
            two reachable rungs (``tool_emulation`` is the anthropic idiom), and
            building both here — rather than at one branch on entry — is what
            lets the descent re-issue instead of aborting the caller's run."""
            msgs: List[dict] = [{"role": "system", "content": system}]
            kw: dict = {"model": self.spec.model, "max_tokens": max_tokens}
            if rung == _STRUCTURED_RUNG_NATIVE:
                msgs.append({"role": "user", "content": user_content})
                kw["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema_dict,
                    },
                }
            else:
                msgs.append(
                    {
                        "role": "user",
                        "content": user_content
                        + _JSON_INSTRUCTION.format(schema=json.dumps(schema_dict)),
                    }
                )
            if extra_body:
                kw["extra_body"] = extra_body
            if stream:
                kw = self._openai_stream_kwargs(kw)
            return msgs, kw

        rung = (
            _STRUCTURED_RUNG_NATIVE
            if self.spec.structured_outputs
            else _STRUCTURED_RUNG_PROMPT_ONLY
        )
        messages, kwargs = _build(rung)

        usage = LLMUsage()
        last_error: Any = None  # Exception (validation) or str (transport decode)
        raw_text = ""
        client = self._transport_client()

        # ``budget`` rather than ``range(attempts)``: a rung descent is not a
        # failed attempt, it is a different request, so it must not consume the
        # caller's retry budget. It can happen at most once (there is exactly
        # one rung below ``native`` on this transport), so the budget is bounded
        # at ``attempts + 1``.
        descended = False
        budget = attempts
        attempt = -1
        while attempt + 1 < budget:
            attempt += 1
            try:
                if attempt == 0:
                    scan_payload: dict = {"messages": messages, **kwargs}
                    self._dlp_scan_request(scan_payload, context=f"structured openai model={self.spec.model}")
                resp = self._openai_completion(
                    client.chat.completions.create,
                    {"messages": messages, **kwargs},
                    stream=stream,
                    idle_timeout=idle_timeout,
                )
                self._usage_from_openai(resp, into=usage)
                # Per ATTEMPT, not per result: ``raw_response`` on the returned
                # StructuredResult is only the last attempt's, so an earlier
                # attempt whose usage never arrived would otherwise be
                # forgotten by a later one that reported normally.
                if not getattr(resp, "krepis_usage_reported", True):
                    usage.usage_unknown = True
                    logger.error(
                        "llm stream: no usage block arrived for a streamed "
                        "call on provider=%s model=%s despite "
                        "stream_options.include_usage — this call's spend is "
                        "UNKNOWN and will not be priced. That is a route "
                        "contract violation, not a zero-cost call.",
                        self.spec.provider, self.spec.model,
                    )
                raw_text = _choice_text(resp, caller_raises_on_empty=True)
                # Deliberately OUTSIDE the retry classification below: a
                # budget exhausted before any content is not an attempt
                # failure, it is a certainty about every remaining attempt.
                _reject_budget_exhausted(
                    resp, raw_text, spec=self.spec,
                    max_tokens=max_tokens, usage=usage,
                )
                # Also outside the retry classification, and computed before
                # JSON validation: an unresolvable served model is a router/
                # transport data-integrity problem, not a validation failure,
                # and cannot be fixed by a corrective retry against the same
                # response. Letting it fall into the validation except below
                # would burn `attempts` retries on an unrecoverable condition
                # and report it as "failed validation" (alpha-engine-config-I6543).
                served_model = (
                    _resolve_group_served_model(resp, spec=self.spec)
                    if self.spec.provider == ROUTER_EDGE_PROVIDER
                    else getattr(resp, "model", self.spec.model)
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
                    budget,
                    last_error,
                )
                if attempt < budget - 1:
                    _retry_backoff_sleep(attempt)
                continue
            except Exception as exc:  # noqa: BLE001 — re-raised unless it is a refusal
                # §7 ladder descent, the ONLY case handled here. Anything else
                # re-raises unchanged: this clause must not become a general
                # transport catch-all.
                if not (
                    rung == _STRUCTURED_RUNG_NATIVE
                    and not descended
                    and _is_response_format_refusal(exc)
                ):
                    raise
                descended = True
                budget += 1
                rung = _STRUCTURED_RUNG_PROMPT_ONLY
                self.dropped_params.append("response_format")
                messages, kwargs = _build(rung)
                logger.error(
                    "llm structured provider=%s model=%s: the endpoint REFUSED "
                    "response_format (%s: %s). Descending the "
                    "model-portability-policy §7 ladder native -> prompt_only "
                    "for this call and recording the drop on the result "
                    "(dropped_params, structured_output_rung). The registry "
                    "declares this deployment can serve strict structured "
                    "output and the endpoint disagrees — that contradiction is "
                    "a registry defect to fix, not a condition to live on "
                    "(alpha-engine-config-I7232).",
                    self.spec.provider,
                    self.spec.model,
                    exc.__class__.__name__,
                    exc,
                )
                continue
            try:
                parsed = parse_and_validate(_extract_json(raw_text))
                return StructuredResult(
                    text=raw_text,
                    model=served_model,
                    provider=self.spec.provider,
                    served_provider=getattr(resp, "provider", None),
                    structured_output_rung=rung,
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
            f"structured output failed validation after {budget} "
            f"attempt(s) at rung={rung}: {last_error}",
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

    @_finalize_result
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
            self._dlp_scan_request(payload, context=f"grounded anthropic model={self.spec.model}")
            msg = self._call_transport(self._transport_client().messages.create, **payload)
            return GroundedResult(
                text=final_text_after_last_tool(getattr(msg, "content", [])),
                model=getattr(msg, "model", self.spec.model),
                provider=self.spec.provider,
                registry_id=self.spec.registry_id,
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
                if attempt == 0:
                    self._dlp_scan_request(kwargs, context=f"grounded openrouter model={self.spec.model}")
                resp = self._call_transport(self._transport_client().chat.completions.create, **kwargs)
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
