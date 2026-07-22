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

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from krepis.anthropic_payload import (
    build_messages_payload,
    build_web_search_tool,
)
from krepis.llm_config import (
    TRANSPORT_ANTHROPIC,
    TRANSPORT_OPENAI,
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


class LLMError(RuntimeError):
    """A call failed after its bounded corrective retries — fail loud.

    Carries ``usage`` (the :class:`LLMUsage` accumulated across the failed
    attempts) so callers can still record the spend of a failed call —
    tokens were consumed even though no valid output was produced.
    """

    def __init__(self, message: str, *, usage: "Optional[LLMUsage]" = None):
        super().__init__(message)
        self.usage = usage


# ── Result types ──────────────────────────────────────────────────────────


@dataclass
class LLMUsage:
    """Normalized token/fee usage for one logical call (all attempts)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    cache_create_1h_tokens: int = 0
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
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: Optional[str] = None,
        client_factory: Optional[Callable[[ModelSpec, str], Any]] = None,
        timeout: float = 180.0,
        max_retries: int = 3,
    ):
        self.spec = spec
        self._api_key = api_key
        self._client_factory = client_factory
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any = None

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

    # ── plain completion ──────────────────────────────────────────────

    def complete(
        self,
        *,
        system: str,
        user_content: str,
        max_tokens: Optional[int] = None,
        cache_system: bool = True,
        extra: Optional[dict] = None,
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
        limit = max_tokens if max_tokens is not None else self.spec.max_tokens

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
        text = (resp.choices[0].message.content or "").strip()
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
        limit = max_tokens if max_tokens is not None else self.spec.max_tokens

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
            except json.JSONDecodeError as exc:
                # The transport returned a non-JSON body on what the SDK
                # treated as a successful transaction (e.g. an OpenRouter
                # gateway hiccup) — this is invisible to the SDK's own
                # ``max_retries`` (status/connection-based) since parsing
                # only fails after the response is already considered
                # final. Treat it as an ordinary bounded-retry attempt
                # failure rather than letting it crash the caller with a
                # raw, context-free JSONDecodeError (live incident
                # 2026-07-20 — krepis#38).
                last_error = (
                    f"transport returned a non-JSON response body "
                    f"({exc.__class__.__name__}: {exc})"
                )
                logger.warning(
                    "llm structured provider=%s model=%s attempt=%d/%d: "
                    "transport returned a non-JSON response body (%s: %s)",
                    self.spec.provider,
                    self.spec.model,
                    attempt + 1,
                    attempts,
                    exc.__class__.__name__,
                    exc,
                )
                continue
            self._usage_from_openai(resp, into=usage)
            raw_text = (resp.choices[0].message.content or "").strip()
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
        limit = max_tokens if max_tokens is not None else self.spec.max_tokens

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
            except json.JSONDecodeError as exc:
                # The transport returned a non-JSON body on what the SDK
                # treated as a successful transaction (e.g. an OpenRouter
                # gateway hiccup) — invisible to the SDK's own
                # ``max_retries`` since parsing only fails after the
                # response is already considered final. Live incident
                # 2026-07-20 (krepis#38): this crashed the caller with a
                # raw, context-free JSONDecodeError instead of engaging
                # the caller's cross-provider fallback, because it isn't
                # a ``RuntimeError``/``LLMError`` subclass.
                last_error = (
                    f"transport returned a non-JSON response body "
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
                continue

            self._usage_from_openai(resp, into=usage)
            choice = resp.choices[0]
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
