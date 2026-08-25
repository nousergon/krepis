"""Provider/model configuration for the :mod:`krepis.llm` adapter.

The adapter's "seamless switch" contract is that WHICH provider+model a
product runs on is *operator configuration*, not code: a
:class:`ModelSpec` is resolved at call time from (in order) an explicit
environment-variable override, an SSM parameter (cached with a short
TTL), or a code default. Flipping the SSM parameter switches a live
consumer between Anthropic, OpenAI, OpenRouter, or any OpenAI-compatible
endpoint within one TTL window — no redeploy; rollback is flipping the
parameter back.

**Wire value formats** (env var and SSM parameter accept the same two):

1. Compact string — ``"provider:model"``, split on the FIRST colon only
   (OpenRouter slugs legitimately contain colons, e.g.
   ``"openrouter:deepseek/deepseek-v4-flash:floor"``).
2. JSON object — any subset of :class:`ModelSpec` fields::

       {"provider": "openrouter", "model": "moonshotai/kimi-k2.6",
        "max_tokens": 8192, "structured_outputs": true}

**Fail-loud semantics** (per ``feedback_no_silent_fails``): a value that
is PRESENT but malformed raises :exc:`LLMConfigError` — a typo'd flip
must abort, not silently serve the old model. An *unreadable* SSM
parameter (network/permission/missing) with a ``default`` provided falls
back to the default WITH a WARN log — failing closed onto known config,
mirroring the BudgetGuard precedent in the Think Tank. No default and no
readable parameter → raise.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMConfigError(RuntimeError):
    """A model spec / provider configuration is malformed or unusable.

    Raised at configuration-resolution time (bad SSM/env value, unknown
    provider with no ``base_url``) and at call time when a request asks
    for a capability the configured transport cannot provide (e.g.
    forcing a server-side web search on an OpenAI-compatible endpoint).
    Never swallowed into a silent fallback.
    """


# ── Provider registry ─────────────────────────────────────────────────────

TRANSPORT_ANTHROPIC = "anthropic"
TRANSPORT_OPENAI = "openai"

#: The provider name :class:`ModelSpec` REFUSES to construct.
#:
#: It used to name a third transport — an in-process LiteLLM Router built
#: inside the consumer, calling each upstream provider directly from it. That
#: transport is gone (alpha-engine-config-I6665); the name survives only as
#: the thing the guard in ``ModelSpec.__post_init__`` recognises, because
#: consumers and stored configs still say it and must get a pointer rather
#: than a confusing "unknown provider needs a base_url".
#:
#: `krepis.router.get_router` still builds a Router — for `resolve_group`,
#: `get_group_primary` and the CLI, none of which issue completions.
REFUSED_IN_PROCESS_PROVIDER = "litellm"

#: Backwards-compatible alias. It no longer names a transport.
TRANSPORT_LITELLM = REFUSED_IN_PROCESS_PROVIDER

#: Provider name emitted for the router-edge route by
#: :func:`krepis.router.resolve_group_spec`.
#:
#: Deliberately NOT ``"litellm"``: :class:`ModelSpec` refuses that name
#: outright (see :data:`REFUSED_IN_PROCESS_PROVIDER`).  This name is absent
#: from :data:`PROVIDER_REGISTRY`, so :class:`ModelSpec` treats it as a custom
#: OpenAI-compatible endpoint — which is what the edge is.
#:
#: Defined HERE, in the module both :mod:`krepis.router` (which stamps it onto
#: the spec) and :mod:`krepis.llm` (which must recognise it to authenticate on
#: the router credential chain) already import, rather than in either of them.
#: ``krepis.router.ROUTER_EDGE_PROVIDER`` re-exports it, so the name keeps
#: working where it is already used.
ROUTER_EDGE_PROVIDER = "litellm_proxy"


@dataclass(frozen=True)
class ProviderDefaults:
    """Transport + connection defaults for a named provider."""

    transport: str  # TRANSPORT_ANTHROPIC | TRANSPORT_OPENAI
    base_url: Optional[str]
    api_key_env: str


# Built-in providers. A ModelSpec may name any OTHER provider (e.g. a
# self-hosted vLLM endpoint) by supplying ``base_url`` + ``api_key_env``
# explicitly — unknown names default to the OpenAI-compatible transport.
PROVIDER_REGISTRY: dict = {
    "anthropic": ProviderDefaults(
        transport=TRANSPORT_ANTHROPIC,
        base_url=None,
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "openai": ProviderDefaults(
        transport=TRANSPORT_OPENAI,
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    "openrouter": ProviderDefaults(
        transport=TRANSPORT_OPENAI,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
}


# ── Model spec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    """One resolved (provider, model) target for the :class:`krepis.llm.LLMClient`.

    Attributes
    ----------
    provider
        Provider name. Built-ins: ``anthropic`` / ``openai`` /
        ``openrouter``. Any other name is treated as a custom
        OpenAI-compatible endpoint and REQUIRES ``base_url`` +
        ``api_key_env``.
    model
        Provider-native model identifier (``claude-haiku-4-5``,
        ``moonshotai/kimi-k2.6``, ``deepseek/deepseek-v4-flash:floor``).
    max_tokens
        Default output-token cap for calls on this spec (overridable
        per call).
    base_url
        Endpoint override. ``None`` uses the registry default for the
        provider.
    structured_outputs
        Whether the provider/model supports strict
        ``response_format=json_schema`` on the OpenAI transport. When
        ``False``, :meth:`krepis.llm.LLMClient.structured` falls back to a
        JSON-instruction prompt + tolerant extraction (Think Tank
        pattern). Ignored on the Anthropic transport (which uses forced
        tool_choice).
    api_key_env
        Environment variable holding the API key. ``None`` uses the
        registry default for the provider.
    reasoning
        OpenRouter's unified reasoning-control object, forwarded verbatim
        into ``extra_body["reasoning"]`` on the OpenAI transport (e.g.
        ``{"effort": "low"}``, ``{"exclude": True}``,
        ``{"max_tokens": 500}``). ``None`` (default) sends no override —
        the model's own default reasoning behavior applies. Anthropic
        transport has no equivalent capability; a non-``None`` value
        there raises :exc:`LLMConfigError` at call time rather than being
        silently dropped.

        **Why this exists** (config#1659 live verification, 2026-07-06):
        a reasoning-capable OpenRouter model (Kimi K2.6) given a long
        system+user prompt spent its ENTIRE output budget on internal
        chain-of-thought and returned essentially empty ``message.content``
        — reproduced even at ``max_tokens=16000`` with
        ``finish_reason="stop"`` (a clean stop, not truncation). Any of
        the three ``reasoning`` variants above fixed it completely
        (verified live); ``exclude`` was cheapest (no reasoning tokens
        billed at all) while producing the longest content. Without this
        knob a reasoning model can silently produce a well-formed, fully
        billed, EMPTY response through this adapter.
    supports_streaming
        Whether the resolved route can serve ``stream=True``. Read by
        :meth:`krepis.llm.LLMClient._require_streaming_route`; a streamed call
        against a route that does not declare it raises
        :exc:`krepis.llm.StreamingUnsupportedError` rather than quietly
        issuing a non-streaming request — the fall back would restore the
        request-deadline failure mode streaming exists to remove
        (alpha-engine-config-I8164).

        Defaults to ``True``, and the two sources of a spec mean different
        things by that. A spec resolved through :mod:`krepis.router` carries
        the registry's own ``capabilities.streaming`` declaration, so an
        undeclared capability resolves to ``False`` and fails closed. A
        hand-built spec keeps the default, which is the caller asserting it
        about an endpoint they chose themselves — the same split
        ``structured_outputs`` already has.
    supports_automatic_prefix_caching
        Whether the provider/model supports server-side automatic prefix
        caching (e.g. DeepSeek, Moonshot/Kimi, Zhipu/GLM). When ``True``,
        the provider caches repeated prompt prefixes transparently with
        no client-side ``cache_control`` markers needed. Defaults to
        ``False`` (conservative) — set explicitly in the SSM JSON spec
        or code default when the model is known to support it. Read by
        :class:`krepis.llm.LLMClient` for cache-aware logging; no
        client-side behavior change is required since the caching is
        automatic. Contrast with ``prompt_caching`` (Anthropic-style
        explicit ``cache_control`` breakpoints) which is transport-level
        and needs no ModelSpec field.
    """

    provider: str
    model: str
    max_tokens: int = 4096
    base_url: Optional[str] = None
    structured_outputs: bool = True
    api_key_env: Optional[str] = None
    reasoning: Optional[dict] = None
    supports_automatic_prefix_caching: bool = False
    supports_streaming: bool = True
    # The registry entry this spec was resolved FROM, when it came from the
    # model registry (``route["registry_id"]``). ``None`` for a hand-built spec.
    #
    # It is not cosmetic and it is not the same as ``model``. Three registry
    # entries — `deepseek-v4-flash`, `-low`, `-max` — all carry the upstream
    # model string `deepseek-v4-flash` while declaring THREE DIFFERENT reasoning
    # configs (`{exclude: true}`, `{effort: low}`, `{effort: max}`). The
    # provider reports the upstream name, so without this field a cost record
    # cannot say which entry was addressed: spend across three distinct
    # configurations collapses into one row, and `{exclude: true}` is
    # indistinguishable from `{effort: max}` downstream — the exact distinction
    # the reasoning-budget class turns on (alpha-engine-config-I6901, I6908).
    registry_id: Optional[str] = None

    def __post_init__(self) -> None:
        # `provider="litellm"` used to select an IN-PROCESS LiteLLM Router,
        # built inside the consumer and calling each upstream provider directly
        # from it. That transport is gone (alpha-engine-config-I6665), but the
        # NAME survives in stored configs and in people's fingers, and a
        # consumer typing it is always making the same mistake: it means "the
        # `high` group", and it used to get "be your own router".
        #
        # Refusing by name rather than letting it fall through to the
        # unknown-provider path is deliberate. Unknown providers are treated as
        # custom OpenAI-compatible endpoints, so without this the failure would
        # be "provider 'litellm' is not a built-in and no base_url was
        # supplied" — true, useless, and pointing at the wrong fix.
        #
        # The objection is stated in full above `krepis.router.
        # resolve_group_spec`. In short, that path (a) egresses straight to the
        # upstream provider, unscanned by the egress proxy; (b) bypasses the
        # authenticated edge, so per-consumer identity, rate limiting and spend
        # attribution never apply; (c) requires `litellm` and a readable model
        # registry inside every consumer.
        #
        # It failed all three ways in production before this guard existed.
        # morning-signal set this provider on 2026-08-02 and every episode for
        # the next six days aborted its primary on `ModuleNotFoundError: No
        # module named 'litellm'` and silently aired on a direct-provider
        # fallback. The package was never installed, so the transport this name
        # selects had never once run — and nothing said so, because the failure
        # was indistinguishable from a provider being down.
        #
        # Raising at CONSTRUCTION rather than at first call is the point: a
        # spec that cannot work should not survive resolution. Deferring it to
        # call time is what let six days pass.
        if self.provider == REFUSED_IN_PROCESS_PROVIDER:
            raise LLMConfigError(
                "ModelSpec(provider='litellm') selects the IN-PROCESS LiteLLM "
                "Router, which calls upstream providers directly from this "
                "consumer — bypassing the egress proxy, the authenticated "
                "router edge, and per-consumer attribution.\n\n"
                "To address a model GROUP, resolve it instead:\n"
                "    from krepis.router import resolve_group_spec\n"
                "    spec, route = resolve_group_spec("
                f"{self.model!r}, exec_context=..., wire='openai')\n\n"
                "That returns a spec pointing at the router edge, with the "
                "model, endpoint and credential decided by the registry. To "
                "call one provider directly and deliberately, name that "
                "provider."
            )

    def _registry_defaults(self) -> Optional[ProviderDefaults]:
        return PROVIDER_REGISTRY.get(self.provider)

    @property
    def transport(self) -> str:
        """``"anthropic"`` or ``"openai"`` — which SDK drives this spec."""
        defaults = self._registry_defaults()
        if defaults is not None:
            return defaults.transport
        # Unknown provider = custom OpenAI-compatible endpoint.
        return TRANSPORT_OPENAI

    def resolved_base_url(self) -> Optional[str]:
        if self.base_url is not None:
            return self.base_url
        defaults = self._registry_defaults()
        if defaults is not None:
            return defaults.base_url
        raise LLMConfigError(
            f"ModelSpec provider {self.provider!r} is not a built-in "
            f"({sorted(PROVIDER_REGISTRY)}) and no base_url was supplied. "
            f"Custom providers must set base_url explicitly."
        )

    def resolved_api_key_env(self) -> str:
        if self.api_key_env is not None:
            return self.api_key_env
        defaults = self._registry_defaults()
        if defaults is not None:
            return defaults.api_key_env
        raise LLMConfigError(
            f"ModelSpec provider {self.provider!r} is not a built-in "
            f"({sorted(PROVIDER_REGISTRY)}) and no api_key_env was supplied. "
            f"Custom providers must set api_key_env explicitly."
        )


_SPEC_JSON_FIELDS = {
    "provider",
    "model",
    "max_tokens",
    "base_url",
    "structured_outputs",
    "api_key_env",
    "reasoning",
    "supports_automatic_prefix_caching",
    "supports_streaming",
}


def parse_model_spec(value: str, *, source: str = "") -> ModelSpec:
    """Parse a wire-format model-spec value (compact string or JSON).

    Raises :exc:`LLMConfigError` on anything malformed — a present-but-
    broken value must abort the flip, never fall through.
    """
    where = f" (from {source})" if source else ""
    text = (value or "").strip()
    if not text:
        raise LLMConfigError(f"empty model spec value{where}")

    if text.startswith("{"):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMConfigError(
                f"model spec JSON failed to parse{where}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise LLMConfigError(
                f"model spec JSON must be an object{where}; got "
                f"{type(raw).__name__}"
            )
        unknown = set(raw) - _SPEC_JSON_FIELDS
        if unknown:
            raise LLMConfigError(
                f"model spec JSON has unknown field(s) {sorted(unknown)}{where}; "
                f"allowed: {sorted(_SPEC_JSON_FIELDS)}"
            )
        if "provider" not in raw or "model" not in raw:
            raise LLMConfigError(
                f"model spec JSON must include 'provider' and 'model'{where}: "
                f"{text!r}"
            )
        try:
            return ModelSpec(**raw)
        except TypeError as exc:  # wrong value types
            raise LLMConfigError(
                f"model spec JSON has invalid field value(s){where}: {exc}"
            ) from exc

    # Compact form: provider:model — split on the FIRST colon only, since
    # OpenRouter slugs carry their own ':floor' / ':online' variant suffix.
    if ":" not in text:
        raise LLMConfigError(
            f"model spec {text!r}{where} is neither JSON nor 'provider:model'"
        )
    provider, model = text.split(":", 1)
    provider, model = provider.strip(), model.strip()
    if not provider or not model:
        raise LLMConfigError(
            f"model spec {text!r}{where} must be 'provider:model' with both "
            f"parts non-empty"
        )
    return ModelSpec(provider=provider, model=model)


# ── Runtime resolution (env → SSM → default) ─────────────────────────────

# param name -> (monotonic expiry, ModelSpec)
_SPEC_CACHE: dict = {}
_SPEC_CACHE_LOCK = threading.Lock()


def clear_spec_cache() -> None:
    """Drop every cached SSM-resolved spec (test seam; mirrors
    ``krepis.secrets`` cache semantics)."""
    with _SPEC_CACHE_LOCK:
        _SPEC_CACHE.clear()


def _read_ssm_parameter(name: str, ssm_client: Any) -> str:
    client = ssm_client
    if client is None:
        import boto3

        from krepis.aws_region import resolve_region

        client = boto3.client("ssm", region_name=resolve_region())
    resp = client.get_parameter(Name=name, WithDecryption=True)
    return resp["Parameter"]["Value"]


def resolve_model_spec(
    ssm_param: str,
    *,
    env_var: Optional[str] = None,
    default: Optional[ModelSpec] = None,
    ttl_seconds: int = 60,
    ssm_client: Any = None,
    max_tokens: Optional[int] = None,
) -> ModelSpec:
    """Resolve the active :class:`ModelSpec` for a product/tier.

    Resolution order (first hit wins):

    1. ``env_var`` — explicit operator/test override, read on EVERY call
       (never cached) so a shell override takes effect immediately.
    2. ``ssm_param`` — the operational flip surface. Cached for
       ``ttl_seconds`` (monotonic clock) so long-running consumers pick
       up a flip within one TTL window without an SSM round-trip per
       request.
    3. ``default`` — the in-code baseline.

    Error semantics:

    - Present-but-malformed value (env OR ssm) → :exc:`LLMConfigError`.
      A typo'd flip aborts loudly rather than silently serving stale
      config.
    - Unreadable SSM (missing param, permission, network) with a
      ``default`` → WARN + default, and the default is cached for the
      TTL so a broken SSM path doesn't add a failing round-trip to every
      call. Without a ``default`` → :exc:`LLMConfigError`.

    ``max_tokens``, when given, overrides the resolved spec's value —
    lets a consumer keep its own output-cap config while the
    provider/model comes from the flip surface.
    """
    spec: Optional[ModelSpec] = None

    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            spec = parse_model_spec(env_value, source=f"env {env_var}")

    if spec is None:
        now = time.monotonic()
        with _SPEC_CACHE_LOCK:
            cached = _SPEC_CACHE.get(ssm_param)
            if cached is not None and cached[0] > now:
                spec = cached[1]
        if spec is None:
            try:
                value = _read_ssm_parameter(ssm_param, ssm_client)
            except Exception as exc:  # noqa: BLE001 — categorized below
                if default is None:
                    raise LLMConfigError(
                        f"SSM parameter {ssm_param!r} could not be read and "
                        f"no default ModelSpec was provided: {exc}"
                    ) from exc
                logger.warning(
                    "resolve_model_spec: SSM parameter %r unreadable (%s); "
                    "falling back to code default %s:%s",
                    ssm_param,
                    exc,
                    default.provider,
                    default.model,
                )
                spec = default
            else:
                # Present-but-malformed raises through (no default rescue).
                spec = parse_model_spec(value, source=f"ssm {ssm_param}")
            with _SPEC_CACHE_LOCK:
                _SPEC_CACHE[ssm_param] = (now + ttl_seconds, spec)

    if max_tokens is not None:
        spec = replace(spec, max_tokens=max_tokens)
    return spec
