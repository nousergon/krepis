"""Model-group Router — single source of truth for LLM model routing.

Reads :file:`LLM_MODEL_REGISTRY.yaml` and builds a :class:`litellm.Router`
with fallback chains. Exposes a CLI for shell scripts and GHA workflows
to resolve model groups to the first healthy model.

Usage::

    # CLI — resolve a group to the best available model (bash/GHA)
    python3 -m krepis.router resolve low
    # → deepseek-v4-flash

    python3 -m krepis.router resolve ultra
    # → moonshotai/kimi-k3

    # List all groups
    python3 -m krepis.router groups

    # Python API
    from krepis.router import get_router, resolve_group
    router = get_router()
    model = resolve_group("high")  # → "deepseek-v4-pro"

Registry lookup
    By default reads ``$LLM_MODEL_REGISTRY_PATH`` (env var).  If unset,
    walks up from *cwd* looking for
    ``private-docs/LLM_MODEL_REGISTRY.yaml`` (alpha-engine-config path).
    Failing that, falls back to the hardcoded model list in
    :func:`_builtin_model_list` — the same list that was previously
    embedded in :func:`krepis.llm._get_router`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_WALK_DEPTH = 8

# ── Resolution contract version ──────────────────────────────────────────
# The dict returned by resolve_group_structured() / `resolve <group> --json`
# is a CROSS-REPO CONTRACT with consumers in alpha-engine-config (groom
# driver, groom_run.sh, disposition audit, reviewed-merge sweep) and
# claude-code-config (the clauder wrapper).  It is versioned, schema'd
# (resolve_schema.json), and evolved ADDITIVELY.
#
# Field renames are the recurring failure mode here: `resolve-group` -> `resolve`
# broke the wrapper (2026-07-24), and `anthropic_base_url` -> `api_base_url`
# broke all four alpha-engine-config consumers (alpha-engine-config-I4453) --
# two of which degraded to an EMPTY base URL and targeted api.anthropic.com.
#
# Rule (model-router-policy R19): emit BOTH names for one release, migrate
# consumers, then remove.  Never a same-commit rename.
RESOLVE_SCHEMA_VERSION = 2

# Fields kept only to avoid breaking not-yet-migrated consumers.  Each entry
# is (deprecated_name, current_name, remove_after_version).
_DEPRECATED_RESOLVE_ALIASES = (
    ("anthropic_base_url", "api_base_url", 3),
)


def _with_compat_aliases(info: dict) -> dict:
    """Add deprecated field aliases to a resolve-contract dict.

    Keeps consumers that have not yet migrated working through one release,
    per the additive-then-remove rule.  Drop an alias by removing its row
    from :data:`_DEPRECATED_RESOLVE_ALIASES` once every consumer is migrated
    and its contract test asserts the new name.
    """
    for old, new, _remove_after in _DEPRECATED_RESOLVE_ALIASES:
        if new in info and old not in info:
            info[old] = info[new]
    return info

_egress_placeholder = "unused-placeholder-see-key-isolation-config3007"

# LiteLLM proxy — the central model router on the dashboard box.  When healthy,
# this is the PREFERRED endpoint for the Claude CLI because it:
#   1. Translates between wire formats (CLI format ↔ provider-native format),
#      making every provider CLI-compatible regardless of its native API format.
#   2. Handles fallback chains with cooldown via litellm.Router.
#   3. Passes through prompt-caching hints and translates them to
#      provider-specific caching mechanisms where applicable.
#   4. Routes all traffic through egress proxies for DLP scanning.
LITELLM_PROXY_URL = "http://127.0.0.1:8980"

# Env-var overrides for every CLI-compatible endpoint (config#4923).
# Each resolves at resolution time from (env var → default constant), so a
# box-level bootstrap can export the override alongside its --port without
# krepis needing a config file or code change.
_ENV_OVERRIDE_MAP: dict[tuple[str, str | None], str] = {
    ("egress_proxy", "deepseek"): "KREPIS_DEEPSEEK_EGRESS_URL",
    ("openrouter", None):         "KREPIS_OPENROUTER_API_URL",
    ("litellm_proxy", None):      "KREPIS_LITELLM_PROXY_URL",
}

# Mapping of (route, provider) tuples to CLI-compatible base URLs.
# Only these combinations speak the wire format the Claude CLI expects.
# Port 8971 strips the /anthropic upstream-prefix (DeepSeek supports the
# CLI wire format natively); ports 8972-8976 serve OpenAI-format (for
# LiteLLM, not CLI-compatible).  The LiteLLM proxy (port 8980) is the
# preferred path and is checked first in _resolve_group_json before
# falling through to these per-provider endpoints.
_CLI_COMPATIBLE_ENDPOINTS: dict[tuple[str, str | None], str] = {
    ("egress_proxy", "deepseek"): "http://127.0.0.1:8971",
    ("openrouter", None):         "https://openrouter.ai/api",
    ("direct", "anthropic"):      "",
    ("litellm_proxy", None):      LITELLM_PROXY_URL,
}


def _resolve_litellm_proxy_url() -> str:
    """LiteLLM proxy URL, from ``KREPIS_LITELLM_PROXY_URL`` env var or the
    module-level :data:`LITELLM_PROXY_URL` constant.

    This is the URL the LiteLLM health probe targets and the ``api_base_url``
    returned on the LiteLLM route — keeping the two in sync with one source
    of truth (config#4923)."""
    return os.environ.get("KREPIS_LITELLM_PROXY_URL", LITELLM_PROXY_URL)


def _resolve_cli_endpoint(route: str, provider: str | None) -> str:
    """Resolve a CLI-compatible endpoint for *(route, provider)* at call time.

    Resolution order:
    1. Per-route env var (``_ENV_OVERRIDE_MAP`` → ``os.environ``)
    2. Module-level ``_CLI_COMPATIBLE_ENDPOINTS`` default

    Returns the endpoint URL or ``""`` for provider-default routes (``direct``).
    Raises :exc:`ValueError` if the combination is not CLI-compatible.
    """
    # 1. Check env override by exact (route, provider) key
    precise_key = (route, provider)
    if provider is not None and precise_key in _ENV_OVERRIDE_MAP:
        override = os.environ.get(_ENV_OVERRIDE_MAP[precise_key])
        if override:
            return override

    # 2. Check env override by (route, None) key (catch-all for the route)
    catchall_key = (route, None)
    if catchall_key in _ENV_OVERRIDE_MAP:
        override = os.environ.get(_ENV_OVERRIDE_MAP[catchall_key])
        if override:
            return override

    # 3. Check exact match in the default map
    if precise_key in _CLI_COMPATIBLE_ENDPOINTS:
        return _CLI_COMPATIBLE_ENDPOINTS[precise_key]

    # 4. Check catch-all by route
    if catchall_key in _CLI_COMPATIBLE_ENDPOINTS:
        return _CLI_COMPATIBLE_ENDPOINTS[catchall_key]

    raise ValueError(
        f"({route!r}, {provider!r}) is not a CLI-compatible endpoint "
        "and cannot be resolved"
    )


def _probe_egress_proxy(url: str, timeout: int = 3) -> bool:
    """Probe an egress proxy's health endpoint.

    Returns ``True`` if the proxy responds with ``200`` at ``/__proxy_health__``.
    Used by :func:`_cli_endpoint_for` to fail at resolve time rather than
    returning a dead endpoint (config#4923).

    Pure / no side-effects beyond the network call.  Tested by mocking
    ``http.client``.
    """
    if not url:
        return False  # empty URL (direct route) is not an egress proxy
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8971

        import http.client as _http

        conn = _http.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/__proxy_health__")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


# ── LiteLLM master key resolution ────────────────────────────────────────
# Same resolution order as the LiteLLM proxy shim and the clauder script:
#   1. LITELLM_MASTER_KEY env var
#   2. secrets.env file (LITELLM_MASTER_KEY=...)
#   3. AWS SSM /symposion/LITELLM_MASTER_KEY
# Returns the key string, or None if unresolvable.

def _resolve_litellm_master_key() -> Optional[str]:
    import os as _os

    # 1. Env var
    _key = _os.environ.get("LITELLM_MASTER_KEY", "").strip()
    if _key:
        return _key

    # 2. secrets.env (same path conventions as the shim)
    _secrets_paths = [
        _os.path.expanduser("~/Development/.llm-routing/secrets.env"),
        _os.path.expanduser("~/.llm-routing/secrets.env"),
    ]
    for _sp in _secrets_paths:
        try:
            with open(_sp) as _sf:
                for _line in _sf:
                    _line = _line.strip()
                    if _line.startswith("LITELLM_MASTER_KEY="):
                        _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                        if _val:
                            return _val
                        break
        except (OSError, IOError):
            continue

    # 3. AWS SSM
    try:
        import subprocess as _sp
        _result = _sp.run(
            ["aws", "ssm", "get-parameter",
             "--name", "/symposion/LITELLM_MASTER_KEY",
             "--with-decryption", "--region", "us-east-1",
             "--profile", "ne-admin",
             "--query", "Parameter.Value", "--output", "text"],
            capture_output=True, text=True, timeout=5,
        )
        if _result.returncode == 0:
            _val = _result.stdout.strip()
            if _val and _val != "None":
                return _val
    except Exception:
        pass

    return None


# ── LiteLLM config staleness check (RETIRED) ────────────────────────────
# alpha-engine-config-I4452: this mtime-glob heuristic is RETIRED.  It
# inferred staleness from filesystem mtimes over a /tmp glob, ran only in
# a laptop CLI path, and nothing on any box called it.  The authoritative
# provenance-based drift checker lives in alpha-engine-config:
#
#   scripts/check_router_config_provenance.py
#
# which compares a CONTENT DIGEST (sha256) stamped into the generated config
# against a fresh hash of the live registry — a fact, not an inference.
#
# The auto-reconcile script (scripts/reconcile_litellm_config.py) runs on
# a 10-minute timer and restarts the router on digest mismatch, satisfying
# the 15-minute propagation SLO (R10) with no human step.
#
# The function has been removed; the caller (get_router) now skips this
# check.  See alpha-engine-config-PR4773 and I4452 for details.

# ── registry file discovery ─────────────────────────────────────────────

def _find_registry() -> Optional[Path]:
    """Resolve the registry file path.

    Lookup order:
    1. ``$LLM_MODEL_REGISTRY_PATH`` (explicit override)
    2. Walk up from cwd for ``private-docs/LLM_MODEL_REGISTRY.yaml``
       (alpha-engine-config convention)

    Returns ``None`` if neither is found — the caller (:func:`get_router`)
    raises :exc:`FileNotFoundError` rather than falling back to a stale
    duplicate.  There is exactly ONE source of truth for model groupings,
    and it lives in ``alpha-engine-config/private-docs/``.
    """
    env_path = os.environ.get("LLM_MODEL_REGISTRY_PATH")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None

    # Walk up from cwd looking for alpha-engine-config/private-docs/
    cwd = Path.cwd().resolve()
    for _ in range(_MAX_WALK_DEPTH):
        candidate = cwd / "private-docs" / "LLM_MODEL_REGISTRY.yaml"
        if candidate.exists():
            return candidate
        # Also check the nested alpha-engine-config path
        candidate2 = cwd / "alpha-engine-config" / "private-docs" / "LLM_MODEL_REGISTRY.yaml"
        if candidate2.exists():
            return candidate2
        parent = cwd.parent
        if parent == cwd:
            break
        cwd = parent

    return None


# ── CLI endpoint helpers ──────────────────────────────────────────────────

def _caching_flags(entry: dict) -> tuple[bool, bool]:
    """Return ``(explicit_breakpoints, automatic_prefix)`` for one model entry.

    The two caching mechanisms are **mutually exclusive** and impose opposite
    client obligations:

    * ``explicit_breakpoints`` — the client must mark cacheable segments with
      ``cache_control: {"type": "ephemeral"}``. Anthropic-wire only.
    * ``automatic_prefix`` — the server caches repeated prefixes
      transparently. The client must send **no** markers; emitting them at a
      provider that does not accept the field is at best a wasted breakpoint
      and at worst a 400.

    This is the ONE place either flag is read, so a mis-declared entry
    produces one consistent answer everywhere rather than a different one per
    call site.

    Forward-compatible with the registry's move to a single
    ``capabilities.caching_mechanism`` enum
    (``explicit_breakpoint`` | ``automatic_prefix`` | ``none``), which makes
    the invalid both-true state unrepresentable rather than merely forbidden
    (alpha-engine-config-I4463). The enum wins when present; the two legacy
    booleans are the fallback until every registry entry carries it.

    An entry declaring BOTH legacy booleans is invalid. Rather than silently
    picking one, this resolves to ``automatic_prefix`` — the safe direction:
    a provider that caches transparently loses nothing by receiving no
    markers, whereas sending markers to one that rejects unknown fields is an
    outage. The registry validator rejects the state at PR time; this is the
    runtime backstop for a registry that predates it.
    """
    caps = entry.get("capabilities") or {}

    mechanism = caps.get("caching_mechanism")
    if mechanism:
        return mechanism == "explicit_breakpoint", mechanism == "automatic_prefix"

    explicit = bool(caps.get("prompt_caching", False))
    automatic = bool(caps.get("automatic_prefix_caching", False))
    if explicit and automatic:
        logger.warning(
            "model %r declares BOTH prompt_caching and automatic_prefix_caching "
            "— these are mutually exclusive; treating it as automatic_prefix "
            "(no client markers), the non-breaking direction",
            entry.get("id", "?"),
        )
        return False, True
    return explicit, automatic


def _cli_endpoint_for(entry: dict) -> str:
    """Return the CLI-compatible base URL for a registry model entry.

    Resolves the endpoint from env var overrides first (config#4923), then
    falls back to the default ``_CLI_COMPATIBLE_ENDPOINTS`` literal.

    For ``egress_proxy`` routes: probes the resolved proxy at
    ``/__proxy_health__`` before returning.  An unreachable proxy raises
    :exc:`ValueError` with a message naming the URL and the env override var
    — so the caller falls through to the next model (same pattern as the
    LiteLLM health check) rather than returning a dead endpoint that would
    surface later as an opaque ConnectionRefused inside an LLM agent.

    Raises :exc:`ValueError` if this entry's route+provider cannot serve
    the wire format the Claude CLI expects (e.g. Gemini/xAI/Moonshot/Zhipu
    egress proxies are OpenAI-format only — use LiteLLM for format translation).
    """
    route = entry.get("route", "")
    provider = entry.get("provider", "")

    endpoint = _resolve_cli_endpoint(route, provider)

    # For egress_proxy routes: probe the proxy before returning.
    # A dead proxy means the route is unusable — fail at resolve time
    # rather than returning an endpoint nobody verified (config#4923).
    if route == "egress_proxy" and endpoint:
        env_var_key = _ENV_OVERRIDE_MAP.get((route, provider))
        env_var_hint = f" (override: ${env_var_key})" if env_var_key else ""
        if not _probe_egress_proxy(endpoint):
            raise ValueError(
                f"Egress proxy at {endpoint!r} is not reachable"
                f"{env_var_hint}. "
                "Set the override to a healthy proxy address, or ensure the "
                "local proxy is running on the configured port."
            )

    return endpoint


def _cli_deployment_id(entry: dict) -> str:
    """Return the model string to set as the CLI model identifier.

    * For *egress_proxy* routes: bare model name (e.g. ``deepseek-v4-flash``)
      — the proxy translates to the upstream model ID.
    * For *openrouter* routes: full OpenRouter slug (e.g.
      ``deepseek/deepseek-v4-flash``) — already stored in the registry's
      *model* field.
    * For *direct* routes: canonical provider model ID.
    """
    route = entry.get("route", "")
    model = entry.get("model", "")

    if route == "egress_proxy":
        return model
    elif route == "openrouter":
        return model
    elif route == "direct" and entry.get("provider") == "anthropic":
        return model
    else:
        return model


# ── registry → Router config ─────────────────────────────────────────────

def _parse_registry(path: Path, openrouter_key: str = "") -> tuple[list[dict], list[dict]]:
    """Parse LLM_MODEL_REGISTRY.yaml into litellm Router model_list + fallbacks.

    Returns (model_list, fallbacks) tuples ready for litellm.Router().
    """
    import yaml as _yaml

    with open(path) as f:
        doc = _yaml.safe_load(f)

    model_list: list[dict] = []
    fallbacks: list[dict] = []
    seen_models: set[str] = set()

    groups = doc.get("model_groups", {})
    models = {m["id"]: m for m in doc.get("models", [])}

    for group_name, group_ids in groups.items():
        fallback_chain: list[str] = []
        for i, mid in enumerate(group_ids):
            entry = models.get(mid)
            if entry is None:
                logger.warning("model %r referenced in group %r not found in models list", mid, group_name)
                continue

            # Primary is named after the GROUP ("low", "med", …) so
            # router.completion(model="low") resolves to the first entry.
            # Fallbacks get qualified: "low-gemini-2.5-flash", etc.
            model_name = group_name if i == 0 else f"{group_name}-{mid}"
            litellm_params = _model_to_litellm_params(entry, openrouter_key)

            if model_name not in seen_models:
                model_list.append({"model_name": model_name, "litellm_params": litellm_params})
                seen_models.add(model_name)

            if i > 0:
                fallback_chain.append(model_name)

        if fallback_chain:
            fallbacks.append({group_name: fallback_chain})

    return model_list, fallbacks


def _model_to_litellm_params(entry: dict, openrouter_key: str) -> dict:
    """Convert a registry model entry to litellm params."""
    provider = entry.get("provider", "")
    route = entry.get("route", "")
    model_id = entry.get("model", "")
    params = entry.get("params", {})

    litellm_params: dict = {}

    # Build the litellm model prefix
    if provider == "anthropic":
        litellm_params["model"] = f"anthropic/{model_id}"
        litellm_params["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    elif provider == "openrouter":
        litellm_params["model"] = f"openrouter/{model_id}"
        litellm_params["api_key"] = openrouter_key
    elif route == "egress_proxy":
        litellm_params["model"] = f"openai/{model_id}"
        litellm_params["api_key"] = _egress_placeholder
        # api_base is driven by the registry — no hardcoded provider→port map.
        # Every egress_proxy entry MUST carry an api_base field pointing at its
        # local proxy instance (e.g. "http://127.0.0.1:8972/v1").
        api_base = entry.get("api_base")
        if api_base:
            litellm_params["api_base"] = api_base
        else:
            entry_id = entry.get("id", model_id)
            logger.warning(
                "egress_proxy entry %r missing api_base — "
                "requests will fail without a base URL", entry_id
            )
    else:
        # Unknown route — treat as generic OpenAI-compatible, no proxy.
        litellm_params["model"] = f"openai/{model_id}"
        litellm_params["api_key"] = _egress_placeholder

    # Apply params from registry
    if "max_tokens" in params:
        litellm_params["max_tokens"] = params["max_tokens"]
    reasoning = params.get("reasoning")
    if reasoning:
        if "extra_body" not in litellm_params:
            litellm_params["extra_body"] = {}
        litellm_params["extra_body"]["reasoning"] = reasoning

    # Multi-tenant egress proxy: set X-Upstream-Host header so the single
    # proxy on port 8990 routes to the correct upstream provider.
    upstream_host = entry.get("upstream_host")
    if upstream_host:
        if "extra_headers" not in litellm_params:
            litellm_params["extra_headers"] = {}
        litellm_params["extra_headers"]["X-Upstream-Host"] = upstream_host

    # Apply RPM/TPM from registry if present
    if "rpm" in entry:
        litellm_params["rpm"] = entry["rpm"]
    if "tpm" in entry:
        litellm_params["tpm"] = entry["tpm"]

    return litellm_params


# ── Router singleton ─────────────────────────────────────────────────────

_router: Any = None
_router_lock: Any = None


def get_router() -> Any:
    """Return the module-level Router singleton, built from LLM_MODEL_REGISTRY.yaml.

    Raises :exc:`FileNotFoundError` if no registry file can be found —
    there is no hardcoded fallback.  The single source of truth lives in
    ``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml``; set
    ``$LLM_MODEL_REGISTRY_PATH`` to point at it explicitly, or run from
    within a repo whose ``private-docs/`` directory contains the file.
    """
    global _router, _router_lock
    if _router is not None:
        return _router

    from threading import Lock as _Lock
    from litellm import Router as _Router

    _router_lock = _Lock()
    with _router_lock:
        if _router is not None:
            return _router

        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        reg_path = _find_registry()
        if not reg_path:
            raise FileNotFoundError(
                "LLM_MODEL_REGISTRY.yaml not found — set "
                "LLM_MODEL_REGISTRY_PATH or run from within a repo "
                "whose private-docs/ directory contains the file.  "
                "The canonical copy lives in "
                "alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml."
            )

        logger.info("building Router from %s", reg_path)
        model_list, fallbacks = _parse_registry(reg_path, openrouter_key)

        _router = _Router(model_list=model_list, fallbacks=fallbacks)
        logger.info("Router initialized: %d models, %d fallback groups", len(model_list), len(fallbacks))
        return _router


# ── group resolution (for shell scripts / GHA) ───────────────────────────

def resolve_group(group: str) -> str:
    """Return the upstream model identifier for *group*'s primary.

    Checks the Router's cooldown state — if the primary model is in
    cooldown (recent failure), returns the first healthy fallback.
    """
    router = get_router()

    # Find the primary model's upstream identifier
    primary_model = _upstream_model_for(router, group)
    if not primary_model:
        # Group not found in model list — return the group name as-is
        return group

    # Check if primary is in cooldown
    deployments = getattr(router, "cooldown_deployments", {})
    primary_key = _deployment_key_for(router, group)
    if primary_key and primary_key in deployments:
        # Primary is in cooldown — try fallbacks
        for fb in router.fallbacks:
            if group in fb:
                for fb_name in fb[group]:
                    fb_key = _deployment_key_for(router, fb_name)
                    if fb_key and fb_key not in deployments:
                        fb_model = _upstream_model_for(router, fb_name)
                        if fb_model:
                            return fb_model
                break  # all fallbacks for this group exhausted

    return primary_model


def group_supports_explicit_cache_breakpoints(group: str) -> bool:
    """True when *group*'s primary honors explicit ``cache_control`` markers.

    The question a client actually needs answered before emitting Anthropic
    ephemeral cache breakpoints. Resolved from the PRIMARY — the entry that
    will serve the request — not from an ``any()`` over the fallback chain,
    which would claim support the served model may not have.

    Returns False when the registry cannot be read: refusing to claim a
    capability we could not verify is the safe direction, since the cost of a
    false negative is a missed cache hit and the cost of a false positive is
    markers sent to a provider that may reject them.
    """
    try:
        reg_path = _find_registry()
        if not reg_path:
            return False
        import yaml as _yaml
        with open(reg_path) as f:
            doc = _yaml.safe_load(f)
        group_ids = (doc.get("model_groups") or {}).get(group) or []
        if not group_ids:
            return False
        models_by_id = {m["id"]: m for m in doc.get("models", [])}
        explicit, _automatic = _caching_flags(models_by_id.get(group_ids[0], {}))
        return explicit
    except Exception:
        logger.warning(
            "could not resolve caching capability for group %r — "
            "assuming no explicit breakpoints", group, exc_info=True,
        )
        return False


def get_group_primary(group: str) -> Optional[str]:
    """Return the litellm model string for *group*'s primary model.

    This is the value that ``resp.model`` will carry when the primary
    (not a fallback) served the request.  Callers compare against it to
    detect whether a fallback was engaged::

        primary = get_group_primary("low")        # "openai/deepseek-v4-flash"
        fallback_used = (resp.model != primary)
    """
    router = get_router()
    for m in router.model_list:
        if m["model_name"] == group:
            return m["litellm_params"]["model"]
    return None


def _upstream_model_for(router: Any, model_name: str) -> Optional[str]:
    """Get the upstream model identifier for a Router model_name."""
    for m in router.model_list:
        if m["model_name"] == model_name:
            return _upstream_model(m["litellm_params"]["model"])
    return None


def _deployment_key_for(router: Any, model_name: str) -> Optional[str]:
    """Find the deployment key for a model in the Router's model list."""
    for m in router.model_list:
        if m["model_name"] == model_name:
            params = m.get("litellm_params", {})
            return params.get("model", "")
    return None


def _upstream_model(litellm_model: str) -> str:
    """Strip the litellm prefix: 'openai/deepseek-v4-flash' → 'deepseek-v4-flash'."""
    parts = litellm_model.split("/", 1)
    return parts[1] if len(parts) > 1 else litellm_model


# ── group resolution (structured, for shell scripts) ────────────────────

def _resolve_group_json(group: str) -> dict:
    """Return full routing info for *group* as a JSON-ready dict.

    Reads the registry directly (bypasses the Router's cooldown state).
    Prefers the LiteLLM proxy when healthy (port resolved from
    ``KREPIS_LITELLM_PROXY_URL`` env var or :data:`LITELLM_PROXY_URL`
    constant) — it handles format translation for all providers, making
    every registry model CLI-compatible.  Falls back to per-provider
    resolution when LiteLLM is unavailable.

    Produces a dict with every field a shell script needs to configure
    the Claude CLI environment: endpoint URL, auth type, provider, route,
    deployment ID, registry ID, capabilities (prompt_caching, etc.), and
    cache pricing.

    The ``api_base_url`` is the CLI-compatible endpoint.  The LiteLLM proxy
    is the preferred path — it translates between CLI wire format and
    provider-native formats.  DeepSeek and OpenRouter are direct-fallback
    paths when LiteLLM is unavailable.  Every endpoint is resolvable via
    an env override (config#4923), so box-level bootstraps can export the
    override alongside their ``--port`` from a single configuration value.

    Raises :exc:`ValueError` if NO model in the group is CLI-compatible.
    """
    import yaml as _yaml

    reg_path = _find_registry()
    if not reg_path:
        raise FileNotFoundError(
            "LLM_MODEL_REGISTRY.yaml not found — set "
            "LLM_MODEL_REGISTRY_PATH or run from within a repo "
            "whose private-docs/ directory contains the file."
        )

    with open(reg_path) as f:
        doc = _yaml.safe_load(f)

    models_by_id: dict[str, dict] = {m["id"]: m for m in doc.get("models", [])}
    group_ids: list[str] = doc.get("model_groups", {}).get(group, [])

    if not group_ids:
        raise ValueError(
            f"Model group {group!r} not found in registry. "
            f"Available groups: {list(doc.get('model_groups', {}).keys())}"
        )

    # ── Prefer LiteLLM proxy (format-translating central router) ──────────
    # When the LiteLLM proxy is healthy, it is ALWAYS the best endpoint for
    # the Claude CLI: it translates between CLI wire format and provider-native
    # formats, making every provider in the registry CLI-compatible regardless
    # of its native API format.  It also handles fallback chains with cooldown
    # and passes through caching hints.  All traffic still flows through egress
    # proxies for DLP scanning (placeholder key → real key injection).
    #
    # The probe URL comes from _resolve_litellm_proxy_url() so the port is
    # configurable via KREPIS_LITELLM_PROXY_URL env var (config#4923).
    #
    # Four SOTA checks gate the LiteLLM path:
    #   1. Proxy health (port reachable at the resolved URL)
    #   2. Master key resolvable (env → secrets.env → SSM)
    #   3. Running config not stale vs registry (LiteLLM boot-time config
    #      newer than or equal to registry mtime)
    #   4. Group exists in the registry (for the cache-capability lookup)
    # Any check failing → fall through to per-provider resolution.
    _litellm_url = _resolve_litellm_proxy_url()
    _litellm_reachable = False
    try:
        from urllib.parse import urlparse as _urlparse
        _parsed = _urlparse(_litellm_url)
        _host = _parsed.hostname or "127.0.0.1"
        _port = _parsed.port or 8980
        import http.client as _http
        conn = _http.HTTPConnection(_host, _port, timeout=2)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        # 200 = no master key set; 401 = master key required but proxy alive.
        _litellm_reachable = resp.status in (200, 401)
    except Exception:
        _litellm_reachable = False

    _litellm_ok = False
    _litellm_skip_reasons: list[str] = []

    if not _litellm_reachable:
        _litellm_skip_reasons.append(
            f"LiteLLM proxy at {_litellm_url} not reachable")
    else:
        # ── Check 2: master key resolvable ───────────────────────────────
        _master_key = _resolve_litellm_master_key()
        if _master_key is None:
            _litellm_skip_reasons.append(
                "LITELLM_MASTER_KEY not resolvable (env → secrets.env → SSM)")
        else:
            # ── Check 3: config staleness (RETIRED — externalized) ──────
            # The mtime-glob heuristic (_litellm_config_is_stale) has been
            # RETIRED.  It inferred staleness from filesystem mtimes over a
            # /tmp glob — an inference, not a fact, and nothing called it.
            #
            # The authoritative check lives in alpha-engine-config:
            #   scripts/check_router_config_provenance.py
            # which compares a content digest (sha256) of the generated config
            # against the live registry.  The auto-reconcile script
            # (scripts/reconcile_litellm_config.py) runs on a 10-minute timer
            # and restarts the router on digest mismatch (15-minute SLO, R10).
            #
            # This krepis path skips the staleness check: the external
            # reconcile is authoritative.  Proceed assuming the config is
            # current — the reconcile loop will fix it within minutes if not.
            _litellm_ok = True

    if _litellm_ok:
        # Caching capability comes from the PRIMARY entry — the model that
        # will actually serve the request — never from an any() over the
        # whole fallback chain.
        #
        # It used to be any(). That declares a capability the served model
        # may not have: a group whose primary does transparent prefix
        # caching but whose fourth fallback honours explicit breakpoints
        # reported "explicit breakpoints supported", so the CLI emitted
        # Anthropic cache_control markers at a provider that never asked for
        # them. Live on `ultra` until 2026-07-27
        # (alpha-engine-config-I4463).
        #
        # Cache pricing below was already primary-derived, so the two are
        # now consistent: everything describing "what serves this group"
        # comes from one entry.
        _primary_entry = models_by_id.get(group_ids[0], {}) if group_ids else {}
        _group_pc, _group_apc = _caching_flags(_primary_entry)
        _cache_pricing = {}
        for _key in ("cost_per_1m_input", "cost_per_1m_output",
                      "cost_per_1m_cache_read", "cost_per_1m_cache_write"):
            if _key in _primary_entry:
                _cache_pricing[_key] = _primary_entry[_key]

        # Derive max_tokens from the primary entry's params
        _params = dict(_primary_entry.get("params", {}))
        # LiteLLM doesn't set a default max_tokens — copy from the primary
        # entry so the Claude CLI has a sensible limit.
        if "max_tokens" not in _params:
            _params["max_tokens"] = 8192

        # The primary model from the registry — displayed to the user so they
        # know which concrete model serves this group.  LiteLLM may internally
        # fall back through the chain; the actual model appears in resp.model.
        _primary_model = _primary_entry.get("model", group)
        _primary_registry_id = _primary_entry.get("id", group)

        _display_name = f"{_primary_model} ({group})"

        return _with_compat_aliases({
            "schema_version": RESOLVE_SCHEMA_VERSION,
            "model": group,
            "display_name": _display_name,
            "provider": "litellm",
            "route": "litellm_proxy",
            "api_base_url": _litellm_url,
            "deployment_id": group,
            "auth_token_type": "litellm_master_key",
            "group": group,
            "registry_id": f"litellm:group:{group}",
            "primary_model": _primary_model,
            "primary_registry_id": _primary_registry_id,
            "capabilities": {
                "web_search": False,
                "tool_choice": False,
                "prompt_caching": _group_pc,
                "automatic_prefix_caching": _group_apc,
                "batches": False,
            },
            "params": _params,
            "cache_pricing": _cache_pricing,
            # supports_prompt_caching: only true for explicit cache_control
            # breakpoints.  automatic_prefix_caching works transparently on
            # the server side without client markers.
            "supports_prompt_caching": _group_pc,
            "automatic_prefix_caching": _group_apc,
            "skipped_entries": [],
        })

    # ── LiteLLM unavailable or gated — fall through to per-provider resolution ──
    # Log skip reasons to stderr so operators can diagnose routing decisions.

    if _litellm_skip_reasons:
        _msg = "; ".join(_litellm_skip_reasons)
        logger.warning("LiteLLM proxy skipped for group %r: %s", group, _msg)

    skips: list[dict] = []

    # Iterate the fallback chain — first CLI-compatible entry wins
    for mid in group_ids:
        entry = models_by_id.get(mid)
        if entry is None:
            continue

        # Skip entries whose provider API doesn't speak the CLI wire format.
        # Only DeepSeek (port 8971, natively CLI-compatible) and OpenRouter
        # (server-side format translation) can serve the CLI directly.
        # All other providers require the LiteLLM proxy for format translation.
        try:
            api_base_url = _cli_endpoint_for(entry)
        except ValueError:
            skips.append({
                "registry_id": mid,
                "provider": entry.get("provider", ""),
                "reason": "Provider-native format only — requires LiteLLM proxy for CLI access",
            })
            continue

        model_str = entry.get("model", "")
        route = entry.get("route", "")
        provider = entry.get("provider", "")
        capabilities = entry.get("capabilities", {})
        params = entry.get("params", {})

        # Determine auth token type
        if route == "egress_proxy":
            auth_token_type = "placeholder"
        elif route == "openrouter":
            auth_token_type = "openrouter_key"
        elif provider == "anthropic":
            auth_token_type = "direct_api_key"
        else:
            auth_token_type = "placeholder"

        # Extract cache pricing for the clauder script to decide caching config
        cache_pricing = {}
        for key in ("cost_per_1m_input", "cost_per_1m_output",
                     "cost_per_1m_cache_read", "cost_per_1m_cache_write"):
            if key in entry:
                cache_pricing[key] = entry[key]

        _entry_pc, _entry_apc = _caching_flags(entry)

        _display_name = f"{model_str} ({group})"

        return _with_compat_aliases({
            "schema_version": RESOLVE_SCHEMA_VERSION,
            "model": model_str,
            "display_name": _display_name,
            "provider": provider,
            "route": route,
            "api_base_url": api_base_url,
            "deployment_id": _cli_deployment_id(entry),
            "auth_token_type": auth_token_type,
            "group": group,
            "registry_id": mid,
            "capabilities": capabilities,
            # Same single reader as the LiteLLM branch, so the two paths
            # cannot disagree about a model's caching mechanism.
            "supports_automatic_prefix_caching": _entry_apc,
            "params": params,
            "cache_pricing": cache_pricing,
            "supports_prompt_caching": _entry_pc,
            "skipped_entries": skips if skips else [],
        })

    raise ValueError(
        f"No model in group {group!r} is CLI-compatible. "
        f"Available models: {group_ids}.  "
        "All provider-native-format entries require the LiteLLM proxy "
        "for CLI access — ensure LiteLLM is running on port 8980."
    )


def resolve_group_structured(group: str) -> dict:
    """Resolve *group* to a full routing decision — the PUBLIC contract.

    This is the supported entry point for programmatic callers.  It returns
    the same dict the ``resolve <group> --json`` CLI prints, carrying every
    field a caller needs to configure an LLM client: endpoint, model,
    auth-token type, capabilities, params, and cache pricing.

    The returned dict conforms to ``resolve_schema.json`` (validated in CI)
    and carries ``schema_version`` (:data:`RESOLVE_SCHEMA_VERSION`).  Callers
    MUST branch on ``schema_version`` rather than probing for fields.

    Raises
    ------
    FileNotFoundError
        No registry file could be located.
    ValueError
        *group* is not in the registry, or no model in it is reachable.

    Notes
    -----
    ``alpha-engine-config``'s groom driver has imported this name since it was
    written, while the module only ever exposed the private
    ``_resolve_group_json`` — so the import failed on every run, the router
    path was dead code, and routing silently fell back to an endpoint exported
    by a bootstrap shell script (alpha-engine-config-I4454).  Consumers must
    bind to a supported public surface; a leading underscore is not one.
    """
    return _resolve_group_json(group)


# ── CLI ──────────────────────────────────────────────────────────────────

def _cli() -> None:
    """Entry point for ``python3 -m krepis.router``."""
    if len(sys.argv) < 2:
        print("Usage: python3 -m krepis.router <command> [args]", file=sys.stderr)
        print("  resolve <group> [--json]  — print first healthy model for group", file=sys.stderr)
        print("  groups                    — list all model groups", file=sys.stderr)
        print("  models                    — list all models in the Router", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "resolve":
        if len(sys.argv) < 3:
            print("Usage: python3 -m krepis.router resolve <low|med|high|ultra> [--json]", file=sys.stderr)
            sys.exit(1)
        group = sys.argv[2]
        want_json = "--json" in sys.argv
        if want_json:
            info = _resolve_group_json(group)
            print(json.dumps(info))
        else:
            model = resolve_group(group)
            print(model)

    elif cmd == "groups":
        router = get_router()
        for fb in router.fallbacks:
            group_name = list(fb.keys())[0]
            print(group_name)

    elif cmd == "models":
        router = get_router()
        for m in router.model_list:
            name = m["model_name"]
            upstream = _upstream_model(m["litellm_params"]["model"])
            print(f"{name} → {upstream}")

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
