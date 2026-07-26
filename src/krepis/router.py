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

_egress_placeholder = "unused-placeholder-see-key-isolation-config3007"

# Mapping of (route, provider) tuples to Anthropic Messages API base URLs.
# Only these combinations speak the wire format the Claude CLI expects.
# Port 8971 carries the /anthropic upstream-prefix (Anthropic-format);
# ports 8972/8973/8974 serve OpenAI-format (for LiteLLM, not CLI-compatible).
# Entries NOT in this dict cannot serve as ANTHROPIC_BASE_URL for the
# Claude CLI and are skipped by _resolve_group_json.
_ANTHROPIC_COMPATIBLE_ENDPOINTS: dict[tuple[str, str | None], str] = {
    ("egress_proxy", "deepseek"): "http://127.0.0.1:8971",
    ("openrouter", None):         "https://openrouter.ai/api",
    ("direct", "anthropic"):      "",
}


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


# ── Anthropic wire-format helpers ─────────────────────────────────────────

def _anthropic_endpoint_for(entry: dict) -> str:
    """Return the Anthropic Messages API base URL for a registry model entry.

    Raises :exc:`ValueError` if this entry's route+provider cannot serve
    the Anthropic Messages API wire format (e.g. Gemini/xAI egress proxies
    are OpenAI-format only).
    """
    route = entry.get("route", "")
    provider = entry.get("provider", "")

    key = (route, provider)
    if key in _ANTHROPIC_COMPATIBLE_ENDPOINTS:
        return _ANTHROPIC_COMPATIBLE_ENDPOINTS[key]

    # Also try with None provider (for route-only lookup like openrouter)
    if (route, None) in _ANTHROPIC_COMPATIBLE_ENDPOINTS:
        return _ANTHROPIC_COMPATIBLE_ENDPOINTS[(route, None)]

    raise ValueError(
        f"Model entry {entry.get('id', '?')!r} "
        f"(route={route!r}, provider={provider!r}) "
        "does not serve the Anthropic Messages API wire format and "
        "cannot be used as ANTHROPIC_BASE_URL for the Claude CLI"
    )


def _anthropic_deployment_id(entry: dict) -> str:
    """Return the model string to set as ``ANTHROPIC_MODEL``.

    * For *egress_proxy* routes: bare model name (e.g. ``deepseek-v4-flash``)
      — the proxy translates to the upstream model ID.
    * For *openrouter* routes: full OpenRouter slug (e.g.
      ``deepseek/deepseek-v4-flash``) — already stored in the registry's
      *model* field.
    * For *anthropic* direct: canonical Anthropic model ID.
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

    Reads the registry directly (bypasses the Router's cooldown state) and
    iterates the group's fallback chain to find the **first model whose
    route+provider serves the Anthropic Messages API wire format**.
    Gemini and xAI egress-proxy ports (8974, 8973) speak OpenAI format
    only and are automatically skipped.

    Produces a dict with every field a shell script needs to configure
    the Claude CLI environment: endpoint URL, auth type, provider, route,
    deployment ID, and registry ID.

    The ``anthropic_base_url`` is the **Claude-CLI-compatible** endpoint:
    Anthropic Messages API format.  Port 8971 (DeepSeek) and OpenRouter
    speak this format; ports 8972/8973/8974 are OpenAI-format LiteLLM
    endpoints and cannot serve ``exec claude`` directly.

    Raises :exc:`ValueError` if NO model in the group is Anthropic-compatible.
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

    # Iterate the fallback chain — first Anthropic-compatible entry wins
    for mid in group_ids:
        entry = models_by_id.get(mid)
        if entry is None:
            continue

        # Skip non-Anthropic-compatible entries (Gemini, xAI)
        try:
            anthropic_base_url = _anthropic_endpoint_for(entry)
        except ValueError:
            continue

        model_str = entry.get("model", "")
        route = entry.get("route", "")
        provider = entry.get("provider", "")

        # Determine auth token type
        if route == "egress_proxy":
            auth_token_type = "placeholder"
        elif route == "openrouter":
            auth_token_type = "openrouter_key"
        elif provider == "anthropic":
            auth_token_type = "anthropic_key"
        else:
            auth_token_type = "placeholder"

        return {
            "model": model_str,
            "provider": provider,
            "route": route,
            "anthropic_base_url": anthropic_base_url,
            "deployment_id": _anthropic_deployment_id(entry),
            "auth_token_type": auth_token_type,
            "group": group,
            "registry_id": mid,
        }

    raise ValueError(
        f"No model in group {group!r} supports the Anthropic Messages API "
        f"wire format.  Available models: {group_ids}.  "
        "Gemini and xAI egress-proxy routes speak OpenAI format only and "
        "cannot serve the Claude CLI directly."
    )


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
