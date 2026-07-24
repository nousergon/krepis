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

# ── builtin fallback (mirrors the pre-registry hardcoded list) ──────────

_egress_placeholder = "unused-placeholder-see-key-isolation-config3007"


def _builtin_model_list(openrouter_key: str = "") -> list[dict]:
    """Hardcoded model list — used when the registry file can't be found."""
    return [
        # low group
        {"model_name": "low", "litellm_params": {"model": "openai/deepseek-v4-flash", "api_base": "http://127.0.0.1:8972/v1", "api_key": _egress_placeholder}},
        {"model_name": "low-gemini-flash", "litellm_params": {"model": "openai/gemini-2.5-flash", "api_base": "http://127.0.0.1:8974/v1beta/openai", "api_key": _egress_placeholder}},
        {"model_name": "low-gpt-oss", "litellm_params": {"model": "openrouter/openai/gpt-oss-120b", "api_key": openrouter_key}},
        {"model_name": "low-gemini-pro", "litellm_params": {"model": "openai/gemini-2.5-pro", "api_base": "http://127.0.0.1:8974/v1beta/openai", "api_key": _egress_placeholder}},
        # med group
        {"model_name": "med", "litellm_params": {"model": "openai/deepseek-v4-flash", "api_base": "http://127.0.0.1:8972/v1", "api_key": _egress_placeholder, "extra_body": {"reasoning": {"effort": "max"}}}},
        {"model_name": "med-openrouter", "litellm_params": {"model": "openrouter/deepseek/deepseek-v4-flash", "api_key": openrouter_key, "extra_body": {"reasoning": {"effort": "max"}}}},
        {"model_name": "med-degrade", "litellm_params": {"model": "openai/deepseek-v4-pro", "api_base": "http://127.0.0.1:8972/v1", "api_key": _egress_placeholder}},
        # high group
        {"model_name": "high", "litellm_params": {"model": "openai/deepseek-v4-pro", "api_base": "http://127.0.0.1:8972/v1", "api_key": _egress_placeholder, "extra_body": {"reasoning": {"effort": "max"}}}},
        {"model_name": "high-openrouter", "litellm_params": {"model": "openrouter/deepseek/deepseek-v4-pro", "api_key": openrouter_key, "extra_body": {"reasoning": {"effort": "max"}}}},
        # ultra group
        {"model_name": "ultra", "litellm_params": {"model": "openrouter/moonshotai/kimi-k3", "api_key": openrouter_key}},
        {"model_name": "ultra-glm", "litellm_params": {"model": "openrouter/zhipuai/glm-5.2", "api_key": openrouter_key}},
        {"model_name": "ultra-degrade", "litellm_params": {"model": "openai/deepseek-v4-pro", "api_base": "http://127.0.0.1:8972/v1", "api_key": _egress_placeholder, "extra_body": {"reasoning": {"effort": "max"}}}},
    ]


def _builtin_fallbacks() -> list[dict]:
    return [
        {"low": ["low-gemini-flash", "low-gpt-oss", "low-gemini-pro"]},
        {"med": ["med-openrouter", "med-degrade"]},
        {"high": ["high-openrouter"]},
        {"ultra": ["ultra-glm", "ultra-degrade"]},
    ]


# ── registry file discovery ─────────────────────────────────────────────

def _find_registry() -> Optional[Path]:
    """Resolve the registry file path."""
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

            model_name = mid if i == 0 else f"{group_name}-{mid}"
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
        # Map provider to egress proxy port
        if provider == "deepseek":
            litellm_params["api_base"] = "http://127.0.0.1:8972/v1"
        elif provider in ("xai", "grok"):
            litellm_params["api_base"] = "http://127.0.0.1:8973/v1"
        elif provider == "gemini":
            litellm_params["api_base"] = "http://127.0.0.1:8974/v1beta/openai"
    else:
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
    """Return the module-level Router singleton, building from registry or builtins."""
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
        if reg_path:
            logger.info("building Router from %s", reg_path)
            model_list, fallbacks = _parse_registry(reg_path, openrouter_key)
        else:
            logger.warning("LLM_MODEL_REGISTRY.yaml not found — using builtin model list")
            model_list = _builtin_model_list(openrouter_key)
            fallbacks = _builtin_fallbacks()

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


# ── CLI ──────────────────────────────────────────────────────────────────

def _cli() -> None:
    """Entry point for ``python3 -m krepis.router``."""
    if len(sys.argv) < 2:
        print("Usage: python3 -m krepis.router <command> [args]", file=sys.stderr)
        print("  resolve <group>  — print first healthy model for group", file=sys.stderr)
        print("  groups           — list all model groups", file=sys.stderr)
        print("  models           — list all models in the Router", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "resolve":
        if len(sys.argv) < 3:
            print("Usage: python3 -m krepis.router resolve <low|med|high|ultra>", file=sys.stderr)
            sys.exit(1)
        group = sys.argv[2]
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
