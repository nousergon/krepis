"""Tests for krepis.router — registry parsing, model resolution, CLI."""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from krepis import router as _router


# ── _parse_registry ──────────────────────────────────────────────────────

REGISTRY_YAML = """
schema_version: 1

model_groups:
  low:
    - deepseek-v4-flash
    - gemini-2.5-flash
    - gpt-oss-120b
    - gemini-2.5-pro
  med:
    - deepseek-v4-flash-max
    - deepseek-v4-flash-openrouter-max
    - deepseek-v4-pro
  high:
    - deepseek-v4-pro-max
    - deepseek-v4-pro-openrouter-max
  ultra:
    - glm-5.2
    - kimi-k3
    - deepseek-v4-pro-max

models:
  - id: deepseek-v4-flash
    name: DeepSeek V4 Flash
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8972/v1
    model: deepseek-v4-flash
    group: low
    group_role: primary
    params:
      max_tokens: 8192
      reasoning:
        exclude: true
    status: active

  - id: gemini-2.5-flash
    name: Gemini 2.5 Flash
    provider: gemini
    route: egress_proxy
    api_base: http://127.0.0.1:8974/v1beta/openai
    model: gemini-2.5-flash
    group: low
    group_role: fallback
    params:
      max_tokens: 4096
    status: active

  - id: gpt-oss-120b
    name: GPT-OSS 120B
    provider: openrouter
    route: openrouter
    model: openai/gpt-oss-120b
    group: low
    group_role: fallback
    params:
      max_tokens: 4096
    status: active

  - id: gemini-2.5-pro
    name: Gemini 2.5 Pro
    provider: gemini
    route: egress_proxy
    api_base: http://127.0.0.1:8974/v1beta/openai
    model: gemini-2.5-pro
    group: low
    group_role: fallback
    params:
      max_tokens: 8192
    status: active

  - id: deepseek-v4-flash-max
    name: DeepSeek V4 Flash Max
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8972/v1
    model: deepseek-v4-flash
    group: med
    group_role: primary
    params:
      max_tokens: 8192
      reasoning:
        effort: max
    status: active

  - id: deepseek-v4-flash-openrouter-max
    name: DeepSeek V4 Flash (OpenRouter, reasoning=max)
    provider: openrouter
    route: openrouter
    model: deepseek/deepseek-v4-flash
    group: med
    group_role: fallback
    params:
      reasoning:
        effort: max
    status: active

  - id: deepseek-v4-pro
    name: DeepSeek V4 Pro
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8972/v1
    model: deepseek-v4-pro
    group: med
    group_role: fallback
    params:
      max_tokens: 8192
    status: active

  - id: deepseek-v4-pro-max
    name: DeepSeek V4 Pro Max
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8972/v1
    model: deepseek-v4-pro
    group: high
    group_role: primary
    params:
      reasoning:
        effort: max
    status: active

  - id: deepseek-v4-pro-openrouter-max
    name: DeepSeek V4 Pro (OpenRouter, reasoning=max)
    provider: openrouter
    route: openrouter
    model: deepseek/deepseek-v4-pro
    group: high
    group_role: fallback
    params:
      reasoning:
        effort: max
    status: active

  - id: kimi-k3
    name: Kimi K3
    provider: openrouter
    route: openrouter
    model: moonshotai/kimi-k3
    group: ultra
    group_role: fallback
    params:
      max_tokens: 16384
    status: active

  - id: glm-5.2
    name: GLM 5.2
    provider: openrouter
    route: openrouter
    model: zhipuai/glm-5.2
    group: ultra
    group_role: primary
    params:
      max_tokens: 16384
    status: active
"""


@pytest.fixture
def registry_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(REGISTRY_YAML)
    yield Path(f.name)
    os.unlink(f.name)


class TestParseRegistry:
    def test_parses_four_groups(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        assert len(fallbacks) == 4  # low, med, high, ultra all have fallbacks

    def test_low_group_primary(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        low_primary = next(m for m in model_list if m["model_name"] == "low")
        assert "openai/deepseek-v4-flash" in low_primary["litellm_params"]["model"]
        assert "8972/v1" in low_primary["litellm_params"]["api_base"]

    def test_low_group_has_fallback_chain(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        low_fb = next(fb for fb in fallbacks if "low" in fb)
        assert len(low_fb["low"]) == 3
        assert "low-gemini-2.5-flash" in low_fb["low"]

    def test_gemini_routes_to_port_8974(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        gemini = next(m for m in model_list if m["model_name"] == "low-gemini-2.5-flash")
        assert "8974/v1beta/openai" in gemini["litellm_params"]["api_base"]
        assert "openai/gemini-2.5-flash" == gemini["litellm_params"]["model"]

    def test_openrouter_model_has_openrouter_prefix(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        ultra = next(m for m in model_list if m["model_name"] == "ultra")
        assert "openrouter/zhipuai/glm-5.2" == ultra["litellm_params"]["model"]

    def test_openrouter_model_uses_openrouter_key(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file, openrouter_key="test-openrouter-key")
        ultra = next(m for m in model_list if m["model_name"] == "ultra")
        assert ultra["litellm_params"]["api_key"] == "test-openrouter-key"

    def test_egress_proxy_model_uses_placeholder_key(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        low = next(m for m in model_list if m["model_name"] == "low")
        assert "placeholder" in low["litellm_params"]["api_key"]

    def test_primary_model_named_as_group_name(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        model_names = {m["model_name"] for m in model_list}
        assert "low" in model_names
        assert "med" in model_names
        assert "high" in model_names
        assert "ultra" in model_names

    def test_reasoning_param_included(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        med_primary = next(m for m in model_list if m["model_name"] == "med")
        extra = med_primary["litellm_params"].get("extra_body", {})
        assert extra.get("reasoning") == {"effort": "max"}

    def test_reasoning_exclude_included(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        low = next(m for m in model_list if m["model_name"] == "low")
        extra = low["litellm_params"].get("extra_body", {})
        assert extra.get("reasoning") == {"exclude": True}


# ── _upstream_model ──────────────────────────────────────────────────────

class TestUpstreamModel:
    def test_strips_openai_prefix(self):
        assert _router._upstream_model("openai/deepseek-v4-flash") == "deepseek-v4-flash"

    def test_strips_openrouter_prefix(self):
        assert _router._upstream_model("openrouter/moonshotai/kimi-k3") == "moonshotai/kimi-k3"

    def test_no_prefix_passthrough(self):
        assert _router._upstream_model("deepseek-v4-flash") == "deepseek-v4-flash"

    def test_anthropic_prefix(self):
        assert _router._upstream_model("anthropic/claude-opus-4-8") == "claude-opus-4-8"


# ── _find_registry ────────────────────────────────────────────────────────

class TestFindRegistry:
    def test_env_var_wins(self, registry_file):
        with mock.patch.dict(os.environ, {"LLM_MODEL_REGISTRY_PATH": str(registry_file)}):
            found = _router._find_registry()
            assert found == registry_file

    def test_env_var_missing_file_returns_none(self):
        with mock.patch.dict(os.environ, {"LLM_MODEL_REGISTRY_PATH": "/nonexistent/path.yaml"}):
            found = _router._find_registry()
            assert found is None

    def test_returns_none_when_no_registry_found(self, monkeypatch, tmp_path):
        """No env var, no private-docs/ walk match → None."""
        monkeypatch.chdir(tmp_path)
        found = _router._find_registry()
        assert found is None


# ── get_router (registry-only, no builtin fallback) ─────────────────────

class TestGetRouter:
    def test_router_loads_from_registry_file(self, registry_file, monkeypatch):
        """When LLM_MODEL_REGISTRY_PATH points at a valid file, the Router builds."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                router = _router.get_router()
            assert router is not None
            model_names = {m["model_name"] for m in router.model_list}
            assert "low" in model_names
            assert "med" in model_names
            assert "high" in model_names
            assert "ultra" in model_names
        finally:
            _router._router = None

    def test_router_raises_when_no_registry_found(self, monkeypatch, tmp_path):
        """Without LLM_MODEL_REGISTRY_PATH or a walk match, get_router raises."""
        monkeypatch.chdir(tmp_path)
        _router._router = None
        try:
            with pytest.raises(FileNotFoundError, match="LLM_MODEL_REGISTRY.yaml not found"):
                _router.get_router()
        finally:
            _router._router = None

    def test_all_models_have_model_param(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                router = _router.get_router()
            for m in router.model_list:
                assert "model" in m["litellm_params"]
        finally:
            _router._router = None

    def test_fallback_groups(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                router = _router.get_router()
            groups = {list(fb.keys())[0] for fb in router.fallbacks}
            assert groups == {"low", "med", "high", "ultra"}
            low_fb = next(fb for fb in router.fallbacks if "low" in fb)
            assert len(low_fb["low"]) == 3
        finally:
            _router._router = None


# ── CLI ──────────────────────────────────────────────────────────────────

class TestCLI:
    def test_help_exits_1(self, capsys):
        with mock.patch.object(sys, "argv", ["krepis.router"]):
            with pytest.raises(SystemExit) as exc:
                _router._cli()
            assert exc.value.code == 1

    def test_resolve_no_args_exits_1(self, capsys):
        with mock.patch.object(sys, "argv", ["krepis.router", "resolve"]):
            with pytest.raises(SystemExit) as exc:
                _router._cli()
            assert exc.value.code == 1

    def test_unknown_command_exits_1(self, capsys):
        with mock.patch.object(sys, "argv", ["krepis.router", "nonexistent"]):
            with pytest.raises(SystemExit) as exc:
                _router._cli()
            assert exc.value.code == 1


# ── _anthropic_endpoint_for ───────────────────────────────────────────────

class TestAnthropicEndpointFor:
    def test_deepseek_egress_proxy_returns_8971(self):
        entry = {"route": "egress_proxy", "provider": "deepseek", "id": "test-ds"}
        assert _router._anthropic_endpoint_for(entry) == "http://127.0.0.1:8971"

    def test_openrouter_returns_openrouter_api(self):
        entry = {"route": "openrouter", "provider": "openrouter", "id": "test-or"}
        assert _router._anthropic_endpoint_for(entry) == "https://openrouter.ai/api"

    def test_openrouter_any_provider_returns_openrouter_api(self):
        """OpenRouter route matches on route alone, regardless of provider."""
        entry = {"route": "openrouter", "provider": "unknown", "id": "test-or2"}
        assert _router._anthropic_endpoint_for(entry) == "https://openrouter.ai/api"

    def test_anthropic_direct_returns_empty(self):
        entry = {"route": "direct", "provider": "anthropic", "id": "test-an"}
        assert _router._anthropic_endpoint_for(entry) == ""

    def test_gemini_egress_proxy_raises_valueerror(self):
        entry = {"route": "egress_proxy", "provider": "gemini", "id": "test-gem"}
        with pytest.raises(ValueError, match="does not serve"):
            _router._anthropic_endpoint_for(entry)

    def test_xai_egress_proxy_raises_valueerror(self):
        entry = {"route": "egress_proxy", "provider": "xai", "id": "test-xai"}
        with pytest.raises(ValueError, match="does not serve"):
            _router._anthropic_endpoint_for(entry)


# ── _anthropic_deployment_id ──────────────────────────────────────────────

class TestAnthropicDeploymentId:
    def test_egress_proxy_returns_bare_model(self):
        entry = {"route": "egress_proxy", "model": "deepseek-v4-flash"}
        assert _router._anthropic_deployment_id(entry) == "deepseek-v4-flash"

    def test_openrouter_returns_full_slug(self):
        entry = {"route": "openrouter", "model": "deepseek/deepseek-v4-flash"}
        assert _router._anthropic_deployment_id(entry) == "deepseek/deepseek-v4-flash"

    def test_anthropic_direct_returns_model_id(self):
        entry = {"route": "direct", "provider": "anthropic", "model": "claude-sonnet-5"}
        assert _router._anthropic_deployment_id(entry) == "claude-sonnet-5"


# ── _resolve_group_json ────────────────────────────────────────────────

class TestResolveGroupDetailed:
    def test_med_returns_deepseek_egress(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("med")
            assert info["model"] == "deepseek-v4-flash"
            assert info["provider"] == "deepseek"
            assert info["route"] == "egress_proxy"
            assert info["anthropic_base_url"] == "http://127.0.0.1:8971"
            assert info["deployment_id"] == "deepseek-v4-flash"
            assert info["auth_token_type"] == "placeholder"
            assert info["registry_id"] == "deepseek-v4-flash-max"
        finally:
            _router._router = None

    def test_high_returns_deepseek_egress(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("high")
            assert info["model"] == "deepseek-v4-pro"
            assert info["provider"] == "deepseek"
            assert info["route"] == "egress_proxy"
            assert info["anthropic_base_url"] == "http://127.0.0.1:8971"
            assert info["deployment_id"] == "deepseek-v4-pro"
            assert info["auth_token_type"] == "placeholder"
            assert info["registry_id"] == "deepseek-v4-pro-max"
        finally:
            _router._router = None

    def test_ultra_returns_openrouter(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("ultra")
            assert info["model"] == "zhipuai/glm-5.2"
            assert info["provider"] == "openrouter"
            assert info["route"] == "openrouter"
            assert info["anthropic_base_url"] == "https://openrouter.ai/api"
            assert info["deployment_id"] == "zhipuai/glm-5.2"
            assert info["auth_token_type"] == "openrouter_key"
            assert info["registry_id"] == "glm-5.2"
        finally:
            _router._router = None

    def test_low_skips_gemini_returns_deepseek(self, registry_file, monkeypatch):
        """Low group primary is gemini (not Anthropic-compat) — skips to deepseek."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("low")
            # Should skip gemini-2.5-flash (not Anthropic-compat) → deepseek-v4-flash
            assert info["provider"] == "deepseek"
            assert info["route"] == "egress_proxy"
            assert info["anthropic_base_url"] == "http://127.0.0.1:8971"
            assert info["registry_id"] == "deepseek-v4-flash"
            assert info["auth_token_type"] == "placeholder"
        finally:
            _router._router = None

    def test_nonexistent_group_raises_valueerror(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                with pytest.raises(ValueError, match="not found in registry"):
                    _router._resolve_group_json("nonexistent")
        finally:
            _router._router = None

    def test_all_keys_present(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("med")
            for key in ("model", "provider", "route", "anthropic_base_url",
                        "deployment_id", "auth_token_type", "group", "registry_id"):
                assert key in info, f"Missing key: {key}"
        finally:
            _router._router = None


# ── CLI resolve --json ──────────────────────────────────────────────────────

class TestCLIResolveGroup:
    def test_json_output(self, registry_file, monkeypatch, capsys):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                with mock.patch.object(sys, "argv", ["krepis.router", "resolve", "med", "--json"]):
                    _router._cli()
            captured = capsys.readouterr()
            import json
            data = json.loads(captured.out)
            assert data["model"] == "deepseek-v4-flash"
            assert data["anthropic_base_url"] == "http://127.0.0.1:8971"
            assert data["auth_token_type"] == "placeholder"
        finally:
            _router._router = None

    def test_plain_output(self, registry_file, monkeypatch, capsys):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                with mock.patch.object(sys, "argv", ["krepis.router", "resolve", "high"]):
                    _router._cli()
            captured = capsys.readouterr()
            assert "deepseek-v4-pro" in captured.out
        finally:
            _router._router = None

    def test_no_group_exits_1(self, capsys):
        with mock.patch.object(sys, "argv", ["krepis.router", "resolve"]):
            with pytest.raises(SystemExit) as exc:
                _router._cli()
            assert exc.value.code == 1
