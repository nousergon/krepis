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

    def test_falls_back_to_shipped_registry(self, monkeypatch, tmp_path):
        """When no env var and no cwd-relative registry, the shipped file is found."""
        monkeypatch.chdir(tmp_path)
        found = _router._find_registry()
        assert found is not None
        assert found.name == "LLM_MODEL_REGISTRY.yaml"


# ── get_router (registry-based, no builtin fallback) ────────────────────

class TestGetRouter:
    def test_router_loads_from_shipped_yaml(self, monkeypatch, tmp_path):
        """The shipped YAML is found and builds a working Router."""
        monkeypatch.chdir(tmp_path)
        # Clear the singleton so it rebuilds from the shipped file
        _router._router = None
        router = _router.get_router()
        assert router is not None
        model_names = {m["model_name"] for m in router.model_list}
        assert "low" in model_names
        assert "med" in model_names
        assert "high" in model_names
        assert "ultra" in model_names

    def test_router_raises_when_no_registry_found(self, monkeypatch, tmp_path):
        """Without the shipped YAML (or any override), get_router raises."""
        monkeypatch.chdir(tmp_path)
        _router._router = None
        # Hide the shipped file by pointing __file__ somewhere else
        real_file = _router.__file__
        try:
            _router.__file__ = str(tmp_path / "nonexistent" / "router.py")
            with pytest.raises(FileNotFoundError, match="LLM_MODEL_REGISTRY.yaml not found"):
                _router.get_router()
        finally:
            _router.__file__ = real_file
            _router._router = None

    def test_all_models_have_model_param(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _router._router = None
        try:
            router = _router.get_router()
            for m in router.model_list:
                assert "model" in m["litellm_params"]
        finally:
            _router._router = None

    def test_fallback_groups(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _router._router = None
        try:
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
