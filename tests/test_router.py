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
        low_primary = next(m for m in model_list if m["model_name"] == "deepseek-v4-flash")
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
        glm = next(m for m in model_list if m["model_name"] == "glm-5.2")
        assert "openrouter/zhipuai/glm-5.2" == glm["litellm_params"]["model"]

    def test_openrouter_model_uses_openrouter_key(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file, openrouter_key="test-openrouter-key")
        glm = next(m for m in model_list if m["model_name"] == "glm-5.2")
        assert glm["litellm_params"]["api_key"] == "test-openrouter-key"

    def test_egress_proxy_model_uses_placeholder_key(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        ds = next(m for m in model_list if m["model_name"] == "deepseek-v4-flash")
        assert "placeholder" in ds["litellm_params"]["api_key"]

    def test_primary_model_named_as_group_name(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        model_names = {m["model_name"] for m in model_list}
        assert "deepseek-v4-flash" in model_names  # low primary
        assert "deepseek-v4-flash-max" in model_names  # med primary
        assert "deepseek-v4-pro-max" in model_names  # high primary
        assert "glm-5.2" in model_names  # ultra primary

    def test_reasoning_param_included(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        flash_max = next(m for m in model_list if m["model_name"] == "deepseek-v4-flash-max")
        extra = flash_max["litellm_params"].get("extra_body", {})
        assert extra.get("reasoning") == {"effort": "max"}

    def test_reasoning_exclude_included(self, registry_file):
        model_list, fallbacks = _router._parse_registry(registry_file)
        ds = next(m for m in model_list if m["model_name"] == "deepseek-v4-flash")
        extra = ds["litellm_params"].get("extra_body", {})
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

    def test_no_env_var_no_registry_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        found = _router._find_registry()
        assert found is None


# ── _builtin_model_list ──────────────────────────────────────────────────

class TestBuiltinModelList:
    def test_has_four_groups(self):
        models = _router._builtin_model_list()
        model_names = {m["model_name"] for m in models}
        assert "low" in model_names
        assert "med" in model_names
        assert "high" in model_names
        assert "ultra" in model_names

    def test_has_gemini_models(self):
        models = _router._builtin_model_list()
        model_names = {m["model_name"] for m in models}
        assert "low-gemini-flash" in model_names
        assert "low-gemini-pro" in model_names

    def test_has_fallback_models(self):
        models = _router._builtin_model_list()
        model_names = {m["model_name"] for m in models}
        assert "med-openrouter" in model_names
        assert "high-openrouter" in model_names
        assert "ultra-kimi" in model_names
        assert "ultra-deepseek" in model_names

    def test_all_models_have_model_param(self):
        models = _router._builtin_model_list()
        for m in models:
            assert "model" in m["litellm_params"]


# ── _builtin_fallbacks ───────────────────────────────────────────────────

class TestBuiltinFallbacks:
    def test_has_all_four_groups(self):
        fallbacks = _router._builtin_fallbacks()
        groups = {list(fb.keys())[0] for fb in fallbacks}
        assert groups == {"low", "med", "high", "ultra"}

    def test_low_has_three_fallbacks(self):
        fallbacks = _router._builtin_fallbacks()
        low = next(fb for fb in fallbacks if "low" in fb)
        assert len(low["low"]) == 3


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
