"""Tests for krepis.router — registry parsing, model resolution, CLI."""

import os
import sys
import tempfile
from unittest import mock
from pathlib import Path
from unittest import mock

import pytest

from krepis import router as _router

# Save reference to the real _probe_egress_proxy before the autouse fixture
# patches it (so TestProbeEgressProxy can call through to the real function).
_original_probe_egress_proxy = _router._probe_egress_proxy

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


@pytest.fixture(autouse=True)
def _patch_egress_probe():
    """Mock the egress proxy health probe so tests don't depend on a running
    proxy (config#4923).  All egress_proxy routes appear healthy."""
    with mock.patch.object(_router, "_probe_egress_proxy", return_value=True):
        yield


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
                m.delenv("LITELLM_MASTER_KEY", raising=False)
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
                m.delenv("LITELLM_MASTER_KEY", raising=False)
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
                m.delenv("LITELLM_MASTER_KEY", raising=False)
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


# ── _cli_endpoint_for ───────────────────────────────────────────────

class TestAnthropicEndpointFor:
    def test_deepseek_egress_proxy_returns_8971(self):
        entry = {"route": "egress_proxy", "provider": "deepseek", "id": "test-ds"}
        assert _router._cli_endpoint_for(entry) == "http://127.0.0.1:8971"

    def test_openrouter_returns_openrouter_api(self):
        entry = {"route": "openrouter", "provider": "openrouter", "id": "test-or"}
        assert _router._cli_endpoint_for(entry) == "https://openrouter.ai/api"

    def test_openrouter_any_provider_returns_openrouter_api(self):
        """OpenRouter route matches on route alone, regardless of provider."""
        entry = {"route": "openrouter", "provider": "unknown", "id": "test-or2"}
        assert _router._cli_endpoint_for(entry) == "https://openrouter.ai/api"

    def test_anthropic_direct_returns_empty(self):
        entry = {"route": "direct", "provider": "anthropic", "id": "test-an"}
        assert _router._cli_endpoint_for(entry) == ""

    def test_gemini_egress_proxy_raises_valueerror(self):
        entry = {"route": "egress_proxy", "provider": "gemini", "id": "test-gem"}
        with pytest.raises(ValueError, match="not a CLI-compatible endpoint"):
            _router._cli_endpoint_for(entry)

    def test_xai_egress_proxy_raises_valueerror(self):
        entry = {"route": "egress_proxy", "provider": "xai", "id": "test-xai"}
        with pytest.raises(ValueError, match="not a CLI-compatible endpoint"):
            _router._cli_endpoint_for(entry)

    def test_egress_proxy_env_override_wins(self):
        """KREPIS_DEEPSEEK_EGRESS_URL env var overrides the hardcoded default."""
        entry = {"route": "egress_proxy", "provider": "deepseek", "id": "test-ds"}
        with mock.patch.dict(os.environ, {"KREPIS_DEEPSEEK_EGRESS_URL": "http://127.0.0.1:9999"}):
            result = _router._cli_endpoint_for(entry)
        assert result == "http://127.0.0.1:9999"

    def test_litellm_proxy_url_resolved_from_env(self):
        """KREPIS_LITELLM_PROXY_URL env var overrides the LITELLM_PROXY_URL constant."""
        with mock.patch.dict(os.environ, {"KREPIS_LITELLM_PROXY_URL": "http://127.0.0.1:9090"}):
            result = _router._resolve_litellm_proxy_url()
        assert result == "http://127.0.0.1:9090"

    def test_litellm_proxy_url_falls_back_to_constant(self):
        """Without env var, _resolve_litellm_proxy_url returns the module constant."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _router._resolve_litellm_proxy_url()
        assert result == _router.LITELLM_PROXY_URL


    def test_openrouter_env_override_wins(self):
        """KREPIS_OPENROUTER_API_URL env var overrides the hardcoded default."""
        entry = {"route": "openrouter", "provider": "openrouter", "id": "test-or"}
        with mock.patch.dict(os.environ, {"KREPIS_OPENROUTER_API_URL": "https://custom.example.com/api"}):
            result = _router._cli_endpoint_for(entry)
        assert result == "https://custom.example.com/api"

    def test_openrouter_env_override_catchall_any_provider(self):
        """KREPIS_OPENROUTER_API_URL applies to any provider on the openrouter route."""
        entry = {"route": "openrouter", "provider": "unknown", "id": "test-or2"}
        with mock.patch.dict(os.environ, {"KREPIS_OPENROUTER_API_URL": "https://custom.example.com/api"}):
            result = _router._cli_endpoint_for(entry)
        assert result == "https://custom.example.com/api"

    def test_litellm_proxy_env_override_in_endpoint_for(self):
        """KREPIS_LITELLM_PROXY_URL is used as the litellm_proxy route endpoint."""
        entry = {"route": "litellm_proxy", "provider": "litellm", "id": "test-lm"}
        with mock.patch.dict(os.environ, {"KREPIS_LITELLM_PROXY_URL": "http://127.0.0.1:9990"}):
            result = _router._cli_endpoint_for(entry)
        assert result == "http://127.0.0.1:9990"

    def test_egress_proxy_env_override_empty_string_ignored(self):
        """An empty env var override string is treated as unset — falls back to default."""
        entry = {"route": "egress_proxy", "provider": "deepseek", "id": "test-ds"}
        with mock.patch.dict(os.environ, {"KREPIS_DEEPSEEK_EGRESS_URL": ""}):
            result = _router._cli_endpoint_for(entry)
        assert result == "http://127.0.0.1:8971"


# ── _probe_egress_proxy ──────────────────────────────────────────────────

class TestProbeEgressProxy:
    def test_returns_true_when_proxy_healthy(self):
        """When the proxy returns 200, _probe_egress_proxy returns True."""
        with mock.patch("http.client.HTTPConnection") as mock_conn:
            mock_resp = mock.MagicMock()
            mock_resp.status = 200
            mock_conn.return_value.getresponse.return_value = mock_resp
            result = _original_probe_egress_proxy("http://127.0.0.1:8971")
        assert result is True

    def test_returns_false_when_proxy_returns_500(self):
        """When the proxy returns a non-200 status, _probe_egress_proxy returns False."""
        with mock.patch("http.client.HTTPConnection") as mock_conn:
            mock_resp = mock.MagicMock()
            mock_resp.status = 500
            mock_conn.return_value.getresponse.return_value = mock_resp
            result = _original_probe_egress_proxy("http://127.0.0.1:8971")
        assert result is False

    def test_returns_false_on_connection_error(self):
        """When the proxy is unreachable, _probe_egress_proxy returns False (no exception)."""
        with mock.patch("http.client.HTTPConnection") as mock_conn:
            mock_conn.return_value.request.side_effect = ConnectionRefusedError
            result = _original_probe_egress_proxy("http://127.0.0.1:8971")
        assert result is False

    def test_returns_false_for_empty_url(self):
        """An empty URL (direct route) is not an egress proxy — returns False."""
        result = _original_probe_egress_proxy("")
        assert result is False

    def test_uses_urlparse_for_port_extraction(self):
        """Custom port in the URL is honoured by urlparse."""
        with mock.patch("http.client.HTTPConnection") as mock_conn:
            mock_resp = mock.MagicMock()
            mock_resp.status = 200
            mock_conn.return_value.getresponse.return_value = mock_resp
            _original_probe_egress_proxy("http://127.0.0.1:9999")
        # Verify the connection was made to the right port
        assert mock_conn.call_args[0][1] == 9999

    def test_default_host_and_port_when_url_has_no_hostname(self):
        """When the URL has no hostname (e.g. missing scheme), defaults to 127.0.0.1:8971."""
        with mock.patch("http.client.HTTPConnection") as mock_conn:
            mock_resp = mock.MagicMock()
            mock_resp.status = 200
            mock_conn.return_value.getresponse.return_value = mock_resp
            _original_probe_egress_proxy("http://:8971")
        assert mock_conn.call_args[0][0] == "127.0.0.1"
        assert mock_conn.call_args[0][1] == 8971


# ── _cli_deployment_id ──────────────────────────────────────────────

class TestAnthropicDeploymentId:
    def test_egress_proxy_returns_bare_model(self):
        entry = {"route": "egress_proxy", "model": "deepseek-v4-flash"}
        assert _router._cli_deployment_id(entry) == "deepseek-v4-flash"

    def test_openrouter_returns_full_slug(self):
        entry = {"route": "openrouter", "model": "deepseek/deepseek-v4-flash"}
        assert _router._cli_deployment_id(entry) == "deepseek/deepseek-v4-flash"

    def test_anthropic_direct_returns_model_id(self):
        entry = {"route": "direct", "provider": "anthropic", "model": "claude-sonnet-5"}
        assert _router._cli_deployment_id(entry) == "claude-sonnet-5"


# ── _resolve_group_json ────────────────────────────────────────────────

class TestResolveGroupDetailed:
    def test_med_returns_deepseek_egress(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("med")
            assert info["model"] == "deepseek-v4-flash"
            assert info["provider"] == "deepseek"
            assert info["route"] == "egress_proxy"
            assert info["api_base_url"] == "http://127.0.0.1:8971"
            assert info["deployment_id"] == "deepseek-v4-flash"
            assert info["auth_token_type"] == "placeholder"
            assert info["registry_id"] == "deepseek-v4-flash-max"
        finally:
            _router._router = None

    def test_high_returns_deepseek_egress(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("high")
            assert info["model"] == "deepseek-v4-pro"
            assert info["provider"] == "deepseek"
            assert info["route"] == "egress_proxy"
            assert info["api_base_url"] == "http://127.0.0.1:8971"
            assert info["deployment_id"] == "deepseek-v4-pro"
            assert info["auth_token_type"] == "placeholder"
            assert info["registry_id"] == "deepseek-v4-pro-max"
        finally:
            _router._router = None

    def test_ultra_returns_openrouter(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("ultra")
            assert info["model"] == "zhipuai/glm-5.2"
            assert info["provider"] == "openrouter"
            assert info["route"] == "openrouter"
            assert info["api_base_url"] == "https://openrouter.ai/api"
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
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("low")
            # Should skip gemini-2.5-flash (not Anthropic-compat) → deepseek-v4-flash
            assert info["provider"] == "deepseek"
            assert info["route"] == "egress_proxy"
            assert info["api_base_url"] == "http://127.0.0.1:8971"
            assert info["registry_id"] == "deepseek-v4-flash"
            assert info["auth_token_type"] == "placeholder"
        finally:
            _router._router = None

    def test_nonexistent_group_raises_valueerror(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                with pytest.raises(ValueError, match="not found in registry"):
                    _router._resolve_group_json("nonexistent")
        finally:
            _router._router = None

    def test_all_keys_present(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("med")
            for key in ("model", "provider", "route", "api_base_url",
                        "deployment_id", "auth_token_type", "group", "registry_id"):
                assert key in info, f"Missing key: {key}"
        finally:
            _router._router = None

    def test_exclude_route_egress_proxy_skips_to_openrouter(self, registry_file, monkeypatch):
        """Med group primary is egress_proxy - with exclude_route=egress_proxy
        it should skip to the OpenRouter fallback (deepseek-v4-flash-openrouter-max)."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router._resolve_group_json("med", exclude_route="egress_proxy")
            assert info["route"] == "openrouter"
            assert info["provider"] == "openrouter"
            assert info["auth_token_type"] == "openrouter_key"
            assert info["deployment_id"] == "deepseek/deepseek-v4-flash"
            assert info["registry_id"] == "deepseek-v4-flash-openrouter-max"
        finally:
            _router._router = None


# ── _resolve_litellm_master_key ────────────────────────────────────────────

class TestResolveLitellmMasterKey:
    def test_env_var_wins(self):
        """LITELLM_MASTER_KEY env var is the highest-priority source."""
        with mock.patch.dict(os.environ, {"LITELLM_MASTER_KEY": "test-master-key-env"}):
            result = _router._resolve_litellm_master_key()
        assert result == "test-master-key-env"

    def test_env_var_empty_ignored(self):
        """An empty or whitespace-only LITELLM_MASTER_KEY is treated as unset."""
        with mock.patch.dict(os.environ, {"LITELLM_MASTER_KEY": "  "}):
            # Prevent falling through to real secrets.env on disk (I5224)
            with mock.patch("os.path.expanduser", return_value="/nonexistent"):
                result = _router._resolve_litellm_master_key()
        # Should fall through to None since no secrets file or SSM is available.
        # Use boolean capture to avoid rendering the key value on failure.
        is_none = result is None
        assert is_none

    def test_secrets_file_found(self, tmp_path):
        """A secrets.env file with LITELLM_MASTER_KEY=... is read."""
        secrets_file = tmp_path / "secrets.env"
        secrets_file.write_text("LITELLM_MASTER_KEY=test-master-key-secrets\n")
        # Ensure no env var interferes; redirect expanduser to our temp dir
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.path.expanduser", return_value=str(secrets_file)):
                result = _router._resolve_litellm_master_key()
        assert result == "test-master-key-secrets"

    def test_secrets_file_quoted_value_stripped(self, tmp_path):
        """Quoted values in secrets.env are stripped."""
        secrets_file = tmp_path / "secrets.env"
        secrets_file.write_text('LITELLM_MASTER_KEY="test-quoted"\n')
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("os.path.expanduser", return_value=str(secrets_file)):
                result = _router._resolve_litellm_master_key()
        assert result == "test-quoted"

    def test_no_key_returns_none(self):
        """When no source has the key, returns None."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _router._resolve_litellm_master_key()
        assert result is None


# ── CLI resolve --json ──────────────────────────────────────────────────────

class TestCLIResolveGroup:
    def test_json_output(self, registry_file, monkeypatch, capsys):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                with mock.patch.object(sys, "argv", ["krepis.router", "resolve", "med", "--json"]):
                    _router._cli()
            captured = capsys.readouterr()
            import json
            data = json.loads(captured.out)
            assert data["model"] == "deepseek-v4-flash"
            assert data["api_base_url"] == "http://127.0.0.1:8971"
            assert data["auth_token_type"] == "placeholder"
        finally:
            _router._router = None

    def test_plain_output(self, registry_file, monkeypatch, capsys):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
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


# ── resolve contract: schema + versioning ────────────────────────────────
# Regression cover for alpha-engine-config-I4453 (the anthropic_base_url ->
# api_base_url rename that broke all four consumers) and I4454 (groom_driver
# importing resolve_group_structured, which had never existed).

import json as _json


def _load_resolve_schema() -> dict:
    schema_path = Path(_router.__file__).parent / "resolve_schema.json"
    return _json.loads(schema_path.read_text())


class TestResolveContract:
    def test_schema_file_ships_with_the_package(self):
        schema = _load_resolve_schema()
        assert schema["type"] == "object"
        assert "schema_version" in schema["required"]

    def test_public_structured_resolver_is_exported(self):
        """groom_driver binds to this name; a private _-prefixed function is
        not a supported surface (alpha-engine-config-I4454)."""
        assert hasattr(_router, "resolve_group_structured")
        assert callable(_router.resolve_group_structured)

    @pytest.mark.parametrize("group", ["low", "med", "high", "ultra"])
    def test_every_group_validates_against_the_schema(
            self, group, registry_file, monkeypatch):
        jsonschema = pytest.importorskip("jsonschema")
        schema = _load_resolve_schema()
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured(group)
            jsonschema.validate(instance=info, schema=schema)
        finally:
            _router._router = None

    @pytest.mark.parametrize("group", ["low", "med", "high", "ultra"])
    def test_required_fields_present_for_every_group(
            self, group, registry_file, monkeypatch):
        """Schema-independent guard, so the contract still holds when
        jsonschema is not installed."""
        schema = _load_resolve_schema()
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured(group)
        finally:
            _router._router = None
        missing = [k for k in schema["required"] if k not in info]
        assert not missing, f"group {group!r} missing contract fields: {missing}"

    def test_carries_schema_version(self, registry_file, monkeypatch):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured("med")
        finally:
            _router._router = None
        assert info["schema_version"] == _router.RESOLVE_SCHEMA_VERSION
        assert _router.RESOLVE_SCHEMA_VERSION >= 2

    def test_deprecated_alias_still_emitted_for_unmigrated_consumers(
            self, registry_file, monkeypatch):
        """Additive-then-remove (model-router-policy R19): the old name is
        emitted alongside the new one for one release so a rename cannot
        break consumers the way I4453 did."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured("med")
        finally:
            _router._router = None
        assert info["anthropic_base_url"] == info["api_base_url"]

    def test_api_base_url_is_never_missing_or_none(
            self, registry_file, monkeypatch):
        """A missing/None base URL silently resolves to api.anthropic.com in
        the Claude CLI -- the failure mode two consumers hit under I4453."""
        for group in ("low", "med", "high", "ultra"):
            _router._router = None
            try:
                with monkeypatch.context() as m:
                    m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                    info = _router.resolve_group_structured(group)
            finally:
                _router._router = None
            assert "api_base_url" in info
            assert info["api_base_url"] is not None

    def test_alias_table_names_a_removal_version(self):
        """Every deprecated alias must declare when it goes away, so the
        compatibility shim cannot become permanent."""
        assert _router._DEPRECATED_RESOLVE_ALIASES
        for old, new, remove_after in _router._DEPRECATED_RESOLVE_ALIASES:
            assert isinstance(remove_after, int)
            assert remove_after > _router.RESOLVE_SCHEMA_VERSION - 1
            assert old != new
