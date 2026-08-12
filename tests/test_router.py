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

# Same trick for the SSM leg: conftest's autouse `_no_ssm_master_key_lookup_from_tests`
# stubs it for the whole suite, so TestLitellmMasterKeyFromSSM — the one class
# that tests the leg itself — has to hold the real function.
_original_litellm_master_key_from_ssm = _router._litellm_master_key_from_ssm

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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
    reachable_from: [laptop, ec2]
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
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        # Chains are dual-keyed (group name + qualified primary name); the
        # GROUP list is the alias map.
        assert set(aliases) == {"low", "med", "high", "ultra"}
        group_keyed = {k for fb in fallbacks for k in fb if k in aliases}
        assert group_keyed == {"low", "med", "high", "ultra"}

    def test_low_group_primary(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        low_primary = next(
            m for m in model_list if m["model_name"] == "low-deepseek-v4-flash"
        )
        assert "openai/deepseek-v4-flash" in low_primary["litellm_params"]["model"]
        assert "8972/v1" in low_primary["litellm_params"]["api_base"]

    def test_low_group_has_fallback_chain(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        low_fb = next(fb for fb in fallbacks if "low" in fb)
        assert len(low_fb["low"]) == 3
        assert "low-gemini-2.5-flash" in low_fb["low"]

    def test_gemini_routes_to_port_8974(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        gemini = next(m for m in model_list if m["model_name"] == "low-gemini-2.5-flash")
        assert "8974/v1beta/openai" in gemini["litellm_params"]["api_base"]
        assert "openai/gemini-2.5-flash" == gemini["litellm_params"]["model"]

    def test_openrouter_model_has_openrouter_prefix(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        ultra = next(m for m in model_list if m["model_name"] == "ultra-glm-5.2")
        assert "openrouter/zhipuai/glm-5.2" == ultra["litellm_params"]["model"]

    def test_openrouter_model_uses_openrouter_key(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file, openrouter_key="test-openrouter-key")
        ultra = next(m for m in model_list if m["model_name"] == "ultra-glm-5.2")
        assert ultra["litellm_params"]["api_key"] == "test-openrouter-key"

    def test_egress_proxy_model_uses_placeholder_key(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        low = next(m for m in model_list if m["model_name"] == "low-deepseek-v4-flash")
        assert "placeholder" in low["litellm_params"]["api_key"]

    def test_groups_addressable_via_alias_map(self, registry_file):
        """Every group stays addressable by its bare name — through the
        ``model_group_alias`` map onto the primary's qualified deployment
        name, never through a deployment NAMED with the bare group name
        (the property test_primary_model_named_as_group_name asserted
        before 0.39.0)."""
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        assert aliases == {
            "low": "low-deepseek-v4-flash",
            "med": "med-deepseek-v4-flash-max",
            "high": "high-deepseek-v4-pro-max",
            "ultra": "ultra-glm-5.2",
        }
        model_names = {m["model_name"] for m in model_list}
        for alias, target in aliases.items():
            assert target in model_names

    def test_no_deployment_is_named_with_a_bare_group_name(self, registry_file):
        """Regression, alpha-engine-config-I6543 (2026-08-09): naming the
        primary deployment with the bare group name made LiteLLM report the
        group ALIAS as ``response.model`` on every healthy primary-served
        call, tripping the served-model guard on wire transports."""
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        model_names = {m["model_name"] for m in model_list}
        assert model_names.isdisjoint(aliases.keys())

    def test_fallback_chains_are_dual_keyed(self, registry_file):
        """Measured on litellm 1.93.0: the fallback lookup key is the model
        name AS ADDRESSED BY THE CALLER — alias resolution never rewrites
        it. The group-name key serves alias-addressed calls; the
        qualified-primary key serves callers addressing the primary
        deployment directly."""
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        by_key = {k: v for fb in fallbacks for k, v in fb.items()}
        for alias, target in aliases.items():
            assert by_key[alias] == by_key[target]

    def test_reasoning_param_included(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        med_primary = next(
            m for m in model_list if m["model_name"] == "med-deepseek-v4-flash-max"
        )
        extra = med_primary["litellm_params"].get("extra_body", {})
        assert extra.get("reasoning") == {"effort": "max"}

    def test_reasoning_exclude_included(self, registry_file):
        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        low = next(m for m in model_list if m["model_name"] == "low-deepseek-v4-flash")
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
            # Groups are addressable through the Router's model_group_alias,
            # never through a deployment named with the bare group name.
            aliases = dict(router.model_group_alias)
            assert set(aliases) == {"low", "med", "high", "ultra"}
            model_names = {m["model_name"] for m in router.model_list}
            assert model_names.isdisjoint(aliases.keys())
            for target in aliases.values():
                assert target in model_names
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
            keys = {k for fb in router.fallbacks for k in fb}
            # Dual-keyed: every group's chain is reachable both by the bare
            # group name (alias-addressed calls) and by the primary's
            # qualified deployment name (qualified-addressed calls).
            assert {"low", "med", "high", "ultra"} <= keys
            assert "low-deepseek-v4-flash" in keys
            low_fb = next(fb for fb in router.fallbacks if "low" in fb)
            assert len(low_fb["low"]) == 3
        finally:
            _router._router = None

    def test_alias_addressed_call_engages_group_keyed_fallbacks(
        self, registry_file, monkeypatch
    ):
        """End-to-end against a REAL litellm Router: an alias-addressed call
        whose primary deployment fails must engage the group's fallback
        chain. This pins the measured litellm behavior the dual-keying
        relies on — the fallback lookup key is the name the caller
        addressed (the alias), not the alias-resolved deployment name."""
        from litellm import Router

        model_list, fallbacks, aliases = _router._parse_registry(registry_file)
        by_name = {m["model_name"]: m for m in model_list}
        by_name["low-deepseek-v4-flash"]["litellm_params"]["mock_response"] = (
            "litellm.InternalServerError"
        )
        by_name["low-gemini-2.5-flash"]["litellm_params"]["mock_response"] = (
            "served-by-fallback"
        )
        router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            model_group_alias=aliases,
            num_retries=0,
        )
        resp = router.completion(
            model="low", messages=[{"role": "user", "content": "hi"}]
        )
        assert resp.choices[0].message.content == "served-by-fallback"


# ── group resolution through the alias map ───────────────────────────────

class TestGroupResolutionThroughAlias:
    """The internal lookups that assumed the primary deployment was NAMED
    with the bare group name (``m["model_name"] == group``) must resolve
    through the model_group_alias instead (alpha-engine-config-I6543)."""

    @pytest.fixture(autouse=True)
    def _fresh_router(self, registry_file, monkeypatch):
        _router._router = None
        monkeypatch.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
        yield
        _router._router = None

    def test_get_group_primary_resolves_through_alias(self):
        assert _router.get_group_primary("low") == "openai/deepseek-v4-flash"
        assert _router.get_group_primary("ultra") == "openrouter/zhipuai/glm-5.2"

    def test_get_group_primary_unknown_group_returns_none(self):
        assert _router.get_group_primary("nonexistent") is None

    def test_resolve_group_returns_primary_upstream_model(self):
        assert _router.resolve_group("low") == "deepseek-v4-flash"

    def test_cli_groups_lists_all_groups(self, capsys):
        with mock.patch.object(sys, "argv", ["krepis.router", "groups"]):
            _router._cli()
        out = capsys.readouterr().out.split()
        assert set(out) == {"low", "med", "high", "ultra"}


# ── served_model_for_deployment ──────────────────────────────────────────

class TestServedModelForDeployment:
    """A response reporting the qualified ``{group}-{mid}`` deployment name
    must be resolvable to the registry entry's upstream model — the
    (model, route) identifier price cards are keyed on."""

    @pytest.fixture(autouse=True)
    def _registry_env(self, registry_file, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))

    def test_qualified_primary_resolves_to_upstream_model(self):
        assert (
            _router.served_model_for_deployment("low-deepseek-v4-flash")
            == "deepseek-v4-flash"
        )

    def test_qualified_fallback_resolves_to_upstream_model(self):
        assert (
            _router.served_model_for_deployment("low-gemini-2.5-flash")
            == "gemini-2.5-flash"
        )

    def test_openrouter_entry_keeps_its_route_correct_slug(self):
        # Cards are per (model, ROUTE): the OpenRouter entry's slug is the
        # pricing key and must come back verbatim, not stripped.
        assert (
            _router.served_model_for_deployment("ultra-kimi-k3")
            == "moonshotai/kimi-k3"
        )

    def test_non_deployment_name_returns_none(self):
        assert _router.served_model_for_deployment("deepseek-v4-flash") is None
        assert _router.served_model_for_deployment("low") is None

    def test_mid_not_in_the_named_group_returns_none(self):
        # "kimi-k3" is an ultra member, not a low member.
        assert _router.served_model_for_deployment("low-kimi-k3") is None

    def test_missing_registry_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLM_MODEL_REGISTRY_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="cannot resolve deployment"):
            _router.served_model_for_deployment("low-deepseek-v4-flash")


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

class TestLitellmProbeSpeaksTheDeclaredScheme:
    """The health probe must speak the scheme its URL declares.

    It was `HTTPConnection` unconditionally with a default port of 8980 —
    correct for exactly one deployment, a loopback proxy on the box. The moment
    the router got a TLS edge (model-router-policy §3.4a) the probe spoke plain
    HTTP at a TLS listener, the handshake failed, and the path was reported
    "not reachable" — indistinguishable from the router being down.

    Measured live 2026-08-03: the router answered `/v1/models` with 23 models
    over `https://router.nousergon.ai:8443` while this probe called it
    unreachable, and resolution fell through to openrouter.ai.
    """

    class _Resp:
        status = 401  # what an authenticating edge returns to an unauthenticated probe

        def read(self):
            return b""

    def _capture(self, monkeypatch):
        seen = {}

        class _Conn:
            def __init__(self, host, port, timeout=None):
                seen.update(host=host, port=port, timeout=timeout,
                            cls=type(self).__name__)

            def request(self, *a, **kw):
                pass

            def getresponse(self):
                return TestLitellmProbeSpeaksTheDeclaredScheme._Resp()

            def close(self):
                pass

        class _Https(_Conn):
            pass

        class _Http(_Conn):
            pass

        monkeypatch.setattr(_router._http_client, "HTTPSConnection", _Https)
        monkeypatch.setattr(_router._http_client, "HTTPConnection", _Http)
        return seen

    def test_https_url_uses_a_tls_connection_and_port_443(self, registry_file, monkeypatch):
        seen = self._capture(monkeypatch)
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                m.setenv("KREPIS_LITELLM_PROXY_URL", "https://router.example.ai")
                m.setenv("LITELLM_MASTER_KEY", "test-key")
                info = _router._resolve_group_json("ultra")
        finally:
            _router._router = None
        assert seen["cls"] == "_Https", (
            "the probe used a plaintext connection for an https:// URL — the "
            "TLS handshake fails and the router is reported unreachable"
        )
        assert seen["port"] == 443, (
            f"an https:// URL with no explicit port probed {seen['port']}, not 443"
        )
        assert info["route"] == "litellm_proxy"

    def test_proxy_route_addresses_the_qualified_primary_not_the_alias(
        self, registry_file, monkeypatch
    ):
        """config-I6727 deliverable 2: on the litellm_proxy route the model to
        ADDRESS is the qualified primary deployment name ({group}-{mid}, the
        #118 naming), never the bare group alias — litellm's proxy stamps the
        client-requested model back onto every non-fallback response, so a
        bare-alias-addressed healthy call reports the alias as resp.model and
        trips the #115 masquerade guard."""
        self._capture(monkeypatch)
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                m.setenv("KREPIS_LITELLM_PROXY_URL", "https://router.example.ai")
                m.setenv("LITELLM_MASTER_KEY", "test-key")
                info = _router.resolve_group_structured("ultra")
        finally:
            _router._router = None
        assert info["route"] == "litellm_proxy"
        assert info["model"] == "ultra-glm-5.2"
        assert info["deployment_id"] == "ultra-glm-5.2"
        assert info["model"] != info["group"] == "ultra"
        # the resolve-time primary facts survive unchanged
        assert info["primary_registry_id"] == "glm-5.2"

    def test_explicit_port_is_honoured(self, registry_file, monkeypatch):
        seen = self._capture(monkeypatch)
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                m.setenv("KREPIS_LITELLM_PROXY_URL", "https://router.example.ai:8443")
                m.setenv("LITELLM_MASTER_KEY", "test-key")
                _router._resolve_group_json("ultra")
        finally:
            _router._router = None
        assert (seen["cls"], seen["port"]) == ("_Https", 8443)

    def test_http_url_still_uses_plaintext_and_8980(self, registry_file, monkeypatch):
        """The loopback case must not regress — R27d permits it."""
        seen = self._capture(monkeypatch)
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                m.setenv("KREPIS_LITELLM_PROXY_URL", "http://127.0.0.1")
                m.setenv("LITELLM_MASTER_KEY", "test-key")
                _router._resolve_group_json("ultra")
        finally:
            _router._router = None
        assert (seen["cls"], seen["port"]) == ("_Http", 8980)

    def test_probe_timeout_allows_for_a_tls_handshake(self, registry_file, monkeypatch):
        """2s suited a loopback proxy. A cold Lambda's TLS handshake to an
        internet edge does not reliably fit in it, and the failure is silent."""
        seen = self._capture(monkeypatch)
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                m.setenv("KREPIS_LITELLM_PROXY_URL", "https://router.example.ai:8443")
                m.setenv("LITELLM_MASTER_KEY", "test-key")
                _router._resolve_group_json("ultra")
        finally:
            _router._router = None
        assert seen["timeout"] >= 5, (
            f"probe timeout is {seen['timeout']}s — too tight for a TLS "
            "handshake from a cold start"
        )


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

    def test_exclude_route_is_gone(self, registry_file, monkeypatch):
        """`exclude_route` was removed in 0.29.0 (model-router-policy R19).

        A consumer narrowing the fallback chain is holding a routing table at
        layer 5. Both callers passed it to mean "this route is not reachable
        from where I am running", which is `exec_context` plus the registry's
        `reachable_from`. Asserting the TypeError rather than deleting the test
        keeps the removal itself under test — a re-added parameter would
        silently restore the ability to route around the LiteLLM proxy.
        """
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                with pytest.raises(TypeError):
                    _router._resolve_group_json("med", exclude_route="egress_proxy")
                with pytest.raises(TypeError):
                    _router.resolve_group_structured("med", exclude_route="egress_proxy")
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


class TestLitellmMasterKeyFromSSM:
    """The SSM leg must work wherever krepis runs, and say so when it does not.

    This leg used to shell out to `aws ssm get-parameter --profile
    ne-laptop-daemon`. Verified 2026-08-03 against the Director Lambda, it
    could not have worked there for three independent reasons — the
    `public.ecr.aws/lambda/python:3.12` image ships no `aws` binary, there is
    no such profile in a Lambda, and `alpha-engine-evaluator-role` grants SSM
    on `/alpha-engine/*` but the parameter is `/symposion/...`. Every one was
    swallowed by `except Exception: pass`, so the LiteLLM route would simply
    have been skipped as unauthenticated the first time the Director could
    actually reach the router.
    """

    def test_uses_boto3_and_no_aws_profile(self, monkeypatch):
        """The credential chain is the execution context's, not a named profile."""
        calls = {}

        class _FakeSSM:
            def get_parameter(self, **kw):
                calls.update(kw)
                return {"Parameter": {"Value": "key-from-ssm"}}

        fake_boto3 = mock.Mock()
        fake_boto3.client.return_value = _FakeSSM()
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
        monkeypatch.delenv("KREPIS_LITELLM_MASTER_KEY_SSM_PARAM", raising=False)

        assert _original_litellm_master_key_from_ssm() == "key-from-ssm"
        assert calls["Name"] == _router.LITELLM_MASTER_KEY_SSM_PARAM
        assert calls["WithDecryption"] is True
        # boto3.client("ssm", region_name=...) — never a profile/session name.
        _, kwargs = fake_boto3.client.call_args
        assert "profile_name" not in kwargs

    def test_parameter_name_is_overridable(self, monkeypatch):
        """A fleet-specific SSM path is a fact about our account, not routing.

        This is what lets a per-consumer credential (R22) replace the shared
        master key without a code change.
        """
        class _FakeSSM:
            def get_parameter(self, **kw):
                return {"Parameter": {"Value": f"key-for-{kw['Name']}"}}

        fake_boto3 = mock.Mock()
        fake_boto3.client.return_value = _FakeSSM()
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
        monkeypatch.setenv(
            "KREPIS_LITELLM_MASTER_KEY_SSM_PARAM", "/alpha-engine/director/LITELLM_KEY")

        assert _original_litellm_master_key_from_ssm() == \
            "key-for-/alpha-engine/director/LITELLM_KEY"

    def test_failure_is_logged_with_its_reason(self, monkeypatch, caplog):
        """Returns None — but "access denied" and "no such parameter" are
        different fixes, and a bare swallow makes them the same event."""
        class _FakeSSM:
            def get_parameter(self, **kw):
                raise RuntimeError("AccessDeniedException: not authorized")

        fake_boto3 = mock.Mock()
        fake_boto3.client.return_value = _FakeSSM()
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

        with caplog.at_level("WARNING"):
            assert _original_litellm_master_key_from_ssm() is None
        assert "AccessDenied" in caplog.text

    def test_missing_boto3_is_reported_not_swallowed(self, monkeypatch, caplog):
        real_import = __import__

        def _no_boto3(name, *a, **kw):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *a, **kw)

        monkeypatch.delitem(sys.modules, "boto3", raising=False)
        monkeypatch.setattr("builtins.__import__", _no_boto3)
        with caplog.at_level("WARNING"):
            assert _original_litellm_master_key_from_ssm() is None
        assert "boto3" in caplog.text


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


# ── Execution-context reachability (model-router-policy R28/R29) ─────────
#
# These tests exist because of a live defect: the Director Lambda's weekly
# `ultra` call resolved to openrouter.ai direct and egressed DLP-unscanned
# (alpha-engine-config-I6183).  Nothing in the chain was unhealthy.  The two
# entries ahead of it were dropped because krepis' own hardcoded endpoint
# table had no row for `moonshot` or `zhipu` — a routing fact invented at
# layer 3 — and the caller compensated with `exclude_route="litellm_proxy"`,
# a routing fact asserted at layer 5.  The registry, which owns endpoints,
# was not consulted about either.

REACHABILITY_REGISTRY_YAML = """
schema_version: 1

model_groups:
  ultra:
    - kimi-direct
    - glm-direct
    - glm-openrouter

models:
  - id: kimi-direct
    name: Kimi (direct Moonshot)
    provider: moonshot
    route: egress_proxy
    api_base: http://127.0.0.1:8990
    endpoints:
      openai: http://127.0.0.1:8990
    reachable_from: [laptop, ec2]
    model: kimi-k3
    group: ultra
    group_role: primary
    params:
      max_tokens: 16384
    status: active

  - id: glm-direct
    name: GLM (direct Zhipu)
    provider: zhipu
    route: egress_proxy
    api_base: http://127.0.0.1:8990
    endpoints:
      openai: http://127.0.0.1:8990
    reachable_from: [laptop, ec2]
    model: glm-5.2
    group: ultra
    group_role: fallback
    params:
      max_tokens: 16384
    status: active

  - id: glm-openrouter
    name: GLM (via OpenRouter)
    provider: openrouter
    route: openrouter
    endpoints:
      anthropic: https://openrouter.ai/api
      openai: https://openrouter.ai/api
    reachable_from: [laptop, ec2]
    model: z-ai/glm-5.2
    group: ultra
    group_role: fallback
    params:
      max_tokens: 16384
    status: active
"""


@pytest.fixture
def reachability_registry():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(REACHABILITY_REGISTRY_YAML)
    yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def _no_litellm():
    """Force the per-provider resolution path deterministically.

    An unresolvable master key is one of the documented LiteLLM skip reasons,
    so this exercises the same branch a real key-less host would.
    """
    with mock.patch.object(_router, "_resolve_litellm_master_key", return_value=None):
        yield


class TestResolveExecContext:
    def test_defaults_to_laptop(self, monkeypatch):
        monkeypatch.delenv("KREPIS_EXEC_CONTEXT", raising=False)
        assert _router._resolve_exec_context() == _router.EXEC_CONTEXT_LAPTOP

    def test_env_var_is_read(self, monkeypatch):
        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")
        assert _router._resolve_exec_context() == "lambda"

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("KREPIS_EXEC_CONTEXT", "lambda")
        assert _router._resolve_exec_context("laptop") == "laptop"

    def test_unknown_context_raises_rather_than_defaulting(self, monkeypatch):
        """Falling back to the default on an unrecognised context would resolve
        an endpoint chosen on a vocabulary mismatch (R29)."""
        monkeypatch.delenv("KREPIS_EXEC_CONTEXT", raising=False)
        with pytest.raises(ValueError, match="Unknown execution context"):
            _router._resolve_exec_context("fargate")

    def test_context_is_never_inferred_from_lambda_env(self, monkeypatch):
        """R29 — declared, not inferred. AWS_LAMBDA_FUNCTION_NAME being set
        must not make krepis believe it is in a Lambda; a wrong guess makes a
        mis-resolution look like a health failure."""
        monkeypatch.delenv("KREPIS_EXEC_CONTEXT", raising=False)
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "alpha-engine-evaluator-director")
        assert _router._resolve_exec_context() == _router.EXEC_CONTEXT_LAPTOP


class TestEntryReachableFrom:
    def test_declared_context_matches(self):
        entry = {"id": "x", "reachable_from": ["laptop", "ec2"]}
        assert _router._entry_reachable_from(entry, "laptop") is True

    def test_undeclared_context_is_filtered(self):
        entry = {"id": "x", "reachable_from": ["laptop", "ec2"]}
        assert _router._entry_reachable_from(entry, "lambda") is False

    def test_missing_field_is_unreachable_from_everywhere(self, caplog):
        """An absent declaration is not permission (R20, fail closed).

        This was permissive as the R19 migration position, to be removed once
        the registry validator enforced the field. It does
        (`validate_llm_model_registry.py`, `test_missing_reachable_from_fails`),
        and the permissive branch had a live cost in the meantime: the
        hand-published S3 copy of the registry the Director reads lagged the
        repo, still carried no `reachable_from`, and the omission was read as
        universal reachability — so a Lambda served `glm-5.2` at openrouter.ai
        DLP-unscanned while logging a healthy route.
        """
        entry = {"id": "legacy-row"}
        with caplog.at_level("ERROR"):
            assert _router._entry_reachable_from(entry, "lambda") is False
            assert _router._entry_reachable_from(entry, "laptop") is False
        assert "reachable_from" in caplog.text
        assert "legacy-row" in caplog.text

    def test_undeclared_entry_is_skipped_with_a_diagnosable_reason(
        self, tmp_path, monkeypatch
    ):
        """Fail closed, but never fail mute — the skip names why."""
        reg = tmp_path / "LLM_MODEL_REGISTRY.yaml"
        reg.write_text(
            "schema_version: 1\n"
            "model_groups:\n"
            "  ultra: [legacy-row]\n"
            "models:\n"
            "  - id: legacy-row\n"
            "    model: some/model\n"
            "    provider: openrouter\n"
            "    route: openrouter\n"
            "    endpoints:\n"
            "      openai: https://openrouter.ai/api\n",
            encoding="utf-8",
        )
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(reg))
                m.setenv("KREPIS_LITELLM_PROXY_URL", "http://127.0.0.1:1")
                with pytest.raises(ValueError) as exc:
                    _router._resolve_group_json(
                        "ultra", exec_context="lambda", wire="openai")
            assert "reachable_from" in str(exc.value)
            assert "legacy-row" in str(exc.value)
        finally:
            _router._router = None


class TestEntryEndpoint:
    def test_reads_declared_wire_endpoint_from_registry(self):
        entry = {
            "id": "x", "route": "egress_proxy", "provider": "moonshot",
            "endpoints": {"openai": "http://127.0.0.1:8990"},
        }
        assert _router._entry_endpoint(entry, "openai") == "http://127.0.0.1:8990"

    def test_provider_absent_from_krepis_tables_still_resolves(self):
        """The I6183 regression. `moonshot` has no row in krepis' legacy table;
        the registry declares its endpoint, so it must resolve anyway. A
        resolver that drops it has invented a routing fact at layer 3 (R29)."""
        entry = {
            "id": "kimi-direct", "route": "egress_proxy", "provider": "moonshot",
            "endpoints": {"openai": "http://127.0.0.1:8990"},
        }
        assert ("egress_proxy", "moonshot") not in _router._LEGACY_CLI_ENDPOINTS
        assert _router._entry_endpoint(entry, "openai") == "http://127.0.0.1:8990"

    def test_undeclared_wire_raises_naming_what_is_declared(self):
        entry = {
            "id": "x", "route": "egress_proxy", "provider": "moonshot",
            "endpoints": {"openai": "http://127.0.0.1:8990"},
        }
        with pytest.raises(ValueError, match="declares no 'anthropic'-wire endpoint"):
            _router._entry_endpoint(entry, "anthropic")

    def test_legacy_api_base_serves_the_openai_wire(self):
        entry = {"id": "x", "route": "egress_proxy", "provider": "deepseek",
                 "api_base": "http://127.0.0.1:8990"}
        assert _router._entry_endpoint(entry, "openai") == "http://127.0.0.1:8990"

    def test_legacy_shim_warns_and_names_the_entry(self, caplog):
        entry = {"id": "unmigrated", "route": "openrouter", "provider": "openrouter"}
        with caplog.at_level("WARNING"):
            url = _router._entry_endpoint(entry, "anthropic")
        assert url == "https://openrouter.ai/api"
        assert "unmigrated" in caplog.text
        assert "R7" in caplog.text

    def test_env_override_beats_the_declared_endpoint(self, monkeypatch):
        monkeypatch.setenv("KREPIS_OPENROUTER_API_URL", "http://localhost:9999")
        entry = {"id": "x", "route": "openrouter", "provider": "openrouter",
                 "endpoints": {"anthropic": "https://openrouter.ai/api"}}
        assert _router._entry_endpoint(entry, "anthropic") == "http://localhost:9999"


class TestResolveFiltersByExecutionContext:
    def test_laptop_resolves_the_declared_primary(
        self, reachability_registry, monkeypatch, _no_litellm
    ):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(reachability_registry))
                info = _router._resolve_group_json(
                    "ultra", exec_context="laptop", wire="openai")
            assert info["registry_id"] == "kimi-direct"
            assert info["exec_context"] == "laptop"
            assert info["wire"] == "openai"
        finally:
            _router._router = None

    def test_lambda_fails_closed_rather_than_reaching_openrouter(
        self, reachability_registry, monkeypatch, _no_litellm
    ):
        """THE I6183 REGRESSION TEST.

        From `lambda` no entry in this chain is reachable — the egress
        proxies are on other hosts and openrouter.ai is off the private-subnet
        route. Before R28/R29 the resolver walked past both direct entries
        (no krepis table row for moonshot/zhipu) and served openrouter.ai
        direct, unscanned. It must now raise.
        """
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(reachability_registry))
                with pytest.raises(ValueError) as exc:
                    _router._resolve_group_json(
                        "ultra", exec_context="lambda", wire="openai")
            msg = str(exc.value)
            assert "lambda" in msg
            # Fail-closed diagnosis: the message carries WHY each entry went.
            assert "kimi-direct" in msg
            assert "glm-openrouter" in msg
        finally:
            _router._router = None

    def test_skipped_entries_names_the_context_not_a_health_failure(
        self, reachability_registry, monkeypatch, _no_litellm
    ):
        """R29 — 'unreachable from here' must be distinguishable from
        'unhealthy'. Collapsing the two is what made the Director's fallback
        log identically to a primary (I6185)."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(reachability_registry))
                # Reachable only from ec2: the two direct entries survive,
                # so resolution succeeds and the skips are observable.
                info = _router._resolve_group_json(
                    "ultra", exec_context="ec2", wire="anthropic")
            reasons = {s["registry_id"]: s["reason"] for s in info["skipped_entries"]}
            assert "kimi-direct" in reasons
            assert "anthropic" in reasons["kimi-direct"]
            assert "not reachable" not in reasons["kimi-direct"].lower()
            assert info["registry_id"] == "glm-openrouter"
        finally:
            _router._router = None

    def test_unknown_wire_is_rejected(self, reachability_registry, monkeypatch):
        with monkeypatch.context() as m:
            m.setenv("LLM_MODEL_REGISTRY_PATH", str(reachability_registry))
            with pytest.raises(ValueError, match="Unknown wire format"):
                _router._resolve_group_json("ultra", wire="grpc")


class TestExcludeRouteIsRemoved:
    def test_the_litellm_route_cannot_be_excluded_by_a_caller(self, registry_file, monkeypatch):
        """R27a.4 — the router route is offered in every context.

        The proxy path is gated on its health probe and on nothing a consumer
        can say. This is the load-bearing half of the removal: `exclude_route`
        existed for exactly one production purpose — letting the Director
        Lambda skip the LiteLLM proxy and egress direct to openrouter.ai while
        the path to the proxy was down — and nothing failed for weeks because
        of it.
        """
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                with pytest.raises(TypeError):
                    _router._resolve_group_json("med", exclude_route="litellm_proxy")
        finally:
            _router._router = None

    def test_no_warning_when_not_passed(self, registry_file, monkeypatch, recwarn):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                _router._resolve_group_json("med")
            assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
        finally:
            _router._router = None


class TestResolveContractCarriesContext:
    def test_exec_context_and_wire_are_emitted(
        self, reachability_registry, monkeypatch, _no_litellm
    ):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(reachability_registry))
                info = _router.resolve_group_structured(
                    "ultra", exec_context="laptop", wire="openai")
            assert info["schema_version"] == _router.RESOLVE_SCHEMA_VERSION
            assert info["exec_context"] == "laptop"
            assert info["wire"] == "openai"
        finally:
            _router._router = None

    def test_contract_still_validates_against_the_schema(
        self, reachability_registry, monkeypatch, _no_litellm
    ):
        """Additive evolution (R19): new fields must not bump the version out
        from under a consumer pinned to 2."""
        import json as _json
        jsonschema = pytest.importorskip("jsonschema")
        schema_path = Path(_router.__file__).parent / "resolve_schema.json"
        schema = _json.loads(schema_path.read_text())
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(reachability_registry))
                info = _router.resolve_group_structured(
                    "ultra", exec_context="laptop", wire="openai")
            jsonschema.validate(info, schema)
            assert info["schema_version"] == 2
        finally:
            _router._router = None


class TestRouterIsNeverContextFiltered:
    """model-router-policy §3.4a R27a — the router is a service, not a location.

    `reachable_from` scopes DIRECT PROVIDER entries only. Applying it to the
    litellm_proxy route would make the router's availability a property of the
    caller's network, which is the inversion §3.4a exists to forbid — and the
    reasoning that produced a 2h20m fleet-wide SSM outage on 2026-08-03
    (nous-ergon-ops-I417): "the Lambda cannot reach the router, so attach the
    Lambda to the router's VPC".

    The pairing is what needs a test. The health gate and the reachability
    filter are each covered above; nothing asserted that the second does not
    apply to the first, which is the shape where both halves pass and the
    behaviour is still wrong.
    """

    def _healthy_litellm(self, monkeypatch):
        """Make every LiteLLM gate pass without a live proxy."""
        import http.client as _http

        class _Resp:
            status = 200

            def read(self):
                return b""

        class _Conn:
            def __init__(self, *a, **k):
                pass

            def request(self, *a, **k):
                pass

            def getresponse(self):
                return _Resp()

            def close(self):
                pass

        monkeypatch.setattr(_http, "HTTPConnection", _Conn)
        monkeypatch.setattr(_router, "_resolve_litellm_master_key",
                            lambda: "test-master-key")

    def test_router_serves_a_context_no_entry_declares(
        self, reachability_registry, monkeypatch
    ):
        """No entry in this fixture declares `lambda`. The router must still
        serve it — availability is a health question, never a reachability
        fact about the caller."""
        self._healthy_litellm(monkeypatch)
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(reachability_registry))
                info = _router._resolve_group_json(
                    "ultra", exec_context="lambda", wire="openai")
            assert info["route"] == "litellm_proxy"
            assert info["exec_context"] == "lambda"
        finally:
            _router._router = None

    def test_no_registry_entry_may_declare_reachability_for_the_router(
        self, reachability_registry
    ):
        """A `route: litellm_proxy` row carrying `reachable_from` would be a
        registry asserting the thing R27a forbids. The resolver never reads
        such a row — the proxy path is synthesised, not looked up — so the
        guard belongs at the registry validator, and this pins the invariant
        the resolver relies on."""
        import yaml as _yaml
        doc = _yaml.safe_load(reachability_registry.read_text())
        proxy_rows = [m for m in doc["models"]
                      if m.get("route") == "litellm_proxy"]
        assert proxy_rows == [], (
            "the litellm_proxy route is synthesised by the resolver, not a "
            "registry row; a row for it would be a second source of truth for "
            "how the router is reached (§3.4a R27f)"
        )

    def test_context_vocabulary_encodes_no_network_posture(self):
        """`lambda_vpc` was the original name and is exactly the invitation
        R27a forbids: a context asserting an attachment makes "attach the
        consumer" read as the natural fix."""
        offenders = [c for c in _router.EXEC_CONTEXTS
                     if "vpc" in c or "subnet" in c or "sg" in c]
        assert offenders == [], (
            f"{offenders} name a network attachment, not a place code runs. "
            "Reaching the router may not depend on network position (R27a)."
        )


# ── resolve_group_spec — the group -> ModelSpec adapter ──────────────────


class TestResolveGroupSpec:
    """`resolve_group_spec` is the supported way a consumer goes from a
    capability tier to a client. Before it existed, three call sites each
    carried their own copy of the `auth_token_type` -> credential-name table
    (`director/agent.py`, `groom_driver.py`, `groomer_krepis_adapter.py`) —
    and this module is the only producer of those values, so a consumer copy
    silently mis-authenticates the day a new one is introduced."""

    def _route(self, **over):
        route = {
            "schema_version": _router.RESOLVE_SCHEMA_VERSION,
            # config-I6727: the wire route resolves to the QUALIFIED primary
            # deployment name; the bare group survives only in "group".
            "model": "med-deepseek-v4-flash-max",
            "display_name": "deepseek-v4-flash-max (med)",
            "provider": "litellm",
            "route": "litellm_proxy",
            "api_base_url": "https://router.example:8443",
            "deployment_id": "med-deepseek-v4-flash-max",
            "auth_token_type": "litellm_master_key",
            "group": "med",
            "registry_id": "litellm:group:med",
            "primary_model": "deepseek-v4-flash",
            "primary_registry_id": "deepseek-v4-flash-max",
            "capabilities": {},
            "params": {"max_tokens": 8192, "structured_outputs": True},
        }
        route.update(over)
        return route

    def _patch(self, monkeypatch, route):
        monkeypatch.setattr(
            _router, "resolve_group_structured", lambda *a, **k: route
        )

    def test_builds_a_spec_from_the_route(self, monkeypatch):
        self._patch(monkeypatch, self._route())
        spec, route = _router.resolve_group_spec("med", exec_context="lambda")
        assert spec.model == "med-deepseek-v4-flash-max"
        assert spec.base_url == "https://router.example:8443"
        assert spec.max_tokens == 8192
        assert route["group"] == "med"

    def test_proxy_route_is_an_openai_endpoint_not_the_in_process_router(
        self, monkeypatch
    ):
        """`resolve_group_structured` reports provider `litellm` for the proxy
        route, and ModelSpec binds that name to TRANSPORT_LITELLM — the
        in-process `get_router()`, which calls each provider DIRECTLY from
        the consumer and reads OPENROUTER_API_KEY from the environment as it
        goes. Emitting it verbatim would egress unscanned to openrouter.ai,
        bypass the authenticated edge entirely, and require litellm plus a
        readable registry inside every consumer — the constraint that
        reverted crucible-evaluator-PR157 (alpha-engine-config-I6059)."""
        from krepis.llm_config import PROVIDER_REGISTRY, TRANSPORT_OPENAI

        self._patch(monkeypatch, self._route())
        spec, _ = _router.resolve_group_spec("med", exec_context="lambda")

        assert spec.provider == _router.ROUTER_EDGE_PROVIDER == "litellm_proxy"
        assert spec.provider not in PROVIDER_REGISTRY, (
            "a name known to PROVIDER_REGISTRY takes that provider's "
            "transport; the edge must resolve as a custom OpenAI-compatible "
            "endpoint"
        )
        assert spec.transport == TRANSPORT_OPENAI
        # base_url + api_key_env are what make a custom endpoint valid.
        assert spec.base_url and spec.api_key_env

    def test_the_proxy_route_carries_the_PRIMARY_registry_id(self, monkeypatch):
        """A cost record has to be able to name the entry that was addressed.

        `model` is the upstream name the provider reports, and three registry
        entries (`deepseek-v4-flash`, `-low`, `-max`) share one such string
        while declaring three different reasoning configs — so `model` alone
        collapses them. On the proxy route `registry_id` is the synthetic
        `litellm:group:{group}`, which names no entry; `primary_registry_id` is
        the closest honest answer — what THIS process addressed, not a claim
        about what the proxy walked to server-side.
        alpha-engine-config-I6908.
        """
        self._patch(monkeypatch, self._route(
            registry_id="litellm:group:med",
            primary_registry_id="deepseek-v4-flash-max",
        ))
        spec, _ = _router.resolve_group_spec("med", exec_context="lambda")
        assert spec.registry_id == "deepseek-v4-flash-max"
        assert not str(spec.registry_id).startswith("litellm:group:"), (
            "the synthetic group id names no registry entry — recording it "
            "would look like attribution while carrying none"
        )

    def test_a_direct_route_carries_its_own_registry_id(self, monkeypatch):
        self._patch(monkeypatch, self._route(
            provider="deepseek", route="egress_proxy",
            auth_token_type="placeholder",
            api_base_url="http://127.0.0.1:8990",
            registry_id="deepseek-v4-flash-max",
        ))
        spec, _ = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.registry_id == "deepseek-v4-flash-max"

    def test_non_proxy_routes_keep_their_provider(self, monkeypatch):
        self._patch(monkeypatch, self._route(
            provider="deepseek", route="egress_proxy",
            auth_token_type="placeholder",
            api_base_url="http://127.0.0.1:8990",
        ))
        spec, _ = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.provider == "deepseek"

    def test_explicit_max_tokens_overrides_the_registry(self, monkeypatch):
        self._patch(monkeypatch, self._route())
        spec, _ = _router.resolve_group_spec(
            "med", exec_context="lambda", max_tokens=4000
        )
        assert spec.max_tokens == 4000

    def test_structured_outputs_override_is_honoured(self, monkeypatch):
        """A call site that has live-verified strict json_schema is unreliable
        for the model serving its group must be able to say so — silently
        taking the registry default would re-break that call site."""
        self._patch(monkeypatch, self._route())
        spec, _ = _router.resolve_group_spec(
            "med", exec_context="lambda", structured_outputs=False
        )
        assert spec.structured_outputs is False

    def test_unknown_schema_version_raises(self, monkeypatch):
        self._patch(monkeypatch, self._route(schema_version=999))
        with pytest.raises(RuntimeError, match="schema_version"):
            _router.resolve_group_spec("med", exec_context="lambda")

    def test_unknown_auth_token_type_raises(self, monkeypatch):
        """Refusing beats guessing: a guessed credential is a real key sent to
        an unintended endpoint."""
        self._patch(monkeypatch, self._route(auth_token_type="new_thing"))
        with pytest.raises(RuntimeError, match="auth_token_type"):
            _router.resolve_group_spec("med", exec_context="lambda")

    def test_placeholder_auth_maps_to_no_credential_name(self, monkeypatch):
        """`placeholder` means the local egress proxy holds the real key. It
        is not a missing credential, and collapsing the two would break every
        direct route from a context that can reach one."""
        self._patch(monkeypatch, self._route(
            auth_token_type="placeholder",
            provider="deepseek",
            route="egress_proxy",
            api_base_url="http://127.0.0.1:8990",
            deployment_id="deepseek-v4-flash",
            registry_id="deepseek-v4-flash-max",
            primary_registry_id="deepseek-v4-flash-max",
        ))
        spec, _ = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.api_key_env is None

    # ── per-consumer router credential ───────────────────────────────────

    def test_router_credential_defaults_to_the_historical_name(self, monkeypatch):
        monkeypatch.delenv(_router.ROUTER_CREDENTIAL_SECRET_ENV, raising=False)
        self._patch(monkeypatch, self._route())
        spec, _ = _router.resolve_group_spec("med", exec_context="lambda")
        assert spec.api_key_env == "LITELLM_MASTER_KEY"

    def test_per_consumer_credential_secret_is_honoured(self, monkeypatch):
        """The edge identifies a consumer BY its credential value, and
        `krepis.secrets` resolves SSM before os.environ — so two consumers
        reading the same secret NAME collapse into one identity at the edge
        no matter how their environments are set."""
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_THINKTANK"
        )
        self._patch(monkeypatch, self._route())
        spec, _ = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.api_key_env == "ROUTER_CONSUMER_THINKTANK"

    def test_per_consumer_credential_does_not_leak_into_other_auth_types(
        self, monkeypatch
    ):
        """It renames the ROUTER credential only. Applying it to a direct
        provider route would point that route at a router credential."""
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_THINKTANK"
        )
        self._patch(monkeypatch, self._route(
            auth_token_type="direct_api_key", provider="anthropic",
            route="direct", api_base_url="",
        ))
        spec, _ = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.api_key_env == "ANTHROPIC_API_KEY"

    def test_blank_env_falls_back_rather_than_naming_an_empty_secret(
        self, monkeypatch
    ):
        monkeypatch.setenv(_router.ROUTER_CREDENTIAL_SECRET_ENV, "   ")
        self._patch(monkeypatch, self._route())
        spec, _ = _router.resolve_group_spec("med", exec_context="lambda")
        assert spec.api_key_env == "LITELLM_MASTER_KEY"


class TestPerConsumerCredentialIsAdmitted:
    """The ADMISSION half of the per-consumer credential
    (alpha-engine-config-I6414).

    ``KREPIS_ROUTER_CREDENTIAL_SECRET`` was honoured only at call time, in
    ``resolve_group_spec``'s ``api_key_env``. The ``litellm_proxy`` candidate is
    admitted or skipped much earlier, in ``_resolve_group_json``, and that check
    called ``_resolve_litellm_master_key`` which looked only for the literal
    ``LITELLM_MASTER_KEY``. So a consumer configured exactly as I6373 intends was
    rejected before its credential was ever consulted.

    The gap is precisely that the two halves were tested separately and each
    passed — the class above covers the call half. These cover the admission
    half and the equivalence between them.
    """

    @staticmethod
    def _no_secrets_env(monkeypatch):
        """Neutralise leg 2 so the test asserts code, not this machine.

        A real ``~/Development/.llm-routing/secrets.env`` exists on the laptop
        and carries ``LITELLM_MASTER_KEY``; without this the default-name cases
        would pass from the filesystem regardless of what the code does.
        """
        monkeypatch.setattr("builtins.open", _raise_oserror)

    def test_only_the_per_consumer_env_var_is_set_and_it_is_admitted(
        self, monkeypatch
    ):
        """The exact I6373 configuration: the per-consumer name declared, the
        credential present under THAT name, and `LITELLM_MASTER_KEY` absent."""
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_THINKTANK"
        )
        monkeypatch.setenv("ROUTER_CONSUMER_THINKTANK", "sk-thinktank-value")
        monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
        self._no_secrets_env(monkeypatch)
        monkeypatch.setattr(
            _router, "_litellm_master_key_from_ssm", _fail_if_called
        )

        assert _router._resolve_litellm_master_key() == "sk-thinktank-value"

    def test_the_shared_key_does_not_satisfy_a_per_consumer_declaration(
        self, monkeypatch
    ):
        """Falling back to `LITELLM_MASTER_KEY` would collapse this consumer
        into the director's identity at the edge — one `X-Router-Consumer`, one
        rate-limit bucket, and revoking one revokes all. That is what
        nous-ergon-ops-PR474 established distinct credentials to prevent, so the
        fallback must NOT exist."""
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_THINKTANK"
        )
        monkeypatch.delenv("ROUTER_CONSUMER_THINKTANK", raising=False)
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-the-shared-director-key")
        self._no_secrets_env(monkeypatch)
        monkeypatch.setattr(
            _router, "_litellm_master_key_from_ssm", lambda name=None: None
        )

        assert _router._resolve_litellm_master_key() is None

    def test_unset_override_resolves_exactly_as_before(self, monkeypatch):
        """The regression guard on every existing consumer."""
        monkeypatch.delenv(_router.ROUTER_CREDENTIAL_SECRET_ENV, raising=False)
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-historical")
        self._no_secrets_env(monkeypatch)
        monkeypatch.setattr(
            _router, "_litellm_master_key_from_ssm", _fail_if_called
        )

        assert _router._resolve_litellm_master_key() == "sk-historical"

    def test_ssm_leg_receives_the_consumer_name(self, monkeypatch):
        """Leg 3 must look up THIS consumer's parameter. Passing the default
        name through would read the director's parameter and admit the route on
        a credential the consumer cannot present at call time — an admission
        that then 401s, which is worse than a clean skip."""
        seen = {}
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_RESEARCH"
        )
        monkeypatch.delenv("ROUTER_CONSUMER_RESEARCH", raising=False)
        monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
        self._no_secrets_env(monkeypatch)
        monkeypatch.setattr(
            _router, "_litellm_master_key_from_ssm",
            lambda name="LITELLM_MASTER_KEY": seen.setdefault("name", name),
        )

        _router._resolve_litellm_master_key()
        assert seen["name"] == "ROUTER_CONSUMER_RESEARCH"

    # ── which SSM parameter the name maps to ─────────────────────────────

    def test_ssm_param_for_the_default_name_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("KREPIS_LITELLM_MASTER_KEY_SSM_PARAM", raising=False)
        seen = _capture_ssm_param(monkeypatch)

        _original_litellm_master_key_from_ssm("LITELLM_MASTER_KEY")
        assert seen["param"] == _router.LITELLM_MASTER_KEY_SSM_PARAM

    def test_ssm_param_for_a_consumer_name_uses_the_fleet_prefix(
        self, monkeypatch
    ):
        from krepis.secrets import SSM_PREFIX

        monkeypatch.delenv("KREPIS_LITELLM_MASTER_KEY_SSM_PARAM", raising=False)
        seen = _capture_ssm_param(monkeypatch)

        _original_litellm_master_key_from_ssm("ROUTER_CONSUMER_THINKTANK")
        assert seen["param"] == (
            f"{SSM_PREFIX.rstrip('/')}/ROUTER_CONSUMER_THINKTANK"
        )

    def test_explicit_ssm_param_override_still_wins(self, monkeypatch):
        """Callers already setting it keep working, whatever their name is."""
        monkeypatch.setenv(
            "KREPIS_LITELLM_MASTER_KEY_SSM_PARAM", "/custom/PARAM"
        )
        seen = _capture_ssm_param(monkeypatch)

        _original_litellm_master_key_from_ssm("ROUTER_CONSUMER_THINKTANK")
        assert seen["param"] == "/custom/PARAM"

    # ── the name is an identifier, not a path ────────────────────────────

    @pytest.mark.parametrize("bad", [
        "../../symposion/LITELLM_MASTER_KEY",   # traversal out of the prefix
        "/alpha-engine/ROUTER_CONSUMER_X",      # absolute, would double-prefix
        "ROUTER CONSUMER",                      # space
        "ROUTER\nCONSUMER",                     # newline into a log record
        "ROUTER-CONSUMER",                      # hyphen is not an env-var char
        "x" * 129,                              # over the length bound
    ])
    def test_a_malformed_name_falls_back_rather_than_naming_a_path(
        self, monkeypatch, bad
    ):
        """The value is operator-supplied and is interpolated into an SSM
        parameter path, so `../../elsewhere/PARAM` reads as a traversal to the
        SSM API rather than as a malformed name."""
        monkeypatch.setenv(_router.ROUTER_CREDENTIAL_SECRET_ENV, bad)
        assert _router.router_credential_secret_name() == "LITELLM_MASTER_KEY"

    def test_a_malformed_name_is_warned_not_swallowed(
        self, monkeypatch, caplog
    ):
        """Falling back silently would authenticate this consumer as whoever
        holds the shared key — the identity collapse distinct credentials exist
        to prevent — and it would look like a working configuration."""
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "../../elsewhere/PARAM"
        )
        with caplog.at_level("WARNING"):
            _router.router_credential_secret_name()
        assert any(
            _router.ROUTER_CREDENTIAL_SECRET_ENV in r.message
            or _router.ROUTER_CREDENTIAL_SECRET_ENV in r.getMessage()
            for r in caplog.records
        )

    def test_a_well_formed_name_is_unchanged(self, monkeypatch):
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_THINKTANK"
        )
        assert (
            _router.router_credential_secret_name()
            == "ROUTER_CONSUMER_THINKTANK"
        )


def _raise_oserror(*_args, **_kwargs):
    raise OSError("secrets.env neutralised for this test")


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError(
        "resolution should have returned before reaching this leg"
    )


def _capture_ssm_param(monkeypatch) -> dict:
    """Record the parameter name `_litellm_master_key_from_ssm` asks SSM for,
    without reaching AWS. Returns the dict the name lands in.

    Callers use `_original_litellm_master_key_from_ssm`, not the module
    attribute: conftest's autouse `_no_ssm_master_key_lookup_from_tests` stubs
    the attribute for the whole suite, so calling it here would exercise the
    stub and assert nothing."""
    seen: dict = {}

    class _FakeSSM:
        @staticmethod
        def get_parameter(Name, WithDecryption):  # noqa: N803 - boto3 kwarg
            seen["param"] = Name
            return {"Parameter": {"Value": "sk-fake"}}

    class _FakeBoto3:
        @staticmethod
        def client(_service, region_name=None):
            return _FakeSSM()

    monkeypatch.setitem(sys.modules, "boto3", _FakeBoto3)
    return seen

    # ── route_is_degraded ────────────────────────────────────────────────

    def test_litellm_proxy_route_is_never_degraded_at_resolve_time(self):
        """The proxy walks the chain itself, so which entry serves is a
        call-time fact. `registry_id` there is the synthetic
        `litellm:group:<g>` and `primary_registry_id` a real model id, so a
        naive comparison fires on EVERY healthy router call — a detector
        whose output does not vary with the condition it names."""
        assert _router.route_is_degraded(self._route()) is False

    def test_per_provider_fallback_is_degraded(self):
        assert _router.route_is_degraded(self._route(
            route="egress_proxy",
            registry_id="deepseek-v4-pro",
            primary_registry_id="deepseek-v4-flash-max",
        )) is True

    def test_per_provider_primary_is_not_degraded(self):
        assert _router.route_is_degraded(self._route(
            route="egress_proxy",
            registry_id="deepseek-v4-flash-max",
            primary_registry_id="deepseek-v4-flash-max",
        )) is False

    def test_missing_primary_on_a_direct_route_reads_as_degraded(self):
        """Absence is not health — principles §2.7."""
        r = self._route(route="egress_proxy", registry_id="x")
        r.pop("primary_registry_id")
        r.pop("primary_model")
        assert _router.route_is_degraded(r) is True


class TestAppConfigFailureIsObservable:
    """`krepis.logging.setup_logging` pins the root logger at INFO with no env
    override, so anything logged below INFO is unobservable in every deployed
    environment that uses it. AppConfig resolution used to log its failure at
    DEBUG — measured 2026-08-04 on alpha-engine-research-runner, where it
    failed on every invocation and the only visible symptom was a
    FileNotFoundError naming LLM_MODEL_REGISTRY_PATH, which is not the cause.
    """

    def _optin(self, monkeypatch):
        monkeypatch.setenv("KREPIS_APPCONFIG_APPLICATION", "alpha-engine")
        monkeypatch.setenv("KREPIS_APPCONFIG_CONFIG_PROFILE", "llm-model-registry")
        monkeypatch.setenv("KREPIS_APPCONFIG_ENVIRONMENT", "production")
        _router._appconfig_cached_path = None
        _router._appconfig_next_poll_s = 0.0

    def test_a_failed_poll_logs_at_warning(self, monkeypatch, caplog):
        import boto3 as _b3

        self._optin(monkeypatch)

        def _boom(*a, **k):
            raise RuntimeError("AccessDenied: simulated")

        monkeypatch.setattr(_b3, "client", _boom)
        with caplog.at_level("WARNING", logger="krepis.router"):
            assert _router._find_registry_from_appconfig() is None
        assert any(r.levelname == "WARNING" for r in caplog.records), (
            "the failure is invisible at INFO, which is where every consumer "
            "of krepis.logging.setup_logging is pinned"
        )
        assert "AppConfig registry resolution FAILED" in caplog.text

    def test_the_warning_says_the_fallback_cannot_help(self, monkeypatch, caplog):
        """\"Falling through to the filesystem walk\" reads as recovery. In a
        Lambda or on a fresh spot box that walk finds nothing, so the message
        has to say so — otherwise the next reader treats the raise that
        follows as the real fault."""
        import boto3 as _b3

        self._optin(monkeypatch)
        monkeypatch.setattr(_b3, "client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        with caplog.at_level("WARNING", logger="krepis.router"):
            _router._find_registry_from_appconfig()
        assert "finds nothing" in caplog.text
        assert "LLM_MODEL_REGISTRY_PATH" in caplog.text

    def test_an_empty_configuration_also_warns(self, monkeypatch, caplog):
        import boto3 as _b3

        self._optin(monkeypatch)

        class _Body:
            def read(self):
                return b""

        class _Client:
            def start_configuration_session(self, **k):
                return {"InitialConfigurationToken": "t"}

            def get_latest_configuration(self, **k):
                return {"Configuration": _Body()}

        monkeypatch.setattr(_b3, "client", lambda *a, **k: _Client())
        with caplog.at_level("WARNING", logger="krepis.router"):
            assert _router._find_registry_from_appconfig() is None
        assert "EMPTY configuration" in caplog.text

    def test_no_warning_when_appconfig_was_never_opted_into(self, monkeypatch, caplog):
        """Environments with a local checkout never set the opt-in and must
        stay quiet — this branch is not reachable for them."""
        monkeypatch.delenv("KREPIS_APPCONFIG_APPLICATION", raising=False)
        with caplog.at_level("WARNING", logger="krepis.router"):
            assert _router._find_registry_from_appconfig() is None
        assert caplog.text == ""


# ── the (url, credential) pair must be able to authenticate ────────────────


class TestUnauthenticatablePairIsRefused:
    """A per-consumer credential paired with the plaintext loopback URL cannot
    authenticate, and `resolve_group_spec` used to return it as a SUCCESSFUL
    resolution — `route == "litellm_proxy"`, `degraded == False`.

    The credential is meaningful only AT the authenticated edge, which
    exchanges it for the router's own key. The process behind that edge knows
    the master key and nothing else, and has no database in which to resolve a
    virtual key, so every call comes back
    `400 {"error":{"message":"No connected db."}}`.

    Measured (alpha-engine-config-I6965): morning-signal declared
    KREPIS_ROUTER_CREDENTIAL_SECRET and not KREPIS_LITELLM_PROXY_URL, took the
    loopback default, and aborted its configured primary on that 400 on EVERY
    scheduled run from 2026-08-09 to 2026-08-12 — airing each episode from a
    fallback that is direct-OpenRouter linkage, which
    alpha-engine-config-I6367 forbids.

    model-router-policy R20: a failed resolution fails CLOSED.
    """

    def _route(self, **over):
        route = {
            "schema_version": _router.RESOLVE_SCHEMA_VERSION,
            "model": "med-deepseek-v4-flash-max",
            "display_name": "deepseek-v4-flash-max (med)",
            "provider": "litellm",
            "route": "litellm_proxy",
            "api_base_url": _router.LITELLM_PROXY_URL,
            "deployment_id": "med-deepseek-v4-flash-max",
            "auth_token_type": "litellm_master_key",
            "group": "med",
            "registry_id": "litellm:group:med",
            "primary_model": "deepseek-v4-flash",
            "primary_registry_id": "deepseek-v4-flash-max",
            "capabilities": {},
            "params": {"max_tokens": 8192, "structured_outputs": True},
        }
        route.update(over)
        return route

    def _patch(self, monkeypatch, route):
        monkeypatch.setattr(
            _router, "resolve_group_structured", lambda *a, **k: route
        )

    def test_consumer_credential_on_the_loopback_process_raises(self, monkeypatch):
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_MORNINGSIGNAL"
        )
        self._patch(monkeypatch, self._route())
        with pytest.raises(RuntimeError) as exc:
            _router.resolve_group_spec("med", exec_context="ec2")
        msg = str(exc.value)
        assert "cannot authenticate" in msg
        # The message must carry BOTH halves and the remedy — a consumer author
        # reading only "authentication failed" learns nothing about which of
        # the two independently-resolved values to change.
        assert _router.LITELLM_PROXY_URL in msg
        assert "ROUTER_CONSUMER_MORNINGSIGNAL" in msg
        assert _router.LITELLM_PROXY_URL_ENV in msg

    def test_master_key_on_the_loopback_process_still_resolves(self, monkeypatch):
        """R27d: a co-tenant consumer may address loopback. The master key is
        what that process can actually validate, so this arrangement is
        legitimate and must not be swept up."""
        monkeypatch.delenv(_router.ROUTER_CREDENTIAL_SECRET_ENV, raising=False)
        self._patch(monkeypatch, self._route())
        spec, route = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.api_key_env == "LITELLM_MASTER_KEY"
        assert route["route"] == "litellm_proxy"

    def test_consumer_credential_on_the_edge_resolves(self, monkeypatch):
        """The whole point of a per-consumer credential — addressed to the edge
        it is meaningful at."""
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_MORNINGSIGNAL"
        )
        self._patch(
            monkeypatch,
            self._route(api_base_url="https://router.nousergon.ai:8443"),
        )
        spec, _ = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.api_key_env == "ROUTER_CONSUMER_MORNINGSIGNAL"
        assert spec.base_url == "https://router.nousergon.ai:8443"

    def test_the_edge_may_itself_be_on_loopback_over_tls(self, monkeypatch):
        """Scheme is half the predicate. An edge terminating TLS on loopback is
        a legitimate deployment; refusing it would make the guard a rule about
        WHERE the consumer runs, which is exactly what R27a forbids."""
        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_MORNINGSIGNAL"
        )
        self._patch(monkeypatch, self._route(api_base_url="https://127.0.0.1:8443"))
        spec, _ = _router.resolve_group_spec("med", exec_context="ec2")
        assert spec.base_url == "https://127.0.0.1:8443"

    @pytest.mark.parametrize(
        "url,loopback",
        [
            ("http://127.0.0.1:8980", True),
            ("http://localhost:8980", True),
            ("http://[::1]:8980", True),
            ("http://127.1.2.3:8980", True),
            ("https://127.0.0.1:8443", False),
            ("https://router.nousergon.ai:8443", False),
            ("http://router.nousergon.ai:8443", False),
            ("", False),
            ("not a url", False),
        ],
    )
    def test_plaintext_loopback_detection(self, url, loopback):
        assert _router._is_plaintext_loopback(url) is loopback
