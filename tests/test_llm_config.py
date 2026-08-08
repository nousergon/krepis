"""Tests for ``krepis.llm_config`` — spec parsing + env→SSM→default resolution."""

import pytest

from krepis.llm_config import (
    LLMConfigError,
    ModelSpec,
    clear_spec_cache,
    parse_model_spec,
    resolve_model_spec,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_spec_cache()
    yield
    clear_spec_cache()


# ── parse_model_spec ──────────────────────────────────────────────────────


class TestParseCompact:
    def test_provider_model(self):
        spec = parse_model_spec("anthropic:claude-haiku-4-5")
        assert spec.provider == "anthropic"
        assert spec.model == "claude-haiku-4-5"

    def test_openrouter_slug_keeps_variant_suffix(self):
        # Split on FIRST colon only — the :floor variant belongs to the model.
        spec = parse_model_spec("openrouter:deepseek/deepseek-v4-flash:floor")
        assert spec.provider == "openrouter"
        assert spec.model == "deepseek/deepseek-v4-flash:floor"

    @pytest.mark.parametrize("bad", ["", "   ", "no-colon", ":model", "provider:"])
    def test_malformed_raises(self, bad):
        with pytest.raises(LLMConfigError):
            parse_model_spec(bad)


class TestParseJson:
    def test_full_object(self):
        spec = parse_model_spec(
            '{"provider": "openrouter", "model": "moonshotai/kimi-k2.6", '
            '"max_tokens": 8192, "structured_outputs": false}'
        )
        assert spec.provider == "openrouter"
        assert spec.model == "moonshotai/kimi-k2.6"
        assert spec.max_tokens == 8192
        assert spec.structured_outputs is False

    def test_unknown_field_raises(self):
        with pytest.raises(LLMConfigError, match="unknown field"):
            parse_model_spec('{"provider": "openai", "model": "x", "nope": 1}')

    def test_missing_required_raises(self):
        with pytest.raises(LLMConfigError, match="provider.*model|'provider' and 'model'"):
            parse_model_spec('{"provider": "openai"}')

    def test_json_array_raises(self):
        with pytest.raises(LLMConfigError):
            parse_model_spec("{broken json")

    def test_reasoning_field_parsed(self):
        spec = parse_model_spec(
            '{"provider": "openrouter", "model": "moonshotai/kimi-k2.6", '
            '"reasoning": {"effort": "low"}}'
        )
        assert spec.reasoning == {"effort": "low"}

    def test_reasoning_defaults_to_none(self):
        spec = parse_model_spec('{"provider": "openai", "model": "x"}')
        assert spec.reasoning is None


# ── ModelSpec transport / resolution ──────────────────────────────────────


class TestModelSpec:
    def test_builtin_transports(self):
        assert ModelSpec("anthropic", "m").transport == "anthropic"
        assert ModelSpec("openai", "m").transport == "openai"
        assert ModelSpec("openrouter", "m").transport == "openai"

    def test_builtin_defaults(self):
        spec = ModelSpec("openrouter", "m")
        assert spec.resolved_base_url() == "https://openrouter.ai/api/v1"
        assert spec.resolved_api_key_env() == "OPENROUTER_API_KEY"

    def test_supports_automatic_prefix_caching_defaults_false(self):
        spec = ModelSpec("anthropic", "claude-haiku-4-5")
        assert spec.supports_automatic_prefix_caching is False

    def test_supports_automatic_prefix_caching_from_json(self):
        spec = parse_model_spec(
            '{"provider": "openrouter", "model": "deepseek/deepseek-v4-flash:floor", '
            '"supports_automatic_prefix_caching": true}'
        )
        assert spec.supports_automatic_prefix_caching is True

    def test_supports_automatic_prefix_caching_ssm_round_trip(self):
        ssm = _FakeSSM({
            "/p/llm": '{"provider": "openai", "model": "gpt-5", '
                      '"supports_automatic_prefix_caching": true}'
        })
        spec = resolve_model_spec("/p/llm", default=ModelSpec("anthropic", "x"), ssm_client=ssm)
        assert spec.supports_automatic_prefix_caching is True

    def test_custom_provider_requires_explicit_fields(self):
        bare = ModelSpec("vllm_spot", "my-model")
        assert bare.transport == "openai"
        with pytest.raises(LLMConfigError, match="base_url"):
            bare.resolved_base_url()
        with pytest.raises(LLMConfigError, match="api_key_env"):
            bare.resolved_api_key_env()

        full = ModelSpec(
            "vllm_spot",
            "my-model",
            base_url="http://10.0.0.12:8000/v1",
            api_key_env="VLLM_API_KEY",
        )
        assert full.resolved_base_url() == "http://10.0.0.12:8000/v1"
        assert full.resolved_api_key_env() == "VLLM_API_KEY"


# ── resolve_model_spec ────────────────────────────────────────────────────


class _FakeSSM:
    """Minimal get_parameter stub; raises KeyError-ish error on missing."""

    def __init__(self, params=None):
        self.params = dict(params or {})
        self.calls = 0

    def get_parameter(self, Name, WithDecryption):  # noqa: N803 — boto3 shape
        self.calls += 1
        if Name not in self.params:
            raise RuntimeError(f"ParameterNotFound: {Name}")
        return {"Parameter": {"Value": self.params[Name]}}


DEFAULT = ModelSpec("anthropic", "claude-haiku-4-5")


class TestResolve:
    def test_env_override_wins(self, monkeypatch):
        ssm = _FakeSSM({"/p/llm": "anthropic:claude-sonnet-4-6"})
        monkeypatch.setenv("P_LLM", "openrouter:moonshotai/kimi-k2.6")
        spec = resolve_model_spec(
            "/p/llm", env_var="P_LLM", default=DEFAULT, ssm_client=ssm
        )
        assert spec.provider == "openrouter"
        assert ssm.calls == 0  # env short-circuits SSM entirely

    def test_ssm_value_used_and_cached(self):
        ssm = _FakeSSM({"/p/llm": "openrouter:deepseek/deepseek-v4-flash:floor"})
        s1 = resolve_model_spec("/p/llm", default=DEFAULT, ssm_client=ssm)
        s2 = resolve_model_spec("/p/llm", default=DEFAULT, ssm_client=ssm)
        assert s1.model == "deepseek/deepseek-v4-flash:floor"
        assert s2 == s1
        assert ssm.calls == 1  # second hit served from TTL cache

    def test_flip_takes_effect_after_ttl(self):
        ssm = _FakeSSM({"/p/llm": "anthropic:claude-haiku-4-5"})
        s1 = resolve_model_spec(
            "/p/llm", default=DEFAULT, ssm_client=ssm, ttl_seconds=0
        )
        ssm.params["/p/llm"] = "openrouter:moonshotai/kimi-k2.6"
        s2 = resolve_model_spec(
            "/p/llm", default=DEFAULT, ssm_client=ssm, ttl_seconds=0
        )
        assert s1.provider == "anthropic"
        assert s2.provider == "openrouter"

    def test_malformed_ssm_raises_even_with_default(self):
        ssm = _FakeSSM({"/p/llm": "garbage-no-colon"})
        with pytest.raises(LLMConfigError):
            resolve_model_spec("/p/llm", default=DEFAULT, ssm_client=ssm)

    def test_unreadable_ssm_falls_back_to_default_with_warning(self, caplog):
        ssm = _FakeSSM({})  # parameter missing
        with caplog.at_level("WARNING"):
            spec = resolve_model_spec("/p/llm", default=DEFAULT, ssm_client=ssm)
        assert spec == DEFAULT
        assert any("unreadable" in r.message for r in caplog.records)

    def test_unreadable_ssm_no_default_raises(self):
        ssm = _FakeSSM({})
        with pytest.raises(LLMConfigError, match="no default"):
            resolve_model_spec("/p/llm", ssm_client=ssm)

    def test_malformed_env_raises(self, monkeypatch):
        monkeypatch.setenv("P_LLM", "nonsense")
        with pytest.raises(LLMConfigError):
            resolve_model_spec(
                "/p/llm", env_var="P_LLM", default=DEFAULT, ssm_client=_FakeSSM()
            )

    def test_max_tokens_override(self):
        ssm = _FakeSSM({"/p/llm": "anthropic:claude-haiku-4-5"})
        spec = resolve_model_spec(
            "/p/llm", default=DEFAULT, ssm_client=ssm, max_tokens=1500
        )
        assert spec.max_tokens == 1500


class TestInProcessLitellmProviderIsRefused:
    """`provider="litellm"` selects the in-process LiteLLM Router, which calls
    upstream providers directly from the consumer. A consumer naming it always
    means "the model group", so the name is refused with a pointer rather than
    honoured.

    Live cost of not doing this (2026-08-08): morning-signal set the provider
    on 2026-08-02 and every episode for six days aborted its primary on
    `ModuleNotFoundError: No module named 'litellm'`, then silently aired on a
    direct-provider fallback. The transport that name selects had never run
    once, and nothing distinguished that from a provider outage.
    """

    def test_construction_raises(self):
        with pytest.raises(LLMConfigError) as exc:
            ModelSpec("litellm", "high")
        assert "resolve_group_spec" in str(exc.value), (
            "the error must name the supported alternative — an error that "
            "only says no leaves the caller to guess, and the guess is what "
            "produced this provider name in the first place"
        )

    def test_the_error_names_the_group_that_was_asked_for(self):
        with pytest.raises(LLMConfigError) as exc:
            ModelSpec("litellm", "ultra")
        assert "'ultra'" in str(exc.value)

    def test_parse_from_json_is_refused_too(self):
        """The production instance came from a JSON config value, not a
        literal — so the guard has to hold on the parse path."""
        with pytest.raises(LLMConfigError):
            parse_model_spec('{"provider": "litellm", "model": "high"}')

    def test_parse_from_colon_form_is_refused_too(self):
        with pytest.raises(LLMConfigError):
            parse_model_spec("litellm:high")

    def test_the_router_edge_provider_is_unaffected(self):
        """`litellm_proxy` is what `resolve_group_spec` emits for the edge —
        a custom OpenAI-compatible endpoint, and the whole point of the guard.
        Refusing it too would block the supported path."""
        spec = ModelSpec(
            "litellm_proxy", "high",
            base_url="https://router.example:8443",
            api_key_env="ROUTER_CONSUMER_TEST",
        )
        assert spec.provider == "litellm_proxy"
        assert spec.resolved_base_url() == "https://router.example:8443"

    def test_named_providers_are_unaffected(self):
        for provider in ("anthropic", "openai", "openrouter"):
            assert ModelSpec(provider, "some-model").provider == provider
