"""Contract tests for the single registry derivation (model-router-policy R6/R6a).

These are contract tests, not value tests: each asserts a rule the derivation
must obey for ANY registry, using a fixture that carries the shapes which have
actually caused incidents. Three of them pin divergences that existed while two
derivations were maintained by hand — a fixture matching only today's registry
would have passed on both sides of every one of them.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from krepis import model_registry as mr

FIXTURE = textwrap.dedent(
    """
    models:
      - id: live-egress
        provider: deepseek
        route: egress_proxy
        model: deepseek-v4-flash
        api_base: http://127.0.0.1:8972/v1
        upstream_host: api.deepseek.com
        status: active
        params:
          max_tokens: 8192
      - id: dead-deprecated
        provider: gemini
        route: egress_proxy
        model: gemini-2.0-flash
        api_base: http://127.0.0.1:8973/v1
        status: deprecated
      - id: dead-unavailable
        provider: zhipu
        route: egress_proxy
        model: qwen3-max
        api_base: http://127.0.0.1:8974/v1
        status: unavailable
      # The shape that produced `anthropic/anthropic/...` + the wrong
      # credential when a derivation branched on provider instead of route.
      - id: anthropic-via-openrouter
        provider: anthropic
        route: openrouter
        model: anthropic/claude-opus-5
        status: active
      - id: anthropic-direct
        provider: anthropic
        route: direct
        model: claude-opus-5
        status: active
      - id: no-status-declared
        provider: xai
        route: egress_proxy
        model: grok-4.5
        api_base: http://127.0.0.1:8975/v1
      - id: sub-floor-tpm
        provider: moonshot
        route: egress_proxy
        model: kimi-k3
        api_base: http://127.0.0.1:8976/v1
        status: active
        tpm: 25000
      - id: reasoning-null-string
        provider: deepseek
        route: egress_proxy
        model: deepseek-v4-pro
        api_base: http://127.0.0.1:8977/v1
        status: active
        params:
          reasoning: "null"
      - id: openrouter-pinned
        provider: anthropic
        route: openrouter
        model: anthropic/claude-opus-5
        status: active
        openrouter_provider_pinning:
          order: [anthropic]
          allow_fallbacks: false
      # Pinning declared on a non-openrouter route must be ignored — this is
      # the provider-vs-route field confusion that caused the
      # `anthropic-via-openrouter` incident above, applied to this field too.
      - id: direct-route-pinning-field-ignored
        provider: anthropic
        route: direct
        model: claude-opus-5
        status: active
        openrouter_provider_pinning:
          order: [anthropic]
      # R25/R26-conformant shape (alpha-engine-config-I6286): OpenRouter
      # reached via the egress proxy, not a direct route. Pinning must still
      # apply — it is keyed on the actual upstream, not on `route` alone.
      - id: openrouter-pinned-via-proxy
        provider: openrouter
        route: egress_proxy
        api_base: http://127.0.0.1:8990
        upstream_host: openrouter.ai
        model: deepseek/deepseek-v4-pro
        status: active
        openrouter_provider_pinning:
          order: [deepseek]
          allow_fallbacks: false

    model_groups:
      low:
        - dead-deprecated
        - live-egress
      allofthemdead:
        - dead-deprecated
        - dead-unavailable
      mixed:
        - live-egress
        - dead-unavailable
        - no-status-declared
    """
)


@pytest.fixture()
def registry(tmp_path: Path) -> mr.Registry:
    p = tmp_path / "LLM_MODEL_REGISTRY.yaml"
    p.write_text(FIXTURE)
    return mr.load_registry(p)


# ── Status filtering (R4) ────────────────────────────────────────────────

def test_both_excluded_statuses_are_filtered_not_just_deprecated(registry):
    """`unavailable` is excluded alongside `deprecated`.

    The forked derivations disagreed on exactly this: the proxy generator
    excluded both, the in-process router excluded neither. Measured against the
    live registry 2026-08-12, that made five `unavailable` deployments
    reachable in-process — an R4 violation on one path only, which is the
    failure R6 predicts.
    """
    live = registry.live_models
    assert "dead-deprecated" not in live
    assert "dead-unavailable" not in live
    assert "live-egress" in live


def test_a_row_with_no_status_counts_as_live(registry):
    """`status` is optional; absence must not silently empty the config."""
    assert "no-status-declared" in registry.live_models


def test_group_filtering_happens_before_the_primary_is_chosen(registry):
    """A group whose FIRST declared member is dead still gets a primary.

    Filtering inside the caller's loop would key the primary off `i == 0` of
    the raw list, so `low` would produce no alias at all and
    `completion(model="low")` would fail with "model not found" — worse than
    the dead member the filter removes.
    """
    assert registry.live_group_ids("low") == ["live-egress"]


def test_a_group_with_no_live_member_yields_nothing_rather_than_a_dead_alias(registry):
    assert registry.live_group_ids("allofthemdead") == []
    assert "allofthemdead" not in dict(registry.iter_live_groups())


def test_live_group_order_follows_the_registry_not_the_models_index(registry):
    """Declared order is the fallback order, so it is load-bearing."""
    assert registry.live_group_ids("mixed") == ["live-egress", "no-status-declared"]


# ── Route-first prefix and credential ────────────────────────────────────

def test_route_decides_the_prefix_not_the_provider(registry):
    """provider: anthropic + route: openrouter must NOT double the prefix.

    Branching on provider produced `anthropic/anthropic/claude-opus-5` — a
    doubled prefix on a model the picker offers. Caught 2026-07-29 in one
    derivation and never carried to the other.
    """
    entry = registry.models["anthropic-via-openrouter"]
    assert mr.litellm_model(entry) == "openrouter/anthropic/claude-opus-5"
    assert mr.api_key_env(entry) == "OPENROUTER_API_KEY"


def test_route_direct_still_honours_the_provider_wire_format(registry):
    entry = registry.models["anthropic-direct"]
    assert mr.litellm_model(entry) == "anthropic/claude-opus-5"
    assert mr.api_key_env(entry) == "ANTHROPIC_API_KEY"


def test_egress_route_carries_only_a_placeholder_credential(registry):
    """R25 key isolation: the real key lives in the proxy, never here."""
    entry = registry.models["live-egress"]
    assert mr.api_key_env(entry) == mr.EGRESS_PLACEHOLDER_ENV
    assert mr.api_key_for(entry, style="value") == mr.EGRESS_PLACEHOLDER_DEFAULT


def test_reference_style_emits_an_indirection_never_a_secret(registry, monkeypatch):
    """A proxy config on disk must carry the NAME, not the resolved value."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-never-be-rendered")
    entry = registry.models["anthropic-via-openrouter"]
    assert mr.api_key_for(entry, style="reference") == "os.environ/OPENROUTER_API_KEY"


def test_overrides_apply_to_values_and_are_ignored_for_references(registry):
    entry = registry.models["anthropic-via-openrouter"]
    ov = {"OPENROUTER_API_KEY": "sk-from-ssm"}
    assert mr.api_key_for(entry, style="value", overrides=ov) == "sk-from-ssm"
    assert mr.api_key_for(entry, style="reference", overrides=ov) == "os.environ/OPENROUTER_API_KEY"


def test_an_unknown_api_key_style_raises_rather_than_defaulting(registry):
    with pytest.raises(ValueError, match="api key style"):
        mr.api_key_for(registry.models["live-egress"], style="whatever")


# ── Deployment params ────────────────────────────────────────────────────

def test_deployment_params_carry_api_base_and_upstream_host_for_egress(registry):
    params = mr.deployment_params(registry.models["live-egress"])
    assert params["api_base"] == "http://127.0.0.1:8972/v1"
    assert params["extra_headers"]["X-Upstream-Host"] == "api.deepseek.com"
    assert params["max_tokens"] == 8192


def test_the_literal_string_null_is_not_forwarded_as_a_reasoning_directive(registry):
    assert mr.extra_body(registry.models["reasoning-null-string"]) is None
    assert "extra_body" not in mr.deployment_params(registry.models["reasoning-null-string"])


def test_openrouter_provider_pinning_is_injected_on_openrouter_routes(registry):
    entry = registry.models["openrouter-pinned"]
    assert mr.extra_body(entry) == {
        "provider": {"order": ["anthropic"], "allow_fallbacks": False}
    }


def test_openrouter_provider_pinning_is_ignored_off_the_openrouter_route(registry):
    # Checked against `route`, not `provider` — this is the field confusion
    # that produced the `anthropic/anthropic/...` incident for litellm_model
    # and api_key_env, applied here too: a direct-route entry with the
    # pinning field set must not have it forwarded.
    entry = registry.models["direct-route-pinning-field-ignored"]
    assert mr.extra_body(entry) is None


def test_openrouter_provider_pinning_is_injected_when_proxied_to_openrouter(registry):
    # alpha-engine-config-I6286: route: egress_proxy with upstream_host:
    # openrouter.ai is the R25/R26-conformant shape. Pinning must still
    # apply — the discriminator is the actual upstream, not `route == "openrouter"`.
    entry = registry.models["openrouter-pinned-via-proxy"]
    assert mr.extra_body(entry) == {
        "provider": {"order": ["deepseek"], "allow_fallbacks": False}
    }


def test_deployment_params_omit_rpm_tpm_and_timeout(registry):
    """Those are per-surface rendering choices, not registry facts.

    Asserted so a future edit that "helpfully" moves them in has to change a
    test that says why they are out.
    """
    params = mr.deployment_params(registry.models["sub-floor-tpm"])
    assert "rpm" not in params
    assert "tpm" not in params
    assert "timeout" not in params


# ── tpm floor ────────────────────────────────────────────────────────────

def test_a_sub_floor_tpm_is_clamped_not_honoured(registry):
    """A tpm below one prompt makes the deployment permanently unroutable."""
    tpm = mr.declared_tpm(registry.models["sub-floor-tpm"], mr.MIN_ADMISSIBLE_TPM)
    assert tpm == mr.MIN_ADMISSIBLE_TPM


def test_an_undeclared_tpm_stays_undeclared(registry):
    """No invented default: it caps usable context while encoding no real limit."""
    assert mr.declared_tpm(registry.models["live-egress"], mr.MIN_ADMISSIBLE_TPM) is None


# ── Discovery fails closed (R1/R20) ──────────────────────────────────────

def test_a_missing_explicit_path_raises_rather_than_falling_back(tmp_path):
    with pytest.raises(mr.RegistryNotFoundError):
        mr.find_registry(tmp_path / "nope.yaml")


def test_a_missing_env_path_raises_rather_than_walking_up(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_REGISTRY_PATH", str(tmp_path / "nope.yaml"))
    with pytest.raises(mr.RegistryNotFoundError):
        mr.find_registry()


def test_discovery_failure_is_distinguishable_from_any_other_file_error(tmp_path, monkeypatch):
    """Callers must be able to tell "no registry" from "could not read one"."""
    monkeypatch.setenv("LLM_MODEL_REGISTRY_PATH", str(tmp_path / "nope.yaml"))
    with pytest.raises(FileNotFoundError) as exc:
        mr.find_registry()
    assert isinstance(exc.value, mr.RegistryNotFoundError)


def test_a_registry_that_is_not_a_mapping_raises(tmp_path):
    p = tmp_path / "LLM_MODEL_REGISTRY.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        mr.load_registry(p)


# ── R6a: the derivation is not forked inside krepis either ───────────────

def test_router_does_not_parse_the_registry_itself():
    """`krepis.router` renders; it must not load YAML.

    The fleet-wide half of this check lives in alpha-engine-config, which is
    the only repo that can see both producers. This half keeps krepis honest
    about its own module boundary.
    """
    source = (Path(mr.__file__).parent / "router.py").read_text()
    assert "safe_load" not in source, (
        "krepis.router parses YAML again — the derivation belongs in "
        "krepis.model_registry (model-router-policy R6)"
    )
