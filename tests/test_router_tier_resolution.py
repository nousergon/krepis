"""Complexity tier → registry model group is ONE mapping, in krepis.

alpha-engine-config-I9297 / spine I9294. Brian's 2026-08-29 ruling: "the
entire nous ergon system should now be running through the krepis router...
we should have no other parallel setups, it should all funnel through the
krepis router."

Before this module existed, five hand-maintained tables turned a complexity
tier into a MODEL ID, two of them behind a silent `except Exception` degrade
onto a hardcoded provider ladder. These tests pin the replacement's three
load-bearing properties: the map names GROUPS and never models, an unknown
tier REFUSES instead of defaulting, and reasoning settings come from the
registry rather than from a second per-tier table.
"""

from __future__ import annotations

import pytest

from krepis import router


class TestTierGroupMap:
    def test_tiers_are_the_complexity_label_vocabulary(self):
        assert router.COMPLEXITY_TIERS == ("low", "mid", "high")
        assert set(router.TIER_GROUPS) == set(router.COMPLEXITY_TIERS)

    def test_map_names_registry_groups_never_models(self):
        """Principle 8: never a model ID, a base URL, or a provider name.

        The tables this replaced held `deepseek-v4-flash` / `deepseek-v4-pro`
        — a vendor's model slugs at the call site. A group name carries no
        vendor, so the guard is that no value looks like one.
        """
        for tier, group in router.TIER_GROUPS.items():
            assert group in ("low", "med", "high", "ultra"), (tier, group)
            assert "/" not in group and "-" not in group, (tier, group)
            for vendor in ("deepseek", "anthropic", "claude", "openai",
                           "glm", "grok", "qwen", "gpt"):
                assert vendor not in group.lower(), (tier, group)

    def test_group_for_tier_resolves_each_tier(self):
        assert router.group_for_tier("low") == "low"
        assert router.group_for_tier("mid") == "med"
        assert router.group_for_tier("high") == "high"

    def test_unknown_tier_refuses_rather_than_defaulting_to_mid(self):
        """Every table this replaced defaulted an unrecognised tier to `mid`.

        A silent default bills the wrong tier and looks healthy. R20: fail
        closed.
        """
        with pytest.raises(ValueError) as exc:
            router.group_for_tier("medium")
        assert "medium" in str(exc.value)
        assert "low" in str(exc.value) and "high" in str(exc.value)

    @pytest.mark.parametrize("bad", ["", None, "ultra", "MID", "high+mid"])
    def test_no_tier_silently_becomes_a_group(self, bad):
        with pytest.raises((ValueError, TypeError)):
            router.group_for_tier(bad)


class TestReasoningParams:
    def test_reads_the_registrys_own_reasoning_block(self):
        route = {"schema_version": router.RESOLVE_SCHEMA_VERSION,
                 "params": {"reasoning": {"effort": "max"}}}
        assert router.reasoning_params(route) == {"effort": "max"}

    def test_absent_reasoning_is_how_the_registry_says_thinking_off(self):
        route = {"schema_version": router.RESOLVE_SCHEMA_VERSION,
                 "params": {"reasoning": None}}
        assert router.reasoning_params(route) is None
        assert router.reasoning_params(
            {"schema_version": router.RESOLVE_SCHEMA_VERSION, "params": {}}
        ) is None

    def test_returns_a_copy_so_a_consumer_cannot_mutate_the_route(self):
        route = {"schema_version": router.RESOLVE_SCHEMA_VERSION,
                 "params": {"reasoning": {"effort": "max"}}}
        router.reasoning_params(route)["effort"] = "low"
        assert route["params"]["reasoning"] == {"effort": "max"}

    def test_unknown_schema_version_refuses_rather_than_probing(self):
        with pytest.raises(RuntimeError) as exc:
            router.reasoning_params({"schema_version": 99, "params": {}})
        assert "schema_version" in str(exc.value)


class TestTierResolutionDelegates:
    def test_resolve_tier_structured_resolves_the_mapped_group(self, monkeypatch):
        seen = {}

        def fake(group, *, exec_context=None, wire=None, requires=()):
            seen.update(group=group, exec_context=exec_context, requires=requires)
            return {"schema_version": router.RESOLVE_SCHEMA_VERSION}

        monkeypatch.setattr(router, "resolve_group_structured", fake)
        router.resolve_tier_structured("mid", exec_context="ec2")
        assert seen["group"] == "med"
        assert seen["exec_context"] == "ec2"

    def test_resolve_tier_structured_fails_closed_on_a_bad_tier(self, monkeypatch):
        """No hardcoded model ladder behind it to fall through to."""
        def fake(*a, **k):  # pragma: no cover - must never be reached
            raise AssertionError("resolution attempted for an unknown tier")

        monkeypatch.setattr(router, "resolve_group_structured", fake)
        with pytest.raises(ValueError):
            router.resolve_tier_structured("nope")

    def test_resolve_tier_spec_delegates_to_resolve_group_spec(self, monkeypatch):
        seen = {}

        def fake(group, **kwargs):
            seen.update(group=group, **kwargs)
            return ("spec", "route")

        monkeypatch.setattr(router, "resolve_group_spec", fake)
        assert router.resolve_tier_spec("high", exec_context="lambda") == ("spec", "route")
        assert seen["group"] == "high"
        assert seen["exec_context"] == "lambda"
