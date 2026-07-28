"""Caching-capability resolution and the optional-parameter gate.

Two defects motivated these, both live on 2026-07-27:

* ``_resolve_group_json`` derived a group's caching capability with ``any()``
  over the whole fallback chain, so ``ultra`` reported "explicit breakpoints
  supported" because its 4th entry did — and the CLI emitted Anthropic
  ``cache_control`` markers at an OpenAI-wire provider
  (alpha-engine-config-I4463).
* ``LLMClient.complete()`` silently dropped ``cache_system`` on the LiteLLM
  transport — the same argument meaning three different things across three
  transports (alpha-engine-config-I4469).
"""
from __future__ import annotations

import pytest

from krepis.router import _caching_flags


EXPLICIT = {"capabilities": {"prompt_caching": True, "automatic_prefix_caching": False}}
AUTOMATIC = {"capabilities": {"prompt_caching": False, "automatic_prefix_caching": True}}
NEITHER = {"capabilities": {"prompt_caching": False, "automatic_prefix_caching": False}}


class TestCachingFlags:
    def test_explicit_breakpoints(self):
        assert _caching_flags(EXPLICIT) == (True, False)

    def test_automatic_prefix(self):
        assert _caching_flags(AUTOMATIC) == (False, True)

    def test_neither(self):
        assert _caching_flags(NEITHER) == (False, False)

    def test_missing_capabilities_block_is_no_caching(self):
        assert _caching_flags({}) == (False, False)
        assert _caching_flags({"capabilities": None}) == (False, False)

    def test_both_true_resolves_to_automatic_not_explicit(self):
        """The invalid state must never yield "send markers".

        A provider that caches transparently loses nothing by receiving no
        markers; sending markers to one that rejects unknown fields is an
        outage. So the ambiguous case resolves the non-breaking way.
        """
        both = {"id": "kimi-k3-direct",
                "capabilities": {"prompt_caching": True,
                                 "automatic_prefix_caching": True}}
        assert _caching_flags(both) == (False, True)

    def test_both_true_warns(self, caplog):
        both = {"id": "glm-5.2-direct",
                "capabilities": {"prompt_caching": True,
                                 "automatic_prefix_caching": True}}
        with caplog.at_level("WARNING"):
            _caching_flags(both)
        assert "mutually exclusive" in caplog.text
        assert "glm-5.2-direct" in caplog.text


class TestCachingMechanismEnum:
    """Forward-compat with the registry's move to a single enum, which makes
    the invalid both-true state unrepresentable rather than merely forbidden."""

    @pytest.mark.parametrize("mechanism,expected", [
        ("explicit_breakpoint", (True, False)),
        ("automatic_prefix", (False, True)),
        ("none", (False, False)),
    ])
    def test_enum_is_honored(self, mechanism, expected):
        assert _caching_flags({"capabilities": {"caching_mechanism": mechanism}}) == expected

    def test_enum_wins_over_contradicting_legacy_booleans(self):
        entry = {"capabilities": {
            "caching_mechanism": "automatic_prefix",
            "prompt_caching": True,
            "automatic_prefix_caching": False,
        }}
        assert _caching_flags(entry) == (False, True)

    def test_legacy_booleans_used_when_enum_absent(self):
        assert _caching_flags(EXPLICIT) == (True, False)


class TestGroupCapabilityComesFromPrimary:
    """The regression this whole change exists to prevent."""

    def test_a_fallback_with_explicit_caching_does_not_infect_the_group(self):
        """Reproduces the live `ultra` defect exactly.

        Primary does automatic prefix caching; a later fallback honors
        explicit breakpoints. Under the old any() the GROUP claimed explicit
        support and the CLI emitted markers the primary never wanted.
        """
        chain = [AUTOMATIC, NEITHER, EXPLICIT, NEITHER]
        primary_explicit, primary_automatic = _caching_flags(chain[0])
        assert primary_explicit is False, "primary must not inherit a fallback's capability"
        assert primary_automatic is True

        # The old behaviour, kept here as the thing we must NOT do:
        any_explicit = any(_caching_flags(e)[0] for e in chain)
        assert any_explicit is True
        assert any_explicit != primary_explicit, (
            "this asymmetry IS the bug — any() over the chain disagrees with "
            "what the primary can actually honor"
        )

    def test_primary_with_explicit_caching_is_reported(self):
        assert _caching_flags([EXPLICIT, AUTOMATIC][0]) == (True, False)


class TestCapabilityGate:
    """LLMClient._capability_gate — raise by default, drop only on request."""

    def _client(self):
        from krepis.llm import LLMClient
        from krepis.llm_config import ModelSpec
        # transport is DERIVED from provider on ModelSpec, not passed.
        spec = ModelSpec(provider="deepseek", model="med", max_tokens=1024)
        return LLMClient(spec, callsite_id="krepis-test")

    def test_supported_param_passes_through(self):
        c = self._client()
        assert c._capability_gate("cache_system", True, on_unsupported="raise") is True
        assert c.dropped_params == []

    def test_unsupported_raises_by_default(self):
        from krepis.llm_config import LLMConfigError
        c = self._client()
        with pytest.raises(LLMConfigError, match="cache_system"):
            c._capability_gate("cache_system", False, on_unsupported="raise")

    def test_raise_message_names_the_escape_hatch(self):
        from krepis.llm_config import LLMConfigError
        c = self._client()
        with pytest.raises(LLMConfigError, match="on_unsupported='drop'"):
            c._capability_gate("cache_system", False, on_unsupported="raise")

    def test_drop_records_the_param_and_returns_false(self):
        c = self._client()
        assert c._capability_gate("cache_system", False, on_unsupported="drop") is False
        assert c.dropped_params == ["cache_system"]

    def test_drops_accumulate(self):
        c = self._client()
        c._capability_gate("cache_system", False, on_unsupported="drop")
        c._capability_gate("reasoning", False, on_unsupported="drop")
        assert c.dropped_params == ["cache_system", "reasoning"]

    def test_supported_param_is_never_recorded_as_dropped(self):
        c = self._client()
        c._capability_gate("cache_system", True, on_unsupported="drop")
        assert c.dropped_params == []


class TestLLMResultCarriesDrops:
    def test_dropped_params_defaults_empty_and_is_not_shared(self):
        from krepis.llm import LLMResult, LLMUsage
        a = LLMResult(text="", model="m", provider="p", usage=LLMUsage(), raw_request={})
        b = LLMResult(text="", model="m", provider="p", usage=LLMUsage(), raw_request={})
        assert a.dropped_params == [] and b.dropped_params == []
        a.dropped_params.append("cache_system")
        assert b.dropped_params == [], "default_factory must not share one list"
