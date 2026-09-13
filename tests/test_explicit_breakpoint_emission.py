"""Marker emission is keyed on the DECLARED capability, never on the transport.

``prompt-caching-policy.md`` §3.6: explicit ``cache_control`` markers are
emitted only when the SERVED model's registry block declares mechanism M1, and
that check is "keyed on the registry capability, never on the transport or the
provider name. Transport is a proxy for mechanism that is wrong in exactly the
case that costs the most." ``model-portability-policy.md`` §2 classifies the
transport form as a Selection -> Transport plane leak.

krepis had the leak. ``LLMClient.complete`` / ``.structured`` /
``.complete_grounded`` constructed a ``cache_control`` block if and only if
``spec.transport == TRANSPORT_ANTHROPIC``. Every other route — the router edge,
OpenRouter, any OpenAI-compatible endpoint — emitted no marker at all, whatever
the model declared. An Anthropic (M1) model reached over an OpenAI-shaped route
therefore got **zero caching, at roughly 10x the cached input rate, silently**:
nothing errors, the output is identical, and the only signal is the invoice.

``krepis-I67``. The tests below pin the four cases the issue names.
"""

from __future__ import annotations

import pytest

from krepis.llm import LLMClient
from krepis.llm_config import ModelSpec

from tests.test_llm import (
    FakeAnthropic,
    FakeOpenAI,
    _anthropic_msg,
    _openai_resp,
    _text_block,
)

# The live-shaped registry fixture lives with the router tests.
from tests.test_router import registry_file  # noqa: F401


def _edge_spec(**kw) -> ModelSpec:
    """A spec on the ROUTER EDGE — OpenAI-shaped wire, arbitrary served model.

    This is the shape the bug is about: the transport says nothing at all
    about the caching mechanism of whatever the proxy routes to.
    """
    base = dict(
        provider="litellm_proxy",
        model="high",
        base_url="https://router.example/v1",
        api_key_env="KREPIS_TEST_KEY",
        max_tokens=64,
    )
    base.update(kw)
    return ModelSpec(**base)


def _anthropic_spec(**kw) -> ModelSpec:
    base = dict(provider="anthropic", model="claude-haiku-4-5", max_tokens=64)
    base.update(kw)
    return ModelSpec(**base)


def _client(spec, fake):
    return LLMClient(
        spec,
        api_key="test-key",
        callsite_id="krepis-test",
        client_factory=lambda _spec, _key: fake,
    )


def _system_parts(kwargs: dict) -> list:
    """The system message's content, always as a list of parts."""
    system = next(m for m in kwargs["messages"] if m["role"] == "system")
    content = system["content"]
    return content if isinstance(content, list) else [content]


def _markers(parts) -> list:
    return [p for p in parts if isinstance(p, dict) and "cache_control" in p]


class TestOpenAIWireEmitsMarkersWhenDeclared:
    """The expensive case: an M1 model reached over an OpenAI-shaped route."""

    def test_declared_m1_emits_cache_control_on_the_system_segment(self):
        fake = FakeOpenAI([_openai_resp("ok")])
        _client(_edge_spec(supports_prompt_caching=True), fake).complete(
            system="STATIC PREFIX", user_content="turn", cache_system=True
        )
        parts = _system_parts(fake.kwargs[0])
        markers = _markers(parts)
        assert len(markers) == 1, f"expected one breakpoint, got {parts!r}"
        assert markers[0]["cache_control"] == {"type": "ephemeral"}
        assert markers[0]["text"] == "STATIC PREFIX"
        assert markers[0]["type"] == "text"

    def test_the_marker_ends_the_stable_segment_not_the_volatile_turn(self):
        """Volatility ordering (prompt-caching-policy §3.1): a breakpoint ends
        a STABLE segment. One on the per-turn content caches nothing and burns
        one of four slots."""
        fake = FakeOpenAI([_openai_resp("ok")])
        _client(_edge_spec(supports_prompt_caching=True), fake).complete(
            system="STATIC PREFIX", user_content="volatile turn"
        )
        user = next(m for m in fake.kwargs[0]["messages"] if m["role"] == "user")
        content = user["content"]
        parts = content if isinstance(content, list) else [content]
        assert _markers(parts) == []

    def test_structured_emits_it_too(self):
        """The structured path builds its own payload; a fix applied to one
        method and not the other is the same defect with a smaller blast
        radius."""
        fake = FakeOpenAI([
            _openai_resp('{"name": "x", "score": 1}', finish_reason="stop")
        ])
        _client(_edge_spec(supports_prompt_caching=True), fake).structured(
            system="STATIC PREFIX",
            user_content="turn",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            schema_name="Spec",
        )
        assert _markers(_system_parts(fake.kwargs[0]))


class TestNoMarkersWhenNotDeclared:
    def test_declared_m2_emits_none_and_records_the_strip(self):
        """Automatic prefix caching needs no markers, and sending them to a
        provider that rejects unknown fields is an outage. The strip is a
        DECLARED degradation path (model-portability I9), not a silent drop."""
        fake = FakeOpenAI([_openai_resp("ok")])
        result = _client(
            _edge_spec(
                supports_prompt_caching=False,
                supports_automatic_prefix_caching=True,
            ),
            fake,
        ).complete(system="STATIC PREFIX", user_content="turn", cache_system=True)
        assert _markers(_system_parts(fake.kwargs[0])) == []
        assert "cache_system" in result.dropped_params

    def test_no_declared_mechanism_emits_none_and_records_the_strip(self):
        """M4. A model declaring neither mechanism is treated as uncached by
        every consumer — the fail-safe direction for correctness and the
        fail-LOUD direction for cost (prompt-caching-policy §2 rule 2)."""
        fake = FakeOpenAI([_openai_resp("ok")])
        result = _client(_edge_spec(supports_prompt_caching=False), fake).complete(
            system="STATIC PREFIX", user_content="turn", cache_system=True
        )
        assert _markers(_system_parts(fake.kwargs[0])) == []
        assert "cache_system" in result.dropped_params

    def test_cache_system_false_is_not_recorded_as_a_drop(self):
        """The caller declining caching is not a degraded call — recording it
        would make `dropped_params` useless as a degradation signal."""
        fake = FakeOpenAI([_openai_resp("ok")])
        result = _client(_edge_spec(supports_prompt_caching=True), fake).complete(
            system="STATIC PREFIX", user_content="turn", cache_system=False
        )
        assert _markers(_system_parts(fake.kwargs[0])) == []
        assert result.dropped_params == []


class TestStructuredDoesNotPolluteTheDegradationSignal:
    """`structured()` asks for caching unconditionally and takes no
    `cache_system` argument, so the strip is this METHOD's preference not
    being honoured, never the caller's request being refused.

    Recording it would put `cache_system` in `dropped_params` on every
    structured call against an M2 model — the healthy majority — which makes
    the field useless as a degradation signal. The log line still fires, so
    the strip is never invisible.
    """

    def test_structured_against_an_m2_model_records_no_drop(self):
        fake = FakeOpenAI([
            _openai_resp('{"name": "x", "score": 1}', finish_reason="stop")
        ])
        result = _client(
            _edge_spec(
                supports_prompt_caching=False,
                supports_automatic_prefix_caching=True,
            ),
            fake,
        ).structured(
            system="STATIC PREFIX",
            user_content="turn",
            schema={"type": "object", "properties": {"name": {"type": "string"}}},
            schema_name="Spec",
        )
        assert _markers(_system_parts(fake.kwargs[0])) == []
        assert "cache_system" not in result.dropped_params


class TestAnthropicTransportKeysOnTheSameFact:
    def test_declared_m1_still_emits_the_native_block(self):
        fake = FakeAnthropic([_anthropic_msg([_text_block("ok")])])
        _client(_anthropic_spec(supports_prompt_caching=True), fake).complete(
            system="STATIC PREFIX", user_content="turn"
        )
        assert fake.payloads[0]["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_an_m2_model_on_the_anthropic_transport_emits_none(self):
        """The mirror of the headline bug, from the other direction: the
        transport does not decide, so declaring M2 strips the marker even on
        the native SDK."""
        fake = FakeAnthropic([_anthropic_msg([_text_block("ok")])])
        result = _client(
            _anthropic_spec(
                supports_prompt_caching=False,
                supports_automatic_prefix_caching=True,
            ),
            fake,
        ).complete(system="STATIC PREFIX", user_content="turn")
        assert "cache_control" not in fake.payloads[0]["system"][0]
        assert "cache_system" in result.dropped_params

    def test_undeclared_keeps_the_provider_registry_default(self):
        """A hand-built spec declares nothing. The default is a fact about the
        PROVIDER, declared once in PROVIDER_REGISTRY — every Anthropic model is
        M1 by construction — not a transport branch in the emission path. A
        router-resolved spec always overrides it."""
        fake = FakeAnthropic([_anthropic_msg([_text_block("ok")])])
        _client(_anthropic_spec(), fake).complete(
            system="STATIC PREFIX", user_content="turn"
        )
        assert fake.payloads[0]["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_undeclared_on_an_openai_shaped_route_emits_none(self):
        """The safe direction: a marker sent to a provider that rejects
        unknown fields is an outage, so an undeclared OpenAI-shaped route gets
        none."""
        fake = FakeOpenAI([_openai_resp("ok")])
        _client(_edge_spec(), fake).complete(
            system="STATIC PREFIX", user_content="turn"
        )
        assert _markers(_system_parts(fake.kwargs[0])) == []


class TestNoTransportBranchDecidesEmission:
    def test_the_same_declaration_produces_markers_on_both_transports(self):
        """Two specs identical but for the TRANSPORT must agree about whether
        a marker is emitted — which is the whole of I67 stated as one
        assertion."""
        openai_fake = FakeOpenAI([_openai_resp("ok")])
        _client(_edge_spec(supports_prompt_caching=True), openai_fake).complete(
            system="STATIC PREFIX", user_content="turn"
        )
        anthropic_fake = FakeAnthropic([_anthropic_msg([_text_block("ok")])])
        _client(_anthropic_spec(supports_prompt_caching=True), anthropic_fake).complete(
            system="STATIC PREFIX", user_content="turn"
        )
        assert _markers(_system_parts(openai_fake.kwargs[0]))
        assert "cache_control" in anthropic_fake.payloads[0]["system"][0]

    def test_no_declared_mechanism_produces_none_on_both_transports(self):
        openai_fake = FakeOpenAI([_openai_resp("ok")])
        _client(_edge_spec(supports_prompt_caching=False), openai_fake).complete(
            system="STATIC PREFIX", user_content="turn"
        )
        anthropic_fake = FakeAnthropic([_anthropic_msg([_text_block("ok")])])
        _client(_anthropic_spec(supports_prompt_caching=False), anthropic_fake).complete(
            system="STATIC PREFIX", user_content="turn"
        )
        assert _markers(_system_parts(openai_fake.kwargs[0])) == []
        assert "cache_control" not in anthropic_fake.payloads[0]["system"][0]


class TestRouterCarriesTheCapabilityOntoTheSpec:
    """The gate is only as good as what reaches it.

    ``_route_to_spec`` dropped BOTH caching flags on the floor: the resolve
    contract has carried ``supports_prompt_caching`` and
    ``automatic_prefix_caching`` since PR69 made them primary-derived, and the
    adapter read neither. So every router-resolved spec arrived at the client
    declaring no mechanism at all.
    """

    @pytest.mark.parametrize("group", ["low", "med", "high", "ultra"])
    def test_resolved_spec_declares_a_caching_mechanism(
        self, group, registry_file, monkeypatch  # noqa: F811
    ):
        from krepis import router as _router

        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                m.setenv("KREPIS_EXEC_CONTEXT", "laptop")
                spec, route = _router.resolve_group_spec(group)
        finally:
            _router._router = None
        assert spec.supports_prompt_caching is not None, (
            "a router-resolved spec must carry the registry's declaration; "
            "None means the gate falls back to the provider default and the "
            "registry had no say"
        )
        assert spec.supports_prompt_caching == bool(
            route.get("supports_prompt_caching")
        )
        assert spec.supports_automatic_prefix_caching == bool(
            route.get("automatic_prefix_caching")
        )
