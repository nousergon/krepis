"""A group is a tier, not a call shape — resolution must honour the difference.

alpha-engine-config-I7904. ``LLM_MODEL_REGISTRY.yaml`` has carried a per-model
``capabilities.tool_choice`` flag since the schema was written, and
``LLM_CALLSITE_REGISTRY.yaml`` has carried ``requires_forced_tool_call`` on the
eval judge's two rows. Neither reached ROUTING. So the judge addressed group
``low``, was handed the member the registry itself says refuses a forced tool
call, and took an identical permanent 400 on all three attempts.

These tests pin the resolution contract: a caller states what its REQUEST SHAPE
needs, the derivation excludes members that do not declare it BEFORE a primary
is chosen, and a group with no such member raises at resolve time rather than
returning a route that cannot work.
"""

from __future__ import annotations

import pytest

from krepis import model_registry as _mr
from krepis import router as _router

_MIXED_CAPABILITY_REGISTRY = """
schema_version: 1

model_groups:
  low:
    - refuses-tools
    - accepts-tools
    - also-accepts-tools
  med:
    - refuses-tools-too
    - parked

models:
  - id: refuses-tools
    provider: deepseek
    route: egress_proxy
    model: deepseek-v4-flash
    api_base: http://127.0.0.1:8971/v1
    reachable_from: [laptop, ec2, lambda]
    endpoints:
      openai: http://127.0.0.1:8971/v1
      anthropic: http://127.0.0.1:8971
    status: active
    params:
      max_tokens: 8192
    capabilities:
      tool_choice: false
  - id: accepts-tools
    provider: zhipu
    route: egress_proxy
    model: glm-4.7-flash
    api_base: http://127.0.0.1:8974/v1
    reachable_from: [laptop, ec2, lambda]
    endpoints:
      openai: http://127.0.0.1:8974/v1
      anthropic: http://127.0.0.1:8974
    status: active
    params:
      max_tokens: 8192
    capabilities:
      tool_choice: true
      streaming: true
  - id: also-accepts-tools
    provider: xai
    route: egress_proxy
    model: grok-4
    api_base: http://127.0.0.1:8975/v1
    reachable_from: [laptop, ec2, lambda]
    endpoints:
      openai: http://127.0.0.1:8975/v1
      anthropic: http://127.0.0.1:8975
    status: active
    capabilities:
      tool_choice: true
  - id: refuses-tools-too
    provider: deepseek
    route: egress_proxy
    model: deepseek-v4-pro
    api_base: http://127.0.0.1:8971/v1
    reachable_from: [laptop, ec2, lambda]
    endpoints:
      openai: http://127.0.0.1:8971/v1
    status: active
    capabilities:
      tool_choice: false
  - id: parked
    provider: zhipu
    route: egress_proxy
    model: glm-5.2
    api_base: http://127.0.0.1:8974/v1
    reachable_from: [laptop, ec2, lambda]
    endpoints:
      openai: http://127.0.0.1:8974/v1
    status: unavailable
    capabilities:
      tool_choice: true
  - id: silent-about-tools
    provider: moonshot
    route: egress_proxy
    model: kimi-k3
    api_base: http://127.0.0.1:8976/v1
    reachable_from: [laptop, ec2, lambda]
    status: active
"""


@pytest.fixture()
def mixed_registry(tmp_path):
    p = tmp_path / "LLM_MODEL_REGISTRY.yaml"
    p.write_text(_MIXED_CAPABILITY_REGISTRY)
    return p


@pytest.fixture()
def registry(mixed_registry):
    return _mr.load_registry(mixed_registry)


class TestTheDerivationFilters:
    def test_without_a_requirement_the_chain_is_unchanged(self, registry):
        assert registry.live_group_ids("low") == [
            "refuses-tools", "accepts-tools", "also-accepts-tools",
        ]

    def test_a_requirement_moves_the_primary_past_the_member_that_refuses(self, registry):
        assert registry.live_group_ids("low", requires=("tool_choice",)) == [
            "accepts-tools", "also-accepts-tools",
        ]

    def test_status_and_capability_are_two_different_rejections(self, registry):
        reasons = dict(registry.capability_rejections("med", requires=("tool_choice",)))
        assert "status is 'unavailable'" in reasons["parked"]
        assert "capabilities.tool_choice" in reasons["refuses-tools-too"]

    def test_an_undeclared_flag_is_not_capability(self, registry):
        """Absence means nobody measured it. Gambling on it is how a forced
        tool call reaches a model that refuses one."""
        entry = registry.models["silent-about-tools"]
        assert _mr.entry_declares_capability(entry, "tool_choice") is False

    def test_a_group_with_no_capable_member_derives_to_nothing(self, registry):
        """The derivation reports emptiness; the RESOLVER is what raises on it
        (see TestResolution). Splitting the two keeps the derivation usable by
        the config generator, which must be able to ASK without failing."""
        assert registry.live_group_ids("med", requires=("tool_choice",)) == []
        err = _mr.CapabilityUnavailableError(
            "med", "tool_choice",
            registry.capability_rejections("med", requires=("tool_choice",)),
        )
        assert "REGISTRY gap" in str(err)
        assert "refuses-tools-too" in str(err)


class TestResolution:
    def _resolve(self, monkeypatch, registry_path, **kw):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_path))
                m.setattr(_router, "_probe_egress_proxy", lambda *a, **k: True)
                m.setattr(
                    _router, "_litellm_edge_admission",
                    lambda: (True, "https://router.example:8443", []),
                )
                return _router.resolve_group_structured("low", exec_context="lambda", **kw)
        finally:
            _router._router = None

    def test_the_unqualified_call_still_gets_the_declared_primary(
        self, monkeypatch, mixed_registry
    ):
        info = self._resolve(monkeypatch, mixed_registry, wire="openai")
        assert info["primary_registry_id"] == "refuses-tools"
        assert info["deployment_id"] == "low-refuses-tools"

    def test_a_forced_tool_call_resolves_to_a_member_that_accepts_one(
        self, monkeypatch, mixed_registry
    ):
        info = self._resolve(
            monkeypatch, mixed_registry, wire="openai", requires=("tool_choice",)
        )
        assert info["primary_registry_id"] == "accepts-tools"
        assert info["deployment_id"] == "low-accepts-tools"

    def test_an_unservable_requirement_raises_at_resolve_time(
        self, monkeypatch, mixed_registry
    ):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(mixed_registry))
                m.setattr(
                    _router, "_litellm_edge_admission",
                    lambda: (True, "https://router.example:8443", []),
                )
                with pytest.raises(_mr.CapabilityUnavailableError) as exc:
                    _router.resolve_group_structured(
                        "med", exec_context="lambda", wire="openai",
                        requires=("tool_choice",),
                    )
        finally:
            _router._router = None
        assert "refuses-tools-too" in str(exc.value)
        assert "no retry and no fallback" in str(exc.value)

    def test_an_unroutable_capability_name_is_refused(self, monkeypatch, mixed_registry):
        """Narrowing a chain for a COST preference (`prompt_caching`) would
        silently reduce depth for something that never rejects a request."""
        with pytest.raises(ValueError, match="not a routable capability"):
            self._resolve(
                monkeypatch, mixed_registry, wire="openai",
                requires=("prompt_caching",),
            )

    def test_the_degraded_walk_names_the_capability_as_its_own_skip_reason(
        self, monkeypatch, mixed_registry
    ):
        """R29: "excluded by status", "lacks a capability", "not reachable from
        here" and "unhealthy" must stay four distinguishable answers."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(mixed_registry))
                m.setenv("KREPIS_LITELLM_PROXY_URL", "http://127.0.0.1:1")
                m.setattr(_router, "_probe_egress_proxy", lambda *a, **k: True)
                info = _router.resolve_group_structured(
                    "low", exec_context="lambda", wire="openai",
                    requires=("tool_choice",),
                )
        finally:
            _router._router = None
        assert info["registry_id"] == "accepts-tools"
        skips = {s["registry_id"]: s["reason"] for s in info["skipped_entries"]}
        assert "capabilities.tool_choice" in skips["refuses-tools"]
        assert "not a health signal" in skips["refuses-tools"].lower()


class TestFallbackChainsAreCapabilityHomogeneous:
    def test_a_capable_primary_never_degrades_onto_a_member_that_refuses(
        self, mixed_registry
    ):
        """The same defect from the other direction: falling a forced tool call
        over onto a member the registry says refuses one turns an availability
        blip into a permanent 400."""
        _model_list, fallbacks, _aliases = _router._parse_registry(mixed_registry)
        chains = {k: v for f in fallbacks for k, v in f.items()}
        assert chains["low-accepts-tools"] == ["low-also-accepts-tools"]
        assert "low-refuses-tools" not in chains["low-accepts-tools"]


class TestThinkingPassthrough:
    """`thinking` and `reasoning` are two different upstream controls.

    Measured against api.deepseek.com through the egress proxy, 2026-08-21,
    `deepseek-v4-flash` with a forced `tool_choice`: no thinking field -> 400,
    `reasoning: {exclude: true}` -> the identical 400, `thinking:
    {type: disabled}` -> 200 with tool_calls. Until the registry could emit the
    third, the only answer a tool-calling consumer had was a different vendor.
    """

    def test_thinking_reaches_extra_body(self):
        assert _mr.extra_body(
            {"route": "egress_proxy", "params": {"thinking": {"type": "disabled"}}}
        ) == {"thinking": {"type": "disabled"}}

    def test_thinking_and_reasoning_are_carried_independently(self):
        got = _mr.extra_body({
            "route": "egress_proxy",
            "params": {"reasoning": {"effort": "low"}, "thinking": {"type": "disabled"}},
        })
        assert got == {"reasoning": {"effort": "low"}, "thinking": {"type": "disabled"}}

    def test_the_yaml_null_string_is_treated_as_absent(self):
        """Same rule `reasoning` already has: a YAML author writing the literal
        four characters means "off", and forwarding them upstream sends a
        string where an object belongs."""
        assert _mr.extra_body(
            {"route": "egress_proxy", "params": {"thinking": "null"}}
        ) is None


class TestStreamingIsARoutableCapability:
    """``streaming`` joined :data:`ROUTABLE_CAPABILITIES` for the same reason
    ``tool_choice`` is there (alpha-engine-config-I8164): a route that cannot
    stream does not serve a streamed request more slowly, it does not serve it
    at all. It is also the one call shape whose absence a successful response
    cannot reveal — a client that fell back to a non-streaming request would
    get a valid completion carrying the request-deadline failure envelope
    streaming was adopted to remove."""

    def test_it_is_declared_routable(self):
        assert "streaming" in _mr.ROUTABLE_CAPABILITIES

    def test_a_requirement_narrows_the_chain_to_the_declaring_member(self, registry):
        assert registry.live_group_ids("low", requires=("streaming",)) == [
            "accepts-tools",
        ]

    def test_silence_about_streaming_is_a_rejection_with_a_reason(self, registry):
        reasons = dict(registry.capability_rejections("low", requires=("streaming",)))
        assert "capabilities.streaming" in reasons["refuses-tools"]
        assert "capabilities.streaming" in reasons["also-accepts-tools"]

    def test_a_group_with_no_streaming_member_raises_at_resolve_time(
        self, monkeypatch, mixed_registry
    ):
        """Fail CLOSED at resolution, before any request is built — not a 400
        the caller re-reads as an availability problem."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(mixed_registry))
                m.setattr(
                    _router, "_litellm_edge_admission",
                    lambda: (True, "https://router.example:8443", []),
                )
                with pytest.raises(_mr.CapabilityUnavailableError):
                    _router.resolve_group_structured(
                        "med", exec_context="lambda", wire="openai",
                        requires=("streaming",),
                    )
        finally:
            _router._router = None


class TestTheDeclarationReachesTheClient:
    """A resolve-time filter is only half of it: an UNQUALIFIED call to a group
    whose primary cannot stream must still hand the client a spec that says so,
    or the refusal happens on the wire instead of in the library."""

    def _resolve(self, monkeypatch, registry_path, group, **kw):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_path))
                m.setattr(_router, "_probe_egress_proxy", lambda *a, **k: True)
                m.setattr(
                    _router, "_litellm_edge_admission",
                    lambda: (True, "https://router.example:8443", []),
                )
                return _router.resolve_group_structured(
                    group, exec_context="lambda", wire="openai", **kw
                )
        finally:
            _router._router = None

    def test_the_group_route_declares_the_primarys_streaming_flag(
        self, monkeypatch, mixed_registry
    ):
        """The PRIMARY's fact, never an any() over the members: a group-level
        capability derived from its membership declares support the served
        model may not have (model-portability-policy §5)."""
        unqualified = self._resolve(monkeypatch, mixed_registry, "low")
        assert unqualified["primary_registry_id"] == "refuses-tools"
        assert unqualified["capabilities"]["streaming"] is False

        qualified = self._resolve(
            monkeypatch, mixed_registry, "low", requires=("streaming",)
        )
        assert qualified["primary_registry_id"] == "accepts-tools"
        assert qualified["capabilities"]["streaming"] is True

    @pytest.mark.parametrize("declared", [True, False])
    def test_the_spec_carries_it_through_to_the_client(self, monkeypatch, declared):
        route = {
            "schema_version": _router.RESOLVE_SCHEMA_VERSION,
            "model": "low-x",
            "provider": "litellm",
            "route": "litellm_proxy",
            "api_base_url": "https://router.example:8443",
            "deployment_id": "low-x",
            "auth_token_type": "litellm_master_key",
            "group": "low",
            "registry_id": "litellm:group:low",
            "primary_model": "x",
            "primary_registry_id": "x",
            "capabilities": {"streaming": declared},
            "params": {"max_tokens": 8192},
        }
        monkeypatch.setattr(
            _router, "resolve_group_structured", lambda *a, **k: route
        )
        spec, _ = _router.resolve_group_spec("low", exec_context="lambda")
        assert spec.supports_streaming is declared

    def test_a_route_silent_about_streaming_resolves_to_false(self, monkeypatch):
        """An undeclared capability is not a capability — absent-means-
        supported is how a request shape reaches a deployment that refuses
        it."""
        route = {
            "schema_version": _router.RESOLVE_SCHEMA_VERSION,
            "model": "low-x",
            "provider": "litellm",
            "route": "litellm_proxy",
            "api_base_url": "https://router.example:8443",
            "deployment_id": "low-x",
            "auth_token_type": "litellm_master_key",
            "group": "low",
            "registry_id": "litellm:group:low",
            "primary_model": "x",
            "primary_registry_id": "x",
            "capabilities": {},
            "params": {},
        }
        monkeypatch.setattr(
            _router, "resolve_group_structured", lambda *a, **k: route
        )
        spec, _ = _router.resolve_group_spec("low", exec_context="lambda")
        assert spec.supports_streaming is False
