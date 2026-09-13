"""``resolve_schema.json`` is generated, and this is what says so.

model-router-policy R17 wants the contract schema'd and stored next to the
producer. R6/R6a's argument — two hand-synchronised descriptions of one fact
drift, invisibly — applies to the schema exactly as it applies to the
derivation. Until this module existed the schema was hand-written, and it had
already drifted: the committed ``exec_context`` enum omitted ``ci``, a context
:data:`krepis.router.EXEC_CONTEXTS` has declared since
alpha-engine-config-I7853, so a resolution from a GitHub Actions runner did not
validate against the contract it satisfies.

The byte-identity test below is the whole control: the committed file must be
exactly what :func:`krepis.resolve_contract.render_resolve_schema` produces, so
a contract change that skips the regeneration is a red check rather than a
schema that silently describes the previous release.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from krepis import resolve_contract as _rc
from krepis import router as _router

# The live-shaped registry fixture lives with the router tests; re-exported
# rather than duplicated, so a registry-shape change lands in one place.
from tests.test_router import registry_file  # noqa: F401

SCHEMA_PATH = Path(_router.__file__).parent / "resolve_schema.json"


class TestSchemaIsGenerated:
    def test_committed_schema_is_byte_identical_to_a_fresh_render(self):
        """The committed bytes ARE the rendered bytes.

        Not "semantically equivalent": byte-identity is what makes the check
        cheap enough to be a required test and impossible to satisfy by
        editing the JSON to agree.
        """
        assert SCHEMA_PATH.read_text() == _rc.render_resolve_schema(), (
            "resolve_schema.json is stale — regenerate it with "
            "`python3 scripts/gen_resolve_schema.py` and commit the result"
        )

    def test_generator_check_mode_agrees_with_this_test(self):
        """The script and the test must render through the same function.

        A generator with its own formatting would let the file be green here
        and stale there.
        """
        import importlib.util

        script = Path(__file__).resolve().parent.parent / "scripts" / "gen_resolve_schema.py"
        spec = importlib.util.spec_from_file_location("_gen_resolve_schema", script)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.SCHEMA_PATH.resolve() == SCHEMA_PATH.resolve()
        assert module.render_resolve_schema is _rc.render_resolve_schema

    def test_schema_declares_its_id_and_dialect(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == _rc.SCHEMA_ID
        assert str(_router.RESOLVE_SCHEMA_VERSION) in _rc.SCHEMA_ID


class TestVocabularyComesFromTheRouter:
    """The drift this module exists to stop, pinned directly."""

    def test_exec_context_enum_is_every_declared_context(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        enum = _enum_of(schema["properties"]["exec_context"])
        assert enum == list(_router.EXEC_CONTEXTS)

    def test_ci_is_in_the_exec_context_enum(self):
        """The instance of the drift. ``ci`` was declared by the router and
        absent from the hand-written schema (alpha-engine-config-I7853)."""
        schema = json.loads(SCHEMA_PATH.read_text())
        assert "ci" in _enum_of(schema["properties"]["exec_context"])

    def test_wire_enum_is_every_declared_wire_format(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        assert _enum_of(schema["properties"]["wire"]) == list(_router.WIRE_FORMATS)

    def test_schema_version_floor_is_the_router_constant(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        assert (
            schema["properties"]["schema_version"]["minimum"]
            == _router.RESOLVE_SCHEMA_VERSION
        )


class TestProducerSatisfiesTheModel:
    """The producer builds dicts; the model describes them. Bind the two."""

    @pytest.mark.parametrize("group", ["low", "med", "high", "ultra"])
    def test_group_resolution_validates_against_the_model(
        self, group, registry_file, monkeypatch
    ):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured(group)
        finally:
            _router._router = None
        _rc.ResolveContract.model_validate(info)

    def test_a_resolution_missing_a_required_field_is_rejected(
        self, registry_file, monkeypatch
    ):
        """Negative control: the model must actually refuse something.

        A validator never observed rejecting is unproven.
        """
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured("med")
        finally:
            _router._router = None
        del info["api_base_url"]
        with pytest.raises(Exception):
            _rc.ResolveContract.model_validate(info)

    def test_an_undeclared_exec_context_is_rejected(
        self, registry_file, monkeypatch
    ):
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured("med")
        finally:
            _router._router = None
        info["exec_context"] = "lambda_vpc"
        with pytest.raises(Exception):
            _rc.ResolveContract.model_validate(info)

    def test_compat_aliases_do_not_fail_validation(self, registry_file, monkeypatch):
        """Additive-then-remove (R19) is unimplementable against a closed
        schema: the deprecated alias must validate for the release it is
        emitted in."""
        _router._router = None
        try:
            with monkeypatch.context() as m:
                m.delenv("LITELLM_MASTER_KEY", raising=False)
                m.setenv("LLM_MODEL_REGISTRY_PATH", str(registry_file))
                info = _router.resolve_group_structured("med")
        finally:
            _router._router = None
        assert "anthropic_base_url" in info
        _rc.ResolveContract.model_validate(info)


def _enum_of(prop: dict) -> list:
    """Enum values of a property, whether or not it is wrapped in ``anyOf``."""
    if "enum" in prop:
        return list(prop["enum"])
    for member in prop.get("anyOf", ()):
        if "enum" in member:
            return list(member["enum"])
    raise AssertionError(f"no enum in {prop!r}")
