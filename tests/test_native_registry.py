"""Native entries never become API deployments or implicit fallbacks."""

from pathlib import Path

import pytest
import yaml

from krepis import model_registry as mr
from krepis import router


@pytest.fixture
def mixed_registry(tmp_path):
    path = tmp_path / "registry.yaml"
    doc = {
        "models": [
            {
                "id": "native-worker",
                "execution_transport": "native",
                "model": "native-fixture",
                "status": "active",
                "capabilities": {"tools": True},
            },
            {
                "id": "api-worker",
                "model": "api-fixture",
                "route": "direct",
                "provider": "fixture",
                "status": "active",
                "capabilities": {"tools": True},
            },
            {
                "id": "native-off",
                "execution_transport": "native",
                "status": "unavailable",
            },
        ],
        "model_groups": {
            "mixed": ["native-worker", "api-worker"],
            "native-only": ["native-worker", "native-off"],
        },
    }
    path.write_text(yaml.safe_dump(doc))
    return mr.load_registry(path)


def test_existing_api_view_excludes_native_and_native_view_is_explicit(mixed_registry):
    assert list(mixed_registry.live_models) == ["api-worker"]
    assert list(mixed_registry.live_models_for("native")) == ["native-worker"]
    assert mixed_registry.live_group_ids("mixed") == ["api-worker"]
    assert mixed_registry.live_group_ids("mixed", transport="native") == [
        "native-worker"
    ]
    assert list(mixed_registry.iter_live_groups()) == [("mixed", ["api-worker"])]
    assert "transport" in mixed_registry.capability_rejections("mixed")[0][1]


@pytest.mark.parametrize("value", [None, "", "Native", "unknown", 1, True, []])
def test_invalid_transport_cannot_silently_default_to_api(value):
    with pytest.raises(ValueError, match="transport"):
        mr.Registry(
            Path("fixture.yaml"),
            {"models": [{"id": "bad", "execution_transport": value}]},
        )


@pytest.mark.parametrize(
    "render", [mr.litellm_model, mr.api_key_env, mr.api_key_for, mr.deployment_params]
)
def test_api_renderers_refuse_native_metadata(render, mixed_registry):
    with pytest.raises(ValueError, match="native"):
        render(mixed_registry.models["native-worker"])


def test_in_process_generation_cannot_emit_native_alias_or_fallback(mixed_registry):
    deployments, fallbacks, aliases = router._parse_registry(mixed_registry.path)
    assert deployments
    assert aliases["mixed"] == "mixed-api-worker"
    assert "native-only" not in aliases
    assert "native-worker" not in str((deployments, fallbacks, aliases))


def test_direct_api_resolution_refuses_native_before_edge_probe(
    mixed_registry, monkeypatch
):
    monkeypatch.setattr(router, "_find_registry", lambda: mixed_registry.path)

    def forbidden():
        pytest.fail("native model reached API edge admission")

    monkeypatch.setattr(router, "_litellm_edge_admission", forbidden)
    with pytest.raises(ValueError, match="native"):
        router._resolve_model_json("native-worker", exec_context="laptop")


def test_degraded_api_group_skips_native_with_durable_reason(
    mixed_registry, monkeypatch
):
    for entry in mixed_registry.doc["models"]:
        entry["reachable_from"] = ["laptop"]
        entry["endpoints"] = {"openai": "http://fixture.invalid/v1"}
    mixed_registry.path.write_text(yaml.safe_dump(mixed_registry.doc))
    monkeypatch.setattr(router, "_find_registry", lambda: mixed_registry.path)
    monkeypatch.setattr(
        router, "_litellm_edge_admission", lambda: (False, "", ["test unavailable"])
    )
    monkeypatch.setattr(
        router, "_entry_endpoint", lambda entry, wire: "http://fixture.invalid/v1"
    )
    result = router._resolve_group_json("mixed", exec_context="laptop")
    assert result["registry_id"] == "api-worker"
    assert any(
        s["registry_id"] == "native-worker" and "transport" in s["reason"]
        for s in result["skipped_entries"]
    )


def test_deployment_metadata_never_resolves_native(mixed_registry, monkeypatch):
    monkeypatch.setattr(router, "_find_registry", lambda: mixed_registry.path)
    assert router.served_model_for_deployment("mixed-native-worker") is None
    assert router.served_model_for_deployment("native-worker") is None
    assert router.served_model_for_deployment("mixed-api-worker") == "api-fixture"


@pytest.mark.parametrize("selector", ["", "unknown", None, True, []])
def test_invalid_transport_selector_fails_for_empty_registry(selector):
    registry = mr.Registry(Path("empty.yaml"), {})
    with pytest.raises(ValueError, match="transport"):
        registry.live_models_for(selector)
    with pytest.raises(ValueError, match="transport"):
        registry.live_group_ids("absent", transport=selector)
    with pytest.raises(ValueError, match="transport"):
        list(registry.iter_live_groups(transport=selector))
