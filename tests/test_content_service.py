"""Server-owned binding and native registry dispatch contracts."""

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from krepis.content_dispatch import (
    AlreadyDispatched,
    ContentResult,
    DispatchJournal,
    RequestConflict,
)
from krepis.model_registry import Registry
from krepis.workload_admission import ContentProfile
from krepis.content_service import ContentService, WorkloadBinding, ServiceRefused


@pytest.fixture
def harness(tmp_path):
    profile = ContentProfile(
        "consumer",
        "r1",
        "content",
        frozenset({"adapter"}),
        frozenset({"processor"}),
        frozenset({"private"}),
        frozenset({"interactive"}),
        100,
        100,
        10,
    )
    binding = WorkloadBinding(profile, "private", "interactive")
    doc = {
        "models": [
            {
                "id": "native",
                "execution_transport": "native",
                "model": "fixture-model",
                "native": {
                    "schema_version": 1,
                    "adapter_id": "adapter",
                    "processors": ["processor"],
                    "execution_class": "content",
                    "billing": "subscription",
                    "qualification_ref": "proof",
                    "deployment_ref": "worker",
                    "served_models": ["observed-model"],
                },
            }
        ],
        "model_groups": {"content": ["native"]},
    }
    state = {"binding": binding, "doc": doc, "authorized": True, "qualified": True}
    calls = []
    qualifications = []
    db = sqlite3.connect(tmp_path / "dispatch.sqlite", isolation_level=None)
    journal = DispatchJournal(db, max_inflight=1)

    def authenticate(credentials):
        if credentials != "valid" or not state["authorized"]:
            raise ServiceRefused("authentication refused")
        return "consumer"

    def qualify(route):
        qualifications.append(route)
        return state["qualified"]

    def execute(route, request, content):
        calls.append((route, request, content))
        return ContentResult("reply", "observed-model", "receipt")

    service = ContentService(
        journal,
        authenticate=authenticate,
        load_binding=lambda principal: state["binding"],
        load_registry=lambda: Registry(Path("fixture"), state["doc"]),
        qualified=qualify,
        execute=execute,
    )
    yield service, state, calls, qualifications, journal
    db.close()


def test_normalizes_from_server_and_records_registry_identity(harness):
    service, state, calls, qualifications, journal = harness
    assert service.submit("valid", request_id="request", content="hi").text == "reply"
    route, request, content = calls[0]
    assert request.data_class == "private" and request.trigger == "interactive"
    assert request.input_bytes == 2 and len(request.execution_id) == 32
    assert route.candidate.registry_id == "native"
    assert len(route.candidate.registry_revision) == 64
    assert route.model == "fixture-model"
    assert len(qualifications) == 2
    assert service.status("valid", "request") == "completed"
    assert service.result("valid", "request").text == "reply"
    with pytest.raises(AlreadyDispatched):
        service.submit("valid", request_id="request", content="hi")
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["", "spoofed", None, {"consumer_id": "consumer"}])
def test_unauthenticated_intake_never_dispatches(harness, bad):
    service, _, calls, _, journal = harness
    with pytest.raises(ServiceRefused):
        service.submit(bad, request_id="request", content="hi")
    assert calls == [] and journal.status("consumer", "request") is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("processors", ["other"]),
        ("billing", "api"),
        ("execution_class", "repository"),
        ("schema_version", True),
        ("schema_version", 2),
        ("processors", []),
        ("processors", "processor"),
        ("adapter_id", ""),
        ("qualification_ref", ""),
        ("deployment_ref", ""),
    ],
)
def test_invalid_or_forbidden_metadata_never_dispatches(harness, field, value):
    service, state, calls, _, journal = harness
    state["doc"]["models"][0]["native"][field] = value
    with pytest.raises((ValueError, ServiceRefused)):
        service.submit("valid", request_id="request", content="hi")
    assert calls == [] and journal.status("consumer", "request") is None


@pytest.mark.parametrize("value", [False, None, 1, "yes"])
def test_qualification_requires_literal_success(harness, value):
    service, state, calls, _, journal = harness
    state["qualified"] = value
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="hi")
    assert calls == [] and journal.status("consumer", "request") is None


def test_changed_model_conflicts_with_previous_request(harness):
    service, state, calls, _, journal = harness
    service.submit("valid", request_id="request", content="hi")
    state["doc"]["models"][0]["model"] = "replacement"
    with pytest.raises(RequestConflict):
        service.submit("valid", request_id="request", content="hi")
    assert len(calls) == 1


def test_api_rows_never_reach_qualification(harness):
    service, state, calls, qualifications, journal = harness
    state["doc"]["models"][0]["execution_transport"] = "api"
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="hi")
    assert qualifications == [] and calls == []


def test_bound_identity_and_classification_cannot_be_downgraded(harness):
    service, state, calls, _, journal = harness
    state["binding"] = replace(
        state["binding"], profile=replace(state["binding"].profile, consumer_id="other")
    )
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="hi")
    assert calls == []


def test_changed_policy_after_admission_refuses_before_effect(harness):
    service, state, calls, _, journal = harness

    def qualify(route):
        state["binding"] = replace(
            state["binding"], profile=replace(state["binding"].profile, revision="r2")
        )
        return True

    service.qualified = qualify
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="hi")
    assert calls == []
    assert journal.status("consumer", "request") is None


def test_failure_never_falls_back(harness):
    service, state, calls, _, journal = harness
    import copy

    alternate = copy.deepcopy(state["doc"]["models"][0])
    alternate["id"] = "alternate"
    state["doc"]["models"].append(alternate)
    state["doc"]["model_groups"]["content"].append("alternate")

    def execute(route, request, content):
        calls.append(route)
        raise RuntimeError("execution failed")

    service.execute = execute
    with pytest.raises(RuntimeError):
        service.submit("valid", request_id="request", content="hi")
    assert len(calls) == 1 and journal.status("consumer", "request") == "uncertain"
    with pytest.raises(AlreadyDispatched):
        service.submit("valid", request_id="request", content="hi")


def test_revoked_principal_cannot_read_results(harness):
    service, state, calls, _, journal = harness
    service.submit("valid", request_id="request", content="hi")
    state["authorized"] = False
    with pytest.raises(ServiceRefused):
        service.result("valid", "request")
    with pytest.raises(ServiceRefused):
        service.status("valid", "request")


def test_refusal_and_completion_are_durable_without_content(harness):
    service, state, calls, _, journal = harness
    with pytest.raises(ServiceRefused):
        service.submit(
            "invalid-secret", request_id="request", content="private-message"
        )
    service.submit("valid", request_id="request", content="private-message")
    rows = journal.db.execute(
        "SELECT outcome,stage FROM content_service_events ORDER BY event_id"
    ).fetchall()
    assert rows == [("refused", "authentication"), ("completed", "completed")]
    dump = "\n".join(journal.db.iterdump())
    assert "invalid-secret" not in dump and "private-message" not in dump


def test_registry_change_during_qualification_is_not_sent(harness):
    service, state, calls, _, journal = harness

    def qualify(route):
        state["doc"]["models"][0]["model"] = "changed"
        return True

    service.qualified = qualify
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="hi")
    assert calls == []


def test_qualification_revocation_before_effect_is_not_sent(harness):
    service, state, calls, _, journal = harness
    answers = iter([True, False])
    service.qualified = lambda route: next(answers)
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="hi")
    assert calls == []


def test_private_size_is_measured_after_utf8_encoding(harness):
    service, state, calls, _, journal = harness
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="é" * 51)
    assert calls == []


def test_legacy_dispatch_binding_hash_is_unchanged(harness):
    import hashlib
    import json
    from dataclasses import asdict
    from krepis.content_dispatch import _binding, _sets
    from krepis.workload_admission import ContentCandidate, ContentRequest

    _, state, _, _, _ = harness
    profile = state["binding"].profile
    request = ContentRequest("r", "r1", "content", "private", "interactive", 2, 100, 10)
    candidate = ContentCandidate(
        "adapter",
        frozenset({"processor"}),
        "content",
        "content",
        "subscription",
        "proof",
    )
    legacy_candidate = asdict(candidate)
    del legacy_candidate["registry_id"]
    del legacy_candidate["registry_revision"]
    raw = json.dumps(
        [asdict(profile), asdict(request), legacy_candidate, "hi"],
        default=_sets,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert (
        _binding(profile, request, candidate, "hi")
        == hashlib.sha256(raw.encode()).hexdigest()
    )


def test_packaged_metadata_schema_matches_valid_and_invalid_inputs(harness):
    import json
    from importlib.resources import files
    import jsonschema

    _, state, _, _, _ = harness
    schema = json.loads(
        files("krepis").joinpath("native_entry_schema.json").read_text()
    )
    metadata = state["doc"]["models"][0]["native"]
    jsonschema.validate(metadata, schema)
    for key, value in [
        ("schema_version", True),
        ("processors", []),
        ("billing", ""),
        ("unknown", "field"),
    ]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(dict(metadata, **{key: value}), schema)


def test_pre_effect_refusal_does_not_consume_global_capacity(harness):
    service, state, calls, _, journal = harness
    answers = iter([True, False])
    service.qualified = lambda route: next(answers)
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="refused", content="hi")
    assert journal.status("consumer", "refused") is None
    service.qualified = lambda route: True
    service.submit("valid", request_id="independent", content="hi")
    assert len(calls) == 1


def test_unqualified_served_model_is_uncertain_and_never_returned(harness):
    service, state, calls, _, journal = harness
    service.execute = lambda route, request, content: ContentResult(
        "reply", "unqualified", "receipt"
    )
    with pytest.raises(ServiceRefused):
        service.submit("valid", request_id="request", content="hi")
    assert journal.status("consumer", "request") == "uncertain"
    assert service.result("valid", "request") is None


def test_duplicate_uncertain_request_does_not_claim_new_uncertainty(harness):
    service, state, calls, _, journal = harness

    def fail(route, request, content):
        raise RuntimeError("failed")

    service.execute = fail
    with pytest.raises(RuntimeError):
        service.submit("valid", request_id="request", content="hi")
    with pytest.raises(AlreadyDispatched):
        service.submit("valid", request_id="request", content="hi")
    rows = journal.db.execute(
        "SELECT outcome FROM content_service_events ORDER BY event_id"
    ).fetchall()
    assert rows == [("uncertain",), ("refused",)]
