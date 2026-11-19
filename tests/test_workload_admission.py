"""Admission rejects prohibited effects before an execution adapter is called."""
from dataclasses import replace
import importlib

import pytest


def api():
    # Assertion identifies the missing additive module before its first implementation.
    assert importlib.util.find_spec("krepis.workload_admission") is not None
    return importlib.import_module("krepis.workload_admission")


def fixtures():
    m = api()
    profile = m.ContentProfile(
        consumer_id="consumer-a", revision="rev-1", group="conversation",
        adapter_ids=frozenset({"native-a"}), processors=frozenset({"processor-a"}),
        data_classes=frozenset({"private"}), triggers=frozenset({"human"}),
        max_input_bytes=1024, max_output_bytes=512, timeout_seconds=30,
    )
    request = m.ContentRequest(
        request_id="req-1", profile_revision="rev-1", group="conversation",
        data_class="private", trigger="human", input_bytes=100,
        output_bytes=256, timeout_seconds=20,
    )
    candidate = m.ContentCandidate(
        adapter_id="native-a", processors=frozenset({"processor-a"}),
        group="conversation", execution_class="content", billing="subscription",
        qualification_ref="qualified-1",
    )
    return m, profile, request, candidate


def test_approved_native_content_is_admitted():
    m, p, r, c = fixtures()
    assert m.admit_content(p, r, c, authenticated_consumer="consumer-a") is None


@pytest.mark.parametrize("field,value", [
    ("adapter_id", "other"),
    ("processors", frozenset({"processor-a", "intermediary"})),
    ("group", "other"), ("execution_class", "tools"), ("billing", "api"),
])
def test_each_fallback_requires_full_eligibility(field, value):
    m, p, r, c = fixtures()
    with pytest.raises(m.AdmissionRefused):
        m.admit_content(p, r, replace(c, **{field: value}), authenticated_consumer="consumer-a")


@pytest.mark.parametrize("field,value", [
    ("profile_revision", "old"), ("group", "other"),
    ("data_class", "public"), ("trigger", "scheduled"),
    ("input_bytes", 1025), ("output_bytes", 513), ("timeout_seconds", 31),
])
def test_request_cannot_widen_profile(field, value):
    m, p, r, c = fixtures()
    with pytest.raises(m.AdmissionRefused):
        m.admit_content(p, replace(r, **{field: value}), c, authenticated_consumer="consumer-a")


def test_authenticated_identity_must_match_server_profile():
    m, p, r, c = fixtures()
    with pytest.raises(m.AdmissionRefused):
        m.admit_content(p, r, c, authenticated_consumer="consumer-b")


def test_model_switch_changes_profile_and_candidate_only():
    m, p, r, c = fixtures()
    switched = replace(p, revision="rev-2", adapter_ids=frozenset({"native-b"}), processors=frozenset({"processor-b"}))
    candidate = replace(c, adapter_id="native-b", processors=frozenset({"processor-b"}))
    with pytest.raises(m.AdmissionRefused):
        m.admit_content(switched, r, candidate, authenticated_consumer="consumer-a")
    current_request = replace(r, profile_revision="rev-2")
    m.admit_content(switched, current_request, candidate, authenticated_consumer="consumer-a")
    with pytest.raises(m.AdmissionRefused):
        m.admit_content(switched, current_request, c, authenticated_consumer="consumer-a")


@pytest.mark.parametrize("field,value", [
    ("adapter_ids", frozenset()), ("processors", frozenset()),
    ("processors", {"mutable"}), ("processors", frozenset({""})),
    ("max_input_bytes", True), ("max_output_bytes", 0),
    ("timeout_seconds", float("inf")), ("revision", " "),
])
def test_invalid_profile_fails_at_construction(field, value):
    _, p, _, _ = fixtures()
    with pytest.raises(ValueError):
        replace(p, **{field: value})


@pytest.mark.parametrize("field,value", [("input_bytes", -1), ("input_bytes", True), ("output_bytes", 0), ("request_id", "")])
def test_invalid_request_fails_at_construction(field, value):
    _, _, r, _ = fixtures()
    with pytest.raises(ValueError):
        replace(r, **{field: value})


def test_unknown_processor_chain_or_qualification_is_not_eligible():
    _, _, _, c = fixtures()
    for changes in ({"processors": frozenset()}, {"qualification_ref": ""}):
        with pytest.raises(ValueError):
            replace(c, **changes)
