import importlib
import json
from pathlib import Path

import pytest
from krepis.workload_admission import ContentCandidate, ContentRequest


def api():
    assert importlib.util.find_spec("krepis.codex_adapter") is not None
    return importlib.import_module("krepis.codex_adapter")


def setup(tmp_path, program=None, evidence=True, cancelled=lambda: False):
    m = api()
    for name in ("home", "credentials", "work"):
        (tmp_path / name).mkdir(mode=0o700)
    binary = tmp_path / "fixture"
    binary.write_text(program or "#!/bin/sh\ncat >/dev/null\nprintf '%s\n' '" + json.dumps({"type":"thread.started","thread_id":"thread-fixture"}) + "' '" + json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"answer"}}) + "' '" + json.dumps({"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}) + "'\n")
    binary.chmod(0o700)
    spec = m.CodexSpec(binary=binary, home=tmp_path/"home", credential_home=tmp_path/"credentials", workdir=tmp_path/"work", model="requested-model", egress_url="http://127.0.0.1:8990/backend-api/codex", upstream_host="chatgpt.com", adapter_id="native", qualification_ref="qualified-1")
    def lookup(execution_id, thread_id, remaining):
        assert (execution_id, thread_id) == ("e" * 32, "thread-fixture")
        assert 0 < remaining <= 2
        return m.ServedEvidence(execution_id, thread_id, "native", "qualified-1", "observed-model", "receipt-1") if evidence else None
    adapter = m.CodexAdapter(spec, served_evidence=lookup, cancelled=cancelled)
    request = ContentRequest("req-1", "r1", "conversation", "private", "human", 5, 100, 2, execution_id="e" * 32)
    candidate = ContentCandidate("native", frozenset({"processor"}), "conversation", "content", "subscription", "qualified-1")
    return m, adapter, request, candidate


def test_native_response_requires_observed_model_evidence(tmp_path):
    _, adapter, request, candidate = setup(tmp_path)
    result = adapter(request, candidate, "hello")
    assert result.text == "answer"
    assert result.served_model == "observed-model"
    assert result.trace_id == "receipt-1"


def test_no_served_identity_is_not_reported_as_success(tmp_path):
    _, adapter, request, candidate = setup(tmp_path, evidence=False)
    with pytest.raises(ValueError, match="evidence"):
        adapter(request, candidate, "hello")


def test_launch_forces_subscription_and_disables_retries_and_material_tools(tmp_path):
    _, adapter, request, _ = setup(tmp_path)
    args = adapter.command(request)
    assert args[:2] == [str(adapter.spec.binary), "exec"]
    assert args[-1] == "-"
    assert "--ignore-user-config" in args and "--strict-config" in args
    assert 'forced_login_method="chatgpt"' in args
    assert "model_providers.controlled.request_max_retries=0" in args
    assert "model_providers.controlled.stream_max_retries=0" in args
    for feature in ("shell_tool", "apps", "plugins", "multi_agent", "view_image", "browser_use", "computer_use"):
        assert "features." + feature + "=false" in args
    assert not any("hello" in a for a in args)


def test_environment_does_not_inherit_operator_credentials(tmp_path, monkeypatch):
    _, adapter, request, candidate = setup(tmp_path, "#!/bin/sh\n[ -z \"$OPENAI_API_KEY\" ] && [ -z \"$AWS_SECRET_ACCESS_KEY\" ] || exit 99\nexit 3\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sensitive-test-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sensitive-test-value")
    with pytest.raises(RuntimeError, match="exit 3"):
        adapter(request, candidate, "hello")


@pytest.mark.parametrize("program", [
    "#!/bin/sh\nprintf 'not json\n'\n",
    '#!/bin/sh\nprintf \'%s\\n\' \'{"type":"turn.completed"}\'\n',
    '#!/bin/sh\nprintf \'%s\\n\' \'{"type":"item.completed","item":{"type":"command_execution"}}\'\n',
])
def test_malformed_or_effectful_output_is_refused(tmp_path, program):
    _, adapter, request, candidate = setup(tmp_path, program)
    with pytest.raises(ValueError):
        adapter(request, candidate, "hello")


def test_timeout_terminates_the_worker(tmp_path):
    from dataclasses import replace
    _, adapter, request, candidate = setup(tmp_path, "#!/bin/sh\nexec sleep 30\n")
    with pytest.raises(TimeoutError):
        adapter(replace(request, timeout_seconds=1), candidate, "hello")


def test_cancellation_before_start_creates_no_worker(tmp_path):
    marker = tmp_path / "effect"
    _, adapter, request, candidate = setup(tmp_path, f"#!/bin/sh\ntouch {marker}\n", cancelled=lambda: True)
    with pytest.raises(InterruptedError):
        adapter(request, candidate, "hello")
    assert not marker.exists()


def test_output_limit_stops_unbounded_native_output(tmp_path):
    _, adapter, request, candidate = setup(tmp_path, "#!/bin/sh\nexec yes oversized\n")
    with pytest.raises(ValueError, match="output"):
        adapter(request, candidate, "hello")


def test_cancellation_still_checked_after_worker_closes_output_pipes(tmp_path):
    marker = tmp_path / "started"
    program = f"#!/bin/sh\nexec >/dev/null 2>&1\ntouch {marker}\nexec sleep 30\n"
    _, adapter, request, candidate = setup(tmp_path, program, cancelled=marker.exists)
    with pytest.raises(InterruptedError):
        adapter(request, candidate, "hello")


@pytest.mark.parametrize("field,value", [("execution_id", "f" * 32), ("thread_id", "other"), ("qualification_ref", "old")])
def test_stale_or_cross_attempt_receipt_is_refused(tmp_path, field, value):
    from dataclasses import replace
    m, adapter, request, candidate = setup(tmp_path)
    receipt = m.ServedEvidence(request.execution_id, "thread-fixture", "native", "qualified-1", "observed", "receipt")
    adapter.served_evidence = lambda *args: replace(receipt, **{field: value})
    with pytest.raises(ValueError, match="evidence"):
        adapter(request, candidate, "hello")


def test_receipt_after_overall_deadline_is_not_success(tmp_path, monkeypatch):
    m, adapter, request, candidate = setup(tmp_path)
    def late(*args):
        monkeypatch.setattr(m.time, "monotonic", lambda: float("inf"))
        return m.ServedEvidence(request.execution_id, "thread-fixture", "native", "qualified-1", "observed", "receipt")
    adapter.served_evidence = late
    with pytest.raises(TimeoutError, match="evidence"):
        adapter(request, candidate, "hello")
