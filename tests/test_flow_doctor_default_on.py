"""0.58.0: flow-doctor is default-on when a flow_doctor_yaml is provided.

Covers the activation precedence (kill switch / explicit enable / pytest guard /
default-on), the deployed→strict / dev→graceful posture, and the
guard_entrypoint / monitor_handler crash-capture helpers.
"""

from __future__ import annotations

import logging
from unittest import mock

import pytest

import krepis.logging as m
from krepis.logging import (
    _flow_doctor_should_activate,
    _is_deployed,
    get_flow_doctor,
    guard_entrypoint,
    monitor_handler,
    setup_logging,
)

# pytest sets PYTEST_CURRENT_TEST for every test, so the default-on path is
# auto-suppressed unless a test opts in via FLOW_DOCTOR_ALLOW_IN_TESTS=1.
_FD_ENV = (
    "FLOW_DOCTOR_ENABLED",
    "FLOW_DOCTOR_DISABLED",
    "FLOW_DOCTOR_ALLOW_IN_TESTS",
    "ALPHA_ENGINE_DEPLOYED",
    "AWS_LAMBDA_FUNCTION_NAME",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in _FD_ENV:
        monkeypatch.delenv(v, raising=False)
    yield
    logging.getLogger().handlers.clear()
    m._fd_instance = None


def _fake_fd_module():
    fake = mock.Mock()
    fake.FlowDoctor.from_config = mock.Mock(return_value=mock.Mock())
    fake.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())
    return fake


def _yaml(tmp_path):
    p = tmp_path / "flow-doctor.yaml"
    p.write_text("flow_name: test\n")
    return str(p)


# --- activation precedence ---------------------------------------------


def test_default_on_activates_with_yaml_when_allowed_in_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ALLOW_IN_TESTS", "1")  # bypass the pytest guard
    fake = _fake_fd_module()
    with mock.patch.dict("sys.modules", {"flow_doctor": fake}):
        setup_logging("test", flow_doctor_yaml=_yaml(tmp_path))
    assert get_flow_doctor() is not None
    # Not deployed + no explicit enable -> strict=False (dev-lenient).
    _, kwargs = fake.FlowDoctor.from_config.call_args
    assert kwargs["strict"] is False


def test_default_on_suppressed_under_pytest_without_optin(tmp_path):
    # PYTEST_CURRENT_TEST is set by the runner; no FLOW_DOCTOR_ALLOW_IN_TESTS.
    fake = _fake_fd_module()
    with mock.patch.dict("sys.modules", {"flow_doctor": fake}):
        setup_logging("test", flow_doctor_yaml=_yaml(tmp_path))
    assert get_flow_doctor() is None


def test_no_yaml_means_no_activation():
    setup_logging("test")
    assert get_flow_doctor() is None


def test_kill_switch_overrides_everything(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ALLOW_IN_TESTS", "1")
    monkeypatch.setenv("FLOW_DOCTOR_DISABLED", "1")
    fake = _fake_fd_module()
    with mock.patch.dict("sys.modules", {"flow_doctor": fake}):
        setup_logging("test", flow_doctor_yaml=_yaml(tmp_path))
    assert get_flow_doctor() is None


def test_should_activate_precedence():
    assert _flow_doctor_should_activate("x.yaml") is False  # pytest guard, no opt-in
    with mock.patch.dict("os.environ", {"FLOW_DOCTOR_ALLOW_IN_TESTS": "1"}):
        assert _flow_doctor_should_activate("x.yaml") is True
        assert _flow_doctor_should_activate(None) is False  # no yaml
    with mock.patch.dict("os.environ", {"FLOW_DOCTOR_ENABLED": "1"}):
        assert _flow_doctor_should_activate("x.yaml") is True  # explicit wins pytest
    with mock.patch.dict("os.environ", {"FLOW_DOCTOR_DISABLED": "1"}):
        assert _flow_doctor_should_activate("x.yaml") is False


def test_collection_time_import_suppressed_without_pytest_env(monkeypatch):
    """Entrypoints call setup_logging at module top, which under pytest runs at
    COLLECTION time — before the runner sets PYTEST_CURRENT_TEST. The
    ``"pytest" in sys.modules`` check must close that gap (2026-06-11: an
    alpha-engine-data test run leaked real alert emails + GitHub issues for
    synthetic fixture tickers through exactly this import-time window)."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    # pytest is necessarily importable-and-imported while this suite runs,
    # so sys.modules alone must suppress the default-on path.
    assert _flow_doctor_should_activate("x.yaml") is False
    monkeypatch.setenv("FLOW_DOCTOR_ALLOW_IN_TESTS", "1")
    assert _flow_doctor_should_activate("x.yaml") is True


# --- deployed / strict posture -----------------------------------------


def test_is_deployed_detection(monkeypatch):
    assert _is_deployed() is False
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "my-fn")
    assert _is_deployed() is True
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME")
    monkeypatch.setenv("ALPHA_ENGINE_DEPLOYED", "1")
    assert _is_deployed() is True


def test_dev_default_on_missing_package_is_graceful(monkeypatch, tmp_path):
    """Default-on in dev must NEVER crash a developer over a missing extra."""
    monkeypatch.setenv("FLOW_DOCTOR_ALLOW_IN_TESTS", "1")  # default-on path, not deployed
    with mock.patch.dict("sys.modules", {"flow_doctor": None}):
        setup_logging("test", flow_doctor_yaml=_yaml(tmp_path))  # no raise
    assert get_flow_doctor() is None


def test_deployed_missing_package_fails_loud(monkeypatch, tmp_path):
    # ALLOW_IN_TESTS bypasses the pytest guard (so it activates); DEPLOYED makes
    # it strict. In a real Lambda/EC2 there is no PYTEST_CURRENT_TEST, so the
    # opt-in isn't needed — default-on + deployed both hold there.
    monkeypatch.setenv("FLOW_DOCTOR_ALLOW_IN_TESTS", "1")
    monkeypatch.setenv("ALPHA_ENGINE_DEPLOYED", "1")  # deployed -> strict
    with mock.patch.dict("sys.modules", {"flow_doctor": None}):
        with pytest.raises(RuntimeError, match="flow-doctor is not installed"):
            setup_logging("test", flow_doctor_yaml=_yaml(tmp_path))


def test_deployed_default_on_passes_strict_true(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ALLOW_IN_TESTS", "1")
    monkeypatch.setenv("ALPHA_ENGINE_DEPLOYED", "1")
    fake = _fake_fd_module()
    with mock.patch.dict("sys.modules", {"flow_doctor": fake}):
        setup_logging("test", flow_doctor_yaml=_yaml(tmp_path))
    _, kwargs = fake.FlowDoctor.from_config.call_args
    assert kwargs["strict"] is True


# --- guard_entrypoint / monitor_handler --------------------------------


def test_guard_entrypoint_noop_when_inactive():
    m._fd_instance = None
    with guard_entrypoint():
        pass  # no fd -> plain passthrough, no crash


def test_guard_entrypoint_reports_and_reraises():
    fd = mock.Mock()
    from contextlib import contextmanager

    @contextmanager
    def _g():
        try:
            yield
        except Exception as exc:
            fd.report(exc)
            raise

    fd.guard = _g
    m._fd_instance = fd
    with pytest.raises(ValueError):
        with guard_entrypoint():
            raise ValueError("boom")
    fd.report.assert_called_once()


def test_monitor_handler_noop_when_inactive():
    m._fd_instance = None

    @monitor_handler
    def handler(event, context):
        return "ok"

    assert handler({}, None) == "ok"


def test_monitor_handler_reports_and_reraises():
    fd = mock.Mock()
    m._fd_instance = fd

    @monitor_handler
    def handler(event, context):
        raise RuntimeError("lambda boom")

    with pytest.raises(RuntimeError):
        handler({}, None)
    fd.report.assert_called_once()


def test_monitor_handler_resolves_fd_at_call_time():
    """Decorated at import (fd None), activated later -> still captures."""
    m._fd_instance = None

    @monitor_handler
    def handler(event, context):
        raise RuntimeError("late boom")

    fd = mock.Mock()
    m._fd_instance = fd  # activated after decoration
    with pytest.raises(RuntimeError):
        handler({}, None)
    fd.report.assert_called_once()
