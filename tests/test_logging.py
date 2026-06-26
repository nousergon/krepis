"""Unit tests for shared logging setup."""

from __future__ import annotations

import json
import logging
import os
from unittest import mock

import pytest

from krepis.logging import (
    JSONFormatter,
    SecretsRedactingFilter,
    _seed_flow_doctor_secrets,
    get_flow_doctor,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Every test starts with a clean root logger."""
    yield
    logging.getLogger().handlers.clear()
    # Reset the module-level singleton so tests don't bleed state.
    import krepis.logging as m
    m._fd_instance = None


# ── JSONFormatter ────────────────────────────────────────────────────────


def test_json_formatter_basic():
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hello %s", args=("world",), exc_info=None,
    )
    out = json.loads(fmt.format(record))
    assert out["level"] == "INFO"
    assert out["msg"] == "hello world"
    assert "ts" in out


def test_json_formatter_with_exception():
    fmt = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=1, msg="failed", args=(), exc_info=sys.exc_info(),
        )
    out = json.loads(fmt.format(record))
    assert "exc" in out
    assert "ValueError" in out["exc"]


def test_json_formatter_with_ctx():
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hi", args=(), exc_info=None,
    )
    record.ctx = {"ticker": "SPY", "qty": 10}
    out = json.loads(fmt.format(record))
    assert out["ctx"] == {"ticker": "SPY", "qty": 10}


# ── setup_logging ────────────────────────────────────────────────────────


def test_setup_logging_text_mode_default(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_JSON_LOGS", raising=False)
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    setup_logging("test")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert not isinstance(root.handlers[0].formatter, JSONFormatter)


def test_setup_logging_json_mode(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_JSON_LOGS", "1")
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    setup_logging("test")
    root = logging.getLogger()
    assert isinstance(root.handlers[0].formatter, JSONFormatter)


def test_setup_logging_clears_existing_handlers(monkeypatch):
    """setup_logging replaces whatever handlers were previously attached
    (pytest's LogCaptureHandler, other loggers, etc.) with exactly one
    fresh StreamHandler."""
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    root = logging.getLogger()
    root.addHandler(logging.StreamHandler())
    root.addHandler(logging.StreamHandler())
    setup_logging("test")
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


# ── Flow Doctor attach ───────────────────────────────────────────────────


def test_flow_doctor_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    setup_logging("test")
    assert get_flow_doctor() is None


def test_flow_doctor_enabled_without_yaml_raises(monkeypatch):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    with pytest.raises(RuntimeError, match="not given a flow_doctor_yaml path"):
        setup_logging("test")


def test_flow_doctor_enabled_missing_yaml_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    missing = tmp_path / "nope.yaml"
    # Even if flow_doctor is installed, the missing yaml must raise before
    # we try to init. Test that path.
    fake_module = mock.Mock()
    fake_module.init = mock.Mock(return_value=mock.Mock())
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())
    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        with pytest.raises(RuntimeError, match="flow-doctor config not found"):
            setup_logging("test", flow_doctor_yaml=str(missing))


def test_flow_doctor_enabled_missing_package_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")
    # Simulate flow_doctor not installed.
    with mock.patch.dict("sys.modules", {"flow_doctor": None}):
        with pytest.raises(RuntimeError, match="flow-doctor is not installed"):
            setup_logging("test", flow_doctor_yaml=str(yaml_path))


def test_flow_doctor_enabled_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")

    fake_instance = mock.Mock()
    fake_module = mock.Mock()
    # flow-doctor 0.6.0+: lib prefers FlowDoctor.from_config() over the
    # removed init() free function (see logging._attach_flow_doctor).
    fake_module.FlowDoctor.from_config = mock.Mock(return_value=fake_instance)
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())

    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        setup_logging("test", flow_doctor_yaml=str(yaml_path))

    assert get_flow_doctor() is fake_instance
    # 0.58.0: from_config receives strict= (True here — FLOW_DOCTOR_ENABLED=1 is
    # an explicit opt-in, so a misconfig should fail loud).
    fake_module.FlowDoctor.from_config.assert_called_once_with(
        config_path=str(yaml_path), strict=True
    )
    # No exclude_patterns passed → kwarg must be absent so we don't
    # silently override the FlowDoctorHandler default.
    _, kwargs = fake_module.FlowDoctorHandler.call_args
    assert "exclude_patterns" not in kwargs


def test_flow_doctor_exclude_patterns_forwarded(monkeypatch, tmp_path):
    """exclude_patterns reaches FlowDoctorHandler when provided.

    Context (2026-04-14): executor suppresses benign IB Error 10197
    alerts via this path. The kwarg must pass through unchanged.
    """
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")

    fake_module = mock.Mock()
    fake_module.init = mock.Mock(return_value=mock.Mock())
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())

    patterns = [r"Error 10197", r"benign warning"]
    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        setup_logging("test", flow_doctor_yaml=str(yaml_path), exclude_patterns=patterns)

    _, kwargs = fake_module.FlowDoctorHandler.call_args
    assert kwargs.get("exclude_patterns") == patterns


def test_flow_doctor_empty_exclude_patterns_not_forwarded(monkeypatch, tmp_path):
    """Empty/None exclude_patterns must not set the kwarg — preserves
    the FlowDoctorHandler default rather than forcing it to ``[]``."""
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")

    fake_module = mock.Mock()
    fake_module.init = mock.Mock(return_value=mock.Mock())
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())

    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        setup_logging("test", flow_doctor_yaml=str(yaml_path), exclude_patterns=[])

    _, kwargs = fake_module.FlowDoctorHandler.call_args
    assert "exclude_patterns" not in kwargs


# ── _seed_flow_doctor_secrets ────────────────────────────────────────────

_FD_YAML = (
    "flow_name: test\n"
    "email:\n"
    "  sender: ${EMAIL_SENDER}\n"
    "  recipients: ${EMAIL_RECIPIENTS}\n"
    "  password: ${GMAIL_APP_PASSWORD}\n"
    "github:\n"
    "  token: ${FLOW_DOCTOR_GITHUB_TOKEN}\n"
)

_FD_VARS = (
    "EMAIL_SENDER",
    "EMAIL_RECIPIENTS",
    "GMAIL_APP_PASSWORD",
    "FLOW_DOCTOR_GITHUB_TOKEN",
)


@pytest.fixture
def _clean_fd_env():
    """Remove the flow-doctor vars before the test and after, so a
    direct ``os.environ[var] = ...`` write by the seed can't leak.
    Also resets the secrets per-process cache + SSM-unavailable latch
    so a real get_secret call in a sibling test can't bleed in."""
    import krepis.secrets as _secrets

    _secrets.clear_cache()
    saved = {k: os.environ.pop(k, None) for k in _FD_VARS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _secrets.clear_cache()


@pytest.fixture
def _patch_get_secret(monkeypatch):
    """Patch ``get_secret`` on the *imported* secrets module object.

    The seed helper resolves it via ``from krepis.secrets
    import get_secret`` at call time. A string-path
    ``monkeypatch.setattr("krepis.secrets.get_secret", …)``
    can resolve a *different* module object than the one in
    ``sys.modules`` (installed copy vs. ``PYTHONPATH=src`` worktree),
    so the patch silently misses and the real SSM-backed get_secret
    runs. Patching the already-imported module object is identity-safe
    — see ``feedback_monkeypatch_over_unittest_mock_patch_in_research``.
    """
    import krepis.secrets as _secrets

    def _apply(fn):
        monkeypatch.setattr(_secrets, "get_secret", fn)

    return _apply


def test_seed_derives_var_set_from_yaml_and_seeds_unset(
    _patch_get_secret, tmp_path, _clean_fd_env
):
    """Every ${VAR} in the yaml is resolved via get_secret and seeded.

    The var set is read from the yaml itself — not a hardcoded list —
    so a yaml-added secret can't silently re-open the env gap.
    """
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text(_FD_YAML)

    seen: list[str] = []

    def fake_get_secret(name, *, required=True, default=None):
        seen.append(name)
        return f"resolved-{name}"

    _patch_get_secret(fake_get_secret)
    _seed_flow_doctor_secrets(str(yaml_path))

    assert sorted(seen) == sorted(_FD_VARS)
    for var in _FD_VARS:
        assert os.environ[var] == f"resolved-{var}"


def test_seed_preset_env_var_wins(_patch_get_secret, tmp_path, _clean_fd_env, monkeypatch):
    """Local dev (not deployed): an already-set env var wins and the
    backend is never touched for it."""
    monkeypatch.delenv("ALPHA_ENGINE_DEPLOYED", raising=False)
    monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text(_FD_YAML)
    os.environ["EMAIL_SENDER"] = "preset@example.com"

    _patch_get_secret(
        lambda name, **kw: pytest.fail(f"get_secret({name}) called for preset var")
        if name == "EMAIL_SENDER" else "resolved"
    )
    _seed_flow_doctor_secrets(str(yaml_path))

    assert os.environ["EMAIL_SENDER"] == "preset@example.com"


def test_seed_deployed_ssm_overwrites_stale_env(
    _patch_get_secret, tmp_path, _clean_fd_env, monkeypatch
):
    """Deployed: SSM is authoritative — a stale sourced value (e.g. a
    rotated PAT from a legacy ~/.alpha-engine.env) is overwritten with
    the live SSM value."""
    monkeypatch.setenv("ALPHA_ENGINE_DEPLOYED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text(_FD_YAML)
    os.environ["FLOW_DOCTOR_GITHUB_TOKEN"] = "stale-from-dotenv"

    _patch_get_secret(
        lambda name, **kw: "fresh-from-ssm"
        if name == "FLOW_DOCTOR_GITHUB_TOKEN" else "resolved"
    )
    _seed_flow_doctor_secrets(str(yaml_path))

    assert os.environ["FLOW_DOCTOR_GITHUB_TOKEN"] == "fresh-from-ssm"


def test_seed_deployed_ssm_miss_keeps_existing(
    _patch_get_secret, tmp_path, _clean_fd_env, monkeypatch
):
    """Deployed but SSM resolves nothing: the existing sourced value is
    left untouched (graceful SSM-miss fallback, not blanked)."""
    monkeypatch.setenv("ALPHA_ENGINE_DEPLOYED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text(_FD_YAML)
    os.environ["FLOW_DOCTOR_GITHUB_TOKEN"] = "sourced-value"

    _patch_get_secret(lambda name, **kw: None)
    _seed_flow_doctor_secrets(str(yaml_path))

    assert os.environ["FLOW_DOCTOR_GITHUB_TOKEN"] == "sourced-value"


def test_seed_unresolvable_secret_left_unset(_patch_get_secret, tmp_path, _clean_fd_env):
    """get_secret → None leaves the var unset so flow-doctor's own
    ConfigError fires loudly (feedback_no_silent_fails)."""
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text(_FD_YAML)

    _patch_get_secret(lambda name, **kw: None)
    _seed_flow_doctor_secrets(str(yaml_path))

    for var in _FD_VARS:
        assert var not in os.environ


def test_seed_backend_exception_swallowed_var_left_unset(
    _patch_get_secret, tmp_path, caplog, _clean_fd_env
):
    """A secrets-backend hiccup is logged at WARNING and the var is
    left unset — never blocks logging setup, never masked with ''."""
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text(_FD_YAML)

    def boom(name, **kw):
        raise RuntimeError("ssm unreachable")

    _patch_get_secret(boom)
    with caplog.at_level(logging.WARNING):
        _seed_flow_doctor_secrets(str(yaml_path))

    for var in _FD_VARS:
        assert var not in os.environ
    assert "flow-doctor secret seed" in caplog.text


def test_seed_missing_yaml_is_noop(_patch_get_secret, tmp_path):
    """An unreadable yaml is a no-op here — _attach_flow_doctor's
    os.path.exists guard reports it with a clearer message."""
    _patch_get_secret(
        lambda name, **kw: pytest.fail("get_secret must not run for missing yaml")
    )
    _seed_flow_doctor_secrets(str(tmp_path / "nope.yaml"))


def test_seed_runs_before_flow_doctor_init(
    monkeypatch, _patch_get_secret, tmp_path, _clean_fd_env
):
    """Integration: the seed populates os.environ *before*
    FlowDoctor.from_config() reads it — the whole point of the fix."""
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text(_FD_YAML)

    _patch_get_secret(lambda name, **kw: f"resolved-{name}")

    env_at_init: dict[str, str] = {}

    def capturing_from_config(*, config_path, strict=True):
        env_at_init.update({v: os.environ.get(v) for v in _FD_VARS})
        return mock.Mock()

    fake_module = mock.Mock()
    # 0.6.0+ path: lib reads the yaml via FlowDoctor.from_config().
    fake_module.FlowDoctor.from_config = mock.Mock(side_effect=capturing_from_config)
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())

    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        setup_logging("test", flow_doctor_yaml=str(yaml_path))

    for var in _FD_VARS:
        assert env_at_init[var] == f"resolved-{var}"


# ── SecretsRedactingFilter ────────────────────────────────────────────────


class _CapturingHandler(logging.Handler):
    """Test helper that captures emitted messages."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _run_through_filter(message: str, *args) -> str:
    """Pipe a single log record through SecretsRedactingFilter + capture."""
    logger = logging.getLogger("_redaction_test")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _CapturingHandler()
    handler.addFilter(SecretsRedactingFilter())
    logger.addHandler(handler)
    logger.warning(message, *args)
    return handler.messages[-1]


def test_redacts_fmp_apikey_in_url():
    msg = (
        "FMP grades-consensus failed for APO: 402 "
        "https://financialmodelingprep.com/stable/grades-consensus?"
        "apikey=1vR8poht91Y6snXRwZdGaAUgqGAalagf&symbol=APO"
    )
    out = _run_through_filter(msg)
    assert "1vR8poht91Y6snXRwZdGaAUgqGAalagf" not in out
    assert "<REDACTED>" in out
    assert "&symbol=APO" in out  # post-key URL params preserved


def test_redacts_apikey_in_args_substitution():
    """logger.warning('...%s', key) — redact AFTER format-string interp."""
    msg = "FMP failed: %s"
    url = "https://api.com/x?apikey=1vR8poht91Y6snXRwZdGaAUgqGAalagf"
    out = _run_through_filter(msg, url)
    assert "1vR8poht91Y6snXRwZdGaAUgqGAalagf" not in out
    assert "<REDACTED>" in out


def test_redacts_aws_access_key_id():
    msg = "AWS credential AKIAIOSFODNN7EXAMPLE in env"
    out = _run_through_filter(msg)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "<AWS_ACCESS_KEY_REDACTED>" in out


def test_redacts_anthropic_api_key():
    msg = "Anthropic call failed with key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv"
    out = _run_through_filter(msg)
    assert "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv" not in out
    assert "<ANTHROPIC_KEY_REDACTED>" in out


def test_redacts_openai_style_key():
    msg = "OpenAI key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
    out = _run_through_filter(msg)
    assert "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789" not in out
    assert "<OPENAI_KEY_REDACTED>" in out


def test_redacts_bearer_token():
    msg = "Authorization: Bearer abc123def456ghi789jkl012"
    out = _run_through_filter(msg)
    assert "abc123def456ghi789jkl012" not in out
    assert "<REDACTED>" in out


def test_redacts_github_pat_classic_and_fine_grained():
    classic = _run_through_filter("Token: ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ab")
    assert "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789ab" not in classic
    assert "<GITHUB_TOKEN_REDACTED>" in classic
    fg = _run_through_filter("Token: github_pat_11ABCDEFG0AbCdEfGhIjKlMnOpQrStUv")
    assert "github_pat_11ABCDEFG0AbCdEfGhIjKlMnOpQrStUv" not in fg
    assert "<GITHUB_TOKEN_REDACTED>" in fg


def test_short_token_below_floor_not_redacted():
    """A ?apikey=abc value below the 16-char floor is NOT redacted —
    false-positive avoidance for short ID-like tokens."""
    out = _run_through_filter("Request: https://api.com/x?apikey=shortkey&y=1")
    assert "shortkey" in out  # under length floor
    assert "<REDACTED>" not in out


def test_no_match_is_unchanged():
    msg = "Normal log line with no secrets"
    out = _run_through_filter(msg)
    assert out == msg


def test_filter_never_raises_on_malformed_record():
    """A filter that crashes is worse than one that no-ops — verify the
    catch-all swallows exceptions silently."""
    f = SecretsRedactingFilter()
    # Synthesise a record whose getMessage will raise — args mismatch.
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="%s %s %s", args=("only-one-arg",), exc_info=None,
    )
    # Should not raise; should return True (record passes through).
    assert f.filter(record) is True


def test_per_record_opt_out_via_no_redact_attr():
    f = SecretsRedactingFilter()
    msg = "apikey=1vR8poht91Y6snXRwZdGaAUgqGAalagf"
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    record.no_redact = True
    f.filter(record)
    # Unmodified since opt-out flag was set
    assert record.getMessage() == msg


def test_setup_logging_attaches_filter_by_default(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_DISABLE_LOG_REDACTION", raising=False)
    setup_logging("test")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert any(
        isinstance(f, SecretsRedactingFilter) for f in handler.filters
    ), "SecretsRedactingFilter must be attached by default"


def test_setup_logging_opt_out_via_env_var(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_DISABLE_LOG_REDACTION", "1")
    setup_logging("test")
    root = logging.getLogger()
    handler = root.handlers[0]
    assert not any(
        isinstance(f, SecretsRedactingFilter) for f in handler.filters
    ), "Filter must NOT be attached when ALPHA_ENGINE_DISABLE_LOG_REDACTION=1"


def test_end_to_end_setup_logging_redacts_emitted_message(monkeypatch, capsys):
    monkeypatch.delenv("ALPHA_ENGINE_DISABLE_LOG_REDACTION", raising=False)
    setup_logging("test")
    logging.getLogger().warning(
        "FMP failed: https://x.com/?apikey=1vR8poht91Y6snXRwZdGaAUgqGAalagf"
    )
    captured = capsys.readouterr()
    assert "1vR8poht91Y6snXRwZdGaAUgqGAalagf" not in (captured.out + captured.err)
    assert "<REDACTED>" in (captured.out + captured.err)
