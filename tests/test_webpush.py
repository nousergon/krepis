"""
Unit tests for ``krepis.webpush``.

Locks down the Web Push send contract: VAPID secret resolution + override
precedence, payload shape, fire-and-forget failure handling (no exceptions
ever propagate — mirrors ``krepis.telegram``'s contract), and the
pywebpush-not-installed no-op path. ``krepis.webpush._webpush`` /
``WebPushException`` are patched directly rather than relying on the real
``pywebpush`` package being installed, since ``webpush`` is an optional
extra CI's default ``.[dev]`` install does not pull in — this keeps the
suite deterministic regardless of what's actually installed locally.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krepis import webpush as wp
from krepis.secrets import clear_cache


class _FakeWebPushException(Exception):
    def __init__(self, message, response=None):
        super().__init__(message)
        self.response = response


SUBSCRIPTION = {
    "endpoint": "https://push.example.com/abc123",
    "keys": {"p256dh": "test-p256dh", "auth": "test-auth"},
}


@pytest.fixture(autouse=True)
def _reset_secrets_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def configured_env(monkeypatch):
    """Resolve the VAPID keypair via env (skip SSM)."""
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    monkeypatch.setenv("WEBPUSH_VAPID_PRIVATE_KEY", "test-private-key")
    monkeypatch.delenv("WEBPUSH_VAPID_SUBJECT", raising=False)


@pytest.fixture
def mock_webpush():
    """Patch the lazily-imported pywebpush send function + exception class,
    regardless of whether the real pywebpush package is installed."""
    with patch.object(wp, "_webpush") as mocked, patch.object(wp, "WebPushException", _FakeWebPushException):
        yield mocked


# ── send_push — happy path ──────────────────────────────────────────────────


class TestSendPushHappyPath:
    def test_returns_true_on_success(self, configured_env, mock_webpush):
        assert wp.send_push(SUBSCRIPTION, title="t", body="b") is True

    def test_calls_pywebpush_with_subscription(self, configured_env, mock_webpush):
        wp.send_push(SUBSCRIPTION, title="t", body="b")
        call = mock_webpush.call_args.kwargs
        assert call["subscription_info"] == SUBSCRIPTION

    def test_payload_shape(self, configured_env, mock_webpush):
        import json

        wp.send_push(SUBSCRIPTION, title="Persona replied", body="Staging or prod?", url="/x", tag="turn-done")
        payload = json.loads(mock_webpush.call_args.kwargs["data"])
        assert payload == {
            "title": "Persona replied",
            "body": "Staging or prod?",
            "url": "/x",
            "tag": "turn-done",
        }

    def test_payload_includes_null_url_and_tag_when_omitted(self, configured_env, mock_webpush):
        import json

        wp.send_push(SUBSCRIPTION, title="t", body="b")
        payload = json.loads(mock_webpush.call_args.kwargs["data"])
        assert payload["url"] is None
        assert payload["tag"] is None

    def test_uses_resolved_private_key(self, configured_env, mock_webpush):
        wp.send_push(SUBSCRIPTION, title="t", body="b")
        assert mock_webpush.call_args.kwargs["vapid_private_key"] == "test-private-key"

    def test_default_subject_when_unconfigured(self, configured_env, mock_webpush):
        wp.send_push(SUBSCRIPTION, title="t", body="b")
        assert mock_webpush.call_args.kwargs["vapid_claims"] == {"sub": wp.VAPID_SUBJECT_DEFAULT}

    def test_configured_subject_used(self, configured_env, mock_webpush, monkeypatch):
        monkeypatch.setenv("WEBPUSH_VAPID_SUBJECT", "mailto:brian@example.com")
        wp.send_push(SUBSCRIPTION, title="t", body="b")
        assert mock_webpush.call_args.kwargs["vapid_claims"] == {"sub": "mailto:brian@example.com"}


# ── send_push — explicit overrides beat secret lookup ───────────────────────


class TestExplicitOverrides:
    def test_explicit_private_key_overrides_secret(self, configured_env, mock_webpush):
        wp.send_push(SUBSCRIPTION, title="t", body="b", vapid_private_key="explicit-key")
        assert mock_webpush.call_args.kwargs["vapid_private_key"] == "explicit-key"

    def test_explicit_subject_overrides_secret(self, configured_env, mock_webpush):
        wp.send_push(SUBSCRIPTION, title="t", body="b", vapid_subject="mailto:override@example.com")
        assert mock_webpush.call_args.kwargs["vapid_claims"] == {"sub": "mailto:override@example.com"}

    def test_works_with_no_env_configured_when_keys_passed_explicitly(self, monkeypatch, mock_webpush):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.delenv("WEBPUSH_VAPID_PRIVATE_KEY", raising=False)
        assert wp.send_push(SUBSCRIPTION, title="t", body="b", vapid_private_key="standalone-key") is True


# ── send_push — secret resolution failures ──────────────────────────────────


class TestSecretResolution:
    def test_missing_private_key_returns_false_no_send(self, monkeypatch, mock_webpush):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.delenv("WEBPUSH_VAPID_PRIVATE_KEY", raising=False)
        assert wp.send_push(SUBSCRIPTION, title="t", body="b") is False
        mock_webpush.assert_not_called()


# ── send_push — pywebpush not installed ─────────────────────────────────────


class TestNotInstalled:
    def test_returns_false_without_calling_anything(self, configured_env, monkeypatch):
        monkeypatch.setattr(wp, "_webpush", None)
        assert wp.send_push(SUBSCRIPTION, title="t", body="b") is False

    def test_logs_warning(self, configured_env, monkeypatch, caplog):
        monkeypatch.setattr(wp, "_webpush", None)
        wp.send_push(SUBSCRIPTION, title="t", body="b")
        assert "pywebpush is not installed" in caplog.text


# ── send_push — failure modes never raise ───────────────────────────────────


class TestFailureSwallowing:
    def test_webpush_exception_returns_false(self, configured_env, mock_webpush):
        mock_webpush.side_effect = _FakeWebPushException("push failed", response=MagicMock(status_code=410))
        assert wp.send_push(SUBSCRIPTION, title="t", body="b") is False

    def test_webpush_exception_logs_status(self, configured_env, mock_webpush, caplog):
        mock_webpush.side_effect = _FakeWebPushException("gone", response=MagicMock(status_code=410))
        wp.send_push(SUBSCRIPTION, title="t", body="b")
        assert "410" in caplog.text

    def test_webpush_exception_without_response_does_not_crash(self, configured_env, mock_webpush):
        mock_webpush.side_effect = _FakeWebPushException("no response attr")
        assert wp.send_push(SUBSCRIPTION, title="t", body="b") is False
