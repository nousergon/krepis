"""
Unit tests for ``krepis.email_sender``.

Locks the email-send contract: secret resolution + argument override,
recipient parsing, Gmail-SMTP primary vs SES fallback selection, html
multipart shape, and fire-and-forget failure handling (no exception ever
propagates to the caller).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krepis import email_sender as es
from krepis.secrets import clear_cache


@pytest.fixture(autouse=True)
def _reset_secrets_cache():
    """Every test starts with an empty secrets cache."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def gmail_env(monkeypatch):
    """Fully configured Gmail-SMTP environment (skip SSM, use env)."""
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    monkeypatch.setenv("EMAIL_SENDER", "bot@gmail.com")
    monkeypatch.setenv("EMAIL_RECIPIENTS", "a@x.com, b@x.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app pass word")


@pytest.fixture
def ses_env(monkeypatch):
    """Configured environment with no Gmail password — forces SES."""
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    monkeypatch.setenv("EMAIL_SENDER", "bot@nousergon.ai")
    monkeypatch.setenv("EMAIL_RECIPIENTS", "a@x.com")
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)


@pytest.fixture
def mock_smtp():
    """Patch smtplib.SMTP with a working context-manager server."""
    with patch.object(es.smtplib, "SMTP") as smtp_cls:
        server = MagicMock()
        smtp_cls.return_value.__enter__.return_value = server
        yield smtp_cls, server


# ── _resolve_recipients ─────────────────────────────────────────────────────


class TestResolveRecipients:
    def test_explicit_arg_wins_over_env(self, gmail_env):
        assert es._resolve_recipients(["only@x.com"]) == ["only@x.com"]

    def test_env_fallback_comma_split_and_strip(self, gmail_env):
        assert es._resolve_recipients(None) == ["a@x.com", "b@x.com"]

    def test_trailing_comma_does_not_yield_empty(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.setenv("EMAIL_RECIPIENTS", "a@x.com,")
        assert es._resolve_recipients(None) == ["a@x.com"]

    def test_no_config_returns_empty(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.delenv("EMAIL_RECIPIENTS", raising=False)
        assert es._resolve_recipients(None) == []


# ── send_email — not configured ─────────────────────────────────────────────


class TestNotConfigured:
    def test_missing_sender_returns_false(self, monkeypatch, mock_smtp):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.delenv("EMAIL_SENDER", raising=False)
        monkeypatch.setenv("EMAIL_RECIPIENTS", "a@x.com")
        assert es.send_email("s", "b") is False
        mock_smtp[0].assert_not_called()

    def test_missing_recipients_returns_false(self, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.setenv("EMAIL_SENDER", "bot@gmail.com")
        monkeypatch.delenv("EMAIL_RECIPIENTS", raising=False)
        assert es.send_email("s", "b") is False


# ── send_email — Gmail SMTP primary ─────────────────────────────────────────


class TestGmailPath:
    def test_returns_true_on_success(self, gmail_env, mock_smtp):
        assert es.send_email("subj", "body") is True

    def test_logs_in_and_sends(self, gmail_env, mock_smtp):
        _, server = mock_smtp
        es.send_email("subj", "body")
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("bot@gmail.com", "apppassword")
        server.sendmail.assert_called_once()
        sender, rcpts, _ = server.sendmail.call_args.args
        assert sender == "bot@gmail.com"
        assert rcpts == ["a@x.com", "b@x.com"]

    def test_html_adds_multipart_alternative(self, gmail_env, mock_smtp):
        _, server = mock_smtp
        es.send_email("subj", "plain", html="<b>hi</b>")
        raw = server.sendmail.call_args.args[2]
        assert "text/plain" in raw and "text/html" in raw

    def test_plain_only_when_no_html(self, gmail_env, mock_smtp):
        _, server = mock_smtp
        es.send_email("subj", "plain")
        raw = server.sendmail.call_args.args[2]
        assert "text/plain" in raw and "text/html" not in raw

    def test_auth_error_returns_false_no_raise(self, gmail_env, mock_smtp):
        import smtplib

        _, server = mock_smtp
        server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad")
        assert es.send_email("s", "b") is False

    def test_generic_smtp_error_returns_false(self, gmail_env, mock_smtp):
        _, server = mock_smtp
        server.sendmail.side_effect = RuntimeError("boom")
        assert es.send_email("s", "b") is False

    def test_explicit_args_override_secrets(self, gmail_env, mock_smtp):
        _, server = mock_smtp
        es.send_email(
            "s", "b", recipients=["override@x.com"], sender="me@x.com"
        )
        sender, rcpts, _ = server.sendmail.call_args.args
        assert sender == "me@x.com" and rcpts == ["override@x.com"]


# ── send_email — SES fallback ───────────────────────────────────────────────


class TestSesFallback:
    def test_ses_used_when_no_app_password(self, ses_env):
        with patch("boto3.client") as bc:
            client = MagicMock()
            bc.return_value = client
            assert es.send_email("subj", "body") is True
            bc.assert_called_once_with("ses", region_name="us-east-1")
            client.send_email.assert_called_once()

    def test_ses_region_override(self, ses_env):
        with patch("boto3.client") as bc:
            bc.return_value = MagicMock()
            es.send_email("s", "b", region="eu-west-1")
            bc.assert_called_once_with("ses", region_name="eu-west-1")

    def test_ses_html_included(self, ses_env):
        with patch("boto3.client") as bc:
            client = MagicMock()
            bc.return_value = client
            es.send_email("s", "b", html="<i>x</i>")
            msg = client.send_email.call_args.kwargs["Message"]
            assert "Html" in msg["Body"] and "Text" in msg["Body"]

    def test_ses_client_error_returns_false(self, ses_env):
        from botocore.exceptions import ClientError

        with patch("boto3.client") as bc:
            client = MagicMock()
            client.send_email.side_effect = ClientError(
                {"Error": {"Message": "denied"}}, "SendEmail"
            )
            bc.return_value = client
            assert es.send_email("s", "b") is False

    def test_ses_generic_error_returns_false(self, ses_env):
        with patch("boto3.client") as bc:
            bc.side_effect = RuntimeError("no creds")
            assert es.send_email("s", "b") is False
