"""Tests for the ``krepis.console`` deep-link chokepoint (config#1300).

Covers the four behaviors the three lifted producers relied on: bare slug,
``?date=`` variant, explicit ``base=`` override (with trailing-slash
tolerance), and the ``CONSOLE_BASE_URL`` env override + its precedence.
"""

from __future__ import annotations

import krepis.console as console
from krepis.console import (
    CONSOLE_BASE_URL_ENV,
    DEFAULT_CONSOLE_BASE_URL,
    console_url,
)


def test_default_base() -> None:
    assert DEFAULT_CONSOLE_BASE_URL == "https://console.nousergon.ai"


def test_slug_only() -> None:
    """Bare landing URL — the predictor's no-date branch."""
    assert console_url("model-zoo") == "https://console.nousergon.ai/model-zoo"


def test_slug_with_date() -> None:
    """Date-keyed deep-link — the executor EOD + predictor dated branches."""
    assert (
        console_url("eod-report", date="2026-06-22")
        == "https://console.nousergon.ai/eod-report?date=2026-06-22"
    )


def test_date_none_is_bare() -> None:
    assert console_url("model-zoo", date=None) == "https://console.nousergon.ai/model-zoo"


def test_base_override() -> None:
    assert (
        console_url("eod-report", date="2026-06-22", base="https://console.example.com")
        == "https://console.example.com/eod-report?date=2026-06-22"
    )


def test_base_override_trailing_slash_stripped() -> None:
    """The executor's local builder ``.rstrip('/')``-ed the base; parity."""
    assert (
        console_url("eod-report", base="https://console.example.com/")
        == "https://console.example.com/eod-report"
    )


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv(CONSOLE_BASE_URL_ENV, "https://staging.console.nousergon.ai")
    assert (
        console_url("model-zoo", date="2026-06-26")
        == "https://staging.console.nousergon.ai/model-zoo?date=2026-06-26"
    )


def test_env_override_trailing_slash_stripped(monkeypatch) -> None:
    monkeypatch.setenv(CONSOLE_BASE_URL_ENV, "https://staging.console.nousergon.ai/")
    assert console_url("model-zoo") == "https://staging.console.nousergon.ai/model-zoo"


def test_explicit_base_beats_env(monkeypatch) -> None:
    """Precedence: explicit ``base=`` arg wins over the env var."""
    monkeypatch.setenv(CONSOLE_BASE_URL_ENV, "https://env.example.com")
    assert (
        console_url("model-zoo", base="https://arg.example.com")
        == "https://arg.example.com/model-zoo"
    )


def test_env_unset_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv(CONSOLE_BASE_URL_ENV, raising=False)
    assert console_url("eod-report").startswith(DEFAULT_CONSOLE_BASE_URL)


def test_module_namespace_callable() -> None:
    """Importable as ``krepis.console.console_url`` (re-export sanity)."""
    assert console.console_url("x") == "https://console.nousergon.ai/x"
