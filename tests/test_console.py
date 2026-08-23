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
    assert DEFAULT_CONSOLE_BASE_URL == "https://dashboard.nousergon.ai"


def test_slug_only() -> None:
    """Bare landing URL — the predictor's no-date branch."""
    assert console_url("model-zoo") == "https://dashboard.nousergon.ai/model-zoo"


def test_slug_with_date() -> None:
    """Date-keyed deep-link — the executor EOD + predictor dated branches."""
    assert (
        console_url("eod-report", date="2026-06-22")
        == "https://dashboard.nousergon.ai/eod-report?date=2026-06-22"
    )


def test_date_none_is_bare() -> None:
    assert console_url("model-zoo", date=None) == "https://dashboard.nousergon.ai/model-zoo"


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
    assert console.console_url("x") == "https://dashboard.nousergon.ai/x"


# --- regression guard: alpha-engine-config#6140 -------------------------
#
# On 2026-08-12 every digest deep-link in the fleet 404'd. Cause: the
# ``console.nousergon.ai`` hostname was cut over to nousergon-console v2
# (a fleet entity index serving only ``/``, ``/search``, ``/registry/<name>``
# and ``/component/<id>``), while every slug built here is a Streamlit page
# ``url_path`` served at ``dashboard.nousergon.ai``. The default was never
# moved, so the links pointed at a host that has none of these routes.
#
# These tests pin the failure mode, not just the current value: they fail if
# the default is ever moved back to a host that cannot serve a page slug.

#: Streamlit ``url_path`` values registered in alpha-engine-dashboard/app.py
#: that producers reach through this module. Measured 404 on console v2 and
#: registered on the Streamlit app, 2026-08-12.
_STREAMLIT_PAGE_SLUGS = (
    "eod-report",
    "director",
    "model-zoo",
    "predictor",
    "analysis",
)


def test_default_is_not_the_v2_console_host() -> None:
    """The v2 console serves no Streamlit page slug — it must never be the default."""
    assert "console.nousergon.ai" not in DEFAULT_CONSOLE_BASE_URL


def test_producer_slugs_resolve_to_the_streamlit_host(monkeypatch) -> None:
    """Every producer deep-link lands on the host that actually serves the page."""
    monkeypatch.delenv(CONSOLE_BASE_URL_ENV, raising=False)
    for slug in _STREAMLIT_PAGE_SLUGS:
        url = console_url(slug, date="2026-08-12")
        assert url.startswith("https://dashboard.nousergon.ai/"), url
        assert "console.nousergon.ai" not in url, url
