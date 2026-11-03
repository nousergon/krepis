"""Environment alert routing (S-20, alpha-engine-config-I12508).

Staging alerts are digest records, never pages; everything that is not an
explicit registry ``digest`` keeps the unchanged page path (fails upward).
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from krepis import alerts


class _Env:
    def __init__(
        self, name: str, routing: object, bucket: str = "staging-crucible-bucket"
    ) -> None:
        self.name = name
        self.region = "us-east-1"
        self._bucket = bucket
        if routing is not _MISSING:
            self.alert_routing = routing

    def bucket(self, role: str) -> str:
        assert role == alerts.DIGEST_BUCKET_ROLE
        return self._bucket


_MISSING = object()


def _install_registry(monkeypatch, envs: dict[str, _Env]) -> None:
    env_mod = types.ModuleType("nousergon_lib.env")

    def resolve(name: str | None = None):
        if name not in envs:
            raise KeyError(f"environment {name!r} is not declared")
        return envs[name]

    env_mod.resolve = resolve  # type: ignore[attr-defined]
    pkg = types.ModuleType("nousergon_lib")
    pkg.env = env_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nousergon_lib", pkg)
    monkeypatch.setitem(sys.modules, "nousergon_lib.env", env_mod)


@pytest.fixture
def s3(monkeypatch):
    client = MagicMock()
    fake = types.ModuleType("boto3")
    fake.client = MagicMock(return_value=client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)
    return client


@pytest.fixture
def no_page_path(monkeypatch):
    """Every surface that can reach a person or a prod store fails the test if touched."""

    def boom(*a, **k):
        raise AssertionError("page path reached under alert_routing=digest")

    for name in (
        "_publish_sns",
        "_publish_telegram",
        "_fetch_source_mutes",
        "_resolve_log_chat",
    ):
        monkeypatch.setattr(alerts, name, boom)
    monkeypatch.setattr(alerts.alert_tiers, "resolve_tier", boom)
    monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
    monkeypatch.delenv("KREPIS_ALERT_DRY_RUN", raising=False)


def test_ne_env_unset_pages(monkeypatch):
    monkeypatch.delenv("NE_ENV", raising=False)
    routing, reason, env = alerts.resolve_alert_routing()
    assert (routing, env) == ("page", None)
    assert "unset" in reason


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(
            lambda mp: (
                mp.setitem(sys.modules, "nousergon_lib", None),
                mp.setitem(sys.modules, "nousergon_lib.env", None),
            ),
            id="lib-absent",
        ),
        pytest.param(lambda mp: _install_registry(mp, {}), id="unknown-environment"),
        pytest.param(
            lambda mp: _install_registry(mp, {"staging": _Env("staging", _MISSING)}),
            id="old-lib-no-field",
        ),
        pytest.param(
            lambda mp: _install_registry(mp, {"staging": _Env("staging", "DIGEST")}),
            id="malformed-value",
        ),
    ],
)
def test_anything_but_an_explicit_digest_fails_upward_to_page(monkeypatch, setup):
    monkeypatch.setenv("NE_ENV", "staging")
    setup(monkeypatch)
    routing, reason, env = alerts.resolve_alert_routing()
    assert (routing, env) == ("page", None)
    assert reason.startswith("fail-upward")


def test_prod_registry_page_keeps_the_page_path(monkeypatch):
    monkeypatch.setenv("NE_ENV", "prod")
    _install_registry(monkeypatch, {"prod": _Env("prod", "page")})
    assert alerts.resolve_alert_routing()[0] == "page"


def test_staging_digest_writes_one_record_and_never_pages(
    monkeypatch, s3, no_page_path
):
    """Holds under all three upward defaults: an unknown source (no registry
    row), an unreadable tier registry and an unset log chat are never reached."""
    monkeypatch.setenv("NE_ENV", "staging")
    monkeypatch.delenv("TELEGRAM_LOG_CHAT_ID", raising=False)
    _install_registry(monkeypatch, {"staging": _Env("staging", "digest")})

    r = alerts.publish(
        "box fell over", severity="critical", source="never-registered-source"
    )

    assert r.alert_routing == "digest"
    assert r.any_ok and r.all_ok
    assert r.telegram_destination is None and r.tier is None
    assert (
        not r.sns.ok
        and not r.telegram.ok
        and "alert_routing=digest" in r.telegram.detail
    )
    s3.put_object.assert_called_once()
    kw = s3.put_object.call_args.kwargs
    assert kw["Bucket"] == "staging-crucible-bucket"
    assert kw["Key"].startswith("alerts/") and kw["Key"].endswith(".json")
    body = json.loads(kw["Body"])
    assert body["env"] == "staging" and body["severity"] == "critical"
    assert (
        body["source"] == "never-registered-source"
        and "box fell over" in body["message"]
    )
    assert body["schema"] == alerts.DIGEST_RECORD_SCHEMA
    assert r.digest_record == f"s3://staging-crucible-bucket/{kw['Key']}"


def test_digest_write_failure_is_total_non_delivery(monkeypatch, s3, no_page_path):
    monkeypatch.setenv("NE_ENV", "staging")
    _install_registry(monkeypatch, {"staging": _Env("staging", "digest")})
    s3.put_object.side_effect = RuntimeError("AccessDenied")

    with pytest.raises(alerts.AlertDeliveryError, match="alert_routing=digest"):
        alerts.publish("x", source="s")

    r = alerts.publish("x", source="s", raise_on_total_failure=False)
    assert not r.any_ok and r.digest_record is None


def test_dry_run_still_wins_over_digest(monkeypatch, s3):
    monkeypatch.setenv("NE_ENV", "staging")
    _install_registry(monkeypatch, {"staging": _Env("staging", "digest")})
    r = alerts.publish("x", source="s", dry_run=True)
    s3.put_object.assert_not_called()
    assert r.alert_routing == "page" and r.digest is None
