#!/usr/bin/env python3
"""Total non-delivery is loud (alpha-engine-config-I9209).

WHAT WENT WRONG
---------------
On 2026-08-29 both laptop identities were IAM-denied on `sns:Publish`, on
`events:PutEvents` for the `nousergon-alerts` bus, AND on `s3:PutObject` for
the `overseer/intake-fallback/` drop zone the primary falls back to. Every
surface an alert could reach was shut, `publish` returned a result nobody
read, and the `router-canary` paged hourly for 119 runs about the `ultra`
class sitting under its prompt-cache floor (alpha-engine-config-I9314) into a
channel that could not deliver.

WHAT THESE TESTS PIN
--------------------
1. Total failure RAISES — the caller cannot continue as if it had paged.
2. PARTIAL failure does not. One channel delivering is the designed behaviour
   and turning it into an exception would page-storm every transient Telegram
   blip. This is the test that keeps the fix from over-reaching.
3. The stable marker `NOUSERGON_ALERT_TOTAL_DELIVERY_FAILURE` is in the
   message, at ERROR level, so a log-scraper can find it and a WARNING filter
   cannot hide it.
4. The opt-out exists and works, because a swallow that cannot be expressed at
   the call site gets expressed as a bare `except` somewhere worse.
"""

from __future__ import annotations

import logging

import pytest

from krepis import alerts, fleet_events


@pytest.fixture(autouse=True)
def _allow_real_path(monkeypatch):
    """Defeat the test-env guard so the fan-out logic actually runs.

    `publish` short-circuits under PYTEST_CURRENT_TEST unless this is set —
    the cross-repo guard that stops a consumer suite paging the operator. This
    module exercises the fan-out against fully stubbed transports, which is
    the one legitimate reason the escape hatch exists.
    """
    monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")


def _stub_transports(monkeypatch, *, sns_ok: bool, telegram_ok: bool, event_ok: bool):
    monkeypatch.setattr(
        alerts, "_resolve_sns_topic_arn",
        lambda explicit: "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts",
    )
    monkeypatch.setattr(
        alerts, "_publish_sns",
        lambda arn, message, subject=None: alerts.ChannelResult(
            ok=sns_ok, detail="stub-sns"
        ),
    )
    monkeypatch.setattr(
        alerts, "_publish_telegram",
        lambda *a, **kw: alerts.ChannelResult(ok=telegram_ok, detail="stub-telegram"),
    )
    monkeypatch.setattr(
        alerts.fleet_events, "emit_alert_event", lambda **kw: event_ok
    )
    # No mute lookup, no dedup marker I/O.
    monkeypatch.setattr(alerts, "_check_source_mute", lambda *a, **kw: (False, ""), raising=False)


def test_every_surface_failing_raises(monkeypatch):
    _stub_transports(monkeypatch, sns_ok=False, telegram_ok=False, event_ok=False)
    with pytest.raises(alerts.AlertDeliveryError) as exc:
        alerts.publish("ultra is below its cache floor", severity="error", source="router-canary")
    assert "NOUSERGON_ALERT_TOTAL_DELIVERY_FAILURE" in str(exc.value)


def test_one_human_channel_surviving_does_not_raise(monkeypatch):
    """SNS denied, Telegram delivered — the operator was reached."""
    _stub_transports(monkeypatch, sns_ok=False, telegram_ok=True, event_ok=False)
    result = alerts.publish("partial", severity="error", source="unit-test")
    assert result.any_ok is True
    assert result.event_emitted is False


def test_the_overseer_event_alone_is_enough_not_to_raise(monkeypatch):
    """Both human channels down but the machine intake got it.

    The Overseer response plane is a real reader, so this is not silence —
    it is a degraded delivery, and degrading is not the failure this raises on.
    """
    _stub_transports(monkeypatch, sns_ok=False, telegram_ok=False, event_ok=True)
    result = alerts.publish("machine-only", severity="error", source="unit-test")
    assert result.any_ok is False
    assert result.event_emitted is True


def test_opt_out_survives_total_failure_and_still_logs_error(monkeypatch, caplog):
    _stub_transports(monkeypatch, sns_ok=False, telegram_ok=False, event_ok=False)
    with caplog.at_level(logging.ERROR, logger="krepis.alerts"):
        result = alerts.publish(
            "opted out", severity="error", source="unit-test",
            raise_on_total_failure=False,
        )
    assert result.any_ok is False
    assert any(
        "NOUSERGON_ALERT_TOTAL_DELIVERY_FAILURE" in r.message
        for r in caplog.records
    ), "the opt-out must not also silence the record"


def test_marker_is_logged_at_error_not_warning(monkeypatch, caplog):
    """A WARNING is the level this failure wore while it was invisible."""
    _stub_transports(monkeypatch, sns_ok=False, telegram_ok=False, event_ok=False)
    with caplog.at_level(logging.WARNING, logger="krepis.alerts"):
        with pytest.raises(alerts.AlertDeliveryError):
            alerts.publish("levels", severity="error", source="unit-test")
    marker_records = [
        r for r in caplog.records
        if "NOUSERGON_ALERT_TOTAL_DELIVERY_FAILURE" in r.getMessage()
    ]
    assert marker_records, "marker was not logged at all"
    assert all(r.levelno >= logging.ERROR for r in marker_records)


def test_dry_run_never_raises(monkeypatch):
    """The dry-run short-circuit returns before any transport is attempted."""
    result = alerts.publish("dry", severity="error", source="unit-test", dry_run=True)
    assert result.any_ok is True
    assert result.event_emitted is None


def test_fleet_events_total_failure_logs_at_error(monkeypatch, caplog):
    """The side channel's own marker moved WARNING -> ERROR."""
    monkeypatch.setenv("NOUSERGON_ALLOW_TEST_EVENTS", "1")
    monkeypatch.setattr(
        fleet_events, "_put_event",
        lambda detail: (_ for _ in ()).throw(RuntimeError("AccessDeniedException PutEvents")),
    )
    monkeypatch.setattr(
        fleet_events, "_write_fallback",
        lambda detail: (_ for _ in ()).throw(RuntimeError("AccessDenied PutObject")),
    )
    with caplog.at_level(logging.WARNING, logger="krepis.fleet_events"):
        assert fleet_events.emit_alert_event(origin="unit-test", body="x") is False
    marker = [
        r for r in caplog.records
        if "NOUSERGON_ALERT_EVENT_EMIT_FAILED" in r.getMessage()
    ]
    assert marker, "the stable marker disappeared"
    assert all(r.levelno >= logging.ERROR for r in marker)
