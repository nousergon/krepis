"""
Unit tests for ``krepis.fleet_events`` and the chokepoint instrumentation.

Pins the Overseer intake contract (alpha-engine-config-I2822): event shape
against the shipped v1 schema, transport fallback order (PutEvents → S3 →
WARNING log, never raises), the test-env guard, severity normalization /
silent-flag proxying, and — critically — the no-double-emit invariant:
one ``alerts.publish`` call produces exactly ONE event even though it
routes through ``telegram.send_message`` internally.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from krepis import alerts, fleet_events, telegram


@pytest.fixture(autouse=True)
def _allow_test_events(monkeypatch):
    """This suite exercises emission against mocked transports."""
    monkeypatch.setenv("NOUSERGON_ALLOW_TEST_EVENTS", "1")
    monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")


@pytest.fixture
def fake_events_boto3():
    """boto3 stub with a mocked events client (and S3 for fallback tests)."""
    events_client = MagicMock()
    events_client.put_events.return_value = {"FailedEntryCount": 0, "Entries": [{}]}
    s3_client = MagicMock()
    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "711398986525"}
    sns_client = MagicMock()
    sns_client.publish.return_value = {"MessageId": "m-1"}

    fake = MagicMock()

    def _client(service: str, **kwargs):
        return {
            "events": events_client,
            "s3": s3_client,
            "sts": sts_client,
            "sns": sns_client,
        }[service]

    fake.client.side_effect = _client
    return fake, events_client, s3_client


def _emitted_detail(events_client) -> dict:
    entries = events_client.put_events.call_args.kwargs["Entries"]
    assert len(entries) == 1
    return json.loads(entries[0]["Detail"])


class TestNormalizeSeverity:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("error", "error"),
            ("CRITICAL", "critical"),
            ("Info", "info"),
            ("warn", "warning"),
            ("warning", "warning"),
            ("bogus-tier", "warning"),
            (None, "info"),
            ("", "info"),
        ],
    )
    def test_mapping(self, raw, expected):
        assert fleet_events.normalize_severity(raw) == expected


class TestEmitAlertEvent:
    def test_put_events_entry_shape(self, fake_events_boto3):
        fake, events_client, _ = fake_events_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            ok = fleet_events.emit_alert_event(
                origin="alerts.publish",
                body="boom",
                severity_raw="error",
                source="unit-test",
                dedup_key="k1",
                channels={"sns": True, "telegram": False},
            )
        assert ok is True
        entry = events_client.put_events.call_args.kwargs["Entries"][0]
        assert entry["Source"] == fleet_events.EVENT_SOURCE
        assert entry["DetailType"] == fleet_events.DETAIL_TYPE
        assert entry["EventBusName"] == fleet_events.DEFAULT_BUS_NAME
        detail = json.loads(entry["Detail"])
        assert detail["schema_version"] == 1
        assert detail["severity"] == "error"
        assert detail["severity_raw"] == "error"
        assert detail["source"] == "unit-test"
        assert detail["dedup_key"] == "k1"
        assert detail["channels"] == {"sns": True, "telegram": False}

    def test_detail_validates_against_shipped_schema(self, fake_events_boto3):
        jsonschema = pytest.importorskip("jsonschema")
        from pathlib import Path

        schema = json.loads(
            (
                Path(fleet_events.__file__).parent / "fleet_event_schema_v1.json"
            ).read_text()
        )
        fake, events_client, _ = fake_events_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            fleet_events.emit_alert_event(
                origin="telegram.send_message",
                body="digest",
                disable_notification=True,
                channels={"sns": None, "telegram": True},
            )
        jsonschema.validate(_emitted_detail(events_client), schema)

    def test_bus_env_override(self, fake_events_boto3, monkeypatch):
        fake, events_client, _ = fake_events_boto3
        monkeypatch.setenv(fleet_events.BUS_ENV, "custom-bus")
        with patch.dict("sys.modules", {"boto3": fake}):
            fleet_events.emit_alert_event(origin="alerts.publish", body="x")
        assert (
            events_client.put_events.call_args.kwargs["Entries"][0]["EventBusName"]
            == "custom-bus"
        )

    def test_body_truncated(self, fake_events_boto3):
        fake, events_client, _ = fake_events_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            fleet_events.emit_alert_event(origin="alerts.publish", body="x" * 9000)
        assert len(_emitted_detail(events_client)["body"]) == fleet_events.MAX_BODY_CHARS

    def test_lambda_env_source_attribution(self, fake_events_boto3, monkeypatch):
        fake, events_client, _ = fake_events_boto3
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "freshness-monitor")
        with patch.dict("sys.modules", {"boto3": fake}):
            fleet_events.emit_alert_event(origin="telegram.send_message", body="x")
        detail = _emitted_detail(events_client)
        assert detail["source"] == "freshness-monitor"
        assert detail["runtime"]["lambda_function_name"] == "freshness-monitor"

    def test_silent_flag_proxies_severity_for_direct_sends(self, fake_events_boto3):
        fake, events_client, _ = fake_events_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            fleet_events.emit_alert_event(
                origin="telegram.send_message", body="x", disable_notification=True
            )
            assert _emitted_detail(events_client)["severity"] == "info"
            fleet_events.emit_alert_event(
                origin="telegram.send_message", body="x", disable_notification=False
            )
            assert _emitted_detail(events_client)["severity"] == "warning"

    def test_put_events_failure_falls_back_to_s3(self, fake_events_boto3):
        fake, events_client, s3_client = fake_events_boto3
        events_client.put_events.side_effect = RuntimeError("AccessDenied")
        with patch.dict("sys.modules", {"boto3": fake}):
            ok = fleet_events.emit_alert_event(origin="alerts.publish", body="boom")
        assert ok is True
        kwargs = s3_client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == fleet_events.DEFAULT_FALLBACK_BUCKET
        assert kwargs["Key"].startswith(fleet_events.FALLBACK_PREFIX + "/")
        payload = json.loads(kwargs["Body"].decode("utf-8"))
        assert payload["detail_type"] == fleet_events.DETAIL_TYPE
        assert payload["detail"]["body"] == "boom"

    def test_rejected_entry_falls_back_to_s3(self, fake_events_boto3):
        fake, events_client, s3_client = fake_events_boto3
        events_client.put_events.return_value = {
            "FailedEntryCount": 1,
            "Entries": [{"ErrorCode": "InternalFailure", "ErrorMessage": "nope"}],
        }
        with patch.dict("sys.modules", {"boto3": fake}):
            assert fleet_events.emit_alert_event(origin="alerts.publish", body="b")
        assert s3_client.put_object.called

    def test_both_transports_failing_never_raises(self, fake_events_boto3, caplog):
        fake, events_client, s3_client = fake_events_boto3
        events_client.put_events.side_effect = RuntimeError("down")
        s3_client.put_object.side_effect = RuntimeError("also down")
        with patch.dict("sys.modules", {"boto3": fake}):
            ok = fleet_events.emit_alert_event(origin="alerts.publish", body="b")
        assert ok is False
        assert "NOUSERGON_ALERT_EVENT_EMIT_FAILED" in caplog.text

    def test_test_env_guard_suppresses(self, fake_events_boto3, monkeypatch):
        fake, events_client, _ = fake_events_boto3
        monkeypatch.delenv("NOUSERGON_ALLOW_TEST_EVENTS")
        with patch.dict("sys.modules", {"boto3": fake}):
            ok = fleet_events.emit_alert_event(origin="alerts.publish", body="b")
        assert ok is False
        events_client.put_events.assert_not_called()


class TestChokepointIntegration:
    """The instrumented chokepoints emit exactly one event per alert."""

    def test_alerts_publish_emits_one_rich_event(self, fake_events_boto3):
        fake, events_client, _ = fake_events_boto3
        with patch.dict("sys.modules", {"boto3": fake}), patch(
            "krepis.telegram.send_message", return_value=True
        ) as tg:
            result = alerts.publish(
                "disk full", severity="critical", source="probe", dedup_key="dk"
            )
        assert result.sns.ok and result.telegram.ok
        assert events_client.put_events.call_count == 1
        detail = _emitted_detail(events_client)
        assert detail["origin"] == "alerts.publish"
        assert detail["severity"] == "critical"
        assert detail["source"] == "probe"
        assert detail["dedup_key"] == "dk"
        assert detail["channels"] == {"sns": True, "telegram": True}
        # The chokepoint passes the RAW message; formatting stays human-side.
        assert detail["body"] == "disk full"

    def test_publish_through_real_send_message_does_not_double_emit(
        self, fake_events_boto3, monkeypatch
    ):
        """End-to-end through the REAL telegram.send_message (mocked HTTP):
        the suppression contextvar must hold the auto-emit hook quiet."""
        fake, events_client, _ = fake_events_boto3
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        resp = MagicMock(status_code=200)
        with patch.dict("sys.modules", {"boto3": fake}), patch(
            "krepis.telegram.requests.post", return_value=resp
        ):
            alerts.publish("boom", severity="error", sns=False)
        assert events_client.put_events.call_count == 1
        assert _emitted_detail(events_client)["origin"] == "alerts.publish"

    def test_dedup_skip_does_not_emit(self, fake_events_boto3):
        fake, events_client, _ = fake_events_boto3
        with patch.dict("sys.modules", {"boto3": fake}), patch.object(
            alerts, "_check_dedup_marker", return_value=(True, "within window")
        ):
            result = alerts.publish("boom", dedup_key="dk", telegram=False)
        assert result.dedup_skipped
        events_client.put_events.assert_not_called()

    def test_direct_send_message_emits_basic_event(self, fake_events_boto3, monkeypatch):
        fake, events_client, _ = fake_events_boto3
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        resp = MagicMock(status_code=200)
        with patch.dict("sys.modules", {"boto3": fake}), patch(
            "krepis.telegram.requests.post", return_value=resp
        ):
            assert telegram.send_message("digest line", disable_notification=True)
        detail = _emitted_detail(events_client)
        assert detail["origin"] == "telegram.send_message"
        assert detail["severity"] == "info"
        assert detail["channels"] == {"sns": None, "telegram": True}

    def test_unconfigured_send_message_does_not_emit(self, fake_events_boto3, monkeypatch):
        fake, events_client, _ = fake_events_boto3
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with patch.dict("sys.modules", {"boto3": fake}), patch(
            "krepis.secrets.get_secret", return_value=None
        ), patch("krepis.telegram.get_secret", return_value=None):
            assert telegram.send_message("x") is False
        events_client.put_events.assert_not_called()

    def test_send_failure_still_emits_with_failed_channel(
        self, fake_events_boto3, monkeypatch
    ):
        fake, events_client, _ = fake_events_boto3
        monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        resp = MagicMock(status_code=500, text="err")
        with patch.dict("sys.modules", {"boto3": fake}), patch(
            "krepis.telegram.requests.post", return_value=resp
        ):
            assert telegram.send_message("x") is False
        assert _emitted_detail(events_client)["channels"]["telegram"] is False

    def test_emission_failure_never_breaks_the_alert(self, monkeypatch):
        """PutEvents AND S3 down: publish still delivers and returns normally."""
        broken = MagicMock()
        broken.client.side_effect = RuntimeError("aws is down")
        sns_ok = MagicMock()
        with patch.dict("sys.modules", {"boto3": broken}), patch.object(
            alerts, "_publish_sns", return_value=alerts.ChannelResult(ok=True)
        ), patch("krepis.telegram.send_message", return_value=True):
            result = alerts.publish("boom", severity="error")
        assert result.any_ok
