"""
Unit tests for ``krepis.alerts``.

Pins the failure-surveillance fan-out contract: per-channel independence
(SNS failure doesn't block Telegram and vice-versa), severity-to-push
mapping (error/critical push, info/warning silent), CLI exit codes
(0 if any channel succeeded, 1 only if both failed), and message
formatting (``[SEVERITY] source: body``).

Designed so the Bash dispatcher consumers — spot_backtest.sh's cleanup
trap, the L117 Lambda-deploying repos' canary-rollback branches — can
rely on stable contract semantics across lib versions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krepis import alerts


@pytest.fixture
def fake_boto3():
    """boto3 stub that returns mocked SNS + STS clients keyed by service."""
    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "711398986525"}
    sns_client = MagicMock()
    sns_client.publish.return_value = {"MessageId": "test-msg-id-abc123"}

    fake = MagicMock()

    def _client(service: str, **kwargs):
        if service == "sts":
            return sts_client
        if service == "sns":
            return sns_client
        raise AssertionError(f"unexpected boto3 client request: {service}")

    fake.client.side_effect = _client
    return fake, sts_client, sns_client


class TestFormatMessage:
    def test_with_source(self):
        assert alerts._format_message("boom", "error", "spot_backtest.sh") == "[ERROR] spot_backtest.sh: boom"

    def test_without_source(self):
        assert alerts._format_message("boom", "warning", None) == "[WARNING] boom"

    def test_severity_uppercased(self):
        assert alerts._format_message("x", "Info", "src") == "[INFO] src: x"


class TestResolveSnsTopicArn:
    def test_explicit_override(self, monkeypatch):
        arn = "arn:aws:sns:us-west-2:000000000000:custom-topic"
        assert alerts._resolve_sns_topic_arn(arn) == arn

    def test_defaults_from_env_and_sts(self, monkeypatch, fake_boto3):
        fake, sts, _ = fake_boto3
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        with patch.object(alerts, "__name__", alerts.__name__):
            with patch.dict("sys.modules", {"boto3": fake}):
                result = alerts._resolve_sns_topic_arn(None)
        assert result == "arn:aws:sns:us-east-1:711398986525:alpha-engine-alerts"
        sts.get_caller_identity.assert_called_once()

    def test_returns_none_when_sts_fails(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        fake = MagicMock()
        fake.client.side_effect = RuntimeError("no creds")
        with patch.dict("sys.modules", {"boto3": fake}):
            assert alerts._resolve_sns_topic_arn(None) is None


class TestPublish:
    def test_both_channels_succeed(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                result = alerts.publish("boom", source="spot_backtest.sh")
        assert result.sns.ok is True
        assert result.telegram.ok is True
        assert result.any_ok is True
        assert result.all_ok is True
        # SNS publish was called with severity-tagged message + readable subject
        kwargs = sns.publish.call_args.kwargs
        assert "[ERROR] spot_backtest.sh: boom" in kwargs["Message"]
        assert kwargs["Subject"].startswith("Alpha Engine alert [ERROR]")

    def test_sns_failure_doesnt_block_telegram(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("topic ARN bad")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                result = alerts.publish("boom", source="x")
        assert result.sns.ok is False
        assert "sns error" in result.sns.detail
        assert result.telegram.ok is True
        assert result.any_ok is True
        assert result.all_ok is False

    def test_telegram_failure_doesnt_block_sns(self, fake_boto3):
        fake, _sts, _sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=False, detail="creds missing")):
                result = alerts.publish("boom", source="x")
        assert result.sns.ok is True
        assert result.telegram.ok is False
        assert result.any_ok is True

    def test_both_failures(self, fake_boto3):
        """Both human channels down. Under `raise_on_total_failure=False` the
        structured result is still returned and still reports the failure.

        REVERSED 2026-08-29 (alpha-engine-config-I9209). This test used to
        assert the default path returned quietly. It cannot: on the default
        path, both human channels failing AND the Overseer intake failing now
        raises `AlertDeliveryError` — see
        `tests/test_alert_total_delivery_failure.py`. The opt-out is what
        preserves the shape this test was written to pin, and pinning it here
        keeps the structured-result contract asserted for the callers that
        legitimately use it.
        """
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("nope")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=False, detail="creds missing")):
                result = alerts.publish(
                    "boom", source="x", raise_on_total_failure=False,
                )
        assert result.any_ok is False
        assert result.all_ok is False

    def test_sns_disabled(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                result = alerts.publish("boom", sns=False)
        sns.publish.assert_not_called()
        assert result.sns.ok is False
        assert "not attempted" in result.sns.detail
        assert result.telegram.ok is True

    def test_telegram_disabled(self, fake_boto3):
        fake, _sts, _sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram") as tg:
                result = alerts.publish("boom", telegram=False)
        tg.assert_not_called()
        assert result.sns.ok is True
        assert result.telegram.ok is False

    def test_severity_push_mapping(self, fake_boto3):
        """error/critical → disable_notification=False (push);
        info/warning → disable_notification=True (no push — still sent,
        see test_non_push_severity_still_delivers_to_telegram below)."""
        fake, _sts, _sns = fake_boto3
        from krepis import telegram as tg_mod

        with patch.dict("sys.modules", {"boto3": fake}):
            for sev, expect_silent in [
                ("error", False),
                ("critical", False),
                ("warning", True),
                ("info", True),
            ]:
                with patch.object(tg_mod, "send_message", return_value=True) as send:
                    alerts.publish("x", severity=sev)
                    silent_kwarg = send.call_args.kwargs.get("disable_notification")
                    assert silent_kwarg is expect_silent, f"severity={sev}: expected silent={expect_silent} got {silent_kwarg}"

    def test_non_push_severity_still_delivers_to_telegram(self, fake_boto3):
        """Pins the corrected contract (alpha-engine-config-I7857): a
        severity outside SEVERITY_PHONE_PUSH is NOT a delivery gate. Both
        SNS and Telegram must still be invoked — the message reaches the
        chat and the SNS-subscribed inbox exactly as at 'error', only
        without a phone buzz. Regressing this back to "info/warning don't
        publish" is the exact bug this issue fixed."""
        fake, _sts, sns = fake_boto3
        from krepis import telegram as tg_mod

        with patch.dict("sys.modules", {"boto3": fake}):
            for sev in ("info", "warning"):
                sns.publish.reset_mock()
                with patch.object(tg_mod, "send_message", return_value=True) as send:
                    result = alerts.publish("x", severity=sev, source="test")

                sns.publish.assert_called_once()
                send.assert_called_once()
                assert result.sns.ok is True
                assert result.telegram.ok is True
                # No-push, not no-send: disable_notification=True, message sent.
                assert send.call_args.kwargs.get("disable_notification") is True
                sent_text = send.call_args.args[0] if send.call_args.args else send.call_args.kwargs.get("message")
                assert sent_text == f"[{sev.upper()}] test: x"

    def test_severity_phone_push_alias_matches_canonical(self):
        """SEVERITY_PUSH is a deprecated alias (CONTRIBUTING.md additive-only
        API rule) for SEVERITY_PHONE_PUSH — same object, same members."""
        assert alerts.SEVERITY_PUSH is alerts.SEVERITY_PHONE_PUSH
        assert alerts.SEVERITY_PHONE_PUSH == frozenset({"error", "critical"})

    def test_sns_subject_truncated_and_sanitized(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True)):
                alerts.publish("body", source="x" * 150)
        subject = sns.publish.call_args.kwargs["Subject"]
        assert len(subject) <= 100
        assert "\n" not in subject

    def test_never_raises_on_a_TRANSPORT_exception(self, fake_boto3):
        """A transport blowing up is caught; only TOTAL non-delivery raises.

        RENAMED AND NARROWED 2026-08-29 (alpha-engine-config-I9209). The old
        name asserted "``publish`` must NEVER raise", which is no longer the
        contract and was the property that let a fleet of laptop detectors emit
        into an IAM-denied channel and report success for months. What survives
        — and what this test now pins — is the half that was always right: an
        exception thrown by a transport must never escape as itself. A
        `RuntimeError` out of boto3 is still turned into a structured
        `ChannelResult`. The deliberate raise on total non-delivery is exercised
        under `raise_on_total_failure=False` here so the two properties stay
        independently tested.

        Surfaced 2026-05-21 (post-v0.24.0 dedup PR): the prior version of
        this test ran with no mocks at all, claiming "no creds + no
        mocks" — but on a laptop with real AWS creds it actually
        published to the live ``alpha-engine-alerts`` SNS topic, firing
        a real ``[ERROR] test: boom`` email to the operator every time
        ``pytest tests/test_alerts.py`` ran. Same defect class as the
        2026-05-13 1015 USD cost-spike and the cost_report storm Brian
        flagged earlier this session — test fixtures leaking into
        production alert channels. The test still verifies the
        never-raises contract, just with explicit boto3 + telegram
        stubs that simulate the "both channels fail" case.
        """
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("simulated SNS unreachable")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=False, detail="simulated"),
            ):
                result = alerts.publish(
                    "boom", source="test", sns_topic_arn=None,
                    raise_on_total_failure=False,
                )
        # Structured result returned despite both channels failing.
        assert isinstance(result, alerts.PublishResult)
        assert result.any_ok is False
        assert result.all_ok is False


class TestCli:
    def test_publish_subcommand_calls_publish(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                rc = alerts.main([
                    "publish",
                    "--message", "boom",
                    "--severity", "error",
                    "--source", "spot_backtest.sh",
                ])
        assert rc == 0
        assert sns.publish.called

    def test_exit_code_1_when_both_channels_fail(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("nope")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=False, detail="creds missing")):
                rc = alerts.main([
                    "publish",
                    "--message", "boom",
                ])
        assert rc == 1

    def test_exit_code_0_when_only_one_channel_ok(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("nope")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                rc = alerts.main([
                    "publish",
                    "--message", "boom",
                ])
        assert rc == 0

    def test_no_sns_flag(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                rc = alerts.main(["publish", "--message", "x", "--no-sns"])
        sns.publish.assert_not_called()
        assert rc == 0

    def test_no_telegram_flag(self, fake_boto3):
        fake, _sts, _sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram") as tg:
                rc = alerts.main(["publish", "--message", "x", "--no-telegram"])
        tg.assert_not_called()
        assert rc == 0

    def test_custom_sns_topic_arn(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        custom = "arn:aws:sns:us-west-2:000000000000:custom"
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True, detail="sent")):
                alerts.main(["publish", "--message", "x", "--sns-topic-arn", custom])
        assert sns.publish.call_args.kwargs["TopicArn"] == custom

    def test_missing_message_arg_fails(self):
        with pytest.raises(SystemExit):
            alerts.main(["publish"])


# ─── Dedup (v0.24.0) ─────────────────────────────────────────────────────────


@pytest.fixture
def fake_boto3_with_s3():
    """boto3 stub extending fake_boto3 with an S3 client + in-memory key store.

    Returns ``(fake, sts, sns, s3, store)`` where ``store`` is a dict
    mapping S3 keys → JSON bodies; tests can pre-populate to simulate
    existing markers + read it back after writes.
    """
    from botocore.exceptions import ClientError

    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "711398986525"}
    sns_client = MagicMock()
    sns_client.publish.return_value = {"MessageId": "test-msg-id-abc123"}

    s3_client = MagicMock()
    store: dict[str, bytes] = {}

    def _get_object(*, Bucket, Key):
        if Key not in store:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "absent"}},
                "GetObject",
            )
        body = MagicMock()
        body.read.return_value = store[Key]
        return {"Body": body}

    def _put_object(*, Bucket, Key, Body, ContentType=None):
        store[Key] = Body if isinstance(Body, bytes) else Body.encode()
        return {"ETag": '"deadbeef"'}

    s3_client.get_object.side_effect = _get_object
    s3_client.put_object.side_effect = _put_object

    fake = MagicMock()

    def _client(service: str, **kwargs):
        if service == "sts":
            return sts_client
        if service == "sns":
            return sns_client
        if service == "s3":
            return s3_client
        raise AssertionError(f"unexpected boto3 client request: {service}")

    fake.client.side_effect = _client
    return fake, sts_client, sns_client, s3_client, store


class TestDedupMarkerKey:
    """``_dedup_marker_key`` is the deterministic S3 key derivation."""

    def test_deterministic_for_same_input(self):
        a = alerts._dedup_marker_key("cost-anomaly-2026-05-09-abc1234")
        b = alerts._dedup_marker_key("cost-anomaly-2026-05-09-abc1234")
        assert a == b

    def test_different_inputs_yield_different_keys(self):
        a = alerts._dedup_marker_key("k1")
        b = alerts._dedup_marker_key("k2")
        assert a != b

    def test_key_format(self):
        k = alerts._dedup_marker_key("anything")
        assert k.startswith(f"{alerts.DEDUP_MARKER_PREFIX}/")
        assert k.endswith(".json")
        # Hashed segment is 16 hex chars
        stem = k.split("/")[-1].removesuffix(".json")
        assert len(stem) == 16
        assert all(c in "0123456789abcdef" for c in stem)

    def test_long_dedup_key_does_not_blow_up_path(self):
        # Even a 10 KB dedup_key produces a fixed-width 16-char hash.
        long_input = "x" * 10240
        k = alerts._dedup_marker_key(long_input)
        assert len(k.split("/")[-1]) == len("XXXXXXXXXXXXXXXX.json")


class TestCheckDedupMarker:
    """Marker check is fail-safe: any uncertainty → ``False`` so caller publishes."""

    def test_nosuchkey_returns_false_with_no_marker_reason(self, fake_boto3_with_s3):
        fake, *_ = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = alerts._check_dedup_marker(
                "alpha-engine-research",
                alerts._dedup_marker_key("never-published"),
                dedup_window_min=60,
            )
        assert within is False
        assert reason == "no marker"

    def test_marker_within_window_returns_true(self, fake_boto3_with_s3):
        from datetime import datetime, timezone

        fake, _sts, _sns, _s3, store = fake_boto3_with_s3
        marker_key = alerts._dedup_marker_key("test-key")
        # Published 5 minutes ago + 60-minute window ⇒ within
        five_min_ago = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        import json as _json
        store[marker_key] = _json.dumps({
            "dedup_key": "test-key",
            "first_published_at": five_min_ago,
            "last_published_at": five_min_ago,
            "publish_count": 1,
        }).encode()
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = alerts._check_dedup_marker(
                "alpha-engine-research", marker_key, dedup_window_min=60,
            )
        assert within is True
        assert "within 60min window" in reason

    def test_marker_expired_returns_false(self, fake_boto3_with_s3):
        from datetime import datetime, timedelta, timezone

        fake, _sts, _sns, _s3, store = fake_boto3_with_s3
        marker_key = alerts._dedup_marker_key("test-key")
        # Published 90 minutes ago + 60-minute window ⇒ expired
        ninety_min_ago = (
            datetime.now(timezone.utc) - timedelta(minutes=90)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        import json as _json
        store[marker_key] = _json.dumps({
            "dedup_key": "test-key",
            "first_published_at": ninety_min_ago,
            "last_published_at": ninety_min_ago,
            "publish_count": 1,
        }).encode()
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = alerts._check_dedup_marker(
                "alpha-engine-research", marker_key, dedup_window_min=60,
            )
        assert within is False
        assert "marker expired" in reason

    def test_window_none_means_forever(self, fake_boto3_with_s3):
        from datetime import datetime, timedelta, timezone

        fake, _sts, _sns, _s3, store = fake_boto3_with_s3
        marker_key = alerts._dedup_marker_key("test-key")
        # Published 30 days ago + window=None ⇒ still suppressed
        long_ago = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        import json as _json
        store[marker_key] = _json.dumps({
            "dedup_key": "test-key",
            "first_published_at": long_ago,
            "last_published_at": long_ago,
            "publish_count": 1,
        }).encode()
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = alerts._check_dedup_marker(
                "alpha-engine-research", marker_key, dedup_window_min=None,
            )
        assert within is True
        assert "forever" in reason

    def test_clienterror_fails_safe_to_publish(self):
        """Transient S3 error other than NoSuchKey → publish anyway."""
        from botocore.exceptions import ClientError

        fake = MagicMock()
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "transient"}},
            "GetObject",
        )
        fake.client.return_value = s3
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = alerts._check_dedup_marker(
                "alpha-engine-research",
                alerts._dedup_marker_key("k"),
                dedup_window_min=60,
            )
        assert within is False
        assert "marker check error" in reason

    def test_corrupt_marker_falls_safe_to_publish(self, fake_boto3_with_s3):
        fake, _sts, _sns, _s3, store = fake_boto3_with_s3
        marker_key = alerts._dedup_marker_key("test-key")
        store[marker_key] = b"{ not json"
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = alerts._check_dedup_marker(
                "alpha-engine-research", marker_key, dedup_window_min=60,
            )
        assert within is False
        assert "marker parse error" in reason


class TestWriteDedupMarker:
    """Marker write is read-modify-write: ``first_published_at`` is stable."""

    def test_first_write_creates_count_1(self, fake_boto3_with_s3):
        fake, _sts, _sns, _s3, store = fake_boto3_with_s3
        marker_key = alerts._dedup_marker_key("fresh")
        with patch.dict("sys.modules", {"boto3": fake}):
            alerts._write_dedup_marker(
                "alpha-engine-research", marker_key,
                dedup_key="fresh", formatted_message="[ERROR] x: boom",
            )
        import json as _json
        payload = _json.loads(store[marker_key])
        assert payload["publish_count"] == 1
        assert payload["dedup_key"] == "fresh"
        assert payload["first_published_at"] == payload["last_published_at"]
        assert payload["message_preview"] == "[ERROR] x: boom"

    def test_second_write_increments_count_preserves_first_published_at(
        self, fake_boto3_with_s3,
    ):
        import json as _json

        fake, _sts, _sns, _s3, store = fake_boto3_with_s3
        marker_key = alerts._dedup_marker_key("recur")
        with patch.dict("sys.modules", {"boto3": fake}):
            alerts._write_dedup_marker(
                "alpha-engine-research", marker_key,
                dedup_key="recur", formatted_message="msg1",
            )
            first_payload = _json.loads(store[marker_key])
            first_published_at = first_payload["first_published_at"]

            # Second write — simulate elapsed time (just rewrite same call)
            alerts._write_dedup_marker(
                "alpha-engine-research", marker_key,
                dedup_key="recur", formatted_message="msg2",
            )
        second_payload = _json.loads(store[marker_key])
        assert second_payload["publish_count"] == 2
        assert second_payload["first_published_at"] == first_published_at
        assert second_payload["message_preview"] == "msg2"

    def test_write_failure_swallowed_does_not_raise(self):
        fake = MagicMock()
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("RMW read failed")
        s3.put_object.side_effect = Exception("AccessDenied")
        fake.client.return_value = s3
        with patch.dict("sys.modules", {"boto3": fake}):
            # Must not raise.
            alerts._write_dedup_marker(
                "alpha-engine-research", "_alerts/_dedup/abc.json",
                dedup_key="k", formatted_message="x",
            )


class TestPublishWithDedup:
    """End-to-end: ``publish(dedup_key=...)`` suppresses repeats within window."""

    def test_first_publish_fires_and_writes_marker(self, fake_boto3_with_s3):
        fake, _sts, sns, _s3, store = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                result = alerts.publish(
                    "anomaly",
                    severity="error",
                    source="cost_report.py",
                    dedup_key="cost-anomaly-2026-05-09-abc1234",
                )
        assert result.dedup_skipped is False
        assert result.any_ok is True
        assert sns.publish.call_count == 1
        # Marker landed in S3.
        marker_key = alerts._dedup_marker_key("cost-anomaly-2026-05-09-abc1234")
        assert marker_key in store

    def test_second_publish_within_window_is_suppressed(self, fake_boto3_with_s3):
        fake, _sts, sns, _s3, _store = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                # First call publishes.
                alerts.publish(
                    "anomaly", source="cost_report.py",
                    dedup_key="recur-key",
                )
                # Reset sns spy so second-call assertions are clean.
                sns.publish.reset_mock()
                # Second call within window suppresses.
                result = alerts.publish(
                    "anomaly", source="cost_report.py",
                    dedup_key="recur-key",
                )
        assert result.dedup_skipped is True
        assert "within 60min window" in result.dedup_reason
        assert result.any_ok is True  # treats suppressed as success
        sns.publish.assert_not_called()

    def test_expired_window_allows_fresh_publish(self, fake_boto3_with_s3):
        """A marker older than the window should allow a fresh publish.

        We simulate "expired" by pre-populating a marker with an
        old timestamp, then calling publish with a 60min window.
        """
        from datetime import datetime, timedelta, timezone
        import json as _json

        fake, _sts, sns, _s3, store = fake_boto3_with_s3
        marker_key = alerts._dedup_marker_key("expired-key")
        ninety_min_ago = (
            datetime.now(timezone.utc) - timedelta(minutes=90)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        store[marker_key] = _json.dumps({
            "dedup_key": "expired-key",
            "first_published_at": ninety_min_ago,
            "last_published_at": ninety_min_ago,
            "publish_count": 1,
        }).encode()

        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                result = alerts.publish(
                    "anomaly", source="cost_report.py",
                    dedup_key="expired-key", dedup_window_min=60,
                )
        assert result.dedup_skipped is False
        assert sns.publish.call_count == 1
        # publish_count incremented + first_published_at preserved
        payload = _json.loads(store[marker_key])
        assert payload["publish_count"] == 2
        assert payload["first_published_at"] == ninety_min_ago

    def test_dedup_key_none_disables_dedup(self, fake_boto3_with_s3):
        """``dedup_key=None`` is the legacy path — marker never touched."""
        fake, _sts, sns, s3, _store = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                result = alerts.publish("anomaly", source="x")  # no dedup_key
        assert result.dedup_skipped is False
        # No S3 marker activity — neither get_object nor put_object was called.
        s3.get_object.assert_not_called()
        s3.put_object.assert_not_called()

    def test_failed_publish_does_not_write_marker(self, fake_boto3_with_s3):
        """A publish that failed in both channels MUST NOT latch out
        future retries by writing a marker."""
        fake, _sts, sns, s3, store = fake_boto3_with_s3
        sns.publish.side_effect = RuntimeError("sns down")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=False, detail="creds"),
            ):
                result = alerts.publish(
                    "anomaly", source="x", dedup_key="no-marker-on-fail",
                    # I9209: the default now raises on total non-delivery.
                    # This test is about the MARKER, not the raise — opting
                    # out keeps it testing the one property it names.
                    raise_on_total_failure=False,
                )
        assert result.any_ok is False
        marker_key = alerts._dedup_marker_key("no-marker-on-fail")
        assert marker_key not in store

    def test_window_none_publishes_once_then_suppresses_indefinitely(
        self, fake_boto3_with_s3,
    ):
        fake, _sts, sns, _s3, _store = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                alerts.publish(
                    "x", source="y", dedup_key="forever", dedup_window_min=None,
                )
                sns.publish.reset_mock()
                result = alerts.publish(
                    "x", source="y", dedup_key="forever", dedup_window_min=None,
                )
        assert result.dedup_skipped is True
        assert "forever" in result.dedup_reason
        sns.publish.assert_not_called()


class TestCliDedup:
    """CLI flag wiring for the new dedup params."""

    def test_dedup_key_flag_passes_through(self, fake_boto3_with_s3):
        fake, _sts, sns, _s3, store = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                rc = alerts.main([
                    "publish", "--message", "x", "--severity", "error",
                    "--dedup-key", "canary-rollback-2026052116",
                ])
        assert rc == 0
        marker_key = alerts._dedup_marker_key("canary-rollback-2026052116")
        assert marker_key in store

    def test_dedup_window_min_zero_maps_to_none(self, fake_boto3_with_s3):
        """CLI convention: --dedup-window-min 0 = forever (Python ``None``)."""
        from datetime import datetime, timedelta, timezone
        import json as _json

        fake, _sts, sns, _s3, store = fake_boto3_with_s3
        # Pre-populate a 30-day-old marker; with --dedup-window-min 0 it
        # should still suppress (forever).
        marker_key = alerts._dedup_marker_key("k")
        long_ago = (
            datetime.now(timezone.utc) - timedelta(days=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        store[marker_key] = _json.dumps({
            "dedup_key": "k",
            "first_published_at": long_ago,
            "last_published_at": long_ago,
            "publish_count": 1,
        }).encode()
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                rc = alerts.main([
                    "publish", "--message", "x",
                    "--dedup-key", "k",
                    "--dedup-window-min", "0",
                ])
        # Exit 0 because dedup_skipped → any_ok=True
        assert rc == 0
        sns.publish.assert_not_called()

    def test_dedup_skipped_stderr_message(self, fake_boto3_with_s3, capsys):
        fake, _sts, _sns, _s3, _store = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                # First call publishes + writes marker.
                alerts.main([
                    "publish", "--message", "x", "--dedup-key", "k",
                ])
                capsys.readouterr()  # drain
                # Second call within window suppresses.
                rc = alerts.main([
                    "publish", "--message", "x", "--dedup-key", "k",
                ])
        captured = capsys.readouterr()
        assert rc == 0
        assert "dedup_skipped=True" in captured.err


class TestTestEnvGuard:
    """The ``PYTEST_CURRENT_TEST`` guard short-circuits real fan-out from
    inside any test process unless ``ALPHA_ENGINE_ALLOW_TEST_ALERTS`` is set
    (L4566). This is the cross-repo chokepoint that stops a consumer suite
    (e.g. alpha-engine's optimizer-shadow tests) from paging the operator."""

    def test_suppressed_in_test_env_without_optin(self, monkeypatch):
        # The autouse conftest sets the escape hatch for the whole lib suite;
        # remove it here to observe the guard's default-on behaviour.
        monkeypatch.delenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", raising=False)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_guard (call)")
        # If the guard fails to short-circuit, these explode the test rather
        # than silently reaching a real channel.
        boom_sns = MagicMock(side_effect=AssertionError("SNS reached in test env!"))
        boom_tg = MagicMock(side_effect=AssertionError("Telegram reached in test env!"))
        monkeypatch.setattr(alerts, "_publish_sns", boom_sns)
        monkeypatch.setattr(alerts, "_publish_telegram", boom_tg)

        result = alerts.publish("boom", source="x", sns=True, telegram=True)

        assert result.sns.ok is False
        assert result.telegram.ok is False
        assert "suppressed in test env" in result.sns.detail
        assert result.any_ok is False
        boom_sns.assert_not_called()
        boom_tg.assert_not_called()

    def test_optin_escape_hatch_re_enables_fanout(self, monkeypatch, fake_boto3):
        fake, _sts, _sns = fake_boto3
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                result = alerts.publish("boom", source="x")
        # Guard did NOT short-circuit — the mocked transports ran.
        assert result.any_ok is True


class TestDryRun:
    """``dry_run=True`` (config-I6759) — verify a call site's argument
    shape without sending anything. Motivated by PR165: a delivery
    verification call site paged Brian with a synthetic ERROR because
    ``publish()`` previously only suppressed fan-out under
    ``PYTEST_CURRENT_TEST``."""

    def test_never_constructs_boto3_client(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        boom = MagicMock(side_effect=AssertionError("boto3.client() reached in dry-run!"))
        fake_boto3_module = MagicMock()
        fake_boto3_module.client = boom
        with patch.dict("sys.modules", {"boto3": fake_boto3_module}):
            result = alerts.publish("boom", source="x", dry_run=True)
        boom.assert_not_called()
        assert result.sns.ok is True
        assert result.telegram.ok is True

    def test_never_touches_telegram_transport(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        with patch.object(
            alerts, "_publish_telegram",
            side_effect=AssertionError("_publish_telegram reached in dry-run!"),
        ) as tg:
            result = alerts.publish("boom", source="x", dry_run=True)
        tg.assert_not_called()
        assert result.telegram.ok is True
        assert result.telegram.detail == "dry-run: would send"

    def test_never_calls_publish_sns(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        with patch.object(
            alerts, "_publish_sns",
            side_effect=AssertionError("_publish_sns reached in dry-run!"),
        ) as sns_fn:
            result = alerts.publish("boom", source="x", dry_run=True)
        sns_fn.assert_not_called()
        assert result.sns.ok is True
        assert result.sns.detail == "dry-run: would send"

    def test_writes_no_dedup_marker(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        with patch.object(
            alerts, "_write_dedup_marker",
            side_effect=AssertionError("_write_dedup_marker reached in dry-run!"),
        ) as write_marker:
            result = alerts.publish(
                "boom", source="x", dedup_key="dry-run-key", dry_run=True,
            )
        write_marker.assert_not_called()
        assert result.dedup_skipped is False
        assert result.sns.ok is True
        assert result.telegram.ok is True

    def test_emits_no_overseer_intake_event(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        from krepis import fleet_events

        with patch.object(
            fleet_events, "emit_alert_event",
            side_effect=AssertionError("emit_alert_event reached in dry-run!"),
        ) as emit:
            alerts.publish("boom", source="x", dry_run=True)
        emit.assert_not_called()

    def test_dry_run_result_shape(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        result = alerts.publish("boom", severity="error", source="x", dry_run=True)
        assert isinstance(result, alerts.PublishResult)
        assert result.sns == alerts.ChannelResult(ok=True, detail="dry-run: would send")
        assert result.telegram == alerts.ChannelResult(ok=True, detail="dry-run: would send")
        assert result.any_ok is True
        assert result.all_ok is True
        assert result.dedup_skipped is False

    def test_dry_run_respects_channel_disable(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        result = alerts.publish("boom", source="x", sns=False, telegram=False, dry_run=True)
        assert result.sns.ok is True
        assert "sns disabled" in result.sns.detail
        assert result.telegram.ok is True
        assert "telegram disabled" in result.telegram.detail

    def test_cli_dry_run_flag_returns_0(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        boom = MagicMock(side_effect=AssertionError("boto3.client() reached in dry-run!"))
        fake_boto3_module = MagicMock()
        fake_boto3_module.client = boom
        with patch.dict("sys.modules", {"boto3": fake_boto3_module}):
            with patch.object(
                alerts, "_publish_telegram",
                side_effect=AssertionError("_publish_telegram reached in dry-run!"),
            ):
                rc = alerts.main([
                    "publish", "--dry-run",
                    "--message", "x", "--severity", "error", "--source", "y",
                ])
        assert rc == 0
        boom.assert_not_called()


# ─── Source-mute (v0.57.0) ───────────────────────────────────────────────────


@pytest.fixture
def fake_boto3_with_ssm():
    """boto3 stub extending fake_boto3 with an SSM client whose
    ``get_parameter`` return value tests control via ``_ssm_value``."""
    sts_client = MagicMock()
    sts_client.get_caller_identity.return_value = {"Account": "711398986525"}
    sns_client = MagicMock()
    sns_client.publish.return_value = {"MessageId": "test-msg-id-abc123"}
    ssm_client = MagicMock()

    fake = MagicMock()

    def _client(service: str, **kwargs):
        if service == "sts":
            return sts_client
        if service == "sns":
            return sns_client
        if service == "ssm":
            return ssm_client
        raise AssertionError(f"unexpected boto3 client request: {service}")

    fake.client.side_effect = _client
    return fake, sts_client, sns_client, ssm_client


def _ssm_value(ssm_client, value: str) -> None:
    ssm_client.get_parameter.return_value = {"Parameter": {"Value": value}}


class TestFetchSourceMutes:
    def test_missing_parameter_returns_empty(self, fake_boto3_with_ssm):
        from botocore.exceptions import ClientError

        fake, _sts, _sns, ssm = fake_boto3_with_ssm
        ssm.get_parameter.side_effect = ClientError(
            {"Error": {"Code": "ParameterNotFound", "Message": "absent"}},
            "GetParameter",
        )
        with patch.dict("sys.modules", {"boto3": fake}):
            assert alerts._fetch_source_mutes(alerts.DEFAULT_MUTE_SSM_PARAM) == []

    def test_malformed_json_fails_open(self, fake_boto3_with_ssm):
        fake, _sts, _sns, ssm = fake_boto3_with_ssm
        _ssm_value(ssm, "not json")
        with patch.dict("sys.modules", {"boto3": fake}):
            assert alerts._fetch_source_mutes(alerts.DEFAULT_MUTE_SSM_PARAM) == []

    def test_non_list_json_fails_open(self, fake_boto3_with_ssm):
        fake, _sts, _sns, ssm = fake_boto3_with_ssm
        _ssm_value(ssm, '{"not": "a list"}')
        with patch.dict("sys.modules", {"boto3": fake}):
            assert alerts._fetch_source_mutes(alerts.DEFAULT_MUTE_SSM_PARAM) == []

    def test_valid_list_parsed(self, fake_boto3_with_ssm):
        fake, _sts, _sns, ssm = fake_boto3_with_ssm
        _ssm_value(
            ssm,
            '[{"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z"}]',
        )
        with patch.dict("sys.modules", {"boto3": fake}):
            entries = alerts._fetch_source_mutes(alerts.DEFAULT_MUTE_SSM_PARAM)
        assert entries == [
            {"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z"}
        ]

    def test_boto3_unavailable_fails_open(self, monkeypatch):
        with patch.dict("sys.modules", {"boto3": None}):
            assert alerts._fetch_source_mutes(alerts.DEFAULT_MUTE_SSM_PARAM) == []


class TestFindLiveMute:
    def test_matches_prefix_and_live(self):
        entries = [
            {
                "source_prefix": "metron",
                "expires_at": "2099-01-01T00:00:00Z",
                "reason": "x",
            }
        ]
        assert alerts._find_live_mute("metron/deploy", entries) == entries[0]

    def test_no_match_different_source(self):
        entries = [{"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z"}]
        assert alerts._find_live_mute("crucible-executor", entries) is None

    def test_expired_entry_does_not_match(self):
        entries = [{"source_prefix": "metron", "expires_at": "2000-01-01T00:00:00Z"}]
        assert alerts._find_live_mute("metron/deploy", entries) is None

    def test_missing_expires_at_does_not_match(self):
        entries = [{"source_prefix": "metron"}]
        assert alerts._find_live_mute("metron/deploy", entries) is None

    def test_unparseable_expires_at_does_not_match(self):
        entries = [{"source_prefix": "metron", "expires_at": "not-a-date"}]
        assert alerts._find_live_mute("metron/deploy", entries) is None

    def test_none_source_never_matches(self):
        entries = [{"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z"}]
        assert alerts._find_live_mute(None, entries) is None

    def test_non_dict_entries_skipped(self):
        assert alerts._find_live_mute("metron/deploy", ["not-a-dict"]) is None


class TestPublishSourceMute:
    def test_suppressed_when_source_matches_live_mute(self, fake_boto3_with_ssm):
        fake, _sts, sns, ssm = fake_boto3_with_ssm
        _ssm_value(
            ssm,
            '[{"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z", '
            '"reason": "focus on crucible"}]',
        )
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                side_effect=AssertionError("telegram reached despite live mute"),
            ):
                result = alerts.publish("deploy failed", source="metron/deploy")
        assert result.muted is True
        assert "metron" in result.mute_reason
        assert result.any_ok is True
        sns.publish.assert_not_called()

    def test_not_suppressed_when_mute_expired(self, fake_boto3_with_ssm):
        fake, _sts, sns, ssm = fake_boto3_with_ssm
        _ssm_value(ssm, '[{"source_prefix": "metron", "expires_at": "2000-01-01T00:00:00Z"}]')
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                result = alerts.publish("deploy failed", source="metron/deploy")
        assert result.muted is False
        sns.publish.assert_called_once()

    def test_not_suppressed_for_non_matching_source(self, fake_boto3_with_ssm):
        fake, _sts, sns, ssm = fake_boto3_with_ssm
        _ssm_value(ssm, '[{"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z"}]')
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                result = alerts.publish("exec failed", source="crucible-executor")
        assert result.muted is False
        sns.publish.assert_called_once()

    def test_no_mute_list_does_not_suppress(self, fake_boto3_with_ssm):
        from botocore.exceptions import ClientError

        fake, _sts, sns, ssm = fake_boto3_with_ssm
        ssm.get_parameter.side_effect = ClientError(
            {"Error": {"Code": "ParameterNotFound", "Message": "absent"}},
            "GetParameter",
        )
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(
                alerts, "_publish_telegram",
                return_value=alerts.ChannelResult(ok=True, detail="sent"),
            ):
                result = alerts.publish("deploy failed", source="metron/deploy")
        assert result.muted is False
        sns.publish.assert_called_once()

    def test_mute_checked_before_dedup(self, fake_boto3_with_ssm):
        """A muted source must not reach the (S3) dedup marker check —
        the fixture only registers sts/sns/ssm clients, so an
        unexpected boto3.client('s3') call raises."""
        fake, _sts, sns, ssm = fake_boto3_with_ssm
        _ssm_value(ssm, '[{"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z"}]')
        with patch.dict("sys.modules", {"boto3": fake}):
            result = alerts.publish(
                "deploy failed", source="metron/deploy",
                dedup_key="metron-deploy-failed",
            )
        assert result.muted is True
        assert result.dedup_skipped is False
        ssm.get_parameter.assert_called_once()

    def test_cli_stderr_reports_muted(self, fake_boto3_with_ssm, capsys):
        fake, _sts, sns, ssm = fake_boto3_with_ssm
        _ssm_value(ssm, '[{"source_prefix": "metron", "expires_at": "2099-01-01T00:00:00Z"}]')
        with patch.dict("sys.modules", {"boto3": fake}):
            rc = alerts.main([
                "publish", "--message", "x", "--source", "metron/deploy",
            ])
        captured = capsys.readouterr()
        assert rc == 0
        assert "muted=True" in captured.err


# ── Condition lifecycle: the open/clear pair (alpha-engine-config-I8105) ─────
#
# Before this, every publisher here was write-once: it emitted on detection and
# emitted nothing when the condition ended, so a page and a live outage were
# indistinguishable downstream. These tests pin the three things that make a
# clear usable rather than decorative — it is PAIRABLE (identity_key), it is
# MACHINE-READABLE (state on the event), and it NEVER PUSHES.


class TestDiffConditions:
    def test_three_way_split(self):
        opened, still_open, cleared = alerts.diff_conditions(
            ["a", "b"], ["b", "c"]
        )
        assert opened == ["c"]
        assert still_open == ["b"]
        assert cleared == ["a"]

    def test_empty_previous_opens_everything(self):
        assert alerts.diff_conditions([], ["x", "y"]) == (["x", "y"], [], [])

    def test_empty_current_clears_everything(self):
        assert alerts.diff_conditions(["x", "y"], []) == ([], [], ["x", "y"])

    def test_both_empty(self):
        assert alerts.diff_conditions([], []) == ([], [], [])

    def test_sorted_for_stable_emission_order(self):
        opened, _still, cleared = alerts.diff_conditions(["z", "m"], ["q", "b"])
        assert opened == ["b", "q"]
        assert cleared == ["m", "z"]

    def test_direction_is_not_symmetric(self):
        # The regression this guards: a call site reading the difference
        # backwards emits a clear for a condition that just STARTED.
        _opened, _still, cleared = alerts.diff_conditions(["was"], ["now"])
        assert cleared == ["was"]


class TestFormatMessageState:
    def test_cleared_carries_visible_marker(self):
        out = alerts._format_message("disk high", "info", "box-health", "cleared")
        assert out == "[INFO] box-health: RESOLVED — disk high"

    def test_opened_renders_exactly_as_before(self):
        assert (
            alerts._format_message("boom", "error", "src", "opened")
            == "[ERROR] src: boom"
        )

    def test_still_open_renders_exactly_as_before(self):
        assert (
            alerts._format_message("boom", "error", "src", "still_open")
            == "[ERROR] src: boom"
        )

    def test_default_state_is_opened(self):
        assert alerts._format_message("boom", "error", "src") == "[ERROR] src: boom"


class TestPublishState:
    def test_unknown_state_raises(self, fake_boto3):
        with pytest.raises(ValueError, match="unknown state"):
            alerts.publish("boom", state="resolved")

    def test_state_and_identity_reach_the_event(self, fake_boto3, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        fake, _sts, _sns = fake_boto3
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return True

        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_fetch_source_mutes", return_value=[]):
                with patch.object(
                    alerts, "_publish_telegram",
                    return_value=alerts.ChannelResult(ok=True, detail="sent"),
                ):
                    with patch.object(alerts.fleet_events, "emit_alert_event", _capture):
                        result = alerts.publish(
                            "boom", source="box-health",
                            state="still_open", identity_key="unit-x-1234",
                        )
        assert captured["state"] == "still_open"
        assert captured["identity_key"] == "unit-x-1234"
        assert result.state == "still_open"
        assert result.identity_key == "unit-x-1234"

    def test_identity_key_defaults_to_dedup_key(self, fake_boto3, monkeypatch):
        # Pre-existing dedup-keyed publishers become pairable without touching
        # their call sites.
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        fake, _sts, _sns = fake_boto3
        captured = {}

        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_fetch_source_mutes", return_value=[]):
                with patch.object(alerts, "_check_dedup_marker", return_value=(False, "")):
                    with patch.object(alerts, "_write_dedup_marker"):
                        with patch.object(
                            alerts, "_publish_telegram",
                            return_value=alerts.ChannelResult(ok=True, detail="sent"),
                        ):
                            with patch.object(
                                alerts.fleet_events, "emit_alert_event",
                                lambda **kw: captured.update(kw),
                            ):
                                result = alerts.publish("boom", dedup_key="dk-1")
        assert captured["identity_key"] == "dk-1"
        assert result.identity_key == "dk-1"

    def test_default_state_is_opened_on_the_event(self, fake_boto3, monkeypatch):
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        fake, _sts, _sns = fake_boto3
        captured = {}
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_fetch_source_mutes", return_value=[]):
                with patch.object(
                    alerts, "_publish_telegram",
                    return_value=alerts.ChannelResult(ok=True, detail="sent"),
                ):
                    with patch.object(
                        alerts.fleet_events, "emit_alert_event",
                        lambda **kw: captured.update(kw),
                    ):
                        alerts.publish("boom")
        assert captured["state"] == "opened"


class TestPublishTelegramSilent:
    def test_silent_true_forces_no_push_at_pushing_severity(self):
        sent = {}

        def _send(msg, disable_notification=False, **kwargs):
            # `**kwargs` absorbs the destination overrides (`chat_id`,
            # `message_thread_id`) the routing tier passes; these cases pin
            # the BUZZ decision, which routing must not disturb.
            sent["disable_notification"] = disable_notification
            sent["chat_id"] = kwargs.get("chat_id")
            return True

        with patch.dict(
            "sys.modules",
            {"krepis.telegram": MagicMock(send_message=_send)},
        ):
            alerts._publish_telegram("m", severity="critical", silent=True)
        assert sent["disable_notification"] is True

    def test_silent_none_keeps_severity_default(self):
        sent = {}

        def _send(msg, disable_notification=False, **kwargs):
            # `**kwargs` absorbs the destination overrides (`chat_id`,
            # `message_thread_id`) the routing tier passes; these cases pin
            # the BUZZ decision, which routing must not disturb.
            sent["disable_notification"] = disable_notification
            sent["chat_id"] = kwargs.get("chat_id")
            return True

        with patch.dict(
            "sys.modules",
            {"krepis.telegram": MagicMock(send_message=_send)},
        ):
            alerts._publish_telegram("m", severity="critical", silent=None)
        assert sent["disable_notification"] is False

    def test_silent_false_forces_push_at_non_pushing_severity(self):
        sent = {}

        def _send(msg, disable_notification=False, **kwargs):
            # `**kwargs` absorbs the destination overrides (`chat_id`,
            # `message_thread_id`) the routing tier passes; these cases pin
            # the BUZZ decision, which routing must not disturb.
            sent["disable_notification"] = disable_notification
            sent["chat_id"] = kwargs.get("chat_id")
            return True

        with patch.dict(
            "sys.modules",
            {"krepis.telegram": MagicMock(send_message=_send)},
        ):
            alerts._publish_telegram("m", severity="info", silent=False)
        # `silent=False` is an EXPLICIT push override, symmetric with
        # `silent=True`, not a synonym for `None`. A daily digest that must be
        # seen is a legitimate `info`-severity push; severity stays the
        # routing key, `silent` is the push key (alpha-engine-config-I9916 —
        # crucible's accountability report passed `silent=False` and arrived
        # silent because this used to read `if silent:`).
        assert sent["disable_notification"] is False

    def test_silent_none_keeps_the_severity_default_at_non_pushing_severity(
        self,
    ):
        sent = {}

        def _send(msg, disable_notification=False, **kwargs):
            sent["disable_notification"] = disable_notification
            return True

        with patch.dict(
            "sys.modules",
            {"krepis.telegram": MagicMock(send_message=_send)},
        ):
            alerts._publish_telegram("m", severity="info", silent=None)
        assert sent["disable_notification"] is True


class TestPublishClear:
    def test_requires_identity_key(self):
        with pytest.raises(ValueError, match="identity_key is required"):
            alerts.publish_clear("all good", identity_key="")

    def test_clear_is_info_cleared_silent_and_undeduped(self, monkeypatch):
        captured = {}

        def _fake_publish(message, **kwargs):
            captured["message"] = message
            captured.update(kwargs)
            return alerts.PublishResult()

        monkeypatch.setattr(alerts, "publish", _fake_publish)
        alerts.publish_clear(
            "timer job failing: x", identity_key="key-1", source="box-health"
        )
        assert captured["severity"] == alerts.CLEAR_SEVERITY == "info"
        assert captured["state"] == "cleared"
        assert captured["identity_key"] == "key-1"
        assert captured["silent"] is True
        # The load-bearing one: a clear must NOT carry the page's dedup key,
        # or the page's own still-live marker swallows its terminator.
        assert captured["dedup_key"] is None

    def test_clear_never_pushes_even_if_info_starts_pushing(
        self, fake_boto3, monkeypatch
    ):
        # Guards the future change that would otherwise make every all-clear
        # in the fleet buzz a phone: widening SEVERITY_PHONE_PUSH.
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(
            alerts, "SEVERITY_PHONE_PUSH",
            frozenset({"info", "warning", "error", "critical"}),
        )
        fake, _sts, _sns = fake_boto3
        sent = {}

        def _send(msg, disable_notification=False, **kwargs):
            sent["disable_notification"] = disable_notification
            sent["text"] = msg
            sent["chat_id"] = kwargs.get("chat_id")
            return True

        with patch.dict("sys.modules", {"boto3": fake, "krepis.telegram": MagicMock(send_message=_send)}):
            with patch.object(alerts, "_fetch_source_mutes", return_value=[]):
                with patch.object(alerts.fleet_events, "emit_alert_event", lambda **kw: True):
                    alerts.publish_clear(
                        "disk high: root >=80% used",
                        identity_key="k", source="box-health",
                    )
        assert sent["disable_notification"] is True
        assert sent["text"].startswith("[INFO] box-health: RESOLVED — ")

    def test_clear_dry_run_sends_nothing_and_reports_state(self):
        result = alerts.publish_clear("x", identity_key="k", dry_run=True)
        assert result.any_ok is True
        assert result.state == "cleared"
        assert result.identity_key == "k"


class TestCliLifecycle:
    def test_clear_subcommand_dry_run_exits_zero(self, capsys):
        rc = alerts.main(
            ["clear", "--message", "ok now", "--identity-key", "k-1", "--dry-run"]
        )
        assert rc == 0
        assert "identity_key='k-1'" in capsys.readouterr().err

    def test_clear_subcommand_requires_identity_key(self):
        with pytest.raises(SystemExit):
            alerts.main(["clear", "--message", "ok now"])

    def test_publish_state_flag_rejects_unknown_value(self):
        with pytest.raises(SystemExit):
            alerts.main(["publish", "--message", "m", "--state", "resolved"])

    def test_publish_forwards_state_and_identity(self, monkeypatch):
        captured = {}

        def _fake_publish(message, **kwargs):
            captured.update(kwargs)
            return alerts.PublishResult(
                sns=alerts.ChannelResult(ok=True, detail="ok")
            )

        monkeypatch.setattr(alerts, "publish", _fake_publish)
        rc = alerts.main(
            [
                "publish", "--message", "m",
                "--state", "still_open", "--identity-key", "id-9",
            ]
        )
        assert rc == 0
        assert captured["state"] == "still_open"
        assert captured["identity_key"] == "id-9"


# ── Delivery DESTINATION tier (alpha-engine-config-I7857) ───────────────────
# These assert the DESTINATION and the delivery decision, never the message
# text. A test that pins the string cannot catch a wrong tier: the same body
# is correct in the operator chat and in the log chat, and the whole defect
# class this tier fixes is a correct message in the wrong channel.


class TestResolveDestination:
    """The pure decision function. No secrets, no transport."""

    def test_incident_severity_goes_to_the_operator_chat(self):
        for sev in sorted(alerts.SEVERITY_PHONE_PUSH):
            dest, _ = alerts.resolve_destination(sev, log_chat_id="-100777")
            assert dest == alerts.DESTINATION_OPERATOR_CHAT

    def test_incident_severity_is_not_diverted_by_a_console_artifact(self):
        # An artifact argument is evidence for a NON-incident finding. It must
        # never be a way to keep a real page out of the incident channel.
        dest, _ = alerts.resolve_destination(
            "critical", console_artifact="s3://console/x.json", log_chat_id="-100777"
        )
        assert dest == alerts.DESTINATION_OPERATOR_CHAT

    def test_non_incident_severity_goes_to_the_log_chat_when_configured(self):
        for sev in ("info", "warning", "notice"):
            dest, _ = alerts.resolve_destination(sev, log_chat_id="-100777")
            assert dest == alerts.DESTINATION_LOG_CHAT

    def test_console_artifact_used_only_when_no_log_chat(self):
        dest, _ = alerts.resolve_destination(
            "info", console_artifact="s3://console/x.json", log_chat_id=None
        )
        assert dest == alerts.DESTINATION_CONSOLE_ONLY

    def test_log_chat_wins_over_console_artifact(self):
        # Both configured is not ambiguous: a chat a human reads beats a
        # surface they must go and look at.
        dest, _ = alerts.resolve_destination(
            "info", console_artifact="s3://console/x.json", log_chat_id="-100777"
        )
        assert dest == alerts.DESTINATION_LOG_CHAT

    def test_neither_configured_falls_back_to_the_operator_chat(self, caplog):
        with caplog.at_level("WARNING"):
            dest, reason = alerts.resolve_destination("info")
        assert dest == alerts.DESTINATION_OPERATOR_CHAT
        assert reason.startswith("fallback")
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_explicit_operator_chat_override(self):
        dest, _ = alerts.resolve_destination(
            "info",
            destination=alerts.DESTINATION_OPERATOR_CHAT,
            log_chat_id="-100777",
        )
        assert dest == alerts.DESTINATION_OPERATOR_CHAT

    def test_explicit_log_chat_without_config_falls_back_loudly(self, caplog):
        with caplog.at_level("WARNING"):
            dest, reason = alerts.resolve_destination(
                "info", destination=alerts.DESTINATION_LOG_CHAT, log_chat_id=None
            )
        assert dest == alerts.DESTINATION_OPERATOR_CHAT
        assert reason.startswith("fallback")
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_explicit_console_only_without_evidence_falls_back_loudly(self, caplog):
        # console_artifact is the EVIDENCE, not a formality: without it,
        # console_only is indistinguishable from a drop.
        with caplog.at_level("WARNING"):
            dest, reason = alerts.resolve_destination(
                "info", destination=alerts.DESTINATION_CONSOLE_ONLY
            )
        assert dest == alerts.DESTINATION_OPERATOR_CHAT
        assert reason.startswith("fallback")
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_no_input_combination_resolves_to_nothing(self):
        # The invariant, exhaustively: every reachable combination names a
        # destination that DELIVERS. There is no "dropped" outcome to return.
        for sev in ("info", "warning", "error", "critical"):
            for dest_arg in (None, *alerts.ALERT_DESTINATIONS):
                for artifact in (None, "s3://console/x.json"):
                    for log_chat in (None, "-100777"):
                        dest, reason = alerts.resolve_destination(
                            sev,
                            destination=dest_arg,
                            console_artifact=artifact,
                            log_chat_id=log_chat,
                        )
                        assert dest in alerts.ALERT_DESTINATIONS
                        assert reason
                        if dest == alerts.DESTINATION_CONSOLE_ONLY:
                            assert artifact, (
                                "console_only without an artifact is a silent drop"
                            )
                        if dest == alerts.DESTINATION_LOG_CHAT:
                            assert log_chat

    def test_unknown_destination_raises(self):
        with pytest.raises(ValueError, match="unknown destination"):
            alerts.resolve_destination("info", destination="somewhere-else")


class TestResolveLogChat:
    """Secret resolution for the log destination."""

    def test_unset_secret_resolves_to_no_log_chat(self, monkeypatch):
        monkeypatch.setattr(
            "krepis.secrets.get_secret", lambda name, **kw: None
        )
        assert alerts._resolve_log_chat() == (None, None)

    def test_chat_and_thread_are_resolved(self, monkeypatch):
        values = {
            alerts.TELEGRAM_LOG_CHAT_PARAM: "-1001234",
            alerts.TELEGRAM_LOG_THREAD_PARAM: "42",
        }
        monkeypatch.setattr(
            "krepis.secrets.get_secret", lambda name, **kw: values.get(name)
        )
        assert alerts._resolve_log_chat() == ("-1001234", 42)

    def test_unparseable_thread_id_degrades_to_the_chat_not_to_nothing(
        self, monkeypatch, caplog
    ):
        values = {
            alerts.TELEGRAM_LOG_CHAT_PARAM: "-1001234",
            alerts.TELEGRAM_LOG_THREAD_PARAM: "general",
        }
        monkeypatch.setattr(
            "krepis.secrets.get_secret", lambda name, **kw: values.get(name)
        )
        with caplog.at_level("WARNING"):
            assert alerts._resolve_log_chat() == ("-1001234", None)
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_thread_lookup_failure_still_yields_the_chat(self, monkeypatch):
        # A readable chat id plus an unreadable topic id must degrade to
        # "the log chat, no topic", not to "no log chat" — the latter would
        # push routine traffic back into the incident channel.
        def _get(name, **kw):
            if name == alerts.TELEGRAM_LOG_CHAT_PARAM:
                return "-1001234"
            raise RuntimeError("ssm unreachable for the thread id")

        monkeypatch.setattr("krepis.secrets.get_secret", _get)
        assert alerts._resolve_log_chat() == ("-1001234", None)

    def test_secrets_backend_failure_is_not_fatal(self, monkeypatch):
        def _boom(name, **kw):
            raise RuntimeError("ssm unreachable")

        monkeypatch.setattr("krepis.secrets.get_secret", _boom)
        assert alerts._resolve_log_chat() == (None, None)


@pytest.fixture
def routed_publish(monkeypatch, fake_boto3):
    """Run a REAL ``alerts.publish`` with every transport captured.

    Returns ``(run, sends, sns_client)``. ``run(log_chat=..., **kwargs)``
    publishes with the log-chat config named by ``log_chat`` (a
    ``(chat_id, thread_id)`` tuple, i.e. what ``_resolve_log_chat`` would
    return) and appends one dict per Telegram send to ``sends`` — so an
    empty ``sends`` list is itself the assertion that no chat was touched.
    """
    fake, _sts, sns_client = fake_boto3
    monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *a, **kw: [])
    monkeypatch.setattr(alerts.fleet_events, "emit_alert_event", lambda **kw: True)
    sends: list[dict] = []

    def _send(
        text,
        disable_notification=False,
        bot_token=None,
        chat_id=None,
        message_thread_id=None,
    ):
        sends.append(
            {
                "text": text,
                "disable_notification": disable_notification,
                "chat_id": chat_id,
                "message_thread_id": message_thread_id,
            }
        )
        return True

    def _run(log_chat=(None, None), **kwargs):
        monkeypatch.setattr(alerts, "_resolve_log_chat", lambda: log_chat)
        with patch.dict(
            "sys.modules",
            {"boto3": fake, "krepis.telegram": MagicMock(send_message=_send)},
        ):
            return alerts.publish("boom", source="box-health", **kwargs)

    return _run, sends, sns_client


class TestPublishDestinationRouting:
    """End-to-end: which chat did the message actually go to?"""

    def test_error_reaches_the_operator_chat_with_the_buzz(self, routed_publish):
        run, sends, _sns = routed_publish
        result = run(log_chat=("-1001234", None), severity="error")
        assert result.telegram_destination == alerts.DESTINATION_OPERATOR_CHAT
        assert len(sends) == 1
        # chat_id=None means send_message resolves TELEGRAM_CHAT_ID — the
        # operator channel — rather than being handed the log override.
        assert sends[0]["chat_id"] is None
        assert sends[0]["disable_notification"] is False

    def test_critical_reaches_the_operator_chat_with_the_buzz(self, routed_publish):
        run, sends, _sns = routed_publish
        result = run(log_chat=("-1001234", None), severity="critical")
        assert result.telegram_destination == alerts.DESTINATION_OPERATOR_CHAT
        assert sends[0]["chat_id"] is None
        assert sends[0]["disable_notification"] is False

    def test_info_with_a_log_chat_goes_there_and_not_to_the_operator(
        self, routed_publish
    ):
        run, sends, _sns = routed_publish
        result = run(log_chat=("-1001234", 42), severity="info")
        assert result.telegram_destination == alerts.DESTINATION_LOG_CHAT
        assert len(sends) == 1
        assert sends[0]["chat_id"] == "-1001234"
        assert sends[0]["message_thread_id"] == 42
        assert "log_chat" in result.telegram.detail

    def test_warning_with_a_log_chat_goes_there_and_not_to_the_operator(
        self, routed_publish
    ):
        run, sends, _sns = routed_publish
        result = run(log_chat=("-1001234", None), severity="warning")
        assert result.telegram_destination == alerts.DESTINATION_LOG_CHAT
        assert sends[0]["chat_id"] == "-1001234"
        assert sends[0]["message_thread_id"] is None

    def test_console_artifact_and_no_log_chat_touches_neither_chat(
        self, routed_publish
    ):
        run, sends, _sns = routed_publish
        result = run(
            log_chat=(None, None),
            severity="info",
            console_artifact="s3://alpha-engine-console/fleet_checks/box-health.json",
        )
        assert result.telegram_destination == alerts.DESTINATION_CONSOLE_ONLY
        assert sends == []
        # Delivered, not failed: the finding is on the console and the SNS
        # record was written. A Bash caller's `|| echo failed` must not fire.
        assert result.telegram.ok is True
        assert result.any_ok is True
        assert "console_only" in result.telegram.detail
        assert "s3://alpha-engine-console/fleet_checks/box-health.json" in (
            result.telegram.detail
        )

    def test_console_only_still_pages_at_an_incident_severity(self, routed_publish):
        run, sends, _sns = routed_publish
        result = run(
            log_chat=(None, None),
            severity="error",
            console_artifact="s3://console/x.json",
        )
        assert result.telegram_destination == alerts.DESTINATION_OPERATOR_CHAT
        assert sends[0]["chat_id"] is None
        assert sends[0]["disable_notification"] is False

    def test_neither_configured_falls_back_to_the_operator_chat_loudly(
        self, routed_publish, caplog
    ):
        # THE invariant (alpha-engine-config-I7857). An unconfigured fleet
        # behaves exactly as it did before this tier existed — noisy, and
        # never silent.
        run, sends, _sns = routed_publish
        with caplog.at_level("WARNING"):
            result = run(log_chat=(None, None), severity="warning")
        assert result.telegram_destination == alerts.DESTINATION_OPERATOR_CHAT
        assert len(sends) == 1
        assert sends[0]["chat_id"] is None
        # Still no buzz — the buzz tier is unchanged by this work.
        assert sends[0]["disable_notification"] is True
        assert result.destination_reason.startswith("fallback")
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_explicit_destination_overrides_the_severity_default(
        self, routed_publish
    ):
        run, sends, _sns = routed_publish
        result = run(
            log_chat=("-1001234", None),
            severity="error",
            destination=alerts.DESTINATION_LOG_CHAT,
        )
        assert result.telegram_destination == alerts.DESTINATION_LOG_CHAT
        assert sends[0]["chat_id"] == "-1001234"

    def test_unknown_destination_raises_from_publish(self, routed_publish):
        run, _sends, _sns = routed_publish
        with pytest.raises(ValueError, match="unknown destination"):
            run(severity="info", destination="nowhere")

    def test_telegram_false_records_no_destination(self, routed_publish):
        run, sends, _sns = routed_publish
        result = run(log_chat=("-1001234", None), severity="info", telegram=False)
        assert sends == []
        assert result.telegram_destination is None


class TestSnsIsUnaffectedByRouting:
    """SNS is the durable record. Routing must not move, mute or reshape it."""

    def _sns_call(self, run, sns_client, **kwargs):
        sns_client.publish.reset_mock()
        run(**kwargs)
        assert sns_client.publish.call_count == 1
        return dict(sns_client.publish.call_args.kwargs)

    def test_identical_across_every_destination(self, routed_publish):
        run, _sends, sns_client = routed_publish
        to_log = self._sns_call(
            run, sns_client, log_chat=("-1001234", None), severity="warning"
        )
        to_console = self._sns_call(
            run,
            sns_client,
            log_chat=(None, None),
            severity="warning",
            console_artifact="s3://console/x.json",
        )
        fallback = self._sns_call(
            run, sns_client, log_chat=(None, None), severity="warning"
        )
        assert to_log == to_console == fallback

    def test_sns_still_publishes_when_telegram_is_console_only(self, routed_publish):
        run, _sends, sns_client = routed_publish
        result = run(
            log_chat=(None, None),
            severity="info",
            console_artifact="s3://console/x.json",
        )
        assert sns_client.publish.call_count == 1
        assert result.sns.ok is True

    def test_incident_and_non_incident_differ_only_by_the_severity_tag(
        self, routed_publish
    ):
        run, _sends, sns_client = routed_publish
        err = self._sns_call(
            run, sns_client, log_chat=("-1001234", None), severity="error"
        )
        warn = self._sns_call(
            run, sns_client, log_chat=("-1001234", None), severity="warning"
        )
        assert err["TopicArn"] == warn["TopicArn"]
        assert err["Message"].replace("[ERROR]", "") == warn["Message"].replace(
            "[WARNING]", ""
        )


class TestPublishClearRouting:
    """A recovery inherits the routing; it is not a special case."""

    def test_clear_goes_to_the_log_chat_when_configured(
        self, monkeypatch, fake_boto3
    ):
        fake, _sts, _sns = fake_boto3
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *a, **kw: [])
        monkeypatch.setattr(alerts.fleet_events, "emit_alert_event", lambda **kw: True)
        monkeypatch.setattr(alerts, "_resolve_log_chat", lambda: ("-1005555", 7))
        sends: list[dict] = []

        def _send(text, disable_notification=False, chat_id=None, message_thread_id=None, bot_token=None):
            sends.append({"chat_id": chat_id, "thread": message_thread_id,
                          "disable_notification": disable_notification})
            return True

        with patch.dict(
            "sys.modules",
            {"boto3": fake, "krepis.telegram": MagicMock(send_message=_send)},
        ):
            result = alerts.publish_clear(
                "disk high cleared", identity_key="k-1", source="box-health"
            )
        assert result.telegram_destination == alerts.DESTINATION_LOG_CHAT
        assert sends[0]["chat_id"] == "-1005555"
        assert sends[0]["thread"] == 7
        assert sends[0]["disable_notification"] is True

    def test_clear_with_a_console_artifact_touches_no_chat(
        self, monkeypatch, fake_boto3
    ):
        fake, _sts, sns_client = fake_boto3
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *a, **kw: [])
        monkeypatch.setattr(alerts.fleet_events, "emit_alert_event", lambda **kw: True)
        monkeypatch.setattr(alerts, "_resolve_log_chat", lambda: (None, None))
        sends: list[dict] = []

        def _send(text, **kw):
            sends.append(kw)
            return True

        with patch.dict(
            "sys.modules",
            {"boto3": fake, "krepis.telegram": MagicMock(send_message=_send)},
        ):
            result = alerts.publish_clear(
                "disk high cleared",
                identity_key="k-1",
                console_artifact="s3://console/box-health.json",
            )
        assert result.telegram_destination == alerts.DESTINATION_CONSOLE_ONLY
        assert sends == []
        # The durable record of the recovery is still written.
        assert sns_client.publish.call_count == 1
        assert result.any_ok is True

    def test_clear_falls_back_to_the_operator_chat_when_unconfigured(
        self, monkeypatch, fake_boto3, caplog
    ):
        fake, _sts, _sns = fake_boto3
        monkeypatch.setenv("ALPHA_ENGINE_ALLOW_TEST_ALERTS", "1")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setattr(alerts, "_fetch_source_mutes", lambda *a, **kw: [])
        monkeypatch.setattr(alerts.fleet_events, "emit_alert_event", lambda **kw: True)
        monkeypatch.setattr(alerts, "_resolve_log_chat", lambda: (None, None))
        sends: list[dict] = []

        def _send(text, disable_notification=False, chat_id=None, message_thread_id=None, bot_token=None):
            sends.append({"chat_id": chat_id, "disable_notification": disable_notification})
            return True

        with patch.dict(
            "sys.modules",
            {"boto3": fake, "krepis.telegram": MagicMock(send_message=_send)},
        ):
            with caplog.at_level("WARNING"):
                result = alerts.publish_clear("cleared", identity_key="k-1")
        assert result.telegram_destination == alerts.DESTINATION_OPERATOR_CHAT
        assert sends[0]["chat_id"] is None
        assert sends[0]["disable_notification"] is True
        assert result.destination_reason.startswith("fallback")
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_clear_passes_routing_arguments_through(self, monkeypatch):
        captured = {}

        def _fake_publish(message, **kwargs):
            captured.update(kwargs)
            return alerts.PublishResult()

        monkeypatch.setattr(alerts, "publish", _fake_publish)
        alerts.publish_clear(
            "x",
            identity_key="k",
            destination=alerts.DESTINATION_LOG_CHAT,
            console_artifact="s3://console/x.json",
        )
        assert captured["destination"] == alerts.DESTINATION_LOG_CHAT
        assert captured["console_artifact"] == "s3://console/x.json"


class TestPublicDestinationApi:
    def test_legacy_severity_names_are_still_exported(self):
        # CONTRIBUTING.md: the public API is additive-only. Consumers pin
        # krepis; a rename breaks them at import time.
        assert alerts.SEVERITY_PHONE_PUSH == frozenset({"error", "critical"})
        assert alerts.SEVERITY_PUSH is alerts.SEVERITY_PHONE_PUSH

    def test_destination_constants_are_public_and_complete(self):
        assert alerts.ALERT_DESTINATIONS == (
            alerts.DESTINATION_OPERATOR_CHAT,
            alerts.DESTINATION_LOG_CHAT,
            alerts.DESTINATION_CONSOLE_ONLY,
        )
        assert alerts.TELEGRAM_LOG_CHAT_PARAM == "TELEGRAM_LOG_CHAT_ID"
        assert alerts.TELEGRAM_LOG_THREAD_PARAM == "TELEGRAM_LOG_MESSAGE_THREAD_ID"

    def test_parameter_constants_are_named_for_what_they_hold(self):
        # They hold a PARAMETER NAME, never a value. The `_PARAM` suffix is
        # what keeps a reader — and CodeQL's clear-text-logging name
        # heuristic — from treating a logged config key as a logged
        # credential. Renaming them back re-raises 3 of the 4 alerts this
        # module was red for.
        assert not any(
            n.endswith("_SECRET")
            for n in dir(alerts)
            if n.startswith("TELEGRAM_")
        )


class TestCliDestination:
    def test_flags_forward_to_publish(self, monkeypatch):
        captured = {}

        def _fake_publish(message, **kwargs):
            captured.update(kwargs)
            return alerts.PublishResult(sns=alerts.ChannelResult(ok=True, detail="ok"))

        monkeypatch.setattr(alerts, "publish", _fake_publish)
        rc = alerts.main(
            [
                "publish", "--message", "m", "--severity", "info",
                "--destination", "console_only",
                "--console-artifact", "s3://console/x.json",
            ]
        )
        assert rc == 0
        assert captured["destination"] == "console_only"
        assert captured["console_artifact"] == "s3://console/x.json"

    def test_clear_flags_forward(self, monkeypatch):
        captured = {}

        def _fake_publish_clear(message, **kwargs):
            captured.update(kwargs)
            return alerts.PublishResult(sns=alerts.ChannelResult(ok=True, detail="ok"))

        monkeypatch.setattr(alerts, "publish_clear", _fake_publish_clear)
        rc = alerts.main(
            [
                "clear", "--message", "m", "--identity-key", "k",
                "--destination", "log_chat",
            ]
        )
        assert rc == 0
        assert captured["destination"] == "log_chat"

    def test_unknown_destination_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            alerts.main(["publish", "--message", "m", "--destination", "nowhere"])

    def test_stderr_names_the_destination(self, monkeypatch, capsys):
        monkeypatch.setattr(
            alerts, "publish",
            lambda message, **kw: alerts.PublishResult(
                sns=alerts.ChannelResult(ok=True, detail="ok"),
                telegram_destination=alerts.DESTINATION_LOG_CHAT,
            ),
        )
        alerts.main(["publish", "--message", "m", "--severity", "info"])
        assert "destination=log_chat" in capsys.readouterr().err


class TestDestinationReasonLeaksNoChatId:
    """Routing telemetry must be safe to serialize into a run log.

    `PublishResult.destination_reason`, `PublishResult.telegram.detail` and
    every log record this module emits are shipped to journald, CloudWatch
    and S3 run logs. A Telegram chat id addresses a channel the bot token can
    post into — credential-adjacent, and `~/Development/CLAUDE.md` ("CLI
    output safety") forbids putting it on those surfaces. CodeQL flagged 4
    high alerts on krepis-PR193 for exactly this class; these pin the fix
    rather than the alert.

    The fake ids below are deliberately distinctive so a passing assertion
    means the value is genuinely absent, not that a substring happened not
    to collide.
    """

    FAKE_CHAT_ID = "-1009876543210XYZZY"
    FAKE_THREAD_ID = 918273645

    def _surfaces(self, result, caplog) -> list[str]:
        """Everything this publish exposed to a log sink, as strings."""
        return [
            result.destination_reason,
            result.telegram.detail,
            result.sns.detail,
            str(result.telegram_destination),
            *[r.getMessage() for r in caplog.records],
        ]

    @pytest.mark.parametrize(
        "severity", ["critical", "error", "warning", "info", "notice", "debug"]
    )
    @pytest.mark.parametrize("log_chat_configured", [True, False])
    @pytest.mark.parametrize("with_artifact", [True, False])
    def test_no_surface_carries_the_resolved_chat_id(
        self, routed_publish, caplog, severity, log_chat_configured, with_artifact
    ):
        run, sends, _sns = routed_publish
        log_chat = (
            (self.FAKE_CHAT_ID, self.FAKE_THREAD_ID)
            if log_chat_configured
            else (None, None)
        )
        kwargs = {"severity": severity}
        if with_artifact:
            kwargs["console_artifact"] = "s3://console/box-health.json"
        with caplog.at_level("DEBUG"):
            result = run(log_chat=log_chat, **kwargs)

        for surface in self._surfaces(result, caplog):
            assert self.FAKE_CHAT_ID not in surface, surface
            assert str(self.FAKE_THREAD_ID) not in surface, surface

        # Not vacuous: the id DID reach the transport whenever routing chose
        # the log chat, so the assertion above is about redaction on the
        # reporting surfaces, not about the id never existing.
        if result.telegram_destination == alerts.DESTINATION_LOG_CHAT:
            assert sends[0]["chat_id"] == self.FAKE_CHAT_ID

        # And the surfaces still say something useful: the destination is
        # always named, and a fallback says whether a log chat was
        # configured — a boolean, not an id.
        assert result.telegram_destination in alerts.ALERT_DESTINATIONS
        assert result.telegram_destination in result.destination_reason or (
            "configured=" in result.destination_reason
            or "incident tier" in result.destination_reason
            or "console_artifact" in result.destination_reason
        )

    def test_an_unparseable_topic_id_is_not_echoed(self, monkeypatch, caplog):
        values = {
            alerts.TELEGRAM_LOG_CHAT_PARAM: self.FAKE_CHAT_ID,
            alerts.TELEGRAM_LOG_THREAD_PARAM: "TOPIC-QUUX-NOT-AN-INT",
        }
        monkeypatch.setattr(
            "krepis.secrets.get_secret", lambda name, **kw: values.get(name)
        )
        with caplog.at_level("DEBUG"):
            chat_id, thread_id = alerts._resolve_log_chat()
        assert (chat_id, thread_id) == (self.FAKE_CHAT_ID, None)
        emitted = " ".join(r.getMessage() for r in caplog.records)
        # The malformed VALUE is what a naive "log it so they can see it"
        # warning would print; it is secrets-resolved, so it must not appear.
        assert "TOPIC-QUUX-NOT-AN-INT" not in emitted
        assert self.FAKE_CHAT_ID not in emitted
        # It still names the parameter to fix — a config key, not a value.
        assert alerts.TELEGRAM_LOG_THREAD_PARAM in emitted
