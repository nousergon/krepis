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
        fake, _sts, sns = fake_boto3
        sns.publish.side_effect = RuntimeError("nope")
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=False, detail="creds missing")):
                result = alerts.publish("boom", source="x")
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
        info/warning → disable_notification=True (silent)."""
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

    def test_sns_subject_truncated_and_sanitized(self, fake_boto3):
        fake, _sts, sns = fake_boto3
        with patch.dict("sys.modules", {"boto3": fake}):
            with patch.object(alerts, "_publish_telegram", return_value=alerts.ChannelResult(ok=True)):
                alerts.publish("body", source="x" * 150)
        subject = sns.publish.call_args.kwargs["Subject"]
        assert len(subject) <= 100
        assert "\n" not in subject

    def test_never_raises_on_publish_exception(self, fake_boto3):
        """``publish`` must NEVER raise — both channels can fail and the
        function still returns a structured PublishResult.

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
                result = alerts.publish("boom", source="test", sns_topic_arn=None)
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
