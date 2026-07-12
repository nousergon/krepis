"""
Unit tests for ``krepis._dedup`` — the shared S3-marker dedup substrate
extracted from ``krepis.alerts`` when ``krepis.email_sender`` became its
second consumer (config#2291).

``krepis.alerts`` and ``krepis.email_sender`` each have their own test
suites pinning their *public* dedup contract through their own module
functions; this file pins the shared mechanism directly so a regression
here is caught at the source rather than only downstream.
"""

from __future__ import annotations

import json as _json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from krepis import _dedup


@pytest.fixture
def fake_boto3_with_s3():
    from botocore.exceptions import ClientError

    s3_client = MagicMock()
    store: dict[str, bytes] = {}

    def _get_object(*, Bucket, Key):
        if Key not in store:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "absent"}}, "GetObject",
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
    fake.client.side_effect = lambda service, **kw: s3_client
    return fake, store


class TestMarkerKey:
    def test_deterministic(self):
        a = _dedup.marker_key("x", marker_prefix="p")
        b = _dedup.marker_key("x", marker_prefix="p")
        assert a == b

    def test_prefix_isolates_namespaces(self):
        """The same dedup_key under two different prefixes must produce
        different marker keys — this is how alerts' and email_sender's
        dedup domains stay independent while sharing the mechanism."""
        a = _dedup.marker_key("same-key", marker_prefix="_alerts/_dedup")
        b = _dedup.marker_key("same-key", marker_prefix="_email/_dedup")
        assert a != b
        assert a.startswith("_alerts/_dedup/")
        assert b.startswith("_email/_dedup/")


class TestCheckMarker:
    def test_no_marker_returns_false(self, fake_boto3_with_s3):
        fake, _store = fake_boto3_with_s3
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = _dedup.check_marker(
                "bucket", _dedup.marker_key("k", marker_prefix="p"),
                dedup_window_min=60,
            )
        assert within is False
        assert reason == "no marker"

    def test_within_window_returns_true(self, fake_boto3_with_s3):
        fake, store = fake_boto3_with_s3
        key = _dedup.marker_key("k", marker_prefix="p")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        store[key] = _json.dumps({
            "dedup_key": "k", "first_published_at": now,
            "last_published_at": now, "publish_count": 1,
        }).encode()
        with patch.dict("sys.modules", {"boto3": fake}):
            within, _reason = _dedup.check_marker("bucket", key, dedup_window_min=60)
        assert within is True

    def test_expired_returns_false(self, fake_boto3_with_s3):
        fake, store = fake_boto3_with_s3
        key = _dedup.marker_key("k", marker_prefix="p")
        old = (datetime.now(timezone.utc) - timedelta(minutes=90)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        store[key] = _json.dumps({
            "dedup_key": "k", "first_published_at": old,
            "last_published_at": old, "publish_count": 1,
        }).encode()
        with patch.dict("sys.modules", {"boto3": fake}):
            within, _reason = _dedup.check_marker("bucket", key, dedup_window_min=60)
        assert within is False

    def test_window_none_is_forever(self, fake_boto3_with_s3):
        fake, store = fake_boto3_with_s3
        key = _dedup.marker_key("k", marker_prefix="p")
        old = (datetime.now(timezone.utc) - timedelta(days=365)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        store[key] = _json.dumps({
            "dedup_key": "k", "first_published_at": old,
            "last_published_at": old, "publish_count": 1,
        }).encode()
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = _dedup.check_marker("bucket", key, dedup_window_min=None)
        assert within is True
        assert "forever" in reason

    def test_corrupt_marker_fails_safe(self, fake_boto3_with_s3):
        fake, store = fake_boto3_with_s3
        key = _dedup.marker_key("k", marker_prefix="p")
        store[key] = b"not json"
        with patch.dict("sys.modules", {"boto3": fake}):
            within, reason = _dedup.check_marker("bucket", key, dedup_window_min=60)
        assert within is False
        assert "marker parse error" in reason

    def test_boto3_missing_fails_safe(self):
        with patch.dict("sys.modules", {"boto3": None}):
            within, reason = _dedup.check_marker("bucket", "p/x.json", dedup_window_min=60)
        assert within is False
        assert "boto3 unavailable" in reason


class TestWriteMarker:
    def test_first_write_creates_count_1(self, fake_boto3_with_s3):
        fake, store = fake_boto3_with_s3
        key = _dedup.marker_key("k", marker_prefix="p")
        with patch.dict("sys.modules", {"boto3": fake}):
            _dedup.write_marker("bucket", key, dedup_key="k", message_preview="hello")
        payload = _json.loads(store[key])
        assert payload["publish_count"] == 1
        assert payload["dedup_key"] == "k"

    def test_second_write_increments_count_preserves_first_published(
        self, fake_boto3_with_s3,
    ):
        fake, store = fake_boto3_with_s3
        key = _dedup.marker_key("k", marker_prefix="p")
        with patch.dict("sys.modules", {"boto3": fake}):
            _dedup.write_marker("bucket", key, dedup_key="k", message_preview="one")
            first_at = _json.loads(store[key])["first_published_at"]
            _dedup.write_marker("bucket", key, dedup_key="k", message_preview="two")
        payload = _json.loads(store[key])
        assert payload["publish_count"] == 2
        assert payload["first_published_at"] == first_at

    def test_write_failure_is_swallowed(self):
        fake = MagicMock()
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("boom")
        fake.client.return_value = s3
        with patch.dict("sys.modules", {"boto3": fake}):
            _dedup.write_marker("bucket", "p/x.json", dedup_key="k", message_preview="m")
        # No exception raised — best-effort.
