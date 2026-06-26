"""
Unit tests for :mod:`krepis.locks`.

Pins the producer-side writer-lock contract:

* First writer's conditional PUT succeeds and creates the lock object.
* Concurrent writer (lock present, TTL fresh) raises
  :exc:`LockHeldByAnotherWriterError` carrying the live holder.
* Context manager releases on normal exit AND on exception (try/finally).
* Soft-TTL self-recovery: stale lock (``ttl_epoch < now``) → delete +
  re-acquire in one attempt.
* Release is best-effort: ``delete_object`` failure logs WARN, never
  masks the inner block's exception.
* Lock body shape is the documented contract
  (``{writer_id, started_at, ttl_epoch, hostname, pid}``).

Pinned so the ae-data ``daily_append`` consumer (and any future
adopter — backfill loops, predictor weight-promote, etc.) can rely
on stable behavior across lib versions.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from krepis import locks
from krepis.locks import (
    DEFAULT_BUCKET,
    DEFAULT_LOCK_KEY,
    LockHeldByAnotherWriterError,
    LockHolder,
    universe_writer_lock,
)


# ── Test fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_s3():
    """An S3 stub that backs a single in-memory key with conditional-PUT
    semantics. Mirrors enough of boto3.client('s3') for the lock module
    to drive end-to-end without botocore stubber boilerplate.
    """

    state: dict = {"body": None}

    client = MagicMock()

    def _put_object(Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and state["body"] is not None:
            err_response = {
                "Error": {"Code": "PreconditionFailed", "Message": "..."},
                "ResponseMetadata": {"HTTPStatusCode": 412},
            }
            raise ClientError(err_response, "PutObject")
        state["body"] = Body
        return {"ETag": '"fake-etag"'}

    def _get_object(Bucket, Key, **kwargs):
        if state["body"] is None:
            err_response = {
                "Error": {"Code": "NoSuchKey", "Message": "..."},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            }
            raise ClientError(err_response, "GetObject")
        body = MagicMock()
        body.read.return_value = (
            state["body"] if isinstance(state["body"], bytes)
            else state["body"].encode("utf-8")
        )
        return {"Body": body}

    def _delete_object(Bucket, Key, **kwargs):
        state["body"] = None
        return {}

    client.put_object.side_effect = _put_object
    client.get_object.side_effect = _get_object
    client.delete_object.side_effect = _delete_object
    client._state = state
    return client


@pytest.fixture
def frozen_time(monkeypatch):
    """Freeze ``_now_epoch`` / ``_now_iso`` to a known instant for
    deterministic ttl_epoch assertions."""
    monkeypatch.setattr(locks, "_now_epoch", lambda: 1_700_000_000)
    monkeypatch.setattr(locks, "_now_iso", lambda: "2026-05-27T12:00:00Z")
    return 1_700_000_000


# ── Acquire / release happy path ─────────────────────────────────────────


class TestAcquireHappyPath:
    def test_first_writer_acquires_and_writes_holder(
        self, fake_s3, frozen_time
    ):
        with universe_writer_lock(
            writer_id="daily_append-test", s3_client=fake_s3
        ) as holder:
            assert holder.writer_id == "daily_append-test"
            assert holder.started_at == "2026-05-27T12:00:00Z"
            assert holder.ttl_epoch == frozen_time + 3600

        # PutObject called with IfNoneMatch=*
        put_call = fake_s3.put_object.call_args
        assert put_call.kwargs["IfNoneMatch"] == "*"
        assert put_call.kwargs["Bucket"] == DEFAULT_BUCKET
        assert put_call.kwargs["Key"] == DEFAULT_LOCK_KEY

        # Body is JSON with the holder shape
        body_bytes = put_call.kwargs["Body"]
        body = json.loads(body_bytes.decode("utf-8"))
        assert set(body.keys()) == {
            "writer_id",
            "started_at",
            "ttl_epoch",
            "hostname",
            "pid",
        }
        assert body["writer_id"] == "daily_append-test"

    def test_release_on_normal_exit(self, fake_s3, frozen_time):
        with universe_writer_lock(
            writer_id="test", s3_client=fake_s3
        ):
            pass
        fake_s3.delete_object.assert_called_once_with(
            Bucket=DEFAULT_BUCKET, Key=DEFAULT_LOCK_KEY
        )

    def test_release_on_exception_in_block(self, fake_s3, frozen_time):
        with pytest.raises(RuntimeError, match="inner blew up"):
            with universe_writer_lock(
                writer_id="test", s3_client=fake_s3
            ):
                raise RuntimeError("inner blew up")
        # The lock MUST be released even when the inner block raised —
        # otherwise the lock leaks until S3 lifecycle purges it.
        fake_s3.delete_object.assert_called_once_with(
            Bucket=DEFAULT_BUCKET, Key=DEFAULT_LOCK_KEY
        )


# ── Contention path ──────────────────────────────────────────────────────


class TestContention:
    def test_second_writer_raises_with_live_holder(
        self, fake_s3, frozen_time
    ):
        # First writer acquires.
        with universe_writer_lock(
            writer_id="first", s3_client=fake_s3
        ):
            # Second writer attempts while first still holds.
            with pytest.raises(LockHeldByAnotherWriterError) as excinfo:
                with universe_writer_lock(
                    writer_id="second", s3_client=fake_s3
                ):
                    pass  # pragma: no cover - shouldn't reach
        assert excinfo.value.holder.writer_id == "first"
        assert excinfo.value.lock_uri == (
            f"s3://{DEFAULT_BUCKET}/{DEFAULT_LOCK_KEY}"
        )

    def test_exception_message_includes_writer_id_and_host(
        self, fake_s3, frozen_time
    ):
        with universe_writer_lock(
            writer_id="first-writer", s3_client=fake_s3
        ):
            with pytest.raises(LockHeldByAnotherWriterError) as excinfo:
                with universe_writer_lock(
                    writer_id="second", s3_client=fake_s3
                ):
                    pass  # pragma: no cover
        msg = str(excinfo.value)
        assert "first-writer" in msg
        assert "host=" in msg
        assert "pid=" in msg


# ── Soft-TTL self-recovery ───────────────────────────────────────────────


class TestSoftTTLSelfRecovery:
    def test_stale_lock_is_overridden(self, fake_s3, monkeypatch):
        """A lock body present with ``ttl_epoch < now`` is treated as
        abandoned. The new writer deletes it and re-acquires."""
        # Prime the bucket with a stale lock (ttl_epoch in the past).
        stale = LockHolder(
            writer_id="dead-process",
            started_at="2026-05-27T00:00:00Z",
            ttl_epoch=1_000_000_000,  # ancient
            hostname="dead-host",
            pid=123,
        )
        fake_s3._state["body"] = stale.to_json().encode("utf-8")

        # Freeze "now" well after stale.ttl_epoch.
        monkeypatch.setattr(locks, "_now_epoch", lambda: 2_000_000_000)
        monkeypatch.setattr(
            locks, "_now_iso", lambda: "2033-01-01T00:00:00Z"
        )

        with universe_writer_lock(
            writer_id="recoverer", s3_client=fake_s3
        ) as holder:
            assert holder.writer_id == "recoverer"

        # The stale lock was deleted as part of the recovery sequence.
        # First delete = stale recovery; second delete = release.
        assert fake_s3.delete_object.call_count == 2

    def test_fresh_lock_not_overridden(self, fake_s3, monkeypatch):
        """A lock body present with ``ttl_epoch > now`` is NOT abandoned.
        The new writer raises rather than overriding."""
        fresh = LockHolder(
            writer_id="active-process",
            started_at="2026-05-27T12:00:00Z",
            ttl_epoch=3_000_000_000,  # far future
            hostname="active-host",
            pid=456,
        )
        fake_s3._state["body"] = fresh.to_json().encode("utf-8")

        monkeypatch.setattr(locks, "_now_epoch", lambda: 2_000_000_000)
        monkeypatch.setattr(
            locks, "_now_iso", lambda: "2033-01-01T00:00:00Z"
        )

        with pytest.raises(LockHeldByAnotherWriterError) as excinfo:
            with universe_writer_lock(
                writer_id="contender", s3_client=fake_s3
            ):
                pass  # pragma: no cover
        assert excinfo.value.holder.writer_id == "active-process"
        # No delete should have been attempted (fresh lock).
        assert fake_s3.delete_object.call_count == 0

    def test_malformed_lock_body_treated_as_stale(
        self, fake_s3, frozen_time
    ):
        """A non-JSON or wrong-shape body is treated as abandoned —
        delete + re-acquire. The S3-lifecycle purger eventually catches
        this anyway, but the immediate self-recovery prevents a fully-
        stuck state."""
        fake_s3._state["body"] = b"NOT_JSON"

        with universe_writer_lock(
            writer_id="recoverer", s3_client=fake_s3
        ) as holder:
            assert holder.writer_id == "recoverer"


# ── Best-effort release ──────────────────────────────────────────────────


class TestBestEffortRelease:
    def test_delete_failure_does_not_mask_inner_exception(
        self, fake_s3, frozen_time
    ):
        """If ``delete_object`` raises during release, the original
        block's exception must propagate unchanged. Lock leaks to S3
        lifecycle TTL; never masks user-facing failure."""

        def _raise_on_delete(**kwargs):
            raise RuntimeError("simulated S3 outage")

        fake_s3.delete_object.side_effect = _raise_on_delete

        with pytest.raises(RuntimeError, match="inner blew up"):
            with universe_writer_lock(
                writer_id="test", s3_client=fake_s3
            ):
                raise RuntimeError("inner blew up")

    def test_delete_failure_on_clean_exit_does_not_raise(
        self, fake_s3, frozen_time, caplog
    ):
        """If the block exits cleanly but ``delete_object`` fails, the
        outer context manager must NOT raise — only log WARN. Inner
        success must reach the caller as success."""
        import logging

        def _raise_on_delete(**kwargs):
            raise RuntimeError("simulated S3 outage")

        fake_s3.delete_object.side_effect = _raise_on_delete

        with caplog.at_level(logging.WARNING, logger="krepis.locks"):
            with universe_writer_lock(
                writer_id="test", s3_client=fake_s3
            ):
                pass  # clean exit

        warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "Failed to release lock" in r.message
        ]
        assert warnings, "Expected a WARN log on release failure"


# ── Custom bucket / key / TTL ────────────────────────────────────────────


class TestCustomization:
    def test_custom_bucket_and_key(self, fake_s3, frozen_time):
        with universe_writer_lock(
            writer_id="test",
            s3_client=fake_s3,
            bucket="my-bucket",
            lock_key="custom/path.lock",
        ):
            pass
        put_call = fake_s3.put_object.call_args
        assert put_call.kwargs["Bucket"] == "my-bucket"
        assert put_call.kwargs["Key"] == "custom/path.lock"

    def test_custom_ttl(self, fake_s3, frozen_time):
        with universe_writer_lock(
            writer_id="test",
            ttl_seconds=60,
            s3_client=fake_s3,
        ) as holder:
            assert holder.ttl_epoch == frozen_time + 60


# ── LockHolder JSON round-trip ───────────────────────────────────────────


class TestLockHolderSerialization:
    def test_to_from_json_round_trip(self):
        holder = LockHolder(
            writer_id="x",
            started_at="2026-05-27T12:00:00Z",
            ttl_epoch=1_700_003_600,
            hostname="h",
            pid=42,
        )
        restored = LockHolder.from_json(holder.to_json())
        assert restored == holder

    def test_from_json_rejects_malformed(self):
        with pytest.raises((json.JSONDecodeError, KeyError, ValueError)):
            LockHolder.from_json("not json")
        with pytest.raises((KeyError, ValueError)):
            LockHolder.from_json('{"writer_id":"x"}')  # missing fields
