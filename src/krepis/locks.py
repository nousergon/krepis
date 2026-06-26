"""
Producer-side write-coordination locks for Alpha Engine modules.

This module ships the single-writer-per-resource primitive that closes
the producer-side half of the same invariant the L4 SF MutualExclusionGuard
(DynamoDB conditional PUT, ROADMAP L274) enforces at the SF-execution
entry point.

**Why both layers.** The SF mutex catches duplicate cron-fired triggers
into the same Step Function (operator double-paste of
``aws stepfunctions start-execution``, EventBridge internal retry-on-
throttle, cross-region replay coincidence). It does NOT catch
operator-launched manual invocations that bypass the SF entirely —
e.g. ``python -m builders.daily_append`` from a forensic / backfill /
dev shell. Without a producer-side lock those manual runs could race
the SF-driven path at ArcticDB exactly like the 2026-05-26 dup-EB-
target incident did (321 unique-symbol ``E_NON_INCREASING_INDEX_VERSION``
races on the same trading instance). The producer-side lock is the
load-bearing surface for the manual-invocation path; the SF mutex is
the load-bearing surface for the SF-entry path. Both layers, the same
"single writer per resource" invariant.

**API.** :func:`universe_writer_lock` is a context manager:

.. code-block:: python

    from krepis.locks import (
        universe_writer_lock,
        LockHeldByAnotherWriterError,
    )

    try:
        with universe_writer_lock(writer_id="daily_append-prod"):
            daily_append(...)
    except LockHeldByAnotherWriterError as exc:
        # `exc.holder` carries the live lock's body so operators can
        # see who owns it without re-reading S3:
        #   {writer_id, started_at, ttl_epoch, hostname, pid}
        logger.error("daily_append refused: lock held by %s", exc.holder)
        raise SystemExit(1)

The first writer's ``put_object`` with ``IfNoneMatch="*"`` succeeds and
creates ``s3://{bucket}/{lock_key}``. Subsequent writers get
``ClientError`` with code ``PreconditionFailed`` (HTTP 412) and this
module translates that into :exc:`LockHeldByAnotherWriterError`. On
normal exit (success OR exception), the context manager
``delete_object``-s the lock — best-effort; never raises, never masks
the inner exception. The S3 lifecycle rule on the ``locks/`` prefix
is the hard backstop (TTL purge for processes that died uncleanly).

**Soft TTL self-recovery.** If a lock is present but its ``ttl_epoch``
has elapsed, :func:`universe_writer_lock` treats it as abandoned:
deletes it, then attempts a fresh conditional PUT. The soft TTL absorbs
the gap between a process dying uncleanly (kill -9, OOM-killer,
SIGTERM cleanup-skip) and the next S3 lifecycle sweep. The hard TTL
(S3 lifecycle ``expires_after_days=1`` on the ``locks/`` prefix) is the
authoritative purger — set it operator-side; this module does not
attempt to manage S3 lifecycle config.

**Why S3, not DynamoDB.** ROADMAP L274's SF mutex uses DynamoDB
conditional PUT — appropriate there because the SF state machine
already has DynamoDB grants for other coordination. The producer-side
lock attaches to a process that already has S3 GetObject/PutObject on
``alpha-engine-research/*`` (every ae-data caller does), so S3 keeps
the IAM surface narrow — no new resource grants required.

**Composes with.**

- :mod:`krepis.alerts` — when ``LockHeldByAnotherWriterError``
  fires from a daemon/cron-launched run, the caller's failure handler
  can publish to SNS+Telegram via :func:`krepis.alerts.publish`.
- ROADMAP L274 — the SF MutualExclusionGuard at the SF entry point.
  This module covers the producer-side path; L274 covers the SF path.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_LOCK_KEY = "locks/universe-writer.lock"
DEFAULT_TTL_SECONDS = 3600


@dataclasses.dataclass(frozen=True)
class LockHolder:
    """The body of a held lock — what was written into S3 at acquire time.

    Operators can inspect this by reading the lock object directly
    (``aws s3 cp s3://{bucket}/{lock_key} -``) to see who owns it
    without invoking this module. The same shape is attached to
    :exc:`LockHeldByAnotherWriterError` so callers see it in
    structured logs.
    """

    writer_id: str
    started_at: str  # ISO-8601 UTC
    ttl_epoch: int  # unix-seconds; soft expiry the next writer honors
    hostname: str
    pid: int

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "LockHolder":
        d = json.loads(payload)
        return cls(
            writer_id=str(d["writer_id"]),
            started_at=str(d["started_at"]),
            ttl_epoch=int(d["ttl_epoch"]),
            hostname=str(d["hostname"]),
            pid=int(d["pid"]),
        )


class LockHeldByAnotherWriterError(RuntimeError):
    """Raised by :func:`universe_writer_lock` when the lock is currently
    held and its soft TTL has not yet elapsed.

    The :attr:`holder` attribute carries the live lock body so the
    caller can log/render it without re-reading S3.
    """

    def __init__(self, holder: LockHolder, lock_uri: str):
        self.holder = holder
        self.lock_uri = lock_uri
        super().__init__(
            f"Lock {lock_uri} held by writer_id={holder.writer_id!r} "
            f"(host={holder.hostname} pid={holder.pid} "
            f"started_at={holder.started_at})"
        )


def _now_epoch() -> int:
    """Indirection for monkeypatching in tests."""
    return int(time.time())


def _now_iso() -> str:
    """Indirection for monkeypatching in tests."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _build_holder(writer_id: str, ttl_seconds: int) -> LockHolder:
    return LockHolder(
        writer_id=writer_id,
        started_at=_now_iso(),
        ttl_epoch=_now_epoch() + ttl_seconds,
        hostname=socket.gethostname(),
        pid=os.getpid(),
    )


def _read_existing_holder(
    s3_client, bucket: str, key: str
) -> LockHolder | None:
    """Read + parse the current lock body, or None if absent / malformed.

    Returns ``None`` on parse failure too — a malformed lock can't be
    interpreted as "held by someone we know about", so we treat it as
    not-held and let the next conditional PUT race fairly. The
    pre-acquire delete in the soft-TTL recovery path also handles
    this case (delete-then-PUT — if the delete races a writer, the PUT
    fails cleanly).
    """
    try:
        from botocore.exceptions import ClientError
    except ImportError:  # pragma: no cover - botocore is a hard dep
        raise

    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "NoSuchKey",
            "404",
        ):
            return None
        raise
    body = obj["Body"].read().decode("utf-8")
    try:
        return LockHolder.from_json(body)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning(
            "Malformed lock body at s3://%s/%s — treating as absent: %s",
            bucket,
            key,
            exc,
        )
        return None


def _try_conditional_put(
    s3_client, bucket: str, key: str, holder: LockHolder
) -> bool:
    """Attempt ``put_object(IfNoneMatch='*')``. Return True on acquired,
    False on PreconditionFailed (lock now held by someone else).
    """
    from botocore.exceptions import ClientError

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=holder.to_json().encode("utf-8"),
            IfNoneMatch="*",
            ContentType="application/json",
        )
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        # S3 returns "PreconditionFailed" on If-None-Match conflict;
        # legacy mocks / older boto3 surface "412" as the bare HTTP
        # status — accept both.
        if code in ("PreconditionFailed", "412"):
            return False
        raise


@contextmanager
def universe_writer_lock(
    writer_id: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    bucket: str = DEFAULT_BUCKET,
    lock_key: str = DEFAULT_LOCK_KEY,
    s3_client=None,
) -> Iterator[LockHolder]:
    """Acquire the universe-writer lock; release on context exit.

    Yields the :class:`LockHolder` so the caller can log it / surface it.

    :param writer_id: Logical name of the writer attempting acquisition
        (e.g. ``"daily_append-prod"``, ``"backfill-260530"``). Carried
        into the lock body so a held lock identifies its owner. NOT
        used for acquisition uniqueness — the lock is single-holder
        regardless of writer_id collision.
    :param ttl_seconds: Soft TTL. If a stale lock exists with an
        elapsed ``ttl_epoch``, this writer will delete + re-acquire.
        Default 1h matches the longest realistic ``daily_append`` run.
    :param bucket: S3 bucket (default ``alpha-engine-research``).
    :param lock_key: Lock object key (default ``locks/universe-writer.lock``).
    :param s3_client: Optional boto3 S3 client override (for tests).
    :raises LockHeldByAnotherWriterError: If the lock is currently held
        and its soft TTL has not elapsed.
    """
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")

    lock_uri = f"s3://{bucket}/{lock_key}"
    holder = _build_holder(writer_id=writer_id, ttl_seconds=ttl_seconds)

    # First attempt — conditional PUT.
    if not _try_conditional_put(s3_client, bucket, lock_key, holder):
        # Lock is held. Inspect it; if soft TTL has elapsed, treat as
        # abandoned and self-recover. Otherwise raise.
        existing = _read_existing_holder(s3_client, bucket, lock_key)
        if existing is not None and existing.ttl_epoch > _now_epoch():
            raise LockHeldByAnotherWriterError(existing, lock_uri)
        # Stale-or-malformed: delete + retry exactly once.
        logger.warning(
            "Stale or malformed lock at %s — overriding "
            "(existing_ttl_epoch=%s, now=%s)",
            lock_uri,
            getattr(existing, "ttl_epoch", None),
            _now_epoch(),
        )
        try:
            s3_client.delete_object(Bucket=bucket, Key=lock_key)
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            logger.warning(
                "Stale-lock delete failed at %s — proceeding anyway: %s",
                lock_uri,
                exc,
            )
        if not _try_conditional_put(s3_client, bucket, lock_key, holder):
            # Someone else raced us to the delete-and-recover. Re-read
            # and raise with the new holder. Do NOT loop — caller
            # should retry deliberately with backoff if they want.
            existing = _read_existing_holder(s3_client, bucket, lock_key)
            if existing is None:
                # Race window with no holder visible — fallthrough to
                # raise with a placeholder so the caller doesn't get
                # an empty-handed exception.
                existing = holder  # not us, but shape-compatible
            raise LockHeldByAnotherWriterError(existing, lock_uri)

    logger.info(
        "Acquired universe-writer lock at %s (writer_id=%s, ttl_epoch=%d)",
        lock_uri,
        holder.writer_id,
        holder.ttl_epoch,
    )

    try:
        yield holder
    finally:
        # Release is best-effort. Never raises, never masks the inner
        # exception. The hard S3-lifecycle TTL is the authoritative
        # purger if this delete misses.
        try:
            s3_client.delete_object(Bucket=bucket, Key=lock_key)
            logger.info("Released universe-writer lock at %s", lock_uri)
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup
            logger.warning(
                "Failed to release lock at %s — relying on S3 lifecycle "
                "TTL: %s",
                lock_uri,
                exc,
            )
