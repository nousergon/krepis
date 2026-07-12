"""
Shared S3-marker-backed dedup substrate.

Extracted from :mod:`krepis.alerts` (v0.24.0) when :mod:`krepis.email_sender`
became the second consumer of the identical "at most one effect per key per
window" pattern (config#2291) — per the ``~/Development/CLAUDE.md`` SOTA
rule (lift to lib when >=2 consumers exist), this now lives once and both
``alerts.publish`` and ``email_sender.send_email`` call it. Each caller owns
its own marker *prefix* (``alerts.publish`` uses ``_alerts/_dedup``;
``email_sender.send_email`` uses ``_email/_dedup``) so the two dedup
namespaces never collide even though the underlying mechanism, S3 bucket,
and marker JSON shape are shared.

**Mechanism.** ``check_marker`` reads
``s3://{bucket}/{marker_prefix}/{sha1(dedup_key)[:16]}.json``; if it exists
and is within ``dedup_window_min`` minutes (``None`` = forever), the caller
should skip its effect. After a successful fresh effect, ``write_marker``
persists (read-modify-write) an incremented ``publish_count`` — the field
name is generic ("count of effects gated by this marker"), not specific to
alerts or email.

**Failure behavior.** Fail-safe throughout: any S3/boto3 error, including a
corrupt/unparseable marker, resolves to "no marker" (i.e. the caller
proceeds with its effect) rather than silently suppressing it. An extra
alert/email is preferable to a dropped one because the marker bucket was
transiently unreachable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Final

logger = logging.getLogger(__name__)

DEFAULT_DEDUP_BUCKET: Final[str] = "alpha-engine-research"


def marker_key(dedup_key: str, *, marker_prefix: str) -> str:
    """Stable S3 key for a dedup_key marker under ``marker_prefix``.

    Hashes the dedup_key so the on-disk path is opaque + bounded length.
    The original dedup_key is preserved inside the marker JSON body for
    debugging.
    """
    digest = hashlib.sha1(dedup_key.encode("utf-8")).hexdigest()[:16]
    return f"{marker_prefix}/{digest}.json"


def check_marker(
    bucket: str,
    key: str,
    *,
    dedup_window_min: int | None,
) -> tuple[bool, str]:
    """Check whether a recent effect for this marker key is still in window.

    Returns ``(within_window, reason)``. ``within_window=True`` means the
    caller should skip its effect (already done within the window);
    ``False`` means proceed.

    Fail-safe: any S3 error other than NoSuchKey returns
    ``(False, "<error description>")`` so the caller proceeds. An extra
    effect (alert / email) is preferable to silently dropping a real one
    because the marker bucket was unreachable.

    ``dedup_window_min=None`` means "forever" — any existing marker
    suppresses subsequent effects indefinitely.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        return False, f"boto3 unavailable: {exc!r}"
    client = boto3.client("s3")
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        payload = json.loads(resp["Body"].read())
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "NoSuchKey":
            return False, "no marker"
        logger.warning(
            "dedup: marker check errored (fail-safe to proceed): %s", exc,
        )
        return False, f"marker check error: {exc!r}"
    except Exception as exc:  # boto3 missing, network, JSON parse
        logger.warning(
            "dedup: marker parse failed (fail-safe to proceed): %s", exc,
        )
        return False, f"marker parse error: {exc!r}"

    if dedup_window_min is None:
        return True, "marker exists; dedup_window_min=None (forever)"

    last_at_str = payload.get("last_published_at") or payload.get("first_published_at")
    if not last_at_str:
        return False, "marker missing timestamp"
    try:
        last_at = datetime.fromisoformat(last_at_str.replace("Z", "+00:00"))
    except ValueError:
        return False, f"marker timestamp unparseable: {last_at_str!r}"

    now = datetime.now(timezone.utc)
    elapsed = now - last_at
    window = timedelta(minutes=dedup_window_min)
    if elapsed < window:
        remaining = window - elapsed
        return True, (
            f"within {dedup_window_min}min window "
            f"(last published {int(elapsed.total_seconds())}s ago; "
            f"{int(remaining.total_seconds())}s remaining)"
        )
    return False, f"marker expired ({int(elapsed.total_seconds())}s ago > {dedup_window_min}min)"


def write_marker(
    bucket: str,
    key: str,
    *,
    dedup_key: str,
    message_preview: str,
) -> None:
    """Persist (or refresh) the dedup marker after a successful effect.

    Read-modify-write: increments ``publish_count`` if the marker already
    exists, otherwise starts a fresh marker. Best-effort — any failure is
    logged at WARNING and swallowed (worst case: one duplicate effect next
    time within the window).
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        logger.warning(
            "dedup: marker write skipped — boto3 unavailable: %s", exc,
        )
        return
    client = boto3.client("s3")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    first_published_at = now_iso
    publish_count = 1
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        prior = json.loads(resp["Body"].read())
        first_published_at = prior.get("first_published_at", now_iso)
        publish_count = int(prior.get("publish_count", 0)) + 1
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code != "NoSuchKey":
            logger.warning(
                "dedup: marker RMW read failed (writing fresh): %s", exc,
            )
    except Exception:  # JSON parse / corrupt marker — overwrite
        pass

    payload = {
        "dedup_key": dedup_key,
        "first_published_at": first_published_at,
        "last_published_at": now_iso,
        "publish_count": publish_count,
        "message_preview": message_preview[:200],
    }
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "dedup: marker write failed (best-effort, swallowed; next call "
            "within window may re-publish): %s", exc,
        )
