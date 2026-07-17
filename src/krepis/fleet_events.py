"""
Structured fleet alert-event emission — the Nousergon Overseer intake feed.

Every operator alert the fleet sends (SNS email, Telegram push/silent) is
human-facing and fire-and-forget: once delivered, nothing owns follow-through.
This module adds a machine-readable side-channel: the two alert chokepoints
(:func:`krepis.alerts.publish` and :func:`krepis.telegram.send_message`) also
emit a versioned, structured **alert event** onto an EventBridge custom bus,
where the Overseer response plane (alpha-engine-config-I2821) consumes it.
Emission is purely additive — delivery behavior of the human channels is
byte-identical whether or not emission succeeds.

**Public API:**

- :func:`emit_alert_event` — best-effort, never raises. Primary transport is
  EventBridge ``PutEvents`` to the bus named by ``NOUSERGON_ALERTS_BUS``
  (default ``nousergon-alerts``); on any failure it falls back to an S3
  drop-zone write (roles fleet-wide already hold write on the shared research
  bucket, so events flow even before an IAM ``events:PutEvents`` grant reaches
  a given role). If both transports fail, a WARNING with the stable marker
  ``NOUSERGON_ALERT_EVENT_EMIT_FAILED`` is logged and the alert's human
  delivery is unaffected.
- :func:`suppress_emission` — context manager used by ``alerts.publish`` so
  its nested ``telegram.send_message`` call does not double-emit; the publish
  call emits one rich event itself.
- :func:`normalize_severity` — maps the fleet's free-form severity strings
  onto the closed v1 enum (``info``/``warning``/``error``/``critical``).

**Event contract.** ``Source="nousergon.krepis"``,
``DetailType="nousergon.alert.v1"``, detail per
:file:`fleet_event_schema_v1.json` (shipped as package data; additive-only
evolution — new optional fields bump the minor semantics, a breaking change
requires ``nousergon.alert.v2`` alongside v1 per the S3-contract-safety rule).

**Test-environment guard.** Mirrors ``alerts.publish``: when
``PYTEST_CURRENT_TEST`` is set, emission is a no-op unless
``NOUSERGON_ALLOW_TEST_EVENTS=1`` (used only by this module's own test suite
against mocked transports), so no consumer test can feed the production
intake queue.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Final, Iterator, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final[int] = 1
EVENT_SOURCE: Final[str] = "nousergon.krepis"
DETAIL_TYPE: Final[str] = "nousergon.alert.v1"

BUS_ENV: Final[str] = "NOUSERGON_ALERTS_BUS"
DEFAULT_BUS_NAME: Final[str] = "nousergon-alerts"

FALLBACK_BUCKET_ENV: Final[str] = "NOUSERGON_ALERTS_FALLBACK_BUCKET"
DEFAULT_FALLBACK_BUCKET: Final[str] = "alpha-engine-research"
FALLBACK_PREFIX: Final[str] = "overseer/intake-fallback"

DEFAULT_REGION: Final[str] = "us-east-1"
MAX_BODY_CHARS: Final[int] = 4000

_VALID_SEVERITIES: Final[frozenset] = frozenset({"info", "warning", "error", "critical"})

# Set by alerts.publish around its Telegram fan-out so send_message's
# auto-emit hook stays quiet — publish emits one rich event itself.
_emission_suppressed: ContextVar[bool] = ContextVar("_emission_suppressed", default=False)


@contextmanager
def suppress_emission() -> Iterator[None]:
    """Suppress auto-emission from nested chokepoints within this context."""
    token = _emission_suppressed.set(True)
    try:
        yield
    finally:
        _emission_suppressed.reset(token)


def emission_suppressed() -> bool:
    """True when a caller higher on the stack owns event emission."""
    return _emission_suppressed.get()


def normalize_severity(raw: Optional[str]) -> str:
    """Map a free-form severity string onto the closed v1 enum.

    ``warn`` folds into ``warning``; anything unrecognized maps to
    ``warning`` (the original string travels in ``severity_raw``, so no
    information is lost — the drain sees both).
    """
    if not raw:
        return "info"
    lowered = raw.strip().lower()
    if lowered == "warn":
        return "warning"
    if lowered in _VALID_SEVERITIES:
        return lowered
    return "warning"


def _resolve_source(explicit: Optional[str]) -> Optional[str]:
    """Attribute the event: explicit arg > env override > Lambda name."""
    return (
        explicit
        or os.environ.get("KREPIS_EVENT_SOURCE")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or None
    )


def _region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )


def _build_detail(
    *,
    origin: str,
    body: str,
    severity_raw: Optional[str],
    source: Optional[str],
    dedup_key: Optional[str],
    channels: Optional[Dict[str, Optional[bool]]],
    disable_notification: Optional[bool],
) -> Dict[str, Any]:
    hostname: Optional[str]
    try:
        hostname = socket.gethostname()
    except Exception:  # pragma: no cover - hostname resolution is best-effort
        hostname = None
    if severity_raw is None and disable_notification is not None:
        # Direct Telegram sends have no severity concept — proxy from the
        # silent flag: silent digests are info-tier, loud pushes warning-tier.
        severity = "info" if disable_notification else "warning"
    else:
        severity = normalize_severity(severity_raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "origin": origin,
        "source": _resolve_source(source),
        "severity": severity,
        "severity_raw": severity_raw,
        "body": body[:MAX_BODY_CHARS],
        "dedup_key": dedup_key,
        "channels": channels,
        "disable_notification": disable_notification,
        "runtime": {
            "lambda_function_name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME"),
            "hostname": hostname,
        },
    }


def _put_event(detail: Dict[str, Any]) -> None:
    import boto3

    client = boto3.client("events", region_name=_region())
    resp = client.put_events(
        Entries=[
            {
                "Source": EVENT_SOURCE,
                "DetailType": DETAIL_TYPE,
                "Detail": json.dumps(detail),
                "EventBusName": os.environ.get(BUS_ENV, DEFAULT_BUS_NAME),
            }
        ]
    )
    if resp.get("FailedEntryCount"):
        entry = resp["Entries"][0]
        raise RuntimeError(
            f"PutEvents rejected: {entry.get('ErrorCode')} {entry.get('ErrorMessage')}"
        )


def _write_fallback(detail: Dict[str, Any]) -> None:
    import boto3

    bucket = os.environ.get(FALLBACK_BUCKET_ENV, DEFAULT_FALLBACK_BUCKET)
    now = datetime.now(timezone.utc)
    key = (
        f"{FALLBACK_PREFIX}/{now:%Y-%m-%d}/"
        f"{now:%H%M%S}-{uuid.uuid4().hex[:8]}.json"
    )
    boto3.client("s3", region_name=_region()).put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(
            {"source": EVENT_SOURCE, "detail_type": DETAIL_TYPE, "detail": detail}
        ).encode("utf-8"),
        ContentType="application/json",
    )


def emit_alert_event(
    *,
    origin: str,
    body: str,
    severity_raw: Optional[str] = None,
    source: Optional[str] = None,
    dedup_key: Optional[str] = None,
    channels: Optional[Dict[str, Optional[bool]]] = None,
    disable_notification: Optional[bool] = None,
) -> bool:
    """Emit one structured alert event to the Overseer intake. Never raises.

    Transport order: EventBridge ``PutEvents`` → S3 drop-zone fallback →
    WARNING log. Returns ``True`` when either transport accepted the event
    (``False`` on total failure or guard suppression) so chokepoint callers
    can observe outcomes without branching on them.

    :param origin: Emitting chokepoint (``alerts.publish`` /
        ``telegram.send_message``).
    :param body: Human-facing alert text (truncated to 4000 chars in the
        event; the human channel always carries the full text).
    :param severity_raw: The caller's free-form severity string; the event
        carries both this and its :func:`normalize_severity` mapping.
    :param source: Origin identifier; falls back to ``KREPIS_EVENT_SOURCE``
        then ``AWS_LAMBDA_FUNCTION_NAME`` env attribution.
    :param dedup_key: The alert's dedup key when the caller used one.
    :param channels: Per-channel delivery outcomes, e.g.
        ``{"sns": True, "telegram": False}``; ``None`` values mean the
        channel was not attempted.
    :param disable_notification: Telegram silent-delivery flag when known.
    """
    # Same defense-in-depth as alerts.publish: a consumer test that reaches
    # this path un-stubbed must not feed the production intake queue.
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get(
        "NOUSERGON_ALLOW_TEST_EVENTS"
    ):
        return False

    detail = _build_detail(
        origin=origin,
        body=body,
        severity_raw=severity_raw,
        source=source,
        dedup_key=dedup_key,
        channels=channels,
        disable_notification=disable_notification,
    )

    try:
        _put_event(detail)
        return True
    except Exception as exc:
        # Swallowed by design: emission is a side-channel and must never
        # block or fail the human alert path. Recording surface: the S3
        # fallback write below (drained by the Overseer alongside the queue).
        logger.warning("fleet_events: PutEvents failed, using S3 fallback: %s", exc)

    try:
        _write_fallback(detail)
        return True
    except Exception as exc:
        # Swallowed by design (same rationale). Recording surface: this
        # stable log marker — the event is lost, the human alert is not.
        logger.warning(
            "fleet_events: emission failed on both transports "
            "[NOUSERGON_ALERT_EVENT_EMIT_FAILED]: %s",
            exc,
        )
    return False


__all__ = [
    "DETAIL_TYPE",
    "DEFAULT_BUS_NAME",
    "EVENT_SOURCE",
    "SCHEMA_VERSION",
    "emission_suppressed",
    "emit_alert_event",
    "normalize_severity",
    "suppress_emission",
]
