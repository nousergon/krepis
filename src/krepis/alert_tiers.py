"""Delivery tier resolution for :mod:`krepis.alerts` — by SOURCE, not by severity.

alpha-engine-config-I6751 Phase 1 (subsumes alpha-engine-config-I6293).

THE DEFECT THIS CLOSES
----------------------
:func:`krepis.alerts.resolve_destination` decides delivery from the *severity
string* a call site chose. Measured (alpha-engine-config-I7857, restated in
that module's own docstring): SNS delivery is byte-identical at every severity,
and all three SNS topics that email cipher813@gmail.com carry
``FilterPolicy: null`` (measured 2026-09-09). So **every publish emails Brian,
whatever severity it names** — 261 messages in the seven days to 2026-09-09
across ``alpha-engine-alerts`` (194), ``crucible-v2-pages`` (54) and
``alpha-engine-alarm-backstop`` (13); ~37/day, never below 13 on any of the
last 14 days.

Severity is the wrong key regardless of the plumbing. Brian's direction
(2026-08-09): *"I only want to be alerted of issues that require addressing
(because something is broken, not generating optimal results)."* That is a
statement about **actionability**, and actionability is a property of the
alert CLASS, not of a string a call site picked. This module makes the class
registry the routing authority; severity survives as diagnostic metadata.

WHERE THE REGISTRY LIVES, AND WHY IT IS NOT IN THIS FILE
--------------------------------------------------------
``nousergon-data/infrastructure/overseer/playbooks.yaml::alert_classes`` — 92
rows, each now carrying ``tier:``. This library does NOT vendor a copy of that
table. A hand-kept twin of a source that could be read is a recorded fleet bug
class: ``alpha-engine-config-I10121``, where ``iam-drift-check`` needed a
hand-written allowlist twin of CloudFormation templates it could have parsed,
and reddened ``main`` four times in three days until the twin was deleted.

Instead ``nousergon-data`` publishes the distilled routing document to
``s3://alpha-engine-research/overseer/alert_tier_registry.json`` on merge and
re-asserts it on a schedule, and this module reads that object. The bucket is
the one :mod:`krepis.alerts` already reads and writes on every deduped publish
(``_alerts/_dedup/``), so no identity needs a new grant to route correctly.

FAIL-LOUD IS *UPWARD*, ALWAYS
-----------------------------
Every unknown resolves to :data:`TIER_PAGE`:

* a source with no registry row — and the emission is marked
  :attr:`TierDecision.registry_drift` so the finding is countable;
* an unreachable, unparseable, or unknown-``schema_version`` registry;
* colliding rows for one source — the STRICTEST of the colliding tiers wins
  (``metron`` is knowingly shared by two classes,
  ``alpha-engine-config-I8995``).

The failure mode this ordering forbids is the one that matters: a routing
layer that goes quiet when it breaks is indistinguishable from a fleet with
nothing to say. Being noisy on ignorance is recoverable; being silent is not.

WHY NOT AN SNS FILTER POLICY
----------------------------
A message-attribute filter policy on the email subscription is cheaper and was
rejected: SNS does **not** match a message that lacks the filtered attribute,
so every one of the ~170 unmigrated ``dedup_key=`` call sites across 12 repos
would have been dropped from email the moment the policy went on — a silent,
unbounded loss, arriving through the door marked "we fixed the noise"
(observability-policy.md §7.2a). Topic-level routing fails the other way: a
publisher this module cannot classify keeps today's behaviour and reaches
Brian.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Final

from krepis import s3_surface

logger = logging.getLogger(__name__)

#: IAM contract for consumers (alpha-engine-config-I8156). This module reads
#: the published tier registry under `overseer/` and reads/writes the
#: consecutive-detection markers under `_alerts/`, both in
#: `alpha-engine-research`. A consumer whose role cannot read `overseer/` still
#: alerts — it just routes everything to PAGE, loudly.
S3_SURFACE = (
    s3_surface.literal("overseer", mode=s3_surface.MODE_READ),
    s3_surface.literal("_alerts"),
)

# ── The three tiers (alpha-engine-config-I6751's approved matrix) ───────────
#: Broken now; action cannot wait for the next console look. Email + Telegram
#: phone push, exactly as every alert behaves today.
TIER_PAGE: Final[str] = "page"
#: Degraded or suboptimal results, or a lifecycle notice under Brian's
#: 2026-08-03 ruling. Telegram without the buzz; no email.
TIER_NOTIFY_SILENT: Final[str] = "notify-silent"
#: Hygiene / conformance / drift — backlog work, nothing broken. Zero
#: notification. The drain's disposition issue IS the remediation path
#: (observability-policy.md §7.4).
TIER_TRACKED_ONLY: Final[str] = "tracked-only"
#: Registry-only: the row's seriousness genuinely varies with the observed
#: facts, so the tier is resolved per emission from the runtime severity
#: (§7.1 — derived where derivable). Never a resolved tier.
TIER_DYNAMIC: Final[str] = "dynamic"

TIERS: Final[tuple[str, ...]] = (TIER_PAGE, TIER_NOTIFY_SILENT, TIER_TRACKED_ONLY)

#: Strictness order, used when several registry rows claim one source.
_STRICTNESS: Final[dict[str, int]] = {
    TIER_TRACKED_ONLY: 0, TIER_NOTIFY_SILENT: 1, TIER_PAGE: 2,
}

# ── Where the published registry lives ─────────────────────────────────────
#: Must match `nousergon-data/infrastructure/overseer/
#: publish_alert_tier_registry.py::REGISTRY_BUCKET` / `REGISTRY_OBJECT`.
REGISTRY_BUCKET: Final[str] = "alpha-engine-research"
REGISTRY_OBJECT: Final[str] = "overseer/alert_tier_registry.json"
#: Refused with a fallback to PAGE when the published document declares
#: anything else — a routing table this code does not understand must not be
#: guessed at.
SUPPORTED_SCHEMA_VERSION: Final[int] = 1

#: Local-file override. Set by the nousergon-data test suite and by any box
#: that stages the object at boot; a readable path here skips S3 entirely.
REGISTRY_PATH_ENV: Final[str] = "KREPIS_ALERT_TIER_REGISTRY_PATH"
#: In-process cache lifetime. The registry changes on merge; 15 minutes bounds
#: how long a just-merged re-route takes to reach a long-lived daemon while
#: costing at most four GETs an hour per process.
CACHE_TTL_SEC: Final[int] = 900

#: The SNS topic every non-`page` emission is redirected to. Measured
#: 2026-09-09: it exists in 711398986525/us-east-1 and has ZERO subscriptions,
#: so it is a durable record with no delivery — which is precisely what
#: "suppress the notification, keep the recording" means (§7.2a). The record
#: is NOT dropped: the message still lands on a topic, and the §7.3 bus event
#: is emitted unchanged.
MUTED_SNS_TOPIC_NAME: Final[str] = "alpha-engine-alerts-muted"

#: Streak markers for `page_after_consecutive`. Same bucket and shape as the
#: dedup markers next door.
STREAK_MARKER_PREFIX: Final[str] = "_alerts/_streak"

#: Severity → tier ladder, used ONLY for rows declared `tier: dynamic`. This
#: is not "severity decides delivery" returning by the back door: it applies
#: solely where the registry says the class's seriousness is a runtime fact.
_DYNAMIC_LADDER: Final[dict[str, str]] = {
    "critical": TIER_PAGE,
    "alarm": TIER_PAGE,
    "error": TIER_NOTIFY_SILENT,
    "warning": TIER_TRACKED_ONLY,
    "warn": TIER_TRACKED_ONLY,
    "info": TIER_TRACKED_ONLY,
}


@dataclass
class TierDecision:
    """The resolved delivery tier for one emission, and why."""

    #: One of :data:`TIERS`. Never ``dynamic`` — that is a registry value.
    tier: str
    #: Human-readable derivation, recorded on the PublishResult and on the
    #: §7.3 bus event so a routed delivery can be told from a fallback one.
    reason: str
    #: The matched row's class name, or ``None`` when nothing matched.
    alert_class: str | None = None
    #: True when the source had no registry row, or the registry could not be
    #: read. Both resolve to PAGE; this flag is what makes the gap COUNTABLE
    #: rather than merely survivable (observability-policy.md §2.2 —
    #: coverage is derived, never hand-listed).
    registry_drift: bool = False
    #: From the matched row. Consecutive detections of the same condition
    #: required before a `page` row actually pages.
    page_after_consecutive: int = 1


_cache: dict[str, object] = {"doc": None, "fetched_at": 0.0}


def _load_registry() -> dict | None:
    """Return the published routing document, or ``None`` if unavailable.

    ``None`` is never a silent outcome — every caller of this function routes
    a ``None`` to :data:`TIER_PAGE` and marks the emission as registry drift.
    The exception is swallowed HERE rather than at the call site because the
    alerting path must not be the thing that raises during an incident; the
    failure mode swallowed is "registry unreadable", and its recording surface
    is the ERROR log line below plus
    :attr:`TierDecision.registry_drift` on every affected emission
    (``~/Development/CLAUDE.md``, *Fail loud and fast*).
    """
    now = time.monotonic()
    if _cache["doc"] is not None and now - float(_cache["fetched_at"]) < CACHE_TTL_SEC:
        return _cache["doc"]  # type: ignore[return-value]

    raw: str | None = None
    local = os.environ.get(REGISTRY_PATH_ENV)
    if local:
        try:
            with open(local, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            logger.error(
                "alert_tiers: %s=%s is set but unreadable (%s) — every alert "
                "will route to PAGE until it is fixed.", REGISTRY_PATH_ENV,
                local, exc,
            )
            return None
    else:
        try:
            import boto3

            body = boto3.client("s3").get_object(
                Bucket=REGISTRY_BUCKET, Key=REGISTRY_OBJECT,
            )["Body"].read()
            raw = body.decode("utf-8")
        except Exception as exc:
            logger.error(
                "alert_tiers: could not read the delivery-tier registry at "
                "s3://%s/%s (%s). EVERY alert routes to PAGE until this is "
                "fixed — that is the intended failure direction, not a "
                "healthy state. Publisher: nousergon-data "
                "infrastructure/overseer/publish_alert_tier_registry.py",
                REGISTRY_BUCKET, REGISTRY_OBJECT, exc,
            )
            return None

    try:
        doc = json.loads(raw)
    except Exception as exc:
        # Deliberately broader than ValueError: a transport that hands back
        # something that is not text at all (a stubbed client, a truncated
        # body) must fail UPWARD to page like every other unknown here, not
        # raise out of the alerting path during an incident.
        logger.error(
            "alert_tiers: the delivery-tier registry is not readable JSON "
            "(%s) — every alert routes to PAGE.", exc,
        )
        return None
    if not isinstance(doc, dict):
        logger.error(
            "alert_tiers: the delivery-tier registry is not a JSON object — "
            "every alert routes to PAGE.",
        )
        return None
    version = doc.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        logger.error(
            "alert_tiers: registry schema_version=%r, this krepis understands "
            "%d — refusing to route on a table it cannot read; every alert "
            "routes to PAGE. Upgrade krepis on this host.",
            version, SUPPORTED_SCHEMA_VERSION,
        )
        return None

    _cache["doc"] = doc
    _cache["fetched_at"] = now
    return doc


def reset_cache() -> None:
    """Drop the in-process registry cache. For tests and long-lived daemons."""
    _cache["doc"] = None
    _cache["fetched_at"] = 0.0


def _match(source: str, entries: list[dict]) -> list[dict]:
    """Rows claiming ``source``: exact matches, else the longest wildcard.

    A registry ``source`` ending in ``*`` (e.g. ``research-runner:*``) is a
    prefix claim. Exact rows always beat wildcard rows; among wildcards the
    LONGEST prefix wins, so a specific claim is never shadowed by a broad one.
    """
    exact = [e for e in entries if e.get("source") == source]
    if exact:
        return exact
    wild = [
        e for e in entries
        if str(e.get("source", "")).endswith("*")
        and source.startswith(str(e["source"])[:-1])
    ]
    if not wild:
        return []
    longest = max(len(str(e["source"])) for e in wild)
    return [e for e in wild if len(str(e["source"])) == longest]


def resolve_tier(source: str | None, severity: str) -> TierDecision:
    """Resolve the delivery tier for one emission. Never raises.

    :param source: The emitter's declared source string — the registry key.
        ``None`` or empty is itself registry drift: an alert nobody can
        attribute cannot be routed, so it pages.
    :param severity: Used ONLY to resolve a row the registry declares
        ``dynamic``, and to name the reason.
    """
    if not source:
        return TierDecision(
            tier=TIER_PAGE,
            reason="no source declared — an unattributable alert cannot be "
                   "routed by class, so it pages (registry drift)",
            registry_drift=True,
        )

    doc = _load_registry()
    if doc is None:
        return TierDecision(
            tier=TIER_PAGE,
            reason="delivery-tier registry unavailable — failing UPWARD to "
                   "page (see the ERROR log line for the cause)",
            registry_drift=True,
        )

    rows = _match(source, list(doc.get("entries") or []))
    if not rows:
        return TierDecision(
            tier=TIER_PAGE,
            reason=(
                f"source={source!r} has no row in the alert-class registry — "
                f"paging, and recording registry drift. Add the row to "
                f"nousergon-data infrastructure/overseer/playbooks.yaml"
            ),
            registry_drift=True,
        )

    resolved: list[tuple[str, dict]] = []
    for row in rows:
        tier = row.get("tier")
        if tier == TIER_DYNAMIC:
            tier = _DYNAMIC_LADDER.get(str(severity).lower())
            if tier is None:
                # An unrecognised severity on a dynamic row is not a reason to
                # go quiet: the row said "it depends", and we cannot tell.
                tier = TIER_PAGE
        if tier not in TIERS:
            logger.error(
                "alert_tiers: registry row %r declares tier=%r, which is not "
                "one of %s — paging.", row.get("class"), row.get("tier"), TIERS,
            )
            tier = TIER_PAGE
        resolved.append((tier, row))

    tier, row = max(resolved, key=lambda pair: _STRICTNESS[pair[0]])
    collision = ""
    if len(resolved) > 1:
        collision = (
            f"; {len(resolved)} rows claim this source "
            f"({', '.join(sorted(str(r.get('class')) for _, r in resolved))}) "
            f"— took the strictest tier"
        )
    dynamic = " (dynamic row resolved from severity)" if row.get("tier") == TIER_DYNAMIC else ""
    return TierDecision(
        tier=tier,
        reason=f"registry class={row.get('class')!r} tier={tier}{dynamic}{collision}",
        alert_class=row.get("class"),
        page_after_consecutive=int(row.get("page_after_consecutive", 1) or 1),
    )


# ── The consecutive-detection gate (cadence-derived, not per-timer) ─────────
# A single failed run of an HOURLY job that its own next run repairs is not
# actionable-now: the retry lands before a human could have acted. Measured
# 2026-09-09 — `metron-deploy-drift` failed once at 18:07 (git ls-remote exit
# 128) and succeeded at 19:07, and box-health paged CRITICAL then
# INFO-RESOLVED for a condition that self-healed in 59 minutes with no action
# available to Brian. The 2026-08-28 episode was the same shape at n=5.
#
# The gate is NOT a per-timer allowlist. The registry derives
# `page_after_consecutive` from the emitting job's cadence against a
# 60-minute operator-response floor (1 when cadence exceeds the floor, 2 when
# it does not), and this function only counts.


def _streak_marker(identity: str) -> str:
    from krepis import _dedup

    return _dedup.marker_key(identity, marker_prefix=STREAK_MARKER_PREFIX)


def consecutive_count(bucket: str, identity: str) -> int | None:
    """How many consecutive OPEN emissions this condition has already had.

    ``None`` means "could not tell", and every caller treats that as "do not
    downgrade" — the gate may only ever make an alert quieter on POSITIVE
    evidence that the condition is fresh. Swallowed failure mode: S3
    unreadable; recording surface: the DEBUG line plus the resulting PAGE.
    """
    try:
        import boto3

        body = boto3.client("s3").get_object(
            Bucket=bucket, Key=_streak_marker(identity),
        )["Body"].read()
        return int(json.loads(body).get("consecutive", 0))
    except Exception as exc:  # missing marker included: this is the first one
        logger.debug("alert_tiers: streak read for %r: %s", identity, exc)
        return 0 if _is_missing(exc) else None


def _is_missing(exc: Exception) -> bool:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return resp.get("Error", {}).get("Code") in ("NoSuchKey", "404")
    return False


def record_open(bucket: str, identity: str, count: int) -> None:
    """Persist the new consecutive count after an OPEN emission."""
    try:
        import boto3

        boto3.client("s3").put_object(
            Bucket=bucket, Key=_streak_marker(identity),
            Body=json.dumps({"consecutive": count, "at": time.time()}).encode(),
            ContentType="application/json",
        )
    except Exception as exc:
        # Worst case is one extra page next time — the gate errs upward by
        # construction, so a failed write can never silence anything.
        logger.debug("alert_tiers: streak write for %r failed: %s", identity, exc)


def clear_streak(bucket: str, identity: str) -> None:
    """Reset the counter when the condition clears (`publish_clear`)."""
    try:
        import boto3

        boto3.client("s3").delete_object(
            Bucket=bucket, Key=_streak_marker(identity),
        )
    except Exception as exc:
        logger.debug("alert_tiers: streak clear for %r failed: %s", identity, exc)
