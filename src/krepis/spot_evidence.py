"""
Spot-teardown chokepoint: preserve the failure's evidence, THEN delete staging.

**The defect this exists to make unrepresentable (alpha-engine-config-I7442).**

Every spot launcher in the fleet ended its EXIT trap with, in this order::

    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" ...
    aws s3 rm "$S3_STAGING" --recursive --quiet 2>/dev/null || true
    echo "  Instance terminated; S3 staging cleaned."

``$S3_STAGING`` is ``s3://alpha-engine-research/tmp/spot_<stage>/<run-id>``, and
``run_ssm`` configures SSM's ``OutputS3KeyPrefix`` as
``<that prefix>/ssm-output``. So the recursive delete above removes the SSM
agent's own upload of the **full** remote stdout and stderr — the only copy of
the workload's output that is not subject to SSM's 24KB
``StandardOutputContent`` cap.

On 2026-08-15 the weekly run's ``PredictorBacktest`` stage failed after ~4
hours. The captured ``_ssm_logs`` copy ended at ``2026-08-15 12:44--output
truncated--``, and the full log it pointed at —
``tmp/spot_predictor-backtest/20260815T123311Z-i-08a4371deec28ef07/ssm-output/``
— was empty. Both copies of the evidence were destroyed by the teardown that
was handling the failure, and why the run failed is permanently unrecoverable.
Six recovery executions over ~11 hours could not target the actual cause.
Measured 2026-08-17: the ``alpha-engine-research`` bucket lifecycle carries
rules for ``staging/`` and ``features/`` **only**, so nothing else bounds
``tmp/`` and hand-deleting it was the only thing keeping it from growing
without limit — which is why the delete is preserved here rather than removed.

**Why this is a module and not a reordered pair of bash lines.** The obligation
is "never delete the only remaining copy of a failure's evidence while handling
that failure", and a comment above two commands does not carry it: the next
author adds a third line, or copies the trap into a fourth launcher (there were
three ``_spot_common.sh`` twins and four surviving monoliths). Here the delete
is reachable **only** from inside :func:`teardown`, only after
:func:`_preserve` has returned ok, and the callers have no ``aws s3 rm`` of
their own. The ordering is a property of the call graph.

**What is retained, and what it costs.** On a SUCCESSFUL run nothing is
preserved — staging is deleted exactly as before, so no successful run's output
starts accruing storage. On a FAILED run the staging prefix is copied to
``s3://{bucket}/_spot_evidence/{slug}/{YYYY-MM-DD}/{run-id}/`` first. That is
the SSM output plus the few staged YAML configs: single-digit MB per failure,
against a stage whose spot instance alone costs dollars per hour. If the copy
fails for any reason the staging prefix is **left in place** and the URI is
printed loudly — retaining an un-lifecycled prefix is a cost, losing the
diagnosis is an outage.

**Never raises, never masks.** Teardown runs on the failure path. Every S3
error is caught and reported; the CLI's exit code reports whether evidence was
preserved, and callers deliberately ignore it in favour of the workload's own
exit status (``fail loud`` applies to the workload, not to the janitor that
must never be the reason a diagnosis is lost twice).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Final, Optional

logger = logging.getLogger(__name__)

#: Durable prefix failure evidence is lifted to. Sits OUTSIDE ``tmp/`` on
#: purpose — ``tmp/`` is the prefix every launcher recursively deletes, and
#: preserving into a sibling of the thing being deleted is how this defect
#: would come back.
DURABLE_PREFIX: Final[str] = "_spot_evidence"

#: Days a preserved failure record is kept. Long enough that a Saturday
#: failure is still diagnosable after the following weekend's re-run and a
#: week of backlog latency; short enough that this never becomes a data set.
#: Only FAILED runs land here, so the steady-state volume is single-digit MB
#: per failure.
EVIDENCE_RETENTION_DAYS: Final[int] = 90

#: Days a leftover ``tmp/spot_<slug>/<run-id>/`` prefix is kept. Reached here
#: from `crucible-backtester#675` (config-I7396), which bounded its own
#: retention with a bash prune using ``date -u -v-Nd`` / ``date -u -d`` CLI
#: fallbacks. That prune retires into this module: the bound belongs beside the
#: thing it bounds, and it now covers all seven launcher families instead of
#: one. Also collects prefixes from runs whose launcher died before its EXIT
#: trap — which the per-repo prune could reach only for its own stage.
STAGING_RETENTION_DAYS: Final[int] = 14


class StagingUriError(ValueError):
    """A ``--staging`` value that is not an ``s3://bucket/key`` URI."""


def parse_s3_uri(uri: str):
    """Split ``s3://bucket/key/prefix`` into ``(bucket, prefix)``.

    Raises :class:`StagingUriError` on anything else. This is the one place
    that is allowed to be strict: a mis-parsed staging URI would either delete
    the wrong prefix or silently delete nothing, and both are worse than a
    loud refusal before any object is touched.
    """
    if not uri or not uri.startswith("s3://"):
        raise StagingUriError("not an s3:// URI: {!r}".format(uri))
    rest = uri[len("s3://") :].strip("/")
    if "/" not in rest:
        raise StagingUriError(
            "refusing to operate on a whole bucket (no key prefix): {!r}".format(uri)
        )
    bucket, prefix = rest.split("/", 1)
    if not bucket or not prefix:
        raise StagingUriError("empty bucket or prefix in {!r}".format(uri))
    return bucket, prefix


def _run_id(prefix: str) -> str:
    """Last path segment of the staging prefix — the launcher's run id.

    Shape: ``<UTC timestamp>-<instance-id>``, e.g.
    ``20260815T123311Z-i-08a4371deec28ef07``. Already unique per attempt, so it
    is the natural leaf of the durable key and no second uniquifier is needed.
    """
    return prefix.rstrip("/").rsplit("/", 1)[-1] or "unknown-run"


def durable_prefix_for(
    slug: str, prefix: str, *, now: Optional[datetime] = None
) -> str:
    """Compute the durable key prefix for one failed run's evidence."""
    now = now or datetime.now(timezone.utc)
    return "{}/{}/{}/{}".format(
        DURABLE_PREFIX, slug or "unknown-stage", now.strftime("%Y-%m-%d"), _run_id(prefix)
    )


def _list_keys(s3, bucket: str, prefix: str):
    """All object keys under ``prefix``. Raises on a genuine S3 error."""
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix.rstrip("/") + "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []) or []:
            keys.append(item["Key"])
        if not resp.get("IsTruncated"):
            return keys
        token = resp.get("NextContinuationToken")
        if not token:
            return keys


def _preserve(s3, bucket: str, prefix: str, dest_prefix: str):
    """Copy every staging object to ``dest_prefix``. Returns ``(ok, detail)``.

    ``ok`` is True only when EVERY object copied (or there were none to copy —
    an empty staging prefix has no evidence to lose). It is the sole gate on
    the delete below, which is what makes the ordering structural rather than
    conventional.
    """
    try:
        keys = _list_keys(s3, bucket, prefix)
    except Exception as exc:
        return False, "list failed: {}: {}".format(type(exc).__name__, exc)
    if not keys:
        return True, "nothing to preserve (staging prefix was empty)"
    base = prefix.rstrip("/") + "/"
    copied = 0
    for key in keys:
        rel = key[len(base) :] if key.startswith(base) else key.rsplit("/", 1)[-1]
        try:
            s3.copy_object(
                Bucket=bucket,
                Key="{}/{}".format(dest_prefix.rstrip("/"), rel),
                CopySource={"Bucket": bucket, "Key": key},
            )
            copied += 1
        except Exception as exc:
            return False, "copy of {} failed after {}/{} objects: {}: {}".format(
                key, copied, len(keys), type(exc).__name__, exc
            )
    return True, "preserved {} object(s)".format(copied)


def _delete_staging(s3, bucket: str, prefix: str):
    """Delete THIS RUN's staging prefix. Returns ``(ok, detail)``.

    Reachable from exactly two places, both inside :func:`teardown`: the
    success branch (nothing to preserve) and the branch guarded by a successful
    :func:`_preserve`. Do not add a third — those two guarded calls ARE the fix
    for alpha-engine-config-I7442, and ``tests/test_spot_evidence.py``
    ``TestOrderingIsStructural`` fails if a third appears. Retention sweeps use
    :func:`_delete_prefix` instead; they operate on OTHER runs' prefixes, which
    have already been through this gate.
    """
    return _delete_prefix(s3, bucket, prefix)


def _delete_prefix(s3, bucket: str, prefix: str):
    """Delete every object under ``prefix``. Returns ``(ok, detail)``."""
    try:
        keys = _list_keys(s3, bucket, prefix)
    except Exception as exc:
        return False, "list failed: {}: {}".format(type(exc).__name__, exc)
    if not keys:
        return True, "nothing to delete"
    deleted = 0
    for chunk_start in range(0, len(keys), 1000):
        chunk = keys[chunk_start : chunk_start + 1000]
        try:
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )
            deleted += len(chunk)
        except Exception as exc:
            return False, "delete failed after {}/{} objects: {}: {}".format(
                deleted, len(keys), type(exc).__name__, exc
            )
    return True, "deleted {} object(s)".format(deleted)


def _child_prefixes(s3, bucket: str, root: str):
    """Immediate child ``CommonPrefixes`` under ``root`` (one S3 LIST, no HEADs)."""
    out = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": root, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("CommonPrefixes", []) or []:
            out.append(item["Prefix"])
        if not resp.get("IsTruncated"):
            return out
        token = resp.get("NextContinuationToken")
        if not token:
            return out


def _leaf_date(child_prefix: str) -> Optional[datetime]:
    """UTC date encoded in a child prefix's leaf, or ``None`` if unparseable.

    Two shapes, both already produced by the writers:
    ``20260815T123311Z-i-08a...`` (a launcher run id) and ``2026-08-15``
    (this module's own durable date component). Parsed in Python rather than
    lexically compared against a shelled-out ``date -u -v-14d``, whose BSD/GNU
    flag split the retired bash prune had to carry a fallback for.
    """
    leaf = child_prefix.rstrip("/").rsplit("/", 1)[-1]
    for fmt, width in (("%Y%m%dT%H%M%SZ", 16), ("%Y-%m-%d", 10)):
        try:
            return datetime.strptime(leaf[:width], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _prune(s3, bucket: str, root: str, *, older_than_days: int, now: datetime):
    """Delete child prefixes of ``root`` whose encoded date is past retention.

    Returns ``(pruned_count, detail)``. Never raises: retention is hygiene, and
    a teardown that fails on it would stop doing the job it exists for. An
    unparseable leaf is LEFT ALONE — deleting on a failure to understand a key
    is how a retention sweep becomes an outage.
    """
    try:
        children = _child_prefixes(s3, bucket, root)
    except Exception as exc:
        return 0, "prune list failed: {}: {}".format(type(exc).__name__, exc)
    pruned = 0
    for child in children:
        stamp = _leaf_date(child)
        if stamp is None:
            continue
        if (now - stamp).days <= older_than_days:
            continue
        ok, _detail = _delete_prefix(s3, bucket, child)
        if ok:
            pruned += 1
    return pruned, "pruned {} prefix(es) under {} older than {}d".format(
        pruned, root, older_than_days
    )


def teardown(
    staging_uri: str,
    *,
    slug: str,
    exit_code: int,
    s3_client=None,
    now: Optional[datetime] = None,
    dry_run: bool = False,
    prune: bool = True,
    staging_retention_days: int = STAGING_RETENTION_DAYS,
    evidence_retention_days: int = EVIDENCE_RETENTION_DAYS,
):
    """Preserve evidence when the run failed, then clean the staging prefix.

    Args:
        staging_uri: ``s3://bucket/tmp/spot_<stage>/<run-id>`` — the prefix the
            launcher staged configs into and pointed SSM's ``OutputS3KeyPrefix``
            beneath.
        slug: stage slug for the durable key (``predictor-backtest``, ...).
        exit_code: the WORKLOAD's exit status. Zero means there is no failure
            whose evidence could be lost, so nothing is preserved and no
            storage accrues for a successful run.
        s3_client: injected boto3 S3 client (tests, and callers that already
            hold one).
        dry_run: report the plan; touch nothing.

    Returns:
        A dict with ``preserved`` / ``deleted`` / ``durable_uri`` /
        ``staging_uri`` / ``detail``. Never raises for an S3 problem — a
        teardown that raises on the failure path replaces the diagnosis it
        exists to protect.
    """
    result = {
        "staging_uri": staging_uri,
        "slug": slug,
        "exit_code": exit_code,
        "preserved": False,
        "deleted": False,
        "durable_uri": None,
        "detail": "",
        "dry_run": dry_run,
        "pruned": None,
    }
    try:
        bucket, prefix = parse_s3_uri(staging_uri)
    except StagingUriError as exc:
        result["detail"] = "refused: {}".format(exc)
        return result

    if s3_client is None:
        try:
            import boto3

            s3_client = boto3.client("s3")
        except Exception as exc:
            result["detail"] = "no S3 client ({}: {}) — staging LEFT IN PLACE".format(
                type(exc).__name__, exc
            )
            return result

    resolved_now = now or datetime.now(timezone.utc)

    def _sweep():
        """Bound both retained trees. Retires `crucible-backtester#675`'s
        per-repo bash prune, which only ever reached its own stage's prefix and
        needed a BSD/GNU `date` fallback to compute the cutoff."""
        if not prune or dry_run:
            return ""
        staging_root = prefix.rstrip("/").rsplit("/", 1)[0] + "/"
        n1, d1 = _prune(
            s3_client,
            bucket,
            staging_root,
            older_than_days=staging_retention_days,
            now=resolved_now,
        )
        n2, d2 = _prune(
            s3_client,
            bucket,
            "{}/{}/".format(DURABLE_PREFIX, slug or "unknown-stage"),
            older_than_days=evidence_retention_days,
            now=resolved_now,
        )
        result["pruned"] = n1 + n2
        return "; {}; {}".format(d1, d2)

    if exit_code == 0:
        if dry_run:
            result["detail"] = "would delete staging (run succeeded; nothing to preserve)"
            return result
        ok, detail = _delete_staging(s3_client, bucket, prefix)
        result["deleted"] = ok
        result["detail"] = "run succeeded; no evidence retained; {}{}".format(
            detail, _sweep()
        )
        return result

    dest = durable_prefix_for(slug, prefix, now=resolved_now)
    result["durable_uri"] = "s3://{}/{}/".format(bucket, dest)
    if dry_run:
        result["detail"] = "would preserve to {} then delete staging".format(
            result["durable_uri"]
        )
        return result

    ok, detail = _preserve(s3_client, bucket, prefix, dest)
    result["preserved"] = ok
    if not ok:
        # The whole point. Staging is the only remaining copy of this failure's
        # evidence, so it is NOT deleted — an un-lifecycled prefix left behind
        # is a cost; a lost diagnosis is a repeat of I7442.
        result["detail"] = (
            "EVIDENCE PRESERVATION FAILED ({}) — staging {} LEFT IN PLACE so the "
            "failure's only remaining log is not destroyed by its own teardown "
            "(alpha-engine-config-I7442)".format(detail, staging_uri)
        )
        return result

    del_ok, del_detail = _delete_staging(s3_client, bucket, prefix)
    result["deleted"] = del_ok
    result["detail"] = "{}; {}{}".format(detail, del_detail, _sweep())
    return result


def _render(result) -> str:
    """One human line for the launcher's stdout — this reaches the SF cause."""
    if result.get("preserved"):
        return "  spot_evidence: failure evidence preserved -> {} ({})".format(
            result.get("durable_uri"), result.get("detail")
        )
    if result.get("exit_code") == 0:
        return "  spot_evidence: {}".format(result.get("detail"))
    return "  spot_evidence: {}".format(result.get("detail"))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m krepis.spot_evidence",
        description=(
            "Spot-teardown chokepoint: on a FAILED run, copy the staging prefix "
            "(which holds SSM's full remote stdout/stderr) to a durable prefix "
            "BEFORE deleting it; on a successful run, just delete it. Replaces "
            "the `aws s3 rm \"$S3_STAGING\" --recursive` line every spot "
            "launcher ran inside its own failure handler "
            "(alpha-engine-config-I7442)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("teardown", help="Preserve-then-clean the staging prefix.")
    p.add_argument(
        "--staging",
        required=True,
        help="s3://bucket/tmp/spot_<stage>/<run-id> — the launcher's staging prefix.",
    )
    p.add_argument("--slug", required=True, help="Stage slug for the durable key.")
    p.add_argument(
        "--exit-code",
        type=int,
        required=True,
        help=(
            "The WORKLOAD's exit status. Non-zero preserves evidence first; "
            "zero deletes staging with nothing retained."
        ),
    )
    p.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    p.add_argument("--dry-run", action="store_true", help="Report the plan only.")
    p.add_argument(
        "--no-prune",
        action="store_true",
        help=(
            "Skip the retention sweep of tmp/spot_<slug>/ and "
            "_spot_evidence/<slug>/ that otherwise runs on every teardown."
        ),
    )
    p.add_argument(
        "--staging-retention-days",
        type=int,
        default=STAGING_RETENTION_DAYS,
        help="Days a leftover staging prefix is kept (default: {}).".format(
            STAGING_RETENTION_DAYS
        ),
    )
    p.add_argument(
        "--evidence-retention-days",
        type=int,
        default=EVIDENCE_RETENTION_DAYS,
        help="Days a preserved failure record is kept (default: {}).".format(
            EVIDENCE_RETENTION_DAYS
        ),
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    result = teardown(
        args.staging,
        slug=args.slug,
        exit_code=args.exit_code,
        dry_run=args.dry_run,
        prune=not args.no_prune,
        staging_retention_days=args.staging_retention_days,
        evidence_retention_days=args.evidence_retention_days,
    )
    if args.json:
        print(json.dumps(result))
    else:
        print(_render(result))
    # Deliberately 0 on every path: the caller's `set -e` EXIT trap must
    # continue to re-exit with the WORKLOAD's status, and a janitor that can
    # overwrite that status is the alpha-engine-config-I7442 class of defect
    # one layer up. Failures are reported in the rendered line and the JSON.
    return 0


if __name__ == "__main__":
    sys.exit(main())
