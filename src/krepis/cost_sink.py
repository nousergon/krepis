"""Persistence for the cost records :mod:`krepis.cost` produces.

**Why this exists.** :meth:`krepis.llm.LLMClient` builds a priced cost record
for every completed call and hands it to a ``cost_sink`` callable. Something
has to write those records somewhere durable. Left to each consumer, that
"something" gets reimplemented per repo and the copies drift — and a cost
sink that drifts fails the way cost telemetry always fails: silently, with
no wrong answer to notice, discovered on an invoice weeks later.

So the sink ships once, here, next to the record producer.

**Nothing fleet-specific is baked in.** Bucket, key prefix and run id are
constructor arguments. This is a public, pip-installable library; it knows
how to write partitioned JSONL to S3 and nothing about anyone's data layout.

Usage::

    from krepis.cost_sink import S3JsonlCostSink
    from krepis.llm import LLMClient

    with S3JsonlCostSink(bucket="my-bucket", prefix="cost_raw",
                         run_id="2026-07-28T09:00Z-abc123") as sink:
        client = LLMClient(spec, callsite_id="my-callsite", cost_sink=sink)
        client.complete(system="...", user_content="...")
    # one PUT per (date, run_id, callsite_id) on exit

**Buffering is the point, not an optimization.** An agentic loop makes
hundreds of calls; a sink that PUT per record would issue hundreds of S3
writes per run. Records accumulate in memory and flush as one JSONL object
per ``(date, run_id, callsite_id)`` — matching the shape the fleet's
retired per-frame tracker used, so downstream aggregators that glob
``*.jsonl`` under a date prefix need no change.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import uuid
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["S3JsonlCostSink", "resolve_run_id"]

# Flush automatically once a single (date, run_id, callsite_id) group reaches
# this many records. Bounds how much is lost if the process dies without a
# clean shutdown — the alternative (buffer everything until close) turns any
# hard kill into total telemetry loss for the run. 200 keeps the common
# agentic loop to a single PUT while capping worst-case loss.
DEFAULT_FLUSH_THRESHOLD = 200


def resolve_run_id(env_var: str = "KREPIS_RUN_ID") -> str:
    """Return a stable run identifier for this process.

    Reads ``env_var`` when set — orchestrators (Step Functions, CI, a
    dispatcher) generally already have a run id worth joining on, and
    inventing a second one destroys that join. Falls back to a random id so
    a sink is never unusable for want of configuration.

    **No format is imposed.** An earlier fleet aggregator required run ids to
    start with an ISO date and silently discarded 100% of production rows for
    17 days when a producer's id format changed
    (alpha-engine-config-I5206). Run ids are opaque here, deliberately.
    """
    value = os.environ.get(env_var, "").strip()
    return value or f"krepis-{uuid.uuid4().hex[:12]}"


class S3JsonlCostSink:
    """Buffer cost records and flush them as partitioned JSONL to S3.

    Key layout: ``{prefix}/{date}/{run_id}/{callsite_id}.{seq}.jsonl``

    ``date`` comes from each record's own ``ts`` (UTC, set by
    :func:`krepis.cost.record_llm_call`), not from flush time — so a record
    lands in the partition describing when the *call* happened, which is
    what a reader comparing spend across days expects. ``seq`` increments per
    flush of a group so a second flush cannot overwrite the first's rows.

    Args:
        bucket: Destination S3 bucket.
        prefix: Key prefix, no trailing slash.
        run_id: Run identifier. Defaults to :func:`resolve_run_id`.
        s3_client: Injected boto3-style client. Constructed lazily when
            omitted, so importing this module costs nothing.
        flush_threshold: Auto-flush a group at this many buffered records.
        register_atexit: Flush on interpreter exit as a safety net for
            callers that forget to close. Disable in tests.

    **Flush failures never raise.** A sink that raises would let a telemetry
    fault take down the work it was measuring — the same trade
    :meth:`krepis.llm.LLMClient._emit_cost_record` makes and for the same
    reason. Failures are logged at ERROR and counted on
    :attr:`flush_errors`, which is the surface a caller asserts on.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        run_id: Optional[str] = None,
        s3_client: Any = None,
        flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
        register_atexit: bool = True,
    ):
        if not bucket or not isinstance(bucket, str):
            raise ValueError("S3JsonlCostSink requires a non-empty bucket")
        if not prefix or not isinstance(prefix, str):
            raise ValueError("S3JsonlCostSink requires a non-empty prefix")
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.run_id = run_id or resolve_run_id()
        self.flush_threshold = max(1, int(flush_threshold))
        self.flush_errors = 0
        self.records_written = 0
        self._s3 = s3_client
        self._buffers: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self._seq: dict[tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()
        self._closed = False
        if register_atexit:
            atexit.register(self.flush)

    # ── sink protocol ────────────────────────────────────────────────

    def __call__(self, record: dict) -> None:
        """Accept one cost record. This is the ``cost_sink`` callable."""
        group = (self._record_date(record), str(record.get("callsite_id") or "unknown"))
        with self._lock:
            self._buffers[group].append(record)
            over = len(self._buffers[group]) >= self.flush_threshold
        if over:
            self.flush()

    def __enter__(self) -> "S3JsonlCostSink":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ── flushing ─────────────────────────────────────────────────────

    def flush(self) -> int:
        """Write every buffered group. Returns the number of objects PUT."""
        with self._lock:
            groups = {k: v for k, v in self._buffers.items() if v}
            self._buffers.clear()
            seqs = {}
            for key in groups:
                seqs[key] = self._seq[key]
                self._seq[key] += 1
        written = 0
        for (date_str, callsite_id), records in groups.items():
            if self._put_group(date_str, callsite_id, seqs[(date_str, callsite_id)], records):
                written += 1
        return written

    def close(self) -> None:
        self.flush()
        self._closed = True

    def _put_group(
        self, date_str: str, callsite_id: str, seq: int, records: list[dict]
    ) -> bool:
        key = f"{self.prefix}/{date_str}/{self.run_id}/{callsite_id}.{seq}.jsonl"
        body = "\n".join(json.dumps(r, default=str) for r in records).encode("utf-8")
        try:
            self._client().put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType="application/x-ndjson",
            )
        except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
            # DELIBERATE non-raising degradation. Rationale, per the
            # fail-loud rule's written-justification requirement:
            # (a) Failure mode swallowed: the S3 PUT failed (credentials,
            #     network, bucket policy).
            # (b) Why the primary deliverable survives: these records
            #     describe work that already completed successfully.
            #     Raising here would let a telemetry fault take down the
            #     work it was measuring.
            # (c) Recording surface: this ERROR log, the `flush_errors`
            #     counter a caller can assert on, and — for consumers who
            #     monitor the destination prefix for freshness — the
            #     absence of the object itself. That last layer is what
            #     catches SUSTAINED loss when logs go unread, and its
            #     absence is what let alpha-engine-config-I5206 run 17 days.
            self.flush_errors += 1
            logger.error(
                "cost sink PUT failed for s3://%s/%s (%d record(s)): %s",
                self.bucket, key, len(records), exc,
            )
            return False
        self.records_written += len(records)
        logger.debug(
            "cost sink wrote %d record(s) to s3://%s/%s",
            len(records), self.bucket, key,
        )
        return True

    # ── helpers ──────────────────────────────────────────────────────

    def _client(self) -> Any:
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3")
        return self._s3

    @staticmethod
    def _record_date(record: dict) -> str:
        """Partition date from the record's own UTC ``ts``.

        Falls back to ``"unknown-date"`` rather than to *today* when ``ts``
        is missing or unparseable. Substituting today's date would file the
        record under a partition it does not belong to — a plausible wrong
        answer, which is harder to notice than an obviously wrong one.
        """
        ts = record.get("ts")
        if isinstance(ts, str) and len(ts) >= 10 and ts[4] == "-" and ts[7] == "-":
            return ts[:10]
        return "unknown-date"
