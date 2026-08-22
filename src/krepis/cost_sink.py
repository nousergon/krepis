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

**Opting in is an environment fact, not a code fact.**
:func:`default_sink_from_env` builds a process-scoped sink from
``KREPIS_COST_SINK_BUCKET`` + ``KREPIS_COST_SINK_PREFIX``, and
:class:`krepis.llm.LLMClient` consults it whenever the caller passed no
``cost_sink``. That inversion is the whole point: while the sink was a
constructor argument, every call site had to remember to pass it, so cost
telemetry covered exactly the call sites whose authors happened to think
of it — measured 2026-08-13, one process (the Think Tank) out of every
LLM-calling stage in the Alpha Engine weekly pipeline
(``alpha-engine-config-I7179``). A new call site now emits by
construction, and *forgetting* is no longer expressible.

Nothing fleet-specific is added by that: with neither variable set the
default is ``None`` and a public consumer pays nothing.
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import threading
import uuid
from collections import defaultdict
from typing import Any, Callable, Optional
from krepis import s3_surface

logger = logging.getLogger(__name__)

__all__ = [
    "S3JsonlCostSink",
    "resolve_run_id",
    "default_sink_from_env",
    "flush_default_sink",
    "flush_cost_on_exit",
    "reset_default_sink_for_tests",
    "BUCKET_ENV_VAR",
    "PREFIX_ENV_VAR",
    "CostSinkConfigError",
]

#: Environment variables that switch the process-default sink on.
#: Both must be set; either alone is a misconfiguration and raises.
BUCKET_ENV_VAR = "KREPIS_COST_SINK_BUCKET"
PREFIX_ENV_VAR = "KREPIS_COST_SINK_PREFIX"

#: Declared S3 surface (``krepis.s3_surface``, alpha-engine-config-I8156).
#: Opting in is an environment fact, not a code fact, so the declaration names
#: the VARIABLE rather than a prefix: a consumer resolves it against its own
#: deploy configuration and checks the value that will actually run. An unset
#: variable resolves to nothing, which is correct — the sink writes nowhere.
S3_SURFACE = (
    s3_surface.from_env_var(PREFIX_ENV_VAR, s3_surface.MODE_READWRITE),
)

_DEFAULT_SINK: "Optional[S3JsonlCostSink]" = None
_DEFAULT_SINK_KEY: "Optional[tuple[str, str, str]]" = None
_DEFAULT_SINK_LOCK = threading.Lock()


class CostSinkConfigError(ValueError):
    """Raised when the default-sink environment is half-configured.

    Deliberately an error rather than a fall-through to "no sink": a
    process that names a bucket and forgets the prefix (or vice versa)
    reads exactly like a process that was never meant to emit, and the
    two are indistinguishable on every surface until an invoice arrives.
    """

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
        # ``run_id`` on the ROW, not only in the key (alpha-engine-config-I7393).
        # The sink already partitions by it — {prefix}/{date}/{run_id}/... — so
        # it is the only layer that knows it; the record builder in cost.py
        # cannot. The contract (nousergon-lib transparency_inventory.yaml
        # `cost_telemetry`) asserts it as a COLUMN, and the aggregator
        # (crucible-research scripts/aggregate_costs.py) builds its DataFrame
        # straight from these rows, so a value present only in the S3 key never
        # reaches the parquet. An explicitly-set run_id on the record wins.
        record.setdefault("run_id", self.run_id)
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


# ── process-default sink ──────────────────────────────────────────────


def default_sink_from_env(
    *, s3_client: Any = None
) -> "Optional[Callable[[dict], None]]":
    """Return the process-scoped default sink, or ``None`` when unconfigured.

    Reads :data:`BUCKET_ENV_VAR` and :data:`PREFIX_ENV_VAR`. Neither set
    → ``None``, and a consumer that never asked for cost telemetry pays
    nothing. Both set → one :class:`S3JsonlCostSink` per process, reused
    across every :class:`krepis.llm.LLMClient` built in it, so a lane that
    constructs a fresh client per request still issues one PUT per
    ``(date, run_id, callsite_id)`` rather than one per call.

    **Exactly one set raises** :exc:`CostSinkConfigError`. Falling back to
    "no sink" on a half-configured environment is how a deploy-time typo
    becomes months of unattributed spend: nothing errors, nothing logs,
    and the destination prefix simply keeps not growing — which is
    indistinguishable from a quiet week. Raising costs a loud failure at
    client construction, *before* the first billable call, and that is
    the trade this fleet has already paid the other way once
    (``alpha-engine-config-I5206``, 17 days of 100% row loss).

    The run id comes from :func:`resolve_run_id`, so an orchestrator that
    exports ``KREPIS_RUN_ID`` gets its own join key and one that does not
    still gets a stable per-process id.
    """
    bucket = os.environ.get(BUCKET_ENV_VAR, "").strip()
    prefix = os.environ.get(PREFIX_ENV_VAR, "").strip()
    if not bucket and not prefix:
        return None
    if not bucket or not prefix:
        missing = BUCKET_ENV_VAR if not bucket else PREFIX_ENV_VAR
        present = PREFIX_ENV_VAR if not bucket else BUCKET_ENV_VAR
        raise CostSinkConfigError(
            f"{present} is set but {missing} is not. The default cost sink "
            f"needs both; a half-configured environment would emit nothing "
            f"while looking exactly like one that was never configured. Set "
            f"{missing}, or unset {present} to disable cost emission "
            f"deliberately."
        )

    # The declared run id, NOT the resolved one: resolve_run_id() mints a
    # fresh random value on every call when KREPIS_RUN_ID is unset, so
    # keying on it would rebuild the sink — and therefore the buffer — per
    # client, which is exactly the one-PUT-per-call shape this class exists
    # to avoid.
    key = (bucket, prefix, os.environ.get("KREPIS_RUN_ID", "").strip())
    global _DEFAULT_SINK, _DEFAULT_SINK_KEY
    with _DEFAULT_SINK_LOCK:
        if _DEFAULT_SINK is None or _DEFAULT_SINK_KEY != key:
            run_id = resolve_run_id()
            _DEFAULT_SINK = S3JsonlCostSink(
                bucket=bucket,
                prefix=prefix,
                run_id=run_id,
                s3_client=s3_client,
                register_atexit=True,
            )
            _DEFAULT_SINK_KEY = key
            logger.info(
                "cost sink: process default active -> s3://%s/%s (run_id=%s)",
                bucket, prefix, run_id,
            )
        return _DEFAULT_SINK


def flush_default_sink() -> int:
    """Flush the process-default sink now. Returns the number of objects PUT.

    **Required at the end of every AWS Lambda handler that makes LLM calls**
    (alpha-engine-config-I7423). The sink buffers to
    :data:`DEFAULT_FLUSH_THRESHOLD` records per ``(date, callsite_id)`` group
    and otherwise relies on the ``atexit`` hook — and **a Lambda container is
    frozen between invocations, not exited, so ``atexit`` does not run.** A
    handler that finishes below the threshold therefore writes NOTHING, and
    the container may be reclaimed hours later without ever unfreezing to a
    normal interpreter shutdown.

    Measured 2026-08-15 on weekly-SF execution ``watch-rerun-2026-08-15-2``:
    ``ReplayConcordance`` ran 812 seconds of DeepSeek calls over 119 artifacts
    — comfortably under the 200-record threshold — and the fan-in coverage
    check reported ``2 stage(s) ran and emitted no cost record ... Observed
    producers: (none)``. The env wiring was correct, the sink was constructed,
    the records were accepted, and every one of them died in the buffer.

    Returns 0 when no default sink is configured, so a caller may invoke this
    unconditionally in a ``finally`` without knowing whether cost telemetry is
    switched on. It never raises: :meth:`S3JsonlCostSink.flush` already
    swallows and counts its own failures, on the principle that a telemetry
    fault must not take down the work it was measuring.
    """
    with _DEFAULT_SINK_LOCK:
        sink = _DEFAULT_SINK
    if sink is None:
        return 0
    return sink.flush()


def flush_cost_on_exit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: flush the process-default cost sink when ``func`` returns.

    The Lambda-handler form of :func:`flush_default_sink`
    (alpha-engine-config-I7423). Wrapping the handler rather than calling the
    flush before each ``return`` is deliberate: a handler grows return paths
    over time — a dry-run short circuit, an early validation bail, an
    exception branch — and a per-return call covers the paths that existed
    when someone last thought about it. The one that gets added next is the
    one that silently stops emitting.

    Flushes on the exception path too, via ``finally``: records already
    accepted describe spend that was really incurred, and losing them because
    the handler failed afterwards is the same loss with a better excuse.

    The flush never raises, so it cannot convert a successful handler into a
    failed one, and it returns 0 when no sink is configured — a consumer with
    cost telemetry switched off pays one function call.

    Usage::

        @flush_cost_on_exit
        def handler(event, context):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            try:
                n = flush_default_sink()
                if n:
                    logger.info("cost sink: flushed %d object(s) on handler exit", n)
            except Exception:  # noqa: BLE001 — telemetry never fails the work
                logger.exception("cost sink: flush on handler exit failed")

    return wrapper


def reset_default_sink_for_tests() -> None:
    """Drop the memoized process default. Tests only."""
    global _DEFAULT_SINK, _DEFAULT_SINK_KEY
    with _DEFAULT_SINK_LOCK:
        _DEFAULT_SINK = None
        _DEFAULT_SINK_KEY = None
