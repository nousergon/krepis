"""Persistence for the ``llm_trace.v1`` records :mod:`krepis.llm_trace` builds.

The sibling of :mod:`krepis.cost_sink`, built on the same three rules:

**Opting in is an environment fact, not a code fact.**
:func:`default_trace_sink_from_env` builds a process-scoped sink from
``KREPIS_TRACE_SINK_BUCKET`` + ``KREPIS_TRACE_SINK_PREFIX``, and
:class:`krepis.llm.LLMClient` consults it whenever the caller passed no
``trace_sink``. Every call site in a configured process is traced by
construction; with neither variable set nothing is traced and a public
consumer pays nothing. Exactly one set raises :exc:`TraceSinkConfigError`,
for the reason :func:`krepis.cost_sink.default_sink_from_env` gives.

**Buffering is the point.** Rows accumulate in memory and flush as one gzip
JSONL object per ``(date, callsite_id)`` group; content blobs flush with
them, or sooner once their bytes pass a threshold.

**Flush failures never raise.** A telemetry fault must not take down the
work it is measuring. Failures are logged at ERROR, counted on
:attr:`S3JsonlTraceSink.flush_errors` / :attr:`S3JsonlTraceSink.blob_errors`,
and the failed records are re-buffered for the next flush, bounded.

Key layout under ``{prefix}`` (nothing fleet-specific is baked in)::

    rows/date=YYYY-MM-DD/callsite=<id>/<run_id>.<writer>.<seq>.jsonl.gz
    blobs/sha256/<ab>/<sha256>.gz

Rows are hive-partitioned, so a reader can prune by date and call site
(``read_json_auto('s3://…/rows/date=2026-10-*/callsite=*/*.jsonl.gz',
hive_partitioning=true)`` in DuckDB). ``<writer>`` is a per-process token:
two processes sharing one run id — concurrent Lambda containers, say — can
never overwrite each other's rows.

Blobs are content-addressed: the key is the sha256 of the *uncompressed*
bytes, written with ``If-None-Match: *``, so a repeated system prompt or a
re-cited chunk is stored once no matter how many calls reference it. The
object's ``media-type`` metadata says how to read the decompressed bytes.
Gzip is written with ``mtime=0``, so the stored bytes are deterministic too.
"""

from __future__ import annotations

import atexit
import functools
import gzip
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from krepis import s3_surface
from krepis.cost_sink import resolve_run_id
from krepis.llm_trace import TraceBlob

logger = logging.getLogger(__name__)

__all__ = [
    "S3JsonlTraceSink",
    "InMemoryTraceSink",
    "TraceSinkConfigError",
    "default_trace_sink_from_env",
    "flush_default_trace_sink",
    "flush_traces_on_exit",
    "reset_default_trace_sink_for_tests",
    "BUCKET_ENV_VAR",
    "PREFIX_ENV_VAR",
    "rows_key",
    "blob_key",
]

#: Environment variables that switch the process-default trace sink on.
BUCKET_ENV_VAR = "KREPIS_TRACE_SINK_BUCKET"
PREFIX_ENV_VAR = "KREPIS_TRACE_SINK_PREFIX"

#: Declared S3 surface (``krepis.s3_surface``): the VARIABLE, not a prefix,
#: because where traces go is the consumer's deploy configuration.
S3_SURFACE = (
    s3_surface.from_env_var(PREFIX_ENV_VAR, s3_surface.MODE_READWRITE),
)

DEFAULT_FLUSH_THRESHOLD = 200
#: Flush pending blobs once they hold this many uncompressed bytes, so a run
#: of large prompts does not hold hundreds of megabytes waiting for rows.
DEFAULT_BLOB_FLUSH_BYTES = 8 * 1024 * 1024
#: Retained-buffer cap after repeated failures, as a multiple of the
#: thresholds above. Past it, the oldest records are dropped and counted.
MAX_RETAINED_MULTIPLE = 10

_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.\-]")

TraceSinkCallable = Callable[[Dict[str, Any], Mapping[str, TraceBlob]], None]

_DEFAULT_SINK: "Optional[S3JsonlTraceSink]" = None
_DEFAULT_SINK_KEY: "Optional[Tuple[str, str, str]]" = None
_DEFAULT_SINK_LOCK = threading.Lock()


class TraceSinkConfigError(ValueError):
    """The default trace-sink environment is half-configured."""


def _safe_segment(value: str) -> str:
    return _SAFE_SEGMENT_RE.sub("_", value or "unknown") or "unknown"


def rows_key(prefix: str, date_str: str, callsite_id: str, run_id: str,
             writer_id: str, seq: int) -> str:
    """Object key for one flushed group of rows."""
    return (
        f"{prefix}/rows/date={date_str}/callsite={_safe_segment(callsite_id)}/"
        f"{_safe_segment(run_id)}.{writer_id}.{seq}.jsonl.gz"
    )


def blob_key(prefix: str, sha256: str) -> str:
    """Object key for one content-addressed blob."""
    return f"{prefix}/blobs/sha256/{sha256[:2]}/{sha256}.gz"


def _gzip(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def _is_precondition_failed(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("PreconditionFailed", "412") or status == 412


def _row_date(row: Mapping[str, Any]) -> str:
    ts = row.get("started_at")
    if isinstance(ts, str) and len(ts) >= 10 and ts[4] == "-" and ts[7] == "-":
        return ts[:10]
    return "unknown-date"


class InMemoryTraceSink:
    """A trace sink that keeps everything in memory.

    For tests, and for a consumer that wants to inspect the trace of a call
    it just made. Implements the same callable protocol as
    :class:`S3JsonlTraceSink`: ``sink(row, blobs)``.
    """

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.blobs: Dict[str, TraceBlob] = {}

    def __call__(self, row: Dict[str, Any], blobs: Mapping[str, TraceBlob]) -> None:
        self.rows.append(row)
        for sha, blob in blobs.items():
            self.blobs.setdefault(sha, blob)


class S3JsonlTraceSink:
    """Buffer trace rows and content blobs, and flush them to S3.

    Args:
        bucket: Destination bucket.
        prefix: Key prefix, no trailing slash (``llm_traces/v1``, say).
        run_id: Run identifier for rows that carry none. Defaults to
            :func:`krepis.cost_sink.resolve_run_id`.
        s3_client: Injected boto3-style client; built lazily when omitted.
        flush_threshold: Auto-flush once a row group holds this many rows.
        blob_flush_bytes: Auto-flush blobs once pending bytes pass this.
        register_atexit: Flush at interpreter exit. Disable in tests.

    Counters a caller can assert on: ``rows_written``, ``blobs_written``,
    ``blobs_deduplicated`` (already present, in this process or in S3),
    ``flush_errors`` (failed row PUTs), ``blob_errors`` (failed blob PUTs),
    ``rows_dropped`` and ``blobs_dropped`` (permanent loss past the retained
    cap or at :meth:`close`).
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        run_id: Optional[str] = None,
        s3_client: Any = None,
        flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
        blob_flush_bytes: int = DEFAULT_BLOB_FLUSH_BYTES,
        register_atexit: bool = True,
    ):
        if not bucket or not isinstance(bucket, str):
            raise ValueError("S3JsonlTraceSink requires a non-empty bucket")
        if not prefix or not isinstance(prefix, str) or not prefix.strip("/"):
            raise ValueError("S3JsonlTraceSink requires a non-empty prefix")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.run_id = run_id or resolve_run_id()
        self.writer_id = uuid.uuid4().hex[:8]
        self.flush_threshold = max(1, int(flush_threshold))
        self.blob_flush_bytes = max(1, int(blob_flush_bytes))
        self.rows_written = 0
        self.blobs_written = 0
        self.blobs_deduplicated = 0
        self.flush_errors = 0
        self.blob_errors = 0
        self.rows_dropped = 0
        self.blobs_dropped = 0
        self._s3 = s3_client
        self._rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self._seq: Dict[Tuple[str, str], int] = defaultdict(int)
        self._blobs: "OrderedDict[str, TraceBlob]" = OrderedDict()
        self._blob_bytes = 0
        self._stored: set = set()
        self._lock = threading.Lock()
        if register_atexit:
            atexit.register(self.close)

    # ── sink protocol ────────────────────────────────────────────────

    def __call__(self, row: Dict[str, Any], blobs: Mapping[str, TraceBlob]) -> None:
        """Accept one row and the blobs it references."""
        if not row.get("run_id"):
            row["run_id"] = self.run_id
        group = (_row_date(row), str(row.get("callsite_id") or "unknown"))
        with self._lock:
            for sha, blob in blobs.items():
                if sha in self._stored or sha in self._blobs:
                    self.blobs_deduplicated += 1
                    continue
                self._blobs[sha] = blob
                self._blob_bytes += len(blob.data)
            self._rows[group].append(row)
            rows_over = len(self._rows[group]) >= self.flush_threshold
            blobs_over = self._blob_bytes >= self.blob_flush_bytes
        if rows_over:
            self.flush()
        elif blobs_over:
            self._flush_blobs()

    def __enter__(self) -> "S3JsonlTraceSink":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ── flushing ─────────────────────────────────────────────────────

    def flush(self) -> int:
        """Write pending blobs, then every buffered row group.

        Returns the number of ROW objects written. Blobs go first so that a
        row is never readable before the content it points at; a blob that
        fails is re-buffered and its row is still written — a row pointing
        at a missing blob is detectable by a reader, a missing row is not.
        """
        self._flush_blobs()
        with self._lock:
            groups = {k: v for k, v in self._rows.items() if v}
            self._rows.clear()
            seqs = {key: self._seq[key] for key in groups}
        written = 0
        failed: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for key, rows in groups.items():
            if self._put_rows(key, seqs[key], rows):
                written += 1
                with self._lock:
                    self._seq[key] = seqs[key] + 1
            else:
                failed[key] = rows
        if failed:
            cap = self.flush_threshold * MAX_RETAINED_MULTIPLE
            with self._lock:
                for key, rows in failed.items():
                    merged = rows + self._rows[key]
                    if len(merged) > cap:
                        overflow = len(merged) - cap
                        merged = merged[overflow:]
                        self.rows_dropped += overflow
                        logger.error(
                            "trace sink: dropping %d row(s) for %s past the "
                            "retained cap (%d) — rows_dropped=%d",
                            overflow, key, cap, self.rows_dropped,
                        )
                    self._rows[key] = merged
        return written

    def close(self) -> None:
        """Flush, then count anything still buffered as permanently lost."""
        self.flush()
        with self._lock:
            rows_left = sum(len(v) for v in self._rows.values())
            blobs_left = len(self._blobs)
            self.rows_dropped += rows_left
            self.blobs_dropped += blobs_left
            self._rows.clear()
            self._blobs.clear()
            self._blob_bytes = 0
        if rows_left or blobs_left:
            logger.warning(
                "trace sink: close() with %d row(s) and %d blob(s) unflushed "
                "after failed writes — PERMANENTLY LOST (flush_errors=%d, "
                "blob_errors=%d)",
                rows_left, blobs_left, self.flush_errors, self.blob_errors,
            )

    def _flush_blobs(self) -> int:
        with self._lock:
            pending = list(self._blobs.items())
            self._blobs.clear()
            self._blob_bytes = 0
        written = 0
        failed: List[Tuple[str, TraceBlob]] = []
        for sha, blob in pending:
            outcome = self._put_blob(sha, blob)
            if outcome is None:
                failed.append((sha, blob))
                continue
            with self._lock:
                self._stored.add(sha)
            if outcome:
                written += 1
            else:
                self.blobs_deduplicated += 1
        if failed:
            cap = self.blob_flush_bytes * MAX_RETAINED_MULTIPLE
            with self._lock:
                merged: "OrderedDict[str, TraceBlob]" = OrderedDict(failed)
                for sha, blob in self._blobs.items():
                    merged.setdefault(sha, blob)
                total = sum(len(b.data) for b in merged.values())
                while merged and total > cap:
                    _, dropped = merged.popitem(last=False)
                    total -= len(dropped.data)
                    self.blobs_dropped += 1
                self._blobs = merged
                self._blob_bytes = total
        return written

    def _put_rows(self, key: Tuple[str, str], seq: int, rows: List[Dict[str, Any]]) -> bool:
        date_str, callsite_id = key
        object_key = rows_key(
            self.prefix, date_str, callsite_id, self.run_id, self.writer_id, seq
        )
        body = "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":"), default=str) + "\n"
            for r in rows
        ).encode("utf-8")
        try:
            self._client().put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=_gzip(body),
                ContentType="application/gzip",
                Metadata={"media-type": "application/x-ndjson"},
            )
        except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
            # DELIBERATE non-raising degradation, per the fail-loud rule's
            # written-justification requirement:
            # (a) swallowed: the S3 PUT failed (credentials, network, policy);
            # (b) the primary deliverable survives: these rows describe calls
            #     that already returned their answers to the caller;
            # (c) recorded on: this ERROR line, `flush_errors`, the re-buffered
            #     rows (retried on the next flush), and the freshness monitor
            #     on the destination prefix, which catches sustained loss.
            self.flush_errors += 1
            logger.error(
                "trace sink PUT failed for s3://%s/%s (%d row(s)): %s",
                self.bucket, object_key, len(rows), exc,
            )
            return False
        self.rows_written += len(rows)
        return True

    def _put_blob(self, sha: str, blob: TraceBlob) -> Optional[bool]:
        """PUT one blob if absent. True = written, False = already there,
        None = failed (counted on ``blob_errors``)."""
        object_key = blob_key(self.prefix, sha)
        try:
            self._client().put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=_gzip(blob.data),
                ContentType="application/gzip",
                Metadata={"media-type": blob.media_type},
                IfNoneMatch="*",
            )
        except Exception as exc:  # noqa: BLE001 — duck-typed boto errors
            if _is_precondition_failed(exc):
                # Content-addressed: an existing object at this key holds
                # these exact bytes. Nothing was lost.
                return False
            # Same deliberate non-raising degradation as `_put_rows`; recorded
            # on this ERROR line, `blob_errors`, and the re-buffered blob.
            self.blob_errors += 1
            logger.error(
                "trace sink blob PUT failed for s3://%s/%s (%d bytes): %s",
                self.bucket, object_key, len(blob.data), exc,
            )
            return None
        self.blobs_written += 1
        return True

    def _client(self) -> Any:
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3")
        return self._s3


# ── process-default sink ──────────────────────────────────────────────


def default_trace_sink_from_env(*, s3_client: Any = None) -> "Optional[S3JsonlTraceSink]":
    """Return the process-scoped default trace sink, or ``None``.

    Reads :data:`BUCKET_ENV_VAR` and :data:`PREFIX_ENV_VAR`. Neither set →
    ``None``. Both set → one :class:`S3JsonlTraceSink` per process, shared by
    every :class:`krepis.llm.LLMClient` built in it. Exactly one set raises
    :exc:`TraceSinkConfigError` at client construction, before any spend.
    """
    bucket = os.environ.get(BUCKET_ENV_VAR, "").strip()
    prefix = os.environ.get(PREFIX_ENV_VAR, "").strip()
    if not bucket and not prefix:
        return None
    if not bucket or not prefix:
        missing = BUCKET_ENV_VAR if not bucket else PREFIX_ENV_VAR
        present = PREFIX_ENV_VAR if not bucket else BUCKET_ENV_VAR
        raise TraceSinkConfigError(
            f"{present} is set but {missing} is not. The default trace sink "
            f"needs both; a half-configured environment would trace nothing "
            f"while looking exactly like one that was never configured. Set "
            f"{missing}, or unset {present} to disable tracing deliberately."
        )
    key = (bucket, prefix, os.environ.get("KREPIS_RUN_ID", "").strip())
    global _DEFAULT_SINK, _DEFAULT_SINK_KEY
    with _DEFAULT_SINK_LOCK:
        if _DEFAULT_SINK is None or _DEFAULT_SINK_KEY != key:
            run_id = resolve_run_id()
            _DEFAULT_SINK = S3JsonlTraceSink(
                bucket=bucket,
                prefix=prefix,
                run_id=run_id,
                s3_client=s3_client,
                register_atexit=True,
            )
            _DEFAULT_SINK_KEY = key
            logger.info(
                "trace sink: process default active -> s3://%s/%s (run_id=%s)",
                bucket, prefix, run_id,
            )
        return _DEFAULT_SINK


def flush_default_trace_sink() -> int:
    """Flush the process-default trace sink now. Returns row objects PUT.

    Required at the end of every AWS Lambda handler that makes LLM calls,
    for the reason :func:`krepis.cost_sink.flush_default_sink` gives: a
    frozen Lambda container never runs ``atexit``. Returns 0 when no default
    sink is configured; never raises.
    """
    with _DEFAULT_SINK_LOCK:
        sink = _DEFAULT_SINK
    if sink is None:
        return 0
    return sink.flush()


def flush_traces_on_exit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: flush the process-default trace sink when ``func`` returns,
    including on the exception path. The Lambda-handler form of
    :func:`flush_default_trace_sink`; never raises."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            try:
                with _DEFAULT_SINK_LOCK:
                    sink = _DEFAULT_SINK
                if sink is None:
                    logger.debug("trace sink: no process-default sink configured")
                else:
                    n = sink.flush()
                    logger.info(
                        "trace sink: flushed %d row object(s) on handler exit "
                        "(flush_errors=%d, blob_errors=%d, rows_dropped=%d)",
                        n, sink.flush_errors, sink.blob_errors, sink.rows_dropped,
                    )
            except Exception:  # noqa: BLE001 — telemetry never fails the work
                logger.exception("trace sink: flush on handler exit failed")

    return wrapper


def reset_default_trace_sink_for_tests() -> None:
    """Drop the memoized process default. Tests only."""
    global _DEFAULT_SINK, _DEFAULT_SINK_KEY
    with _DEFAULT_SINK_LOCK:
        _DEFAULT_SINK = None
        _DEFAULT_SINK_KEY = None
