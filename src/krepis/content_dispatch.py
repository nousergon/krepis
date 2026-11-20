"""Durable, no-replay orchestration for an admitted content execution.

The caller owns authentication, trusted normalization, current registry/profile
resolution, quota admission, worker sandboxing, timeouts and cancellation. The
adapter must enforce those execution bounds and report actual served identity.
This module does not launch a process or implement a provider transport.

Supply a dedicated autocommit SQLite connection per worker on a synchronous,
local durable filesystem with private file/directory permissions. All workers
share the same journal and capacity. Protect/replicate the journal as private
consumer data: completed responses are stored here. Restoring an older journal
requires external reconciliation before dispatch is enabled.

A started record is an unresolved effect, NOT a healthy worker heartbeat.
Started/uncertain records retain capacity and are never automatically expired,
cleared or retried. Recovery needs external effect evidence; this primitive
intentionally exposes no unverified reset operation.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import sqlite3
import uuid
from typing import Callable, Optional, cast

from .workload_admission import (
    ContentCandidate, ContentProfile, ContentRequest, admit_content,
)


class AlreadyDispatched(ValueError):
    """This request has an execution record; retrieve/reconcile, never replay."""


class RequestConflict(ValueError):
    """The same consumer request identity names different execution inputs."""


class CapacityUnavailable(ValueError):
    """Unresolved attempts consume the journal's configured execution capacity."""


@dataclass(frozen=True)
class ContentResult:
    text: str
    served_model: str
    trace_id: str

    def __post_init__(self) -> None:
        _text(self.text, "result text", allow_empty=True)
        _text(self.served_model, "served model")
        _text(self.trace_id, "trace identity")
        if len(self.served_model) > 512 or len(self.trace_id) > 512:
            raise ValueError("result identity exceeds bound")


def _text(value: object, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError("invalid " + name)


def _sets(value: object) -> list[str]:
    if isinstance(value, frozenset):
        members = cast("frozenset[object]", value)
        if all(isinstance(v, str) for v in members):
            return sorted(cast("frozenset[str]", members))
    raise TypeError("unsupported binding field")


def _binding(profile: ContentProfile, request: ContentRequest,
             candidate: ContentCandidate, content: str) -> str:
    candidate_fields = asdict(candidate)
    if not candidate.registry_id and not candidate.registry_revision:
        # Preserve the durable binding of callers predating registry integration.
        del candidate_fields["registry_id"]
        del candidate_fields["registry_revision"]
    raw = json.dumps([asdict(profile), asdict(request), candidate_fields, content],
                     default=_sets, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DispatchJournal:
    """A journal shared across adapters; capacity is fixed in durable metadata."""

    def __init__(self, db: sqlite3.Connection, *, max_inflight: int) -> None:
        if type(max_inflight) is not int or not 1 <= max_inflight <= 2**31 - 1:
            raise ValueError("invalid capacity")
        self.db = db
        self._durable()
        db.executescript("""
            CREATE TABLE IF NOT EXISTS content_dispatch_config (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                version INTEGER NOT NULL, capacity INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS content_dispatch_attempts (
                consumer TEXT NOT NULL, request_id TEXT NOT NULL,
                binding TEXT NOT NULL, adapter_id TEXT NOT NULL,
                qualification_ref TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('started','completed','uncertain')),
                result_json TEXT,
                PRIMARY KEY(consumer,request_id));
        """)
        db.execute("INSERT OR IGNORE INTO content_dispatch_config VALUES(1,2,?)", (max_inflight,))
        db.execute("BEGIN IMMEDIATE")
        try:
            version, capacity = db.execute("SELECT version,capacity FROM content_dispatch_config WHERE singleton=1").fetchone()
            if version not in {1, 2} or capacity != max_inflight:
                raise ValueError("journal version/capacity mismatch")
            columns = {row[1] for row in db.execute("PRAGMA table_info(content_dispatch_attempts)")}
            if "execution_id" not in columns:
                db.execute("ALTER TABLE content_dispatch_attempts ADD COLUMN execution_id TEXT")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS content_dispatch_execution_id ON content_dispatch_attempts(execution_id)")
            db.execute("UPDATE content_dispatch_config SET version=2 WHERE singleton=1")
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        self.max_inflight = max_inflight

    def verify_durability(self) -> None:
        """Refuse unavailable durability before a service-owned journal write."""
        self._durable()

    def _durable(self) -> None:
        if self.db.isolation_level is not None or self.db.in_transaction:
            raise ValueError("independent autocommit connection required")
        main = next(row[2] for row in self.db.execute("PRAGMA database_list") if row[1] == "main")
        mode = self.db.execute("PRAGMA main.journal_mode").fetchone()[0]
        sync = self.db.execute("PRAGMA main.synchronous").fetchone()[0]
        if not main or mode not in {"wal", "delete", "truncate", "persist"} or sync < 2:
            raise ValueError("durable synchronous disk-backed journal required")

    def capacity_available(self) -> bool:
        """Point-in-time capacity, never a reservation or provider quota claim."""
        self.verify_durability()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            # A read-only/query-only connection can pass durability and SELECT.
            # Check local write admission without adding or modifying any record.
            self.db.execute("UPDATE content_dispatch_attempts SET status=status WHERE 0")
            count = self.db.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM content_dispatch_attempts "
                "WHERE status!='completed' LIMIT ?)", (self.max_inflight,)).fetchone()[0]
            return count < self.max_inflight
        finally:
            self.db.execute("ROLLBACK")

    def _claim(self, consumer: str, request_id: str, binding: str,
               candidate: ContentCandidate, preflight: Optional[Callable[[], None]] = None) -> str:
        self._durable()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute(
                "SELECT binding FROM content_dispatch_attempts WHERE consumer=? AND request_id=?",
                (consumer, request_id)).fetchone()
            if existing:
                if existing[0] != binding:
                    raise RequestConflict("request identity reused with different inputs")
                raise AlreadyDispatched("request already has a durable attempt")
            count = self.db.execute(
                "SELECT COUNT(*) FROM content_dispatch_attempts WHERE status!='completed'").fetchone()[0]
            if count >= self.max_inflight:
                raise CapacityUnavailable("unresolved attempt capacity exhausted")
            if preflight is not None:
                # Trusted, bounded, effect-free check under the capacity lock.
                # Refusal rolls back without claiming an execution or quota slot.
                preflight()
            execution_id = uuid.uuid4().hex
            self.db.execute("INSERT INTO content_dispatch_attempts "
                            "(consumer,request_id,binding,adapter_id,qualification_ref,status,result_json,execution_id) "
                            "VALUES(?,?,?,?,?,'started',NULL,?)",
                            (consumer, request_id, binding, candidate.adapter_id, candidate.qualification_ref, execution_id))
            self.db.execute("COMMIT")
            return execution_id
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _finish(self, consumer: str, request_id: str, status: str,
                result: Optional[ContentResult] = None) -> None:
        self._durable()
        encoded = json.dumps(asdict(result), ensure_ascii=False) if result is not None else None
        cursor = self.db.execute(
            "UPDATE content_dispatch_attempts SET status=?,result_json=? "
            "WHERE consumer=? AND request_id=? AND status='started'",
            (status, encoded, consumer, request_id))
        if cursor.rowcount != 1:
            raise ValueError("attempt is no longer started")

    def status(self, authenticated_consumer: str, request_id: str) -> Optional[str]:
        """Consumer-scoped state. Identity must come from server authentication."""
        row = self.db.execute(
            "SELECT status FROM content_dispatch_attempts WHERE consumer=? AND request_id=?",
            (authenticated_consumer, request_id)).fetchone()
        return cast(str, row[0]) if row else None

    def result(self, authenticated_consumer: str, request_id: str) -> Optional[ContentResult]:
        """Return only that consumer's completed response; never trigger execution."""
        row = self.db.execute(
            "SELECT result_json FROM content_dispatch_attempts "
            "WHERE consumer=? AND request_id=? AND status='completed'",
            (authenticated_consumer, request_id)).fetchone()
        if not row:
            return None
        values = cast(dict[str, str], json.loads(row[0]))
        return ContentResult(**values)

    def dispatch(
        self,
        profile: ContentProfile,
        request: ContentRequest,
        candidate: ContentCandidate,
        *,
        authenticated_consumer: str,
        content: str,
        adapter: Callable[[ContentRequest, ContentCandidate, str], object],
        preflight: Optional[Callable[[], None]] = None,
    ) -> ContentResult:
        """Commit an attempt before its external effect; never retry on any failure.

        The optional preflight runs under the claim lock and must perform only
        bounded local policy/evidence reads, never sends or external effects.
        Refusal does not claim capacity; the service records that refusal.
        No write lock is held while the adapter runs. Even cancellation is uncertain:
        the upstream may already have consumed quota. A persistence failure after a
        result leaves the started record blocking replay, rather than returning an
        unrecorded success. Exception details and prompts are not stored here.
        """
        _text(content, "content", allow_empty=True)
        admit_content(profile, request, candidate, authenticated_consumer=authenticated_consumer)
        if len(content.encode("utf-8")) != request.input_bytes:
            raise ValueError("input size does not match complete rendered content")
        if request.execution_id:
            raise ValueError("caller must not supply execution identity")
        execution_id = self._claim(authenticated_consumer, request.request_id,
                       _binding(profile, request, candidate, content), candidate, preflight)
        try:
            result = adapter(replace(request, execution_id=execution_id), candidate, content)
            if not isinstance(result, ContentResult):
                raise ValueError("adapter must return a normalized ContentResult")
            if len(result.text.encode("utf-8")) > request.output_bytes:
                raise ValueError("adapter result exceeds output bound")
        except BaseException:
            # Persist only the uncertain state, never the potentially sensitive error.
            # A failure to persist is raised; the durable started row still blocks replay.
            self._finish(authenticated_consumer, request.request_id, "uncertain")
            raise
        self._finish(authenticated_consumer, request.request_id, "completed", result)
        return result

def dispatch_content(
    journal: DispatchJournal,
    profile: ContentProfile,
    request: ContentRequest,
    candidate: ContentCandidate,
    *,
    authenticated_consumer: str,
    content: str,
    adapter: Callable[[ContentRequest, ContentCandidate, str], object],
) -> ContentResult:
    """Dispatch through the journal; see DispatchJournal.dispatch for guarantees."""
    return journal.dispatch(profile, request, candidate,
                            authenticated_consumer=authenticated_consumer,
                            content=content, adapter=adapter)
