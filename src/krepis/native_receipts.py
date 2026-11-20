"""Durable physical-send claims for isolated native execution.

The trusted dispatcher authorizes an execution before starting the worker. The
compelled egress boundary claims it before sending and completes it only with
observed response metadata. Workers MUST NOT have access to this database.
Filesystem ownership, expiry clock synchronization, backup and network compulsion
are deployment requirements. A claimed execution is uncertain until completed;
there is deliberately no expiry/reset/retry API for an already claimed send.

This module performs no networking and cannot attest a caller's honesty. Only
trusted egress code may call complete; a model name from the request is not
served-model evidence. No prompt, response content or credential is stored.
"""
from dataclasses import dataclass
import math
import re
import sqlite3
import time
from typing import Optional
import uuid


class ReceiptRefused(ValueError):
    """No physical send or receipt transition is authorized."""


@dataclass(frozen=True)
class TransportReceipt:
    execution_id: str
    thread_id: str
    adapter_id: str
    qualification_ref: str
    served_model: str
    receipt_id: str
    response_id: str


def _text(value: object) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512 or any(ord(c) < 32 for c in value):
        raise ValueError("bounded receipt identity required")


def _execution(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("durable execution identity required")


class ReceiptStore:
    """Independent disk-backed SQLite connection; share files, not connections.

    Use initialize=False with a mode=ro, timeout=0 connection for readers.
    Construct before entering the execution callback; read() bounds each lookup."""

    def __init__(self, db: sqlite3.Connection, *, initialize: bool = True) -> None:
        self.db = db
        self._durable()
        if initialize:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS main.native_receipt_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL);
                INSERT OR IGNORE INTO main.native_receipt_schema VALUES(1,1);
                CREATE TABLE IF NOT EXISTS main.native_receipts (
                    execution_id TEXT PRIMARY KEY, adapter_id TEXT NOT NULL,
                    qualification_ref TEXT NOT NULL, upstream TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('authorized','claimed','completed')),
                    receipt_id TEXT UNIQUE, thread_id TEXT, served_model TEXT, response_id TEXT);
            """)
        if db.execute("SELECT version FROM main.native_receipt_schema WHERE singleton=1").fetchone()[0] != 1:
            raise ValueError("unsupported native receipt schema")

    def _durable(self) -> None:
        if self.db.isolation_level is not None or self.db.in_transaction:
            raise ValueError("independent autocommit connection required")
        main = next(row[2] for row in self.db.execute("PRAGMA database_list") if row[1] == "main")
        mode = self.db.execute("PRAGMA main.journal_mode").fetchone()[0]
        sync = self.db.execute("PRAGMA main.synchronous").fetchone()[0]
        if not main or mode not in {"wal", "delete", "truncate", "persist"} or sync < 2:
            raise ValueError("durable synchronous disk-backed journal required")

    def authorize(self, execution_id: str, adapter_id: str, qualification_ref: str,
                  upstream: str, *, expires_at: float) -> None:
        """Trusted dispatcher only; a duplicate identity is never re-authorized."""
        _execution(execution_id)
        for value in (adapter_id, qualification_ref, upstream):
            _text(value)
        if isinstance(expires_at, bool) or not math.isfinite(expires_at) or expires_at <= time.time():
            raise ValueError("future finite authorization deadline required")
        self._durable()
        try:
            _ = self.db.execute(
                "INSERT INTO main.native_receipts(execution_id,adapter_id,qualification_ref,upstream,expires_at,state) "
                "VALUES(?,?,?,?,?,'authorized')",
                (execution_id, adapter_id, qualification_ref, upstream, expires_at))
        except sqlite3.IntegrityError as exc:
            raise ReceiptRefused("execution already authorized") from exc

    def claim(self, execution_id: str, thread_id: str, upstream: str) -> str:
        """Commit before the first outbound byte. Never retry after this succeeds."""
        _execution(execution_id)
        _text(thread_id)
        _text(upstream)
        self._durable()
        receipt = uuid.uuid4().hex
        _ = self.db.execute("BEGIN IMMEDIATE")
        try:
            # Evaluate expiry AFTER acquiring the write lock, not before a
            # potentially blocking SQLite busy wait.
            result = self.db.execute(
                "UPDATE main.native_receipts SET state='claimed',receipt_id=?,thread_id=? "
                "WHERE execution_id=? AND upstream=? AND state='authorized' AND expires_at>?",
                (receipt, thread_id, execution_id, upstream, time.time()))
            if result.rowcount != 1:
                raise ReceiptRefused("execution unavailable, expired or already claimed")
            _ = self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                _ = self.db.execute("ROLLBACK")
            raise
        return receipt

    def complete(self, receipt_id: str, *, served_model: str, response_id: str) -> None:
        """Trusted egress only, after decoding a complete successful response."""
        for value in (receipt_id, served_model, response_id):
            _text(value)
        self._durable()
        result = self.db.execute(
            "UPDATE main.native_receipts SET state='completed',served_model=?,response_id=? "
            "WHERE receipt_id=? AND state='claimed'",
            (served_model, response_id, receipt_id))
        if result.rowcount != 1:
            raise ReceiptRefused("receipt is not a pending physical send")

    def deadline(self, execution_id: str) -> float:
        """Trusted authorization's absolute Unix deadline for transport fencing."""
        _execution(execution_id)
        row = self.db.execute(
            "SELECT expires_at FROM main.native_receipts WHERE execution_id=?", (execution_id,)).fetchone()
        if row is None:
            raise ReceiptRefused("execution is not authorized")
        return float(row[0])

    def counts(self) -> dict[str, int]:
        """Durable state counts; claimed includes every unresolved physical send."""
        result = {"authorized": 0, "claimed": 0, "completed": 0}
        for state, count in self.db.execute("SELECT state,COUNT(*) FROM main.native_receipts GROUP BY state"):
            if state not in result:
                raise ValueError("unknown receipt state")
            result[state] = int(count)
        return result

    def status(self, execution_id: str) -> Optional[str]:
        _execution(execution_id)
        row = self.db.execute("SELECT state FROM main.native_receipts WHERE execution_id=?", (execution_id,)).fetchone()
        return str(row[0]) if row else None

    def read(self, execution_id: str, thread_id: str, remaining_seconds: float) -> Optional[TransportReceipt]:
        """One bounded local read; missing/unfinished evidence is never success.

        The supplied connection is dedicated to this reader: busy_timeout is
        limited to the smaller of the remaining budget or 50ms. No polling or
        network operation occurs. Deadline expiry after a read refuses evidence.
        """
        _execution(execution_id)
        _text(thread_id)
        if (isinstance(remaining_seconds, bool) or not math.isfinite(remaining_seconds)
                or remaining_seconds <= 0):
            raise ValueError("positive finite read budget required")
        deadline = time.monotonic() + remaining_seconds
        _ = self.db.execute("PRAGMA busy_timeout=" + str(int(min(remaining_seconds, 0.05) * 1000)))
        row = self.db.execute(
            "SELECT execution_id,thread_id,adapter_id,qualification_ref,served_model,receipt_id,response_id "
            "FROM main.native_receipts WHERE execution_id=? AND thread_id=? AND state='completed'",
            (execution_id, thread_id)).fetchone()
        if time.monotonic() >= deadline:
            raise TimeoutError("receipt read deadline exceeded")
        return TransportReceipt(*row) if row else None
