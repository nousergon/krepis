"""Bounded read-only observations of native dispatch/service evidence.

For trusted operator monitoring, not a consumer-facing endpoint. The caller
protects the database path and all parent directories from replacement. No
prompts, results, request IDs, consumer IDs or exception details are returned.
An observation is historical evidence, NEVER a worker heartbeat or readiness
verdict. Combine it with independently fresh process/qualification observations.
Missing, locked, corrupt, unsupported or over-budget evidence raises; a monitor
must render that as unavailable, never as zero activity or healthy. No effects
are cleared or replayed. This reader neither initializes nor migrates a store.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import stat
import time
from typing import Optional


@dataclass(frozen=True)
class ContentObservation:
    schema_version: int
    observed_at: float
    evidence: str
    dispatch_counts: dict[str, int]
    service_counts: dict[str, int]
    capacity: int
    capacity_remaining: int
    latest_event_at: Optional[float]
    oldest_unfinished_at: Optional[float]


def observe_content(path: Path, *, now: float, max_records: int = 100000,
                    timeout_seconds: float = 1.0) -> ContentObservation:
    """Read one coherent snapshot with a per-table row and elapsed-time budget.

    The timeout bounds SQLite VM work and nonblocking lock acquisition; it cannot
    bound a stalled kernel/filesystem. Operated monitors must also bound the
    reader process. Use a local durable filesystem, not a network filesystem.
    ``now`` is the operator's current Unix time; future event times refuse.
    """
    if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
        raise ValueError('finite current time required')
    if type(max_records) is not int or not 1 <= max_records <= 100000:
        raise ValueError('invalid observation row bound')
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 5):
        raise ValueError('invalid observation time bound')
    deadline = time.monotonic() + timeout_seconds

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError('observation deadline exceeded')

    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('regular journal file required')
    db = sqlite3.connect(path.absolute().as_uri() + '?mode=ro', uri=True,
                         isolation_level=None, timeout=0)
    try:
        db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        schemas = {
            'content_dispatch_config': {'singleton': 'INTEGER', 'version': 'INTEGER', 'capacity': 'INTEGER'},
            'content_dispatch_attempts': {'consumer': 'TEXT', 'request_id': 'TEXT', 'binding': 'TEXT',
                'adapter_id': 'TEXT', 'qualification_ref': 'TEXT', 'status': 'TEXT',
                'result_json': 'TEXT', 'execution_id': 'TEXT'},
            'content_service_events': {'event_id': 'INTEGER', 'created_at': 'REAL', 'consumer': 'TEXT',
                'request_id': 'TEXT', 'registry_revision': 'TEXT', 'outcome': 'TEXT', 'stage': 'TEXT',
                'error_type': 'TEXT', 'rejections': 'TEXT'},
        }
        for table, columns in schemas.items():
            physical = db.execute("SELECT count(*) FROM sqlite_master WHERE name=? AND type='table'", (table,)).fetchone()[0]
            predicates = ' OR '.join('(name=? AND upper(type)=?)' for _ in columns)
            parameters = [table] + [value for pair in columns.items() for value in pair]
            count = db.execute('SELECT count(*) FROM pragma_table_info(?) WHERE ' + predicates, parameters).fetchone()[0]
            if physical != 1 or count != len(columns):
                raise ValueError('unsupported physical journal schema')
        config = db.execute("""SELECT CASE WHEN singleton=1 AND version=2
            AND typeof(capacity)='integer' AND capacity BETWEEN 1 AND 2147483647
            THEN capacity ELSE NULL END FROM content_dispatch_config LIMIT 2""").fetchall()
        if len(config) != 1 or config[0][0] is None:
            raise ValueError('unsupported journal metadata')
        capacity: int = config[0][0]
        dispatch: dict[str, int] = dict.fromkeys(('started', 'completed', 'uncertain'), 0)
        events: dict[str, int] = dict.fromkeys(('started', 'completed', 'refused', 'uncertain'), 0)
        for index, (status,) in enumerate(db.execute(
                """SELECT CASE status WHEN 'started' THEN 'started' WHEN 'completed' THEN 'completed'
                WHEN 'uncertain' THEN 'uncertain' ELSE NULL END FROM content_dispatch_attempts LIMIT ?""", (max_records + 1,))):
            check_deadline()
            if index >= max_records:
                raise ValueError('dispatch observation row bound exceeded')
            if status not in dispatch:
                raise ValueError('unsupported dispatch status')
            dispatch[status] += 1
        latest: Optional[float] = None
        unfinished: Optional[float] = None
        for index, (created_at, outcome) in enumerate(db.execute(
                """SELECT CASE WHEN typeof(created_at) IN ('integer','real') THEN created_at ELSE NULL END,
                CASE outcome WHEN 'started' THEN 'started' WHEN 'completed' THEN 'completed'
                WHEN 'refused' THEN 'refused' WHEN 'uncertain' THEN 'uncertain' ELSE NULL END
                FROM content_service_events LIMIT ?""", (max_records + 1,))):
            check_deadline()
            if index >= max_records:
                raise ValueError('service observation row bound exceeded')
            if outcome not in events:
                raise ValueError('unsupported service outcome')
            if (type(created_at) not in (int, float) or not math.isfinite(created_at)
                    or not 0 <= created_at <= now):
                raise ValueError('invalid event time')
            events[outcome] += 1
            latest = created_at if latest is None else max(latest, created_at)
            if outcome == 'started':
                unfinished = created_at if unfinished is None else min(unfinished, created_at)
        unresolved = dispatch['started'] + dispatch['uncertain']
        if unresolved > capacity:
            raise ValueError('journal exceeds recorded capacity')
        check_deadline()
        return ContentObservation(1, float(now), 'present' if sum(dispatch.values()) + sum(events.values()) else 'empty',
                                  dispatch, events, capacity, capacity - unresolved, latest, unfinished)
    finally:
        db.close()
