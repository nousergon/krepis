"""Durable dispatch must record intent before invoking an external adapter."""
import importlib
import sqlite3
from dataclasses import replace

import pytest
from krepis.workload_admission import ContentCandidate, ContentProfile, ContentRequest


def api():
    assert importlib.util.find_spec("krepis.content_dispatch") is not None
    return importlib.import_module("krepis.content_dispatch")


def connection(path):
    db = sqlite3.connect(str(path), isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db


def fixture(tmp_path, limit=1):
    m = api()
    db = connection(tmp_path / "journal.sqlite")
    journal = m.DispatchJournal(db, max_inflight=limit)
    p = ContentProfile("consumer", "r1", "conversation", frozenset({"native"}),
                       frozenset({"processor"}), frozenset({"private"}),
                       frozenset({"human"}), 100, 100, 30)
    r = ContentRequest("request", "r1", "conversation", "private", "human", 5, 100, 20)
    c = ContentCandidate("native", frozenset({"processor"}), "conversation",
                         "content", "subscription", "qualification-1")
    return m, db, journal, p, r, c


def dispatch(m, j, p, r, c, adapter, content="hello", consumer="consumer"):
    return m.dispatch_content(j, p, r, c, authenticated_consumer=consumer,
                              content=content, adapter=adapter)


def test_intent_is_durable_before_adapter_and_response_survives_restart(tmp_path):
    m, db, j, p, r, c = fixture(tmp_path)
    def adapter(request, candidate, content):
        observer = m.DispatchJournal(connection(tmp_path / "journal.sqlite"), max_inflight=1)
        assert observer.status("consumer", "request") == "started"
        assert (replace(request, execution_id=""), candidate, content) == (r, c, "hello")
        return m.ContentResult("answer", "served-model", "trace-1")
    result = dispatch(m, j, p, r, c, adapter)
    db.close()
    reopened = m.DispatchJournal(connection(tmp_path / "journal.sqlite"), max_inflight=1)
    assert reopened.result("consumer", "request") == result
    assert reopened.result("other", "request") is None


def test_completed_request_never_replays(tmp_path):
    m, _, j, p, r, c = fixture(tmp_path)
    calls = []
    def adapter(*args):
        calls.append(args)
        return m.ContentResult("answer", "served", "trace")
    dispatch(m, j, p, r, c, adapter)
    with pytest.raises(m.AlreadyDispatched):
        dispatch(m, j, p, r, c, adapter)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, TimeoutError])
def test_failure_stays_uncertain_and_reserves_capacity(tmp_path, failure):
    m, _, j, p, r, c = fixture(tmp_path)
    def adapter(*args):
        raise failure("sensitive provider detail")
    with pytest.raises(failure):
        dispatch(m, j, p, r, c, adapter)
    assert j.status("consumer", "request") == "uncertain"
    with pytest.raises(m.CapacityUnavailable):
        dispatch(m, j, p, replace(r, request_id="new"), c, adapter)
    with pytest.raises(m.AlreadyDispatched):
        dispatch(m, j, p, r, c, adapter)


@pytest.mark.parametrize("change", ["content", "revision", "candidate"])
def test_request_identity_cannot_be_reused_for_changed_execution(tmp_path, change):
    m, _, j, p, r, c = fixture(tmp_path)
    adapter = lambda *args: m.ContentResult("ok", "served", "trace")
    dispatch(m, j, p, r, c, adapter)
    content = "hello"
    if change == "content":
        content = "world"
    elif change == "revision":
        p, r = replace(p, revision="r2"), replace(r, profile_revision="r2")
    else:
        c = replace(c, qualification_ref="qualification-2")
    with pytest.raises(m.RequestConflict):
        dispatch(m, j, p, r, c, adapter, content)


def test_shared_capacity_prevents_overlapping_connections(tmp_path):
    m, _, j, p, r, c = fixture(tmp_path)
    second = m.DispatchJournal(connection(tmp_path / "journal.sqlite"), max_inflight=1)
    def adapter(*args):
        with pytest.raises(m.CapacityUnavailable):
            dispatch(m, second, p, replace(r, request_id="second"), c, lambda *a: pytest.fail("executed"))
        return m.ContentResult("ok", "served", "trace")
    dispatch(m, j, p, r, c, adapter)
    dispatch(m, second, p, replace(r, request_id="second"), c, lambda *a: m.ContentResult("ok", "served", "trace2"))


def test_capacity_cannot_change_between_workers(tmp_path):
    m, _, _, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="capacity"):
        m.DispatchJournal(connection(tmp_path / "journal.sqlite"), max_inflight=2)


@pytest.mark.parametrize("bad_result", [None, "plain string", "overflow"])
def test_invalid_output_never_becomes_success(tmp_path, bad_result):
    m, _, j, p, r, c = fixture(tmp_path)
    result = m.ContentResult("x" * 101, "served", "trace") if bad_result == "overflow" else bad_result
    with pytest.raises(ValueError):
        dispatch(m, j, p, r, c, lambda *a: result)
    assert j.status("consumer", "request") == "uncertain"
    assert j.result("consumer", "request") is None


@pytest.mark.parametrize("content,consumer", [("too long", "consumer"), ("hello", "other")])
def test_admission_failure_has_no_external_effect(tmp_path, content, consumer):
    m, _, j, p, r, c = fixture(tmp_path)
    with pytest.raises(ValueError):
        dispatch(m, j, p, r, c, lambda *a: pytest.fail("executed"), content, consumer)
    assert j.status("consumer", "request") is None


def test_memory_database_is_refused():
    m = api()
    with pytest.raises(ValueError, match="durable"):
        m.DispatchJournal(sqlite3.connect(":memory:", isolation_level=None), max_inflight=1)


def test_ambiguous_completion_is_not_resent_after_database_failure(tmp_path):
    m, db, j, p, r, c = fixture(tmp_path)
    def adapter(*args):
        db.close()
        return m.ContentResult("answer", "served", "trace")
    with pytest.raises(sqlite3.ProgrammingError):
        dispatch(m, j, p, r, c, adapter)
    reopened = m.DispatchJournal(connection(tmp_path / "journal.sqlite"), max_inflight=1)
    assert reopened.status("consumer", "request") == "started"
    with pytest.raises(m.AlreadyDispatched):
        dispatch(m, reopened, p, r, c, lambda *a: pytest.fail("replayed"))


def test_simultaneous_workers_cannot_both_claim_one_slot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    from threading import Barrier, Event

    m, _, _, p, r, c = fixture(tmp_path)
    barrier, release, observed = Barrier(3), Event(), Queue()
    def worker(identity):
        db = connection(tmp_path / "journal.sqlite")
        journal = m.DispatchJournal(db, max_inflight=1)
        def adapter(*args):
            observed.put("started")
            assert release.wait(5), "test failed to release adapter"
            return m.ContentResult("ok", "served", identity)
        try:
            barrier.wait(timeout=5)
            dispatch(m, journal, p, replace(r, request_id=identity), c, adapter)
        except m.CapacityUnavailable:
            observed.put("refused")
        finally:
            db.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, name) for name in ("a", "b")]
        try:
            barrier.wait(timeout=5)
            assert sorted([observed.get(timeout=5), observed.get(timeout=5)]) == ["refused", "started"]
        finally:
            release.set()
        for future in futures:
            future.result(timeout=5)


def test_execution_identity_is_durable_and_unique_across_consumers(tmp_path):
    m, db, j, p, r, c = fixture(tmp_path)
    observed = []
    def adapter(request, *args):
        identity = request.execution_id
        row = db.execute("SELECT execution_id FROM content_dispatch_attempts WHERE execution_id=?", (identity,)).fetchone()
        assert row == (identity,)
        observed.append(identity)
        return m.ContentResult("ok", "served", identity)
    dispatch(m, j, p, r, c, adapter)
    dispatch(m, j, replace(p, consumer_id="other"), r, c, adapter, consumer="other")
    assert len(set(observed)) == 2
    assert all(len(identity) == 32 for identity in observed)


def test_v1_journal_migrates_without_replaying_or_losing_result(tmp_path):
    import json
    m = api()
    db = connection(tmp_path / "legacy.sqlite")
    db.executescript("""
      CREATE TABLE content_dispatch_config(singleton INTEGER PRIMARY KEY, version INTEGER, capacity INTEGER);
      INSERT INTO content_dispatch_config VALUES(1,1,1);
      CREATE TABLE content_dispatch_attempts(consumer TEXT,request_id TEXT,binding TEXT,adapter_id TEXT,qualification_ref TEXT,status TEXT,result_json TEXT,PRIMARY KEY(consumer,request_id));
    """)
    result = {"text":"old answer","served_model":"observed","trace_id":"old-receipt"}
    db.execute("INSERT INTO content_dispatch_attempts VALUES(?,?,?,?,?,?,?)", ("consumer","old","binding","native","q","completed",json.dumps(result)))
    journal = m.DispatchJournal(db, max_inflight=1)
    assert journal.result("consumer", "old") == m.ContentResult(**result)
    assert db.execute("SELECT version FROM content_dispatch_config").fetchone() == (2,)
    assert db.execute("SELECT execution_id FROM content_dispatch_attempts").fetchone() == (None,)


def test_intake_cannot_choose_durable_execution_identity(tmp_path):
    m, _, j, p, r, c = fixture(tmp_path)
    with pytest.raises(ValueError, match="caller"):
        dispatch(m, j, p, replace(r, execution_id="e" * 32), c, lambda *a: pytest.fail("executed"))
