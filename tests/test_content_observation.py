"""Monitoring reads evidence without mutating effects or disclosing content."""
import importlib
import sqlite3
import pytest
from tests.test_content_service import harness as _service_harness
service_harness = _service_harness


@pytest.fixture
def harness(service_harness):
    return service_harness


def observe(path, **kwargs):
    assert importlib.util.find_spec('krepis.content_observation') is not None
    return importlib.import_module('krepis.content_observation').observe_content(path, now=1000, **kwargs)


def test_empty_is_explicit_and_observation_never_initializes(tmp_path, harness):
    path = tmp_path / 'dispatch.sqlite'
    before = path.read_bytes()
    result = observe(path)
    assert result.evidence == 'empty'
    assert result.dispatch_counts == {'started': 0, 'completed': 0, 'uncertain': 0}
    assert result.capacity_remaining == 1
    assert result.latest_event_at is None
    assert path.read_bytes() == before
    missing = tmp_path / 'absent.sqlite'
    with pytest.raises((ValueError, OSError, sqlite3.Error)):
        observe(missing)
    assert not missing.exists()


def test_durable_counts_preserve_uncertainty_without_content(tmp_path, harness, monkeypatch):
    service, _, _, _, journal = harness
    monkeypatch.setattr('krepis.content_service.time.time', lambda: 900)
    service.submit('valid', request_id='private-request', content='private prompt')
    def fail(*args):
        raise RuntimeError('private failure')
    service.execute = fail
    with pytest.raises(RuntimeError):
        service.submit('valid', request_id='second', content='hi')
    result = observe(tmp_path / 'dispatch.sqlite')
    assert result.evidence == 'present'
    assert result.dispatch_counts == {'started': 0, 'completed': 1, 'uncertain': 1}
    assert result.service_counts == {'started': 0, 'completed': 1, 'refused': 0, 'uncertain': 1}
    assert result.capacity_remaining == 0
    assert result.latest_event_at == 900
    assert result.oldest_unfinished_at is None
    assert 'private' not in repr(result) and 'reply' not in repr(result)
    assert journal.status('consumer', 'second') == 'uncertain'
    assert observe(tmp_path / 'dispatch.sqlite') == result


def test_unfinished_event_is_visible(tmp_path, harness):
    harness[-1].db.execute("INSERT INTO content_service_events(created_at,outcome,stage) VALUES(900,'started','authentication')")
    result = observe(tmp_path / 'dispatch.sqlite')
    assert result.oldest_unfinished_at == 900
    assert result.service_counts['started'] == 1


@pytest.mark.parametrize('timestamp', [-1, 1001, float('inf')])
def test_invalid_event_time_refuses(tmp_path, harness, timestamp):
    harness[-1].db.execute("INSERT INTO content_service_events(created_at,outcome,stage) VALUES(?,'started','authentication')", (timestamp,))
    with pytest.raises(ValueError):
        observe(tmp_path / 'dispatch.sqlite')


def test_row_budget_refuses_partial_totals(tmp_path, harness):
    for _ in range(3):
        harness[-1].db.execute("INSERT INTO content_service_events(created_at,outcome,stage) VALUES(900,'started','authentication')")
    with pytest.raises(ValueError, match='bound'):
        observe(tmp_path / 'dispatch.sqlite', max_records=2)


@pytest.mark.parametrize('sql', [
    'UPDATE content_dispatch_config SET version=99',
    'DROP TABLE content_service_events',
    "INSERT INTO content_service_events(created_at,outcome,stage) VALUES(900,'unknown','authentication')",
])
def test_unknown_or_missing_evidence_refuses(tmp_path, harness, sql):
    harness[-1].db.execute(sql)
    with pytest.raises((ValueError, sqlite3.Error)):
        observe(tmp_path / 'dispatch.sqlite')


def test_locked_database_refuses(tmp_path, harness):
    db = harness[-1].db
    db.execute('BEGIN EXCLUSIVE')
    try:
        with pytest.raises(sqlite3.OperationalError):
            observe(tmp_path / 'dispatch.sqlite')
    finally:
        db.execute('ROLLBACK')


def test_expired_read_budget_refuses(tmp_path, harness, monkeypatch):
    assert importlib.util.find_spec('krepis.content_observation') is not None
    module = importlib.import_module('krepis.content_observation')
    ticks = iter([0, 2])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks, 2))
    with pytest.raises((TimeoutError, sqlite3.OperationalError)):
        module.observe_content(tmp_path / 'dispatch.sqlite', now=1000, timeout_seconds=1)


def test_malformed_large_value_is_not_materialized(tmp_path, harness):
    import tracemalloc
    harness[-1].db.execute("INSERT INTO content_service_events(created_at,outcome,stage) VALUES(900,zeroblob(4000000),'authentication')")
    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            observe(tmp_path / 'dispatch.sqlite', max_records=1)
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 1000000
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize('replacement', [
    "CREATE VIEW content_dispatch_attempts AS SELECT 'completed' AS status WHERE 0",
    'CREATE TABLE content_dispatch_attempts(status TEXT)',
])
def test_missing_physical_producer_schema_refuses(tmp_path, harness, replacement):
    db = harness[-1].db
    db.execute('DROP TABLE content_dispatch_attempts')
    db.execute(replacement)
    with pytest.raises(ValueError):
        observe(tmp_path / 'dispatch.sqlite')


def test_observation_during_execution_sees_capacity_claim(tmp_path, harness, monkeypatch):
    service = harness[0]
    monkeypatch.setattr('krepis.content_service.time.time', lambda: 900)
    original = service.execute
    def inspect(*args):
        result = observe(tmp_path / 'dispatch.sqlite')
        assert result.dispatch_counts['started'] == 1
        assert result.service_counts['started'] == 1
        assert result.capacity_remaining == 0
        assert result.oldest_unfinished_at == 900
        return original(*args)
    service.execute = inspect
    service.submit('valid', request_id='request', content='hi')
    assert observe(tmp_path / 'dispatch.sqlite').capacity_remaining == 1
