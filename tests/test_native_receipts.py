"""Physical sends are once-only; observed model identity survives process loss."""
import importlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
import pytest

EXECUTION = 'a' * 32

def api():
    assert importlib.util.find_spec('krepis.native_receipts') is not None
    return importlib.import_module('krepis.native_receipts')

def connect(path):
    return sqlite3.connect(str(path), isolation_level=None, timeout=0.1)

def fixture(tmp_path):
    m = api()
    path = tmp_path / 'receipts.sqlite'
    db = connect(path)
    store = m.ReceiptStore(db)
    store.authorize(EXECUTION, 'adapter', 'qualification', 'upstream.test', expires_at=time.time()+60)
    return m, path, db, store

def test_claim_is_visible_before_send_and_survives_restart(tmp_path):
    m, path, db, store = fixture(tmp_path)
    receipt = store.claim(EXECUTION, 'thread', 'upstream.test')
    db.close()
    observer = m.ReceiptStore(connect(path))
    assert observer.status(EXECUTION) == 'claimed'
    assert observer.read(EXECUTION, 'thread', 0.1) is None
    with pytest.raises(m.ReceiptRefused):
        observer.claim(EXECUTION, 'thread', 'upstream.test')
    observer.complete(receipt, served_model='actual', response_id='response')
    evidence = observer.read(EXECUTION, 'thread', 0.1)
    assert evidence.execution_id == EXECUTION
    assert evidence.thread_id == 'thread'
    assert evidence.adapter_id == 'adapter'
    assert evidence.qualification_ref == 'qualification'
    assert evidence.served_model == 'actual'
    assert evidence.receipt_id == receipt
    assert observer.read(EXECUTION, 'other-thread', 0.1) is None
    assert observer.read('b'*32, 'thread', 0.1) is None
    with pytest.raises(m.ReceiptRefused):
        observer.complete(receipt, served_model='different', response_id='other')

@pytest.mark.parametrize('execution,upstream', [('b'*32,'upstream.test'),(EXECUTION,'forbidden.test')])
def test_unregistered_or_wrong_destination_never_claims(tmp_path, execution, upstream):
    m, _, _, store = fixture(tmp_path)
    with pytest.raises(m.ReceiptRefused):
        store.claim(execution, 'thread', upstream)
    assert store.status(EXECUTION) == 'authorized'

def test_expired_and_reauthorized_execution_refused(tmp_path):
    m, _, db, store = fixture(tmp_path)
    with pytest.raises(m.ReceiptRefused):
        store.authorize(EXECUTION, 'adapter', 'qualification', 'upstream.test', expires_at=time.time()+60)
    db.execute('UPDATE native_receipts SET expires_at=0')
    with pytest.raises(m.ReceiptRefused):
        store.claim(EXECUTION, 'thread', 'upstream.test')

def test_concurrent_workers_cannot_send_twice(tmp_path):
    m, path, _, _ = fixture(tmp_path)
    def claim(_):
        with connect(path) as db:
            try:
                return m.ReceiptStore(db).claim(EXECUTION, 'thread', 'upstream.test')
            except m.ReceiptRefused:
                return None
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(claim, range(2)))
    assert sum(value is not None for value in results) == 1

def test_busy_reader_obeys_small_deadline(tmp_path):
    m, path, db, _ = fixture(tmp_path)
    reader = m.ReceiptStore(connect(path))
    db.execute('BEGIN EXCLUSIVE')
    start = time.monotonic()
    with pytest.raises((TimeoutError, sqlite3.OperationalError)):
        reader.read(EXECUTION, 'thread', 0.02)
    assert time.monotonic() - start < 0.2
    db.execute('ROLLBACK')

@pytest.mark.parametrize('budget', [0, -1, float('inf'), float('nan'), True])
def test_invalid_read_deadline_refused(tmp_path, budget):
    _, _, _, store = fixture(tmp_path)
    with pytest.raises(ValueError):
        store.read(EXECUTION, 'thread', budget)

def test_volatile_or_async_storage_refused(tmp_path):
    m = api()
    with pytest.raises(ValueError):
        m.ReceiptStore(connect(':memory:'))
    db = connect(tmp_path / 'bad.sqlite')
    db.execute('PRAGMA synchronous=OFF')
    with pytest.raises(ValueError):
        m.ReceiptStore(db)

@pytest.mark.parametrize('execution', ['', 'b'*31, 'G'*32])
def test_invalid_execution_identity_refused(tmp_path, execution):
    _, _, _, store = fixture(tmp_path)
    with pytest.raises(ValueError):
        store.authorize(execution, 'adapter', 'qualification', 'upstream.test', expires_at=time.time()+60)

def test_authorization_expiring_while_waiting_for_lock_cannot_send(tmp_path):
    import threading
    m, path, db, _ = fixture(tmp_path)
    db.execute('UPDATE native_receipts SET expires_at=?', (time.time()+0.08,))
    waiting = threading.Event()
    locked = threading.Event()
    def attempt():
        worker = connect(path)
        worker.execute('PRAGMA busy_timeout=1000')
        store = m.ReceiptStore(worker)
        waiting.set()
        assert locked.wait(1)
        try:
            store.claim(EXECUTION, 'thread', 'upstream.test')
            return 'sent'
        except m.ReceiptRefused:
            return 'refused'
        finally:
            worker.close()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(attempt)
        assert waiting.wait(1)
        db.execute('BEGIN IMMEDIATE')
        locked.set()
        time.sleep(0.15)
        db.execute('COMMIT')
        assert future.result() == 'refused'

def test_temporary_shadow_table_cannot_capture_durable_claim(tmp_path):
    m, path, db, store = fixture(tmp_path)
    db.execute('CREATE TEMP TABLE native_receipts AS SELECT * FROM main.native_receipts')
    store.claim(EXECUTION, 'thread', 'upstream.test')
    observer = m.ReceiptStore(connect(path))
    assert observer.status(EXECUTION) == 'claimed'

def test_dedicated_reader_opens_without_write_access(tmp_path):
    m, path, _, store = fixture(tmp_path)
    receipt = store.claim(EXECUTION, 'thread', 'upstream.test')
    store.complete(receipt, served_model='actual', response_id='response')
    readonly = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, isolation_level=None, timeout=0)
    reader = m.ReceiptStore(readonly, initialize=False)
    assert reader.read(EXECUTION, 'thread', 0.1).served_model == 'actual'

def test_transport_deadline_comes_from_durable_authorization(tmp_path):
    m, _, db, store = fixture(tmp_path)
    deadline = db.execute('SELECT expires_at FROM native_receipts WHERE execution_id=?',(EXECUTION,)).fetchone()[0]
    assert store.deadline(EXECUTION) == deadline
    with pytest.raises(m.ReceiptRefused):
        store.deadline('b'*32)

def test_durable_counts_include_unresolved_sends(tmp_path):
    _, _, _, store = fixture(tmp_path)
    assert store.counts() == {'authorized':1,'claimed':0,'completed':0}
    receipt = store.claim(EXECUTION,'thread','upstream.test')
    assert store.counts() == {'authorized':0,'claimed':1,'completed':0}
    store.complete(receipt,served_model='actual',response_id='r')
    assert store.counts() == {'authorized':0,'claimed':0,'completed':1}
