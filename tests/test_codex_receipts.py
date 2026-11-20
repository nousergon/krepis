"""Observed response identity is mandatory; request identity is not evidence."""
import importlib
import json
import pytest

def api():
    assert importlib.util.find_spec('krepis.codex_receipts') is not None
    return importlib.import_module('krepis.codex_receipts')

def event(kind='response.completed', **fields):
    return ('data: ' + json.dumps({'type':kind, 'response':{'id':'r1','model':'served','status':'completed', **fields}}) + '\n\n').encode()

def test_chunked_sse_returns_observed_model_and_response():
    parser = api().CodexReceiptDecoder()
    data = event('response.created',status='in_progress') + event() + b'data: [DONE]\n\n'
    for byte in data:
        parser.feed(bytes([byte]))
    assert parser.finish() == ('served', 'r1')

@pytest.mark.parametrize('data', [b'', event('response.created'), event(model=None), event(id=''), event(status='failed'), event()[:-1], b'data: garbage\n\n', b'data: [DONE]\n\n', event()+event(), event()+b'data: {"type":"error"}\n\n', event('response.created',model='other')+event(), event('response.created',id='other')+event()])
def test_incomplete_malformed_conflicting_or_failed_response_is_not_a_receipt(data):
    parser = api().CodexReceiptDecoder()
    with pytest.raises(ValueError):
        parser.feed(data)
        parser.finish()


def test_oversized_event_poisons_parser_without_retaining_full_content():
    parser = api().CodexReceiptDecoder(max_event_bytes=128, max_total_bytes=256)
    with pytest.raises(ValueError):
        parser.feed(b'data: '+ b'x'*129)
    with pytest.raises(ValueError):
        parser.finish()


def test_total_stream_limit_includes_ignored_comments():
    parser = api().CodexReceiptDecoder(max_event_bytes=128, max_total_bytes=256)
    with pytest.raises(ValueError):
        for _ in range(100):
            parser.feed(b': heartbeat\n\n')


def test_crlf_and_multiline_sse_data():
    parser = api().CodexReceiptDecoder()
    parser.feed(b'event: response.completed\r\ndata: {"type":"response.completed",\r\ndata: "response":{"id":"r1","model":"served","status":"completed"}}\r\n\r\n')
    assert parser.finish() == ('served','r1')


def test_duplicate_json_model_fields_refused():
    parser = api().CodexReceiptDecoder()
    with pytest.raises(ValueError):
        parser.feed(b'data: {"type":"response.completed","response":{"id":"r1","model":"a","model":"b","status":"completed"}}\n\n')

@pytest.mark.parametrize('prefix', [b': ping\rdata: {"type":"error"}\r\r\n\n', b'event: error\n', b'event: response.created\n'])
def test_hidden_error_or_contradictory_named_event_refused(prefix):
    parser = api().CodexReceiptDecoder()
    with pytest.raises(ValueError):
        parser.feed(prefix + event())
        parser.finish()


def test_nonfinite_json_is_refused():
    parser = api().CodexReceiptDecoder()
    with pytest.raises(ValueError):
        parser.feed(event().replace(b'"response":', b'"nested":NaN,"response":'))

def test_many_short_lines_do_not_materialize_unbounded_line_list():
    import tracemalloc
    parser = api().CodexReceiptDecoder(max_event_bytes=1)
    data = b'\n' * (1024 * 1024)
    tracemalloc.start()
    try:
        parser.feed(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024

@pytest.mark.parametrize('data', [event('response.created', status='failed')+event(),event('response.created', error={'message':'failed'})+event(),event(incomplete_details={'reason':'max_output_tokens'})])
def test_failed_lifecycle_metadata_is_refused(data):
    parser = api().CodexReceiptDecoder()
    with pytest.raises(ValueError):
        parser.feed(data)
        parser.finish()
