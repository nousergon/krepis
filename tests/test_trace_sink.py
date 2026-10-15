"""Tests for :mod:`krepis.trace_sink` — hermetic, against a fake S3 client."""

from __future__ import annotations

import gzip
import json

import pytest

from krepis import trace_sink as ts
from krepis.llm_trace import TraceBlob, sha256_hex


class _ClientError(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3:
    """put_object fake: stores objects, honours If-None-Match, can fail."""

    def __init__(self):
        self.objects = {}
        self.calls = []
        self.fail_rows = 0
        self.fail_blobs = 0

    def put_object(self, **kw):
        self.calls.append(kw)
        key = kw["Key"]
        if "/rows/" in key and self.fail_rows:
            self.fail_rows -= 1
            raise RuntimeError("s3 down")
        if "/blobs/" in key and self.fail_blobs:
            self.fail_blobs -= 1
            raise RuntimeError("s3 down")
        if kw.get("IfNoneMatch") == "*" and key in self.objects:
            raise _ClientError("PreconditionFailed", 412)
        self.objects[key] = kw

    def rows(self):
        out = []
        for key, obj in sorted(self.objects.items()):
            if "/rows/" in key:
                out.extend(
                    json.loads(line)
                    for line in gzip.decompress(obj["Body"]).decode().splitlines()
                )
        return out


def _row(callsite="cs", started="2026-10-14T13:30:00.000Z", **extra):
    row = {"schema_version": "llm_trace.v1", "callsite_id": callsite,
           "started_at": started}
    row.update(extra)
    return row


def _blob(text):
    data = text.encode()
    return sha256_hex(data), TraceBlob(data, "text/plain; charset=utf-8")


def _sink(s3, **kw):
    kw.setdefault("register_atexit", False)
    return ts.S3JsonlTraceSink(bucket="b", prefix="llm_traces/v1/", run_id="run-1",
                               s3_client=s3, **kw)


class TestKeys:
    def test_rows_are_hive_partitioned_and_writer_scoped(self):
        key = ts.rows_key("p", "2026-10-14", "thinktank thesis", "r/1", "abcd1234", 3)
        assert key == ("p/rows/date=2026-10-14/callsite=thinktank_thesis/"
                       "r_1.abcd1234.3.jsonl.gz")

    def test_blobs_are_content_addressed(self):
        sha = "ab" + "0" * 62
        assert ts.blob_key("p", sha) == f"p/blobs/sha256/ab/{sha}.gz"


class TestConstruction:
    def test_bucket_and_prefix_are_required(self):
        with pytest.raises(ValueError, match="bucket"):
            ts.S3JsonlTraceSink(bucket="", prefix="p", register_atexit=False)
        with pytest.raises(ValueError, match="prefix"):
            ts.S3JsonlTraceSink(bucket="b", prefix="/", register_atexit=False)

    def test_prefix_slashes_are_trimmed(self):
        assert _sink(FakeS3()).prefix == "llm_traces/v1"


class TestFlush:
    def test_rows_and_blobs_land_where_declared(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sha, blob = _blob("hello")
        sink(_row(), {sha: blob})
        assert sink.flush() == 1
        keys = list(s3.objects)
        assert keys[0] == f"llm_traces/v1/blobs/sha256/{sha[:2]}/{sha}.gz"
        assert keys[1].startswith("llm_traces/v1/rows/date=2026-10-14/callsite=cs/run-1.")
        assert gzip.decompress(s3.objects[keys[0]]["Body"]) == b"hello"
        assert s3.objects[keys[0]]["IfNoneMatch"] == "*"
        assert s3.objects[keys[0]]["Metadata"] == {"media-type": "text/plain; charset=utf-8"}
        assert s3.rows() == [_row(run_id="run-1")]
        assert (sink.rows_written, sink.blobs_written) == (1, 1)

    def test_gzip_is_deterministic(self):
        s3a, s3b = FakeS3(), FakeS3()
        for s3 in (s3a, s3b):
            sha, blob = _blob("same bytes")
            sink = _sink(s3)
            sink(_row(), {sha: blob})
            sink.flush()
        blob_a = [o["Body"] for k, o in s3a.objects.items() if "/blobs/" in k]
        blob_b = [o["Body"] for k, o in s3b.objects.items() if "/blobs/" in k]
        assert blob_a == blob_b

    def test_a_row_keeps_its_own_run_id(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_row(run_id="other"), {})
        sink.flush()
        assert s3.rows()[0]["run_id"] == "other"

    def test_groups_split_by_date_and_callsite(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_row("a"), {})
        sink(_row("b"), {})
        sink(_row("a", started="2026-10-15T00:00:00.000Z"), {})
        sink(_row("a", started=None), {})
        assert sink.flush() == 4
        assert any("date=unknown-date" in k for k in s3.objects)

    def test_a_second_flush_never_overwrites_the_first(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sink(_row(), {})
        sink.flush()
        sink(_row(), {})
        sink.flush()
        assert len([k for k in s3.objects if "/rows/" in k]) == 2

    def test_the_row_threshold_auto_flushes(self):
        s3 = FakeS3()
        sink = _sink(s3, flush_threshold=2)
        sink(_row(), {})
        assert s3.objects == {}
        sink(_row(), {})
        assert len(s3.rows()) == 2

    def test_the_blob_threshold_flushes_blobs_only(self):
        s3 = FakeS3()
        sink = _sink(s3, blob_flush_bytes=4)
        sha, blob = _blob("hello")
        sink(_row(), {sha: blob})
        assert [k for k in s3.objects if "/blobs/" in k]
        assert s3.rows() == []

    def test_flush_with_nothing_buffered_writes_nothing(self):
        s3 = FakeS3()
        assert _sink(s3).flush() == 0
        assert s3.calls == []


class TestDedup:
    def test_a_blob_is_written_once_per_process(self):
        s3 = FakeS3()
        sink = _sink(s3)
        sha, blob = _blob("system prompt")
        sink(_row(), {sha: blob})
        sink(_row(), {sha: blob})
        sink.flush()
        sink(_row(), {sha: blob})
        sink.flush()
        assert len([c for c in s3.calls if "/blobs/" in c["Key"]]) == 1
        assert sink.blobs_deduplicated == 2

    def test_a_blob_already_in_s3_is_not_an_error(self):
        s3 = FakeS3()
        sha, blob = _blob("shared chunk")
        s3.objects[ts.blob_key("llm_traces/v1", sha)] = {"Body": b"x"}
        sink = _sink(s3)
        sink(_row(), {sha: blob})
        sink.flush()
        assert sink.blob_errors == 0
        assert sink.blobs_deduplicated == 1
        assert sink.blobs_written == 0


class TestFailuresNeverRaise:
    def test_a_failed_row_put_is_counted_and_retried(self, caplog):
        s3 = FakeS3()
        s3.fail_rows = 1
        sink = _sink(s3)
        sink(_row(), {})
        assert sink.flush() == 0
        assert sink.flush_errors == 1
        assert "trace sink PUT failed" in caplog.text
        assert sink.flush() == 1
        assert len(s3.rows()) == 1

    def test_a_failed_blob_put_is_counted_retried_and_the_row_still_lands(self):
        s3 = FakeS3()
        s3.fail_blobs = 1
        sink = _sink(s3)
        sha, blob = _blob("body")
        sink(_row(), {sha: blob})
        assert sink.flush() == 1
        assert sink.blob_errors == 1
        assert len(s3.rows()) == 1
        sink.flush()
        assert sink.blobs_written == 1

    def test_the_retained_row_buffer_is_capped(self):
        s3 = FakeS3()
        s3.fail_rows = 10 ** 6
        sink = _sink(s3, flush_threshold=1)
        for _ in range(ts.MAX_RETAINED_MULTIPLE + 3):
            sink(_row(), {})
        assert sink.rows_dropped == 3

    def test_the_retained_blob_buffer_is_capped(self):
        s3 = FakeS3()
        s3.fail_blobs = 10 ** 6
        sink = _sink(s3, blob_flush_bytes=1)
        for i in range(ts.MAX_RETAINED_MULTIPLE + 2):
            sha, blob = _blob(f"b{i}")
            sink(_row(), {sha: blob})
            sink._flush_blobs()
        assert sink.blobs_dropped > 0

    def test_close_counts_what_could_not_be_written(self, caplog):
        s3 = FakeS3()
        s3.fail_rows = s3.fail_blobs = 10 ** 6
        sink = _sink(s3)
        sha, blob = _blob("x")
        sink(_row(), {sha: blob})
        sink.close()
        assert (sink.rows_dropped, sink.blobs_dropped) == (1, 1)
        assert "PERMANENTLY LOST" in caplog.text

    def test_a_context_manager_closes(self):
        s3 = FakeS3()
        with _sink(s3) as sink:
            sink(_row(), {})
        assert len(s3.rows()) == 1


class TestInMemorySink:
    def test_it_keeps_rows_and_first_blob(self):
        sink = ts.InMemoryTraceSink()
        sha, blob = _blob("x")
        sink(_row(), {sha: blob})
        sink(_row(), {sha: TraceBlob(b"other", "x")})
        assert len(sink.rows) == 2
        assert sink.blobs[sha] == blob


class TestDefaultSink:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        for var in (ts.BUCKET_ENV_VAR, ts.PREFIX_ENV_VAR, "KREPIS_RUN_ID"):
            monkeypatch.delenv(var, raising=False)
        ts.reset_default_trace_sink_for_tests()
        yield
        ts.reset_default_trace_sink_for_tests()

    def test_unconfigured_is_none(self):
        assert ts.default_trace_sink_from_env() is None
        assert ts.flush_default_trace_sink() == 0

    @pytest.mark.parametrize("var", [ts.BUCKET_ENV_VAR, ts.PREFIX_ENV_VAR])
    def test_half_configured_raises(self, monkeypatch, var):
        monkeypatch.setenv(var, "x")
        with pytest.raises(ts.TraceSinkConfigError, match="is set but"):
            ts.default_trace_sink_from_env()

    def test_configured_is_one_sink_per_process(self, monkeypatch):
        monkeypatch.setenv(ts.BUCKET_ENV_VAR, "bucket")
        monkeypatch.setenv(ts.PREFIX_ENV_VAR, "llm_traces/v1")
        monkeypatch.setenv("KREPIS_RUN_ID", "run-9")
        s3 = FakeS3()
        first = ts.default_trace_sink_from_env(s3_client=s3)
        assert first is ts.default_trace_sink_from_env()
        assert (first.bucket, first.prefix, first.run_id) == ("bucket", "llm_traces/v1", "run-9")
        first(_row(), {})
        assert ts.flush_default_trace_sink() == 1

    def test_the_decorator_flushes_on_success_and_on_error(self, monkeypatch):
        monkeypatch.setenv(ts.BUCKET_ENV_VAR, "bucket")
        monkeypatch.setenv(ts.PREFIX_ENV_VAR, "p")
        s3 = FakeS3()
        sink = ts.default_trace_sink_from_env(s3_client=s3)

        @ts.flush_traces_on_exit
        def ok():
            sink(_row(), {})
            return "done"

        @ts.flush_traces_on_exit
        def boom():
            sink(_row(), {})
            raise RuntimeError("handler failed")

        assert ok() == "done"
        with pytest.raises(RuntimeError, match="handler failed"):
            boom()
        assert len(s3.rows()) == 2

    def test_the_decorator_is_a_no_op_without_a_sink(self):
        @ts.flush_traces_on_exit
        def handler():
            return 1

        assert handler() == 1

    def test_the_decorator_swallows_a_flush_crash(self, monkeypatch, caplog):
        monkeypatch.setenv(ts.BUCKET_ENV_VAR, "bucket")
        monkeypatch.setenv(ts.PREFIX_ENV_VAR, "p")
        sink = ts.default_trace_sink_from_env(s3_client=FakeS3())
        monkeypatch.setattr(sink, "flush", lambda: 1 / 0)

        @ts.flush_traces_on_exit
        def handler():
            return 1

        assert handler() == 1
        assert "flush on handler exit failed" in caplog.text


def test_the_sink_declares_its_s3_surface_as_the_variable():
    from krepis import s3_surface

    assert {e.kind for e in ts.S3_SURFACE} == {s3_surface.KIND_ENV}
    assert ts.S3_SURFACE[0].variable == ts.PREFIX_ENV_VAR


def test_an_untyped_client_error_is_not_mistaken_for_a_precondition():
    assert not ts._is_precondition_failed(RuntimeError("x"))
    assert ts._is_precondition_failed(_ClientError("412"))
