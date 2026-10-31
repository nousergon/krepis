"""Tests for ``krepis.llm_capture`` — SFT v3 record building + gated sink."""

import json

import pytest

from krepis.llm import GroundedResult, LLMResult, LLMUsage, StructuredResult
from datetime import datetime, timezone

from krepis import llm_trace as lt
from krepis.llm_capture import (
    SFT_SCHEMA_VERSION,
    SftCaptureWriteError,
    SftProjectionError,
    append_sft_jsonl,
    build_sft_record,
    capture_enabled,
    capture_llm_call,
    content_hash,
    sft_record_from_trace,
)


@pytest.fixture(autouse=True)
def _capture_flags_clear(monkeypatch):
    monkeypatch.delenv("LLM_SFT_CAPTURE_ENABLED", raising=False)
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", raising=False)


def _anthropic_result():
    return LLMResult(
        text="the answer",
        model="claude-haiku-4-5",
        provider="anthropic",
        usage=LLMUsage(input_tokens=100, output_tokens=50),
        raw_request={
            "model": "claude-haiku-4-5",
            "max_tokens": 256,
            "system": [{"type": "text", "text": "You are a judge.",
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "score this"}],
        },
        raw_response=None,
    )


def _openrouter_result():
    return StructuredResult(
        text='{"score": 4}',
        model="deepseek/deepseek-v4-flash",
        provider="openrouter",
        usage=LLMUsage(input_tokens=10, output_tokens=5,
                       provider_cost_usd=0.0001),
        raw_request={
            "model": "deepseek/deepseek-v4-flash:floor",
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": "score this"},
            ],
            "response_format": {"type": "json_schema"},
        },
        raw_response=None,
        data={"score": 4},
    )


class TestBuildRecord:
    def test_envelope_shape(self):
        rec = build_sft_record(
            _anthropic_result(), producer="mnemon_judge",
            meta={"memory_id": 42}, cost_usd=0.001,
        )
        assert rec["schema_version"] == SFT_SCHEMA_VERSION == 3
        assert rec["producer"] == "mnemon_judge"
        assert rec["model"] == "claude-haiku-4-5"
        assert rec["output_text"] == "the answer"
        assert rec["cost_usd"] == 0.001
        assert rec["meta"]["memory_id"] == 42
        assert rec["meta"]["provider"] == "anthropic"
        assert rec["usage"]["input_tokens"] == 100
        assert rec["provenance"]["source"] == "live"
        assert rec["provenance"]["content_hash"] == content_hash(
            rec["input_messages"]
        )

    def test_anthropic_system_normalized_into_messages(self):
        rec = build_sft_record(_anthropic_result(), producer="p")
        assert rec["input_messages"][0] == {
            "role": "system", "content": "You are a judge.",
        }
        assert rec["input_messages"][1]["role"] == "user"
        # system stays OUT of invocation_params (it's part of the input)
        assert "system" not in rec["invocation_params"]
        assert rec["invocation_params"]["max_tokens"] == 256

    def test_openai_messages_pass_through(self):
        rec = build_sft_record(_openrouter_result(), producer="p")
        assert rec["input_messages"][0]["role"] == "system"
        assert rec["structured_output"] == {"score": 4}
        assert rec["invocation_params"]["response_format"] == {
            "type": "json_schema"
        }

    def test_grounded_search_payload_in_meta(self):
        result = GroundedResult(
            text="t", model="m", provider="anthropic",
            usage=LLMUsage(), raw_request={"messages": []},
            searches=[{"query": "q", "urls": [], "result_count": 0,
                       "error": None}],
            citations=[{"url": "https://x"}],
        )
        rec = build_sft_record(result, producer="p")
        assert rec["meta"]["searches"][0]["query"] == "q"
        assert rec["meta"]["citations"] == [{"url": "https://x"}]

    def test_empty_producer_raises(self):
        with pytest.raises(ValueError, match="producer"):
            build_sft_record(_anthropic_result(), producer="  ")

    def test_bad_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            build_sft_record(
                _anthropic_result(), producer="p", source="imagined"
            )


class TestGateAndSink:
    def test_disabled_is_noop(self, tmp_path):
        sink = tmp_path / "sft.jsonl"
        wrote = capture_llm_call(
            _anthropic_result(), producer="p", sink_path=sink
        )
        assert wrote is False
        assert not sink.exists()

    @pytest.mark.parametrize(
        "var", ["LLM_SFT_CAPTURE_ENABLED", "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED"]
    )
    def test_either_flag_enables(self, monkeypatch, tmp_path, var):
        monkeypatch.setenv(var, "1")
        assert capture_enabled()
        sink = tmp_path / "nested" / "sft.jsonl"
        wrote = capture_llm_call(
            _openrouter_result(), producer="p", sink_path=sink,
            meta={"edition": "am"},
        )
        assert wrote is True
        lines = sink.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["producer"] == "p"
        assert rec["meta"]["edition"] == "am"

    def test_append_accumulates(self, tmp_path):
        sink = tmp_path / "sft.jsonl"
        r1 = build_sft_record(_anthropic_result(), producer="p")
        assert append_sft_jsonl(sink, [r1]) == 1
        assert append_sft_jsonl(sink, [r1, r1]) == 2
        assert len(sink.read_text().strip().splitlines()) == 3

    def test_write_failure_raises(self, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("file, not dir")
        rec = build_sft_record(_anthropic_result(), producer="p")
        with pytest.raises(SftCaptureWriteError):
            append_sft_jsonl(blocker / "sub" / "sft.jsonl", [rec])


class TestContentHashParity:
    def test_matches_nousergon_lib_algorithm(self):
        # Byte-identical canonicalization to nousergon_lib.sft.content_hash:
        # json.dumps(sort_keys=True, ensure_ascii=False, default=str) sha256.
        import hashlib
        msgs = [{"role": "user", "content": "héllo"}]
        expected = hashlib.sha256(
            json.dumps(msgs, sort_keys=True, ensure_ascii=False,
                       default=str).encode("utf-8")
        ).hexdigest()
        assert content_hash(msgs) == expected


# ── projection from an llm_trace.v1 row (alpha-engine-config-I12312) ───────

_T0 = datetime(2026, 10, 31, 9, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 10, 31, 9, 0, 1, 250000, tzinfo=timezone.utc)


def _trace(result, *, capture=lt.CAPTURE_FULL, error=None):
    return lt.build_trace_record(
        result,
        callsite_id="unit-test-callsite",
        context=lt.TraceContext(trace_id="ab" * 16, run_id="run-1", source="env"),
        span_id="cd" * 8,
        started_at=_T0,
        ended_at=_T1,
        duration_ms=1250.0,
        request_model="requested-model",
        capture=capture,
        error=error,
    )


def _scanned(result):
    result.dlp_verdict = "ok"  # bodies are kept only for a scanned body
    return result


# Fields a live capture and a projection of the same call must agree on.
_PARITY_FIELDS = (
    "schema_version", "producer", "model", "input_messages",
    "invocation_params", "output_text", "structured_output",
)


class TestProjectionFromTrace:
    @pytest.mark.parametrize("make", [_anthropic_result, _openrouter_result])
    def test_matches_live_capture_of_the_same_call(self, make):
        result = _scanned(make())
        row, blobs = _trace(result)
        live = build_sft_record(result, producer="p")
        projected = sft_record_from_trace(row, blobs, producer="p")
        for name in _PARITY_FIELDS:
            assert projected[name] == live[name], name
        assert projected["provenance"] == live["provenance"]
        assert projected["meta"]["provider"] == live["meta"]["provider"]

    def test_is_deterministic_and_joins_back_to_its_trace(self):
        row, blobs = _trace(_scanned(_anthropic_result()))
        first = sft_record_from_trace(row, blobs, producer="p")
        assert first == sft_record_from_trace(row, blobs, producer="p")
        assert first["captured_at"] == row["ended_at"]
        assert first["meta"]["trace"] == {
            "trace_id": "ab" * 16, "span_id": "cd" * 8,
            "callsite_id": "unit-test-callsite", "run_id": "run-1",
        }

    def test_output_message_is_the_traced_output_body(self):
        row, blobs = _trace(_scanned(_openrouter_result()))
        rec = sft_record_from_trace(row, blobs, producer="p")
        assert rec["output_message"] == {"text": '{"score": 4}',
                                         "data": {"score": 4}}

    def test_usage_carries_only_what_the_trace_measured(self):
        result = _scanned(_anthropic_result())
        result.usage = LLMUsage(input_tokens=100, output_tokens=50,
                                cache_read_tokens=7, cache_create_tokens=3,
                                reasoning_tokens=2, attempts=1,
                                provider_cost_usd=0.5, web_fetch_requests=4)
        row, blobs = _trace(result)
        usage = sft_record_from_trace(row, blobs, producer="p")["usage"]
        assert usage == {
            "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 7,
            "cache_create_tokens": 3, "reasoning_tokens": 2,
            "budget_escalations": 0, "attempts": 1,
            "web_search_requests": 0, "usage_unknown": False,
        }
        # Never reported as a zero the trace did not record.
        assert "provider_cost_usd" not in usage
        assert "web_fetch_requests" not in usage

    def test_web_citations_land_in_meta(self):
        result = _scanned(GroundedResult(
            text="t", model="m", provider="anthropic", usage=LLMUsage(),
            raw_request={"messages": [{"role": "user", "content": "q"}]},
            citations=[{"url": "https://x", "title": "X", "snippet": "s"}],
        ))
        row, blobs = _trace(result)
        rec = sft_record_from_trace(row, blobs, producer="p")
        assert rec["meta"]["citations"] == [
            {"url": "https://x", "title": "X", "snippet": "s"}]

    def test_accepts_raw_bytes_as_read_back_from_the_store(self):
        row, blobs = _trace(_scanned(_anthropic_result()))
        as_bytes = {sha: blob.data for sha, blob in blobs.items()}
        assert (sft_record_from_trace(row, as_bytes, producer="p")
                == sft_record_from_trace(row, blobs, producer="p"))

    def test_hash_only_row_is_refused(self):
        row, blobs = _trace(_scanned(_anthropic_result()),
                            capture=lt.CAPTURE_HASH_ONLY)
        assert blobs == {}
        with pytest.raises(SftProjectionError, match="hash_only"):
            sft_record_from_trace(row, blobs, producer="p")

    def test_unscanned_body_downgraded_to_hash_only_is_refused(self):
        result = _anthropic_result()  # dlp_verdict None -> hash_only
        row, blobs = _trace(result)
        with pytest.raises(SftProjectionError, match="hash_only"):
            sft_record_from_trace(row, blobs, producer="p")

    def test_a_call_that_raised_is_refused(self):
        row, blobs = _trace(_scanned(_anthropic_result()),
                            error=RuntimeError("boom"))
        with pytest.raises(SftProjectionError, match="no completion"):
            sft_record_from_trace(row, blobs, producer="p")

    def test_missing_blob_is_refused(self):
        row, blobs = _trace(_scanned(_anthropic_result()))
        out_sha = row["krepis"]["content"]["output_sha256"]
        partial = {k: v for k, v in blobs.items() if k != out_sha}
        with pytest.raises(SftProjectionError, match="not in the blobs"):
            sft_record_from_trace(row, partial, producer="p")

    def test_blob_that_does_not_hash_to_its_address_is_refused(self):
        row, blobs = _trace(_scanned(_anthropic_result()))
        in_sha = row["krepis"]["content"]["input_sha256"]
        tampered = dict(blobs)
        tampered[in_sha] = b'{"messages": []}'
        with pytest.raises(SftProjectionError, match="does not hash"):
            sft_record_from_trace(row, tampered, producer="p")

    def test_wrong_schema_version_is_refused(self):
        row, blobs = _trace(_scanned(_anthropic_result()))
        with pytest.raises(SftProjectionError, match="llm_trace.v1"):
            sft_record_from_trace(dict(row, schema_version="llm_trace.v0"),
                                  blobs, producer="p")

    def test_producer_and_source_validated_like_build(self):
        row, blobs = _trace(_scanned(_anthropic_result()))
        with pytest.raises(ValueError):
            sft_record_from_trace(row, blobs, producer=" ")
        with pytest.raises(ValueError):
            sft_record_from_trace(row, blobs, producer="p", source="bogus")
