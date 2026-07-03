"""Tests for ``krepis.llm_capture`` — SFT v3 record building + gated sink."""

import json

import pytest

from krepis.llm import GroundedResult, LLMResult, LLMUsage, StructuredResult
from krepis.llm_capture import (
    SFT_SCHEMA_VERSION,
    SftCaptureWriteError,
    append_sft_jsonl,
    build_sft_record,
    capture_enabled,
    capture_llm_call,
    content_hash,
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
