"""Tests for :mod:`krepis.llm_trace` — the ``llm_trace.v1`` record builder.

The golden-record tests at the bottom are the contract: they pin the WHOLE
row, byte for byte after canonical serialization, for fixed inputs. A change
that moves any field fails here and has to be argued for — readers of
``llm_traces/v1/`` (evals, DuckDB views, the reconciliation job) dispatch on
``schema_version`` and must never see the meaning of a v1 row drift.

To regenerate after a DELIBERATE schema change (which should also bump the
schema version):

    KREPIS_REGENERATE_GOLDEN=1 PYTHONPATH=src python -m pytest tests/test_llm_trace.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from krepis import llm_trace as lt
from krepis.llm import GroundedResult, LLMResult, LLMUsage, StructuredResult

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
REGENERATE = os.environ.get("KREPIS_REGENERATE_GOLDEN") == "1"

T0 = datetime(2026, 10, 14, 13, 30, 0, 123456, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 14, 13, 30, 2, 456000, tzinfo=timezone.utc)
SPAN = "00f067aa0ba902b7"


def _blocks():
    return [
        lt.SourceBlock(
            cid="c1",
            kind="rag_chunk",
            source_id="rag.chunks/8d0c1f2e",
            text="[8-K | 2026-09-04]\nRevenue grew 12% year over year.",
            document_id="rag.documents/41aa",
            uri="https://www.sec.gov/Archives/example-8k.htm",
            version="filed:2026-09-04",
            retrieval={"method": "hybrid", "rank": 1, "score": 0.81},
        ),
        lt.SourceBlock(
            cid="c2",
            kind="s3_object",
            source_id="s3://bucket/news/QLYS.json",
            text='{"headline": "Qualys raises guidance"}',
            version="3HL4kqtJlcpXroDTDmJ+rmSpXd3dIbrHY",
        ),
    ]


# ── sources ──────────────────────────────────────────────────────────────────


class TestRenderSources:
    def test_offsets_point_at_exactly_the_injected_text(self):
        rendered = lt.render_sources(_blocks())
        for source in rendered.sources:
            assert rendered.text[source.char_start:source.char_end] == source.block.text
            assert source.content_sha256 == lt.sha256_hex(source.block.text)

    def test_delimiters_are_stable(self):
        rendered = lt.render_sources(_blocks()[:1])
        assert rendered.text.startswith('<source cid="c1">\n')
        assert rendered.text.endswith("\n</source>")

    def test_rendering_is_deterministic(self):
        assert lt.render_sources(_blocks()) == lt.render_sources(_blocks())

    def test_non_ascii_offsets_are_code_points(self):
        block = lt.SourceBlock(cid="c1", kind="doc", source_id="d", text="résumé — ✓")
        rendered = lt.render_sources([block])
        (only,) = rendered.sources
        assert rendered.text[only.char_start:only.char_end] == "résumé — ✓"

    def test_a_duplicate_cid_raises(self):
        block = _blocks()[0]
        with pytest.raises(ValueError, match="duplicate cid"):
            lt.render_sources([block, block])

    def test_mappings_are_accepted(self):
        rendered = lt.render_sources(
            [{"cid": "c9", "kind": "metric", "source_id": "m/x", "text": "42"}]
        )
        assert rendered.sources[0].block.cid == "c9"


class TestSourceBlockValidation:
    @pytest.mark.parametrize("cid", ["", "has space", "x" * 65, None])
    def test_a_bad_cid_is_rejected(self, cid):
        with pytest.raises(ValueError, match="cid"):
            lt.SourceBlock(cid=cid, kind="doc", source_id="d", text="t")

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            lt.SourceBlock(cid="c1", kind="vibes", source_id="d", text="t")

    def test_an_empty_source_id_is_rejected(self):
        with pytest.raises(ValueError, match="source_id"):
            lt.SourceBlock(cid="c1", kind="doc", source_id=" ", text="t")

    def test_non_string_text_is_rejected(self):
        with pytest.raises(ValueError, match="text"):
            lt.SourceBlock(cid="c1", kind="doc", source_id="d", text=b"bytes")

    def test_from_mapping_refuses_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown keys"):
            lt.SourceBlock.from_mapping(
                {"cid": "c1", "kind": "doc", "source_id": "d", "text": "t", "score": 1}
            )


class TestBindSources:
    def test_none_leaves_the_prompt_alone(self):
        assert lt.bind_sources("question", None) == ("question", ())

    def test_an_empty_list_leaves_the_prompt_alone(self):
        assert lt.bind_sources("question", []) == ("question", ())

    def test_a_list_is_rendered_and_prepended(self):
        content, sources = lt.bind_sources("question", _blocks())
        assert content.endswith("\n\nquestion")
        for source in sources:
            assert content[source.char_start:source.char_end] == source.block.text

    def test_a_pre_rendered_section_is_located_in_the_template(self):
        rendered = lt.render_sources(_blocks())
        template = f"Context follows.\n{rendered.text}\nNow answer."
        content, sources = lt.bind_sources(template, rendered)
        assert content == template
        for source in sources:
            assert content[source.char_start:source.char_end] == source.block.text

    def test_an_empty_pre_rendered_section_binds_nothing(self):
        assert lt.bind_sources("q", lt.render_sources([])) == ("q", ())

    @pytest.mark.parametrize("copies", [0, 2])
    def test_a_section_that_is_not_there_exactly_once_raises(self, copies):
        rendered = lt.render_sources(_blocks())
        with pytest.raises(ValueError, match=f"occurs {copies} times"):
            lt.bind_sources("q " + rendered.text * copies, rendered)

    @pytest.mark.parametrize("bad", ["a string", b"bytes", 42])
    def test_a_non_sequence_raises(self, bad):
        with pytest.raises(TypeError):
            lt.bind_sources("q", bad)


class TestCitedCids:
    def test_markers_are_found_in_order(self):
        assert lt.cited_cids("x [c2] y [c1] z [c2]", ["c1", "c2"]) == ["c2", "c1"]

    def test_comma_lists_are_split(self):
        assert lt.cited_cids("claim [c1, c3]", ["c1", "c2", "c3"]) == ["c1", "c3"]

    def test_unknown_markers_are_not_citations(self):
        assert lt.cited_cids("step [1] and [c7]", ["c1"]) == []

    def test_empty_text(self):
        assert lt.cited_cids("", ["c1"]) == []


# ── ids and context ──────────────────────────────────────────────────────────


class TestTraceContext:
    def test_a_valid_traceparent_wins(self):
        tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        ctx = lt.resolve_trace_context("run-1", environ={"TRACEPARENT": tp,
                                                         "KREPIS_TRACE_ID": "a" * 32})
        assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert ctx.parent_span_id == "00f067aa0ba902b7"
        assert ctx.source == "traceparent"
        assert ctx.run_id == "run-1"

    @pytest.mark.parametrize("tp", [
        "garbage",
        "00-" + "0" * 32 + "-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    ])
    def test_an_invalid_traceparent_is_ignored(self, tp, caplog):
        ctx = lt.resolve_trace_context("run-1", environ={"TRACEPARENT": tp})
        assert ctx.source == "run_id"
        assert "malformed" in caplog.text

    def test_the_env_trace_id_is_used(self):
        ctx = lt.resolve_trace_context(None, environ={"KREPIS_TRACE_ID": "AB" * 16})
        assert ctx.trace_id == "ab" * 16
        assert ctx.source == "env"

    def test_a_malformed_env_trace_id_is_ignored(self, caplog):
        ctx = lt.resolve_trace_context("r", environ={"KREPIS_TRACE_ID": "nope"})
        assert ctx.source == "run_id"
        assert "malformed" in caplog.text

    def test_the_run_id_derivation_is_stable_and_w3c_shaped(self):
        a = lt.resolve_trace_context("run-7", environ={})
        b = lt.resolve_trace_context("run-7", environ={})
        assert a.trace_id == b.trace_id == lt.derive_trace_id("run-7")
        assert len(a.trace_id) == 32 and int(a.trace_id, 16)
        assert lt.derive_trace_id("run-8") != a.trace_id

    def test_no_run_gets_a_random_trace(self):
        draws = iter([b"\x00" * 16, b"\x01" * 16])
        ctx = lt.resolve_trace_context(None, environ={}, rand=lambda n: next(draws))
        assert ctx.trace_id == "01" * 16
        assert ctx.source == "random"

    def test_span_ids_are_never_zero(self):
        draws = iter([b"\x00" * 8, b"\x02" * 8])
        assert lt.new_span_id(rand=lambda n: next(draws)) == "02" * 8


class TestCaptureMode:
    def test_default_is_full(self):
        assert lt.resolve_capture_mode({}) == lt.CAPTURE_FULL

    def test_hash_only_is_honoured(self):
        assert lt.resolve_capture_mode({"KREPIS_TRACE_CAPTURE": "HASH_ONLY"}) == "hash_only"

    def test_an_unknown_value_is_the_restrictive_mode(self, caplog):
        assert lt.resolve_capture_mode({"KREPIS_TRACE_CAPTURE": "fulll"}) == "hash_only"
        assert "unknown" in caplog.text


def test_isoformat_utc_has_millisecond_precision_and_z():
    assert lt.isoformat_utc(T0) == "2026-10-14T13:30:00.123Z"
    assert lt.isoformat_utc(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05.000Z"


def test_canonical_json_is_order_independent_and_tolerates_objects():
    assert lt.canonical_json_bytes({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'.encode()
    assert lt.canonical_json_bytes({"x": object}).startswith(b'{"x":"<class')


def test_the_emission_failure_counter_counts():
    before = lt.emission_failures()
    assert lt.record_emission_failure() == before + 1
    assert lt.emission_failures() == before + 1


# ── the record ───────────────────────────────────────────────────────────────


def _complete_result(user_content):
    return LLMResult(
        text="Revenue rose [c1]; guidance went up [c2].",
        model="deepseek-v4-flash",
        provider="litellm_proxy",
        usage=LLMUsage(
            input_tokens=1200, output_tokens=300, cache_read_tokens=800,
            reasoning_tokens=40, attempts=1,
        ),
        raw_request={
            "model": "med-deepseek-v4-flash-max",
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": "You are an analyst."},
                {"role": "user", "content": user_content},
            ],
        },
        registry_id="deepseek-v4-flash-max",
        served_deployment="med-deepseek-v4-flash-max",
        dlp_verdict="ok",
        finish_reason="stop",
        response_id="chatcmpl-abc123",
        trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
        span_id=SPAN,
    )


def _build_complete():
    user_content, sources = lt.bind_sources("Write the thesis for QLYS.", _blocks())
    result = _complete_result(user_content)
    return lt.build_trace_record(
        result,
        callsite_id="thinktank-thesis",
        context=lt.TraceContext(
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            run_id="a2e236a07e24",
            source="run_id",
        ),
        span_id=SPAN,
        started_at=T0,
        ended_at=T1,
        duration_ms=2332.5444,
        request_model="med-deepseek-v4-flash-max",
        sources=sources,
        system="You are an analyst.",
        subject={"kind": "ticker", "id": "QLYS"},
        prompt_id="thinktank_thesis",
        prompt_version="0.3.0",
        capture=lt.CAPTURE_FULL,
        library_version="0.0.0-test",
    ), result


def _build_grounded():
    result = GroundedResult(
        text="Answer from the web.",
        model="claude-haiku-4-5",
        provider="anthropic",
        usage=LLMUsage(input_tokens=50, output_tokens=20, web_search_requests=1),
        raw_request={"model": "claude-haiku-4-5", "max_tokens": 512,
                     "messages": [{"role": "user", "content": "q"}]},
        dlp_verdict="dlp_disabled",
        citations=[{"url": "https://example.com/a", "title": "A", "snippet": "alpha"},
                   {"url": "https://example.com/b", "title": None, "snippet": None}],
    )
    return lt.build_trace_record(
        result,
        callsite_id="morning-signal-generation",
        context=lt.TraceContext(trace_id="ab" * 16, source="env"),
        span_id="cd" * 8,
        started_at=T0,
        ended_at=T1,
        duration_ms=10,
        request_model="claude-haiku-4-5",
        library_version="0.0.0-test",
    )


def _golden(name, payload):
    path = GOLDEN_DIR / name
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if REGENERATE:
        path.write_text(text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8")), payload


class TestGoldenRecords:
    def test_complete_with_sources_full_capture(self):
        (row, blobs), _ = _build_complete()
        expected, actual = _golden(
            "llm_trace_v1_complete_with_sources.json",
            {"row": row, "blob_shas": sorted(blobs)},
        )
        assert actual == expected

    def test_grounded_hash_only_after_an_unscanned_body(self):
        row, blobs = _build_grounded()
        expected, actual = _golden(
            "llm_trace_v1_grounded_hash_only.json",
            {"row": row, "blob_shas": sorted(blobs)},
        )
        assert actual == expected


class TestRecordContract:
    def test_the_record_is_deterministic(self):
        assert _build_complete()[0] == _build_complete()[0]

    def test_every_hash_addresses_its_blob(self):
        (row, blobs), _ = _build_complete()
        for sha, blob in blobs.items():
            assert lt.sha256_hex(blob.data) == sha
        assert row["content"]["input_sha256"] in blobs
        assert row["content"]["output_sha256"] in blobs
        for citation in row["citations"]:
            assert citation["content_sha256"] in blobs

    def test_the_input_blob_is_the_exact_request(self):
        (row, blobs), result = _build_complete()
        body = json.loads(blobs[row["content"]["input_sha256"]].data)
        assert body == result.raw_request

    def test_spans_index_the_sent_user_message(self):
        (row, _), result = _build_complete()
        sent = result.raw_request["messages"][1]["content"]
        for citation, block in zip(row["citations"], _blocks()):
            span = citation["span"]
            assert span["message_index"] == 1
            assert sent[span["char_start"]:span["char_end"]] == block.text

    def test_citation_in_output_is_parsed_not_judged(self):
        (row, _), _ = _build_complete()
        assert [c["cited_in_output"] for c in row["citations"]] == [True, True]

    def test_structured_data_citations_count(self):
        user_content, sources = lt.bind_sources("q", _blocks())
        result = StructuredResult(
            text="{}", model="m", provider="openrouter", usage=LLMUsage(),
            raw_request={}, dlp_verdict="ok", structured_output_rung="native",
            data={"claims": [{"text": "guidance up", "evidence": "[c2]"}]},
        )
        row, blobs = lt.build_trace_record(
            result, callsite_id="x", context=lt.TraceContext(trace_id="ab" * 16),
            span_id=SPAN, started_at=T0, ended_at=T1, duration_ms=1,
            request_model="m", sources=sources,
        )
        assert [c["cited_in_output"] for c in row["citations"]] == [False, True]
        output = json.loads(blobs[row["content"]["output_sha256"]].data)
        assert output["data"] == result.data
        assert row["route"]["structured_output_rung"] == "native"

    def test_hash_only_writes_no_blobs_but_keeps_every_hash(self):
        user_content, sources = lt.bind_sources("q", _blocks())
        result = _complete_result(user_content)
        row, blobs = lt.build_trace_record(
            result, callsite_id="x", context=lt.TraceContext(trace_id="ab" * 16),
            span_id=SPAN, started_at=T0, ended_at=T1, duration_ms=1,
            request_model="m", sources=sources, capture=lt.CAPTURE_HASH_ONLY,
        )
        assert blobs == {}
        assert row["content"]["capture"] == "hash_only"
        assert row["content"]["capture_reason"] == "configured"
        assert all(c["content_sha256"] for c in row["citations"])

    @pytest.mark.parametrize("verdict", [None, "dlp_disabled", "scan_error"])
    def test_an_unscanned_body_is_never_kept(self, verdict):
        result = LLMResult(text="t", model="m", provider="p", usage=LLMUsage(),
                           raw_request={"secret?": "maybe"}, dlp_verdict=verdict)
        row, blobs = lt.build_trace_record(
            result, callsite_id="x", context=lt.TraceContext(trace_id="ab" * 16),
            span_id=SPAN, started_at=T0, ended_at=T1, duration_ms=1,
            request_model="m",
        )
        assert blobs == {}
        assert row["content"]["capture_reason"] == f"dlp_verdict={verdict}"

    def test_the_edge_request_id_is_set_only_on_the_router_edge(self):
        (row, _), _ = _build_complete()
        assert row["route"]["edge_request_id"] == "chatcmpl-abc123"
        grounded_row, _ = _build_grounded()
        assert grounded_row["route"]["edge_request_id"] is None

    def test_a_bad_span_id_or_capture_mode_raises(self):
        result = LLMResult(text="t", model="m", provider="p", usage=LLMUsage(),
                           raw_request={})
        kw = dict(callsite_id="x", context=lt.TraceContext(trace_id="ab" * 16),
                  started_at=T0, ended_at=T1, duration_ms=1, request_model="m")
        with pytest.raises(ValueError, match="span_id"):
            lt.build_trace_record(result, span_id="xyz", **kw)
        with pytest.raises(ValueError, match="capture"):
            lt.build_trace_record(result, span_id=SPAN, capture="off", **kw)
