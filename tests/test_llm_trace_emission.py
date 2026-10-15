"""``LLMClient`` emits one ``llm_trace.v1`` row per public call.

These pin the chokepoint, not the record shape (``test_llm_trace.py`` owns
that): every public method traces through ``_finalize_result``, sources are
bound into the exact user content that is sent, the router edge is tagged,
the cost row joins the trace row, and emission never raises.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import krepis.llm as _llm
from krepis import llm_trace as lt
from krepis import trace_sink as ts
from krepis.llm import LLMClient, SearchOptions
from krepis.llm_config import ROUTER_EDGE_PROVIDER, ModelSpec
from krepis.session_dlp import DLP_OK, DLPVerdict

OPENROUTER_SPEC = ModelSpec("openrouter", "moonshotai/kimi-k2.6", max_tokens=1024)
ANTHROPIC_SPEC = ModelSpec("anthropic", "claude-haiku-4-5", max_tokens=1024)
EDGE_SPEC = ModelSpec(
    provider=ROUTER_EDGE_PROVIDER,
    model="low-deepseek-v4-flash",
    base_url="https://router.example.invalid:8443",
    api_key_env="ROUTER_CONSUMER_THINKTANK",
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    for var in (ts.BUCKET_ENV_VAR, ts.PREFIX_ENV_VAR, "KREPIS_RUN_ID",
                "KREPIS_TRACE_ID", "TRACEPARENT", "KREPIS_TRACE_CAPTURE",
                "KREPIS_COST_SINK_BUCKET", "KREPIS_COST_SINK_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    # A body that was scanned and passed — the only case in which content is
    # kept. The scan itself is session_dlp's to test.
    monkeypatch.delenv("KREPIS_DLP_DISABLED", raising=False)
    monkeypatch.setattr(_llm, "check_request", lambda _b: DLPVerdict(DLP_OK, "", 1.0, 0.0))
    ts.reset_default_trace_sink_for_tests()
    yield
    ts.reset_default_trace_sink_for_tests()


class FakeOpenAI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        return self._responses.pop(0)


class FakeAnthropic:
    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **payload):
        self.payloads.append(payload)
        return self._responses.pop(0)


def _usage():
    return SimpleNamespace(prompt_tokens=100, completion_tokens=50,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=0))


def _resp(content, model="moonshotai/kimi-k2.6", rid="gen-123"):
    return SimpleNamespace(
        id=rid,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                 finish_reason="stop")],
        usage=_usage(),
        model=model,
    )


def _client(spec, fake, **kw):
    kw.setdefault("trace_sink", ts.InMemoryTraceSink())
    return LLMClient(spec, callsite_id="krepis-trace-test",
                     client_factory=lambda _s, _k: fake, api_key="k", **kw)


def _blocks():
    return [
        lt.SourceBlock(cid="c1", kind="rag_chunk", source_id="rag.chunks/1",
                       text="Revenue grew 12%."),
        lt.SourceBlock(cid="c2", kind="s3_object", source_id="s3://b/k.json",
                       text='{"guidance": "raised"}'),
    ]


class TestEveryCallIsTraced:
    def test_complete_emits_one_row_with_ids_latency_and_response_id(self):
        sink = ts.InMemoryTraceSink()
        fake = FakeOpenAI([_resp("plain answer")])
        result = _client(OPENROUTER_SPEC, fake, trace_sink=sink).complete(
            system="sys", user_content="q", subject={"kind": "ticker", "id": "QLYS"},
            prompt_id="p", prompt_version="1",
        )
        (row,) = sink.rows
        assert row["schema_version"] == "llm_trace.v1"
        assert row["callsite_id"] == "krepis-trace-test"
        assert row["trace_id"] == result.trace_id and len(result.trace_id) == 32
        assert row["span_id"] == result.span_id and len(result.span_id) == 16
        assert result.duration_ms is not None and result.duration_ms >= 0
        assert result.response_id == "gen-123"
        assert row["gen_ai"]["response.id"] == "gen-123"
        assert row["gen_ai"]["request.model"] == "moonshotai/kimi-k2.6"
        assert row["subject"] == {"kind": "ticker", "id": "QLYS"}
        assert row["prompt"]["template_id"] == "p"
        assert row["prompt"]["system_sha256"] == lt.sha256_hex("sys")
        assert row["content"]["capture"] == "full"
        sent = json.loads(sink.blobs[row["content"]["input_sha256"]].data)
        assert sent == fake.kwargs[0]

    def test_ids_are_stamped_even_with_no_trace_sink(self):
        fake = FakeOpenAI([_resp("x")])
        client = LLMClient(OPENROUTER_SPEC, callsite_id="c",
                           client_factory=lambda _s, _k: fake)
        result = client.complete(system="s", user_content="u")
        assert result.trace_id and result.span_id and result.duration_ms is not None
        assert result.trace_emission_error is None

    def test_a_run_shares_one_trace_and_each_call_has_its_own_span(self, monkeypatch):
        monkeypatch.setenv("KREPIS_RUN_ID", "run-42")
        sink = ts.InMemoryTraceSink()
        fake = FakeOpenAI([_resp("a"), _resp("b")])
        client = _client(OPENROUTER_SPEC, fake, trace_sink=sink)
        client.complete(system="s", user_content="1")
        client.complete(system="s", user_content="2")
        first, second = sink.rows
        assert first["trace_id"] == second["trace_id"] == lt.derive_trace_id("run-42")
        assert first["span_id"] != second["span_id"]
        assert first["run_id"] == "run-42"

    def test_structured_is_traced_and_data_citations_resolve(self):
        sink = ts.InMemoryTraceSink()
        fake = FakeOpenAI([_resp('{"claim": "guidance raised [c2]"}')])
        result = _client(OPENROUTER_SPEC, fake, trace_sink=sink).structured(
            system="s", user_content="q", schema={"type": "object"},
            schema_name="claims", sources=_blocks(),
        )
        (row,) = sink.rows
        assert result.data == {"claim": "guidance raised [c2]"}
        assert [c["cited_in_output"] for c in row["citations"]] == [False, True]
        assert row["route"]["structured_output_rung"] == "native"

    def test_grounded_is_traced_with_its_web_citations(self):
        sink = ts.InMemoryTraceSink()
        msg = SimpleNamespace(
            id="msg_01",
            model="claude-haiku-4-5",
            content=[SimpleNamespace(type="text", text="grounded answer")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0,
                                  cache_creation=None, server_tool_use=None),
        )
        fake = FakeAnthropic([msg])
        _client(ANTHROPIC_SPEC, fake, trace_sink=sink).complete_grounded(
            system="s", user_content="q", search=SearchOptions(),
        )
        (row,) = sink.rows
        assert row["gen_ai"]["response.id"] == "msg_01"
        assert row["gen_ai"]["provider.name"] == "anthropic"


class TestSourcesReachTheWire:
    def test_a_source_list_is_prepended_and_its_offsets_index_the_sent_message(self):
        sink = ts.InMemoryTraceSink()
        fake = FakeOpenAI([_resp("Revenue grew [c1].")])
        _client(OPENROUTER_SPEC, fake, trace_sink=sink).complete(
            system="s", user_content="What happened?", sources=_blocks(),
        )
        sent = fake.kwargs[0]["messages"][1]["content"]
        assert sent.endswith("What happened?")
        (row,) = sink.rows
        for citation, block in zip(row["citations"], _blocks()):
            span = citation["span"]
            assert sent[span["char_start"]:span["char_end"]] == block.text
            assert citation["content_sha256"] == lt.sha256_hex(block.text)
            assert sink.blobs[citation["content_sha256"]].data == block.text.encode()
        assert [c["cited_in_output"] for c in row["citations"]] == [True, False]

    def test_a_pre_rendered_section_is_located_in_the_callers_template(self):
        sink = ts.InMemoryTraceSink()
        rendered = lt.render_sources(_blocks())
        template = f"Filings:\n{rendered.text}\n\nWrite the thesis."
        fake = FakeOpenAI([_resp("ok")])
        _client(OPENROUTER_SPEC, fake, trace_sink=sink).complete(
            system="s", user_content=template, sources=rendered,
        )
        sent = fake.kwargs[0]["messages"][1]["content"]
        assert sent == template
        span = sink.rows[0]["citations"][1]["span"]
        assert sent[span["char_start"]:span["char_end"]] == _blocks()[1].text

    def test_a_misplaced_pre_rendered_section_fails_before_any_spend(self):
        fake = FakeOpenAI([_resp("never")])
        with pytest.raises(ValueError, match="occurs 0 times"):
            _client(OPENROUTER_SPEC, fake).complete(
                system="s", user_content="no sources here",
                sources=lt.render_sources(_blocks()),
            )
        assert fake.kwargs == []

    def test_anthropic_payload_carries_the_rendered_sources(self):
        sink = ts.InMemoryTraceSink()
        msg = SimpleNamespace(
            id="msg_02", model="claude-haiku-4-5",
            content=[SimpleNamespace(type="text", text="answer [c2]")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0,
                                  cache_creation=None, server_tool_use=None),
        )
        fake = FakeAnthropic([msg])
        _client(ANTHROPIC_SPEC, fake, trace_sink=sink).complete(
            system="s", user_content="q", sources=_blocks(),
        )
        sent = fake.payloads[0]["messages"][0]["content"]
        assert sent.startswith('<source cid="c1">')
        span = sink.rows[0]["citations"][0]["span"]
        assert sent[span["char_start"]:span["char_end"]] == _blocks()[0].text
        assert [c["cited_in_output"] for c in sink.rows[0]["citations"]] == [False, True]


class TestDlpInterplay:
    def test_an_unscanned_body_is_traced_hash_only(self, monkeypatch):
        monkeypatch.setenv("KREPIS_DLP_DISABLED", "1")
        sink = ts.InMemoryTraceSink()
        fake = FakeOpenAI([_resp("x")])
        _client(OPENROUTER_SPEC, fake, trace_sink=sink).complete(
            system="s", user_content="u", sources=_blocks(),
        )
        (row,) = sink.rows
        assert sink.blobs == {}
        assert row["content"]["capture"] == "hash_only"
        assert row["content"]["capture_reason"] == "dlp_verdict=dlp_disabled"
        assert row["dlp"]["verdict"] == "dlp_disabled"

    def test_the_capture_env_can_restrict_to_hashes(self, monkeypatch):
        monkeypatch.setenv("KREPIS_TRACE_CAPTURE", "hash_only")
        sink = ts.InMemoryTraceSink()
        _client(OPENROUTER_SPEC, FakeOpenAI([_resp("x")]), trace_sink=sink).complete(
            system="s", user_content="u",
        )
        assert sink.blobs == {}
        assert sink.rows[0]["content"]["capture"] == "hash_only"


class TestRouterEdgeAttribution:
    def test_edge_requests_carry_callsite_run_and_trace(self, monkeypatch):
        monkeypatch.setenv("KREPIS_RUN_ID", "run-edge")
        fake = FakeOpenAI([_resp("hi", model="deepseek/deepseek-v4-flash", rid="chatcmpl-9")])
        sink = ts.InMemoryTraceSink()
        result = _client(EDGE_SPEC, fake, trace_sink=sink).complete(
            system="s", user_content="u",
        )
        metadata = fake.kwargs[0]["extra_body"]["metadata"]
        assert metadata["tags"] == ["callsite:krepis-trace-test"]
        assert metadata["spend_logs_metadata"] == {
            "callsite_id": "krepis-trace-test",
            "run_id": "run-edge",
            "trace_id": result.trace_id,
            "span_id": result.span_id,
        }
        assert sink.rows[0]["route"]["edge_request_id"] == "chatcmpl-9"

    def test_structured_edge_requests_are_tagged_too(self):
        fake = FakeOpenAI([_resp('{"a": 1}', model="deepseek/deepseek-v4-flash")])
        _client(EDGE_SPEC, fake).structured(
            system="s", user_content="u", schema={"type": "object"}, schema_name="x",
        )
        assert "spend_logs_metadata" in fake.kwargs[0]["extra_body"]["metadata"]

    def test_non_edge_requests_carry_no_metadata(self):
        fake = FakeOpenAI([_resp("hi")])
        _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert "metadata" not in fake.kwargs[0].get("extra_body", {})


class TestJoins:
    def test_the_cost_row_carries_the_trace_and_span_ids(self):
        records = []
        fake = FakeOpenAI([_resp("x")])
        result = _client(OPENROUTER_SPEC, fake, cost_sink=records.append).complete(
            system="s", user_content="u",
        )
        assert records[0]["trace_id"] == result.trace_id
        assert records[0]["span_id"] == result.span_id

    def test_a_streamed_call_keeps_its_response_id(self):
        chunks = [
            SimpleNamespace(id="chatcmpl-s1", model="moonshotai/kimi-k2.6", usage=None,
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="he"),
                                                     finish_reason=None)]),
            SimpleNamespace(id="chatcmpl-s1", model="moonshotai/kimi-k2.6", usage=None,
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="y"),
                                                     finish_reason="stop")]),
            SimpleNamespace(id="chatcmpl-s1", model="moonshotai/kimi-k2.6",
                            usage=_usage(), choices=[]),
        ]
        sink = ts.InMemoryTraceSink()
        result = _client(OPENROUTER_SPEC, FakeOpenAI([iter(chunks)]),
                         trace_sink=sink).complete(system="s", user_content="u",
                                                   stream=True)
        assert result.text == "hey"
        assert result.response_id == "chatcmpl-s1"
        assert sink.rows[0]["route"]["streamed"] is True


class TestEmissionNeverRaises:
    def test_a_sink_that_raises_is_logged_counted_and_recorded(self, caplog):
        def broken(_row, _blobs):
            raise RuntimeError("disk full")

        before = lt.emission_failures()
        fake = FakeOpenAI([_resp("still answered")])
        result = _client(OPENROUTER_SPEC, fake, trace_sink=broken).complete(
            system="s", user_content="u",
        )
        assert result.text == "still answered"
        assert result.trace_emission_error == "disk full"
        assert lt.emission_failures() == before + 1
        assert "trace emission failed" in caplog.text

    def test_a_bad_subject_degrades_the_trace_not_the_call(self):
        fake = FakeOpenAI([_resp("ok")])
        result = _client(OPENROUTER_SPEC, fake).complete(
            system="s", user_content="u", subject="not-a-mapping",
        )
        assert result.text == "ok"
        assert result.trace_emission_error

    def test_an_unresolvable_context_degrades_to_a_random_trace(self, monkeypatch, caplog):
        def boom(*_a, **_k):
            raise RuntimeError("bad env")

        monkeypatch.setattr(_llm, "resolve_trace_context", boom)
        fake = FakeOpenAI([_resp("ok")])
        result = _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert len(result.trace_id) == 32
        assert "could not resolve trace context" in caplog.text

    def test_a_frozen_result_does_not_turn_a_failure_into_a_raise(self):
        class Frozen:
            span_id = "00f067aa0ba902b7"
            duration_ms = 1.0
            model = "m"

            def __setattr__(self, name, value):
                raise AttributeError("frozen")

        client = _client(OPENROUTER_SPEC, FakeOpenAI([]),
                         trace_sink=lambda *_: 1 / 0)
        client._emit_trace_record(
            Frozen(), context=lt.TraceContext(trace_id="ab" * 16),
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
        )


class TestEnvironmentActivation:
    def test_both_variables_build_the_process_default_sink(self, monkeypatch):
        monkeypatch.setenv(ts.BUCKET_ENV_VAR, "alpha-engine-research")
        monkeypatch.setenv(ts.PREFIX_ENV_VAR, "llm_traces/v1")
        client = LLMClient(OPENROUTER_SPEC, callsite_id="c",
                           client_factory=lambda _s, _k: FakeOpenAI([]))
        assert isinstance(client._trace_sink, ts.S3JsonlTraceSink)
        assert client._trace_sink is ts.default_trace_sink_from_env()

    def test_a_half_configured_environment_fails_before_any_spend(self, monkeypatch):
        monkeypatch.setenv(ts.BUCKET_ENV_VAR, "alpha-engine-research")
        with pytest.raises(ts.TraceSinkConfigError):
            LLMClient(OPENROUTER_SPEC, callsite_id="c")

    def test_unconfigured_means_no_sink(self):
        assert LLMClient(OPENROUTER_SPEC, callsite_id="c")._trace_sink is None
