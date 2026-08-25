"""Streaming completions through ``krepis.llm.LLMClient``.

Why this path exists (``alpha-engine-config-I8164``): every Director plan
failure in August 2026 had one shape — a non-streaming completion ran into a
request deadline, returned nothing, and the entire partial generation was
discarded. Four attempts on 2026-08-22, two on 2026-08-14, two on 2026-08-08.
Streaming replaces "how long may the whole generation take" with "how long may
the model be SILENT", which is the only condition a client can diagnose from
outside — so a slow call completes and a hung one is caught FASTER than a slow
one was before.

These lock the four properties that make that true rather than merely claimed:
the inter-chunk budget actually fires, a completed stream returns the SAME
shape as the non-streaming path, ``finish_reason`` reaches the result, and a
streamed call is still attributed — priced when usage arrives, and explicitly
UNPRICED (never zero) when it does not.
"""

import json
import threading
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from krepis.llm import (
    BudgetExhaustedError,
    LLMClient,
    StreamIdleTimeoutError,
    StreamingUnsupportedError,
    StreamTotalTimeoutError,
)
from krepis.llm_config import ModelSpec


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("KREPIS_DLP_DISABLED", "1")


OPENROUTER_SPEC = ModelSpec("openrouter", "moonshotai/kimi-k2.6", max_tokens=1024)
ANTHROPIC_SPEC = ModelSpec("anthropic", "claude-haiku-4-5", max_tokens=1024)


class Probe(BaseModel):
    name: str
    score: int


# ── doubles ───────────────────────────────────────────────────────────────


def _chunk(content=None, finish_reason=None, usage=None, model="moonshotai/kimi-k2.6"):
    """One OpenAI-wire streaming chunk.

    The usage-bearing final chunk really does carry an EMPTY ``choices`` on
    every OpenAI-compatible route, so the double must too — an accumulator that
    reads a choice before checking usage would pass against a friendlier fake
    and drop every token count in production.
    """
    choices = []
    if content is not None or finish_reason is not None:
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage, model=model)


def _usage(prompt=100, completion=50):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )


def _text_stream(pieces, *, finish_reason="stop", usage=_usage()):
    chunks = [_chunk(content=p) for p in pieces]
    chunks.append(_chunk(finish_reason=finish_reason))
    if usage is not None:
        chunks.append(_chunk(usage=usage))
    return chunks


class _StallingStream:
    """Yields *pieces*, then blocks forever — a live socket that went quiet.

    A dead stream and a slow one are indistinguishable to a total-duration
    deadline; this is the case the idle budget is meant to separate, so the
    double blocks rather than raising. The event lets the test release the
    worker thread at teardown instead of leaking a blocked one per test.
    """

    def __init__(self, pieces):
        self.pieces = pieces
        self.released = threading.Event()
        self.closed = False

    def __iter__(self):
        for piece in self.pieces:
            yield _chunk(content=piece)
        self.released.wait(30)

    def close(self):
        self.closed = True
        self.released.set()


class _RelentlessStream:
    """Yields a chunk every *interval* seconds, FOREVER — never idle.

    This is alpha-engine-config-I8348's exact failure shape: a route that
    keeps producing chunks fast enough to keep resetting the idle clock, so
    an idle-only bound never fires and only a TOTAL bound can catch it.
    """

    def __init__(self, interval):
        self.interval = interval
        self.closed = False
        self._stop = threading.Event()

    def __iter__(self):
        i = 0
        while not self._stop.wait(self.interval):
            i += 1
            yield _chunk(content=f"c{i}")

    def close(self):
        self.closed = True
        self._stop.set()


class FakeOpenAI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


class FakeAnthropic:
    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **payload):
        self.payloads.append(payload)
        return self._responses.pop(0)


def _client(spec, fake, **kw):
    return LLMClient(
        spec,
        callsite_id="krepis-stream-test",
        client_factory=lambda _spec, _key: fake,
        **kw,
    )


# ── the inter-chunk budget ────────────────────────────────────────────────


class TestInterChunkIdleTimeoutFires:
    """The bound is SILENCE BETWEEN chunks, not total duration."""

    def test_a_stalled_stream_aborts_on_the_idle_budget(self):
        stream = _StallingStream(["par", "tial "])
        fake = FakeOpenAI([stream])
        client = _client(OPENROUTER_SPEC, fake)

        with pytest.raises(StreamIdleTimeoutError) as excinfo:
            client.complete(
                system="s", user_content="u", stream=True, idle_timeout=0.25
            )

        err = excinfo.value
        assert err.partial_text == "par" + "tial ", (
            "the partial generation must ride the exception — discarding it "
            "is the failure mode this whole path exists to remove"
        )
        assert err.chunks == 2
        assert err.idle_timeout == 0.25
        assert err.finish_reason is None, (
            "a stream that died never reported how it ended; claiming a "
            "finish_reason here would invent one"
        )
        assert stream.closed, "the abandoned stream must be closed, not leaked"

    def test_slow_but_alive_completes(self):
        """Chunks slower than the budget INDIVIDUALLY are fine; only the gap
        between them counts. A total-duration deadline cannot express this."""
        import time

        def _dribble():
            for piece in ("a", "b", "c", "d"):
                time.sleep(0.05)
                yield _chunk(content=piece)
            yield _chunk(finish_reason="stop")
            yield _chunk(usage=_usage())

        fake = FakeOpenAI([_dribble()])
        client = _client(OPENROUTER_SPEC, fake)
        result = client.complete(
            system="s", user_content="u", stream=True, idle_timeout=0.5
        )
        assert result.text == "abcd"

    def test_a_transport_error_mid_stream_propagates(self):
        """An idle budget must not convert a real provider failure into a
        timeout — the two have different fixes."""
        def _explode():
            yield _chunk(content="a")
            raise RuntimeError("upstream reset")

        fake = FakeOpenAI([_explode()])
        client = _client(OPENROUTER_SPEC, fake)
        with pytest.raises(RuntimeError, match="upstream reset"):
            client.complete(
                system="s", user_content="u", stream=True, idle_timeout=5
            )

    def test_idle_timeout_at_or_above_the_transport_timeout_warns(self, caplog):
        """A knob that cannot fire is a knob that quietly does nothing."""
        fake = FakeOpenAI([_text_stream(["hi"])])
        client = _client(OPENROUTER_SPEC, fake, timeout=10.0)
        with caplog.at_level("WARNING"):
            client.complete(
                system="s", user_content="u", stream=True, idle_timeout=10.0
            )
        assert any("binds first" in r.getMessage() for r in caplog.records)

    def test_zero_idle_timeout_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="stream_idle_timeout must be > 0"):
            LLMClient(
                OPENROUTER_SPEC, callsite_id="x", stream_idle_timeout=0
            )


# ── the TOTAL budget (alpha-engine-config-I8348) ──────────────────────────


class TestTotalDurationBoundFires:
    """The idle budget bounds SILENCE. A route that keeps emitting chunks
    resets the idle clock forever, so only a TOTAL bound can stop it."""

    def test_a_relentless_stream_aborts_on_the_total_budget(self):
        stream = _RelentlessStream(0.02)
        fake = FakeOpenAI([stream])
        client = _client(OPENROUTER_SPEC, fake)

        with pytest.raises(StreamTotalTimeoutError) as excinfo:
            client.complete(
                system="s", user_content="u", stream=True,
                idle_timeout=0.25, total_timeout=0.5,
            )

        err = excinfo.value
        assert err.total_timeout == 0.5
        assert err.elapsed >= 0.5, (
            "elapsed must be at or past the budget by construction — an "
            "exception claiming it fired early would misreport the bound"
        )
        assert err.partial_text, (
            "the partial generation must ride the exception, same contract "
            "as the idle path — discarding it is what I8164 removed"
        )
        assert err.chunks > 0
        assert stream.closed, "the abandoned stream must be closed, not leaked"

    def test_the_idle_bound_still_fires_when_a_generous_total_is_set(self):
        """Setting a total budget must not disarm the idle budget — they are
        two independent bounds with different fixes."""
        stream = _StallingStream(["par", "tial "])
        fake = FakeOpenAI([stream])
        client = _client(OPENROUTER_SPEC, fake)

        with pytest.raises(StreamIdleTimeoutError):
            client.complete(
                system="s", user_content="u", stream=True,
                idle_timeout=0.25, total_timeout=30,
            )

    def test_unset_total_timeout_leaves_the_stream_unbounded(self):
        """The default must not hand an existing consumer a ceiling it never
        asked for: with no total budget, a slow-but-alive stream completes."""
        import time

        def _dribble():
            for piece in ("a", "b", "c", "d"):
                time.sleep(0.05)
                yield _chunk(content=piece)
            yield _chunk(finish_reason="stop")
            yield _chunk(usage=_usage())

        fake = FakeOpenAI([_dribble()])
        client = _client(OPENROUTER_SPEC, fake)
        result = client.complete(
            system="s", user_content="u", stream=True, idle_timeout=0.5
        )
        assert result.text == "abcd"

    def test_a_total_budget_at_or_below_the_idle_budget_is_rejected(self):
        """A total bound the idle bound always wins is a knob that does
        nothing — reject it at the call rather than let it read as armed."""
        fake = FakeOpenAI([_RelentlessStream(0.02)])
        client = _client(OPENROUTER_SPEC, fake)
        with pytest.raises(ValueError, match="total_timeout"):
            client.complete(
                system="s", user_content="u", stream=True,
                idle_timeout=5, total_timeout=5,
            )

    def test_a_non_positive_client_level_total_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="stream_total_timeout"):
            _client(OPENROUTER_SPEC, FakeOpenAI([]), stream_total_timeout=0)



# ── same shape as the non-streaming path ──────────────────────────────────


class TestStreamedResultMatchesNonStreamed:
    """A completed stream is re-assembled into the SAME object the
    non-streaming path returns — so every guard downstream of the transport
    call (empty-content diagnostics, budget exhaustion, served-model
    resolution, usage extraction) runs unchanged on both."""

    def test_complete_openai(self):
        streamed = _client(
            OPENROUTER_SPEC, FakeOpenAI([_text_stream(["Hel", "lo ", "world"])])
        ).complete(system="s", user_content="u", stream=True)

        plain_resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello world"),
                    finish_reason="stop",
                )
            ],
            usage=_usage(),
            model="moonshotai/kimi-k2.6",
        )
        plain = _client(OPENROUTER_SPEC, FakeOpenAI([plain_resp])).complete(
            system="s", user_content="u"
        )

        assert streamed.text == plain.text == "Hello world"
        assert streamed.model == plain.model
        assert streamed.provider == plain.provider
        assert streamed.finish_reason == plain.finish_reason == "stop"
        assert streamed.usage.input_tokens == plain.usage.input_tokens == 100
        assert streamed.usage.output_tokens == plain.usage.output_tokens == 50
        assert streamed.usage.usage_unknown is False
        assert type(streamed) is type(plain)

        assert streamed.streamed is True and plain.streamed is False
        assert streamed.stream_chunks == 5

    def test_structured_openai_parses_the_assembled_body_at_stream_end(self):
        """The schema constraint and the stream are not in tension: fragments
        that are not JSON at any point in flight become one object at the
        end, and are validated exactly as the non-streaming body is."""
        pieces = ['{"na', 'me": "atl', 'as", "sc', 'ore": 7}']
        fake = FakeOpenAI([_text_stream(pieces)])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True,
        )
        assert result.parsed == Probe(name="atlas", score=7)
        assert result.data == {"name": "atlas", "score": 7}
        assert result.structured_output_rung == "native"
        assert result.streamed is True
        assert result.finish_reason == "stop"

    def test_complete_anthropic(self):
        events = _anthropic_text_events(["Hel", "lo"])
        result = _client(ANTHROPIC_SPEC, FakeAnthropic([events])).complete(
            system="s", user_content="u", stream=True
        )
        assert result.text == "Hello"
        assert result.finish_reason == "end_turn"
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 42, (
            "message_delta's output count is CUMULATIVE — summing it with "
            "message_start's would double-count the generation"
        )
        assert result.usage.usage_unknown is False

    def test_structured_anthropic_assembles_the_forced_tool_call(self):
        events = _anthropic_tool_events(
            "Probe", ['{"name":', ' "atlas",', ' "score": 7}']
        )
        result = _client(ANTHROPIC_SPEC, FakeAnthropic([events])).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True,
        )
        assert result.parsed == Probe(name="atlas", score=7)
        assert result.structured_output_rung == "tool_emulation"
        assert result.streamed is True


# ── finish_reason ─────────────────────────────────────────────────────────


class TestFinishReasonIsSurfaced:
    """It separates "the model finished" from "the budget ran out" from "the
    stream died" — and was previously read only inside the failure paths, so a
    successful call recorded nothing about how it ended."""

    @pytest.mark.parametrize("reason", ["stop", "length", "tool_calls"])
    def test_openai_streamed(self, reason):
        fake = FakeOpenAI([_text_stream(["body"], finish_reason=reason)])
        result = _client(OPENROUTER_SPEC, fake).complete(
            system="s", user_content="u", stream=True
        )
        assert result.finish_reason == reason

    def test_openai_non_streamed_too(self):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="body"), finish_reason="length"
                )
            ],
            usage=_usage(),
            model="moonshotai/kimi-k2.6",
        )
        result = _client(OPENROUTER_SPEC, FakeOpenAI([resp])).complete(
            system="s", user_content="u"
        )
        assert result.finish_reason == "length"

    def test_anthropic_stop_reason(self):
        events = _anthropic_text_events(["x"], stop_reason="max_tokens")
        result = _client(ANTHROPIC_SPEC, FakeAnthropic([events])).complete(
            system="s", user_content="u", stream=True
        )
        assert result.finish_reason == "max_tokens"


# ── attribution ───────────────────────────────────────────────────────────


class TestStreamedCallsAreStillAttributed:
    def test_usage_is_requested_on_the_wire(self):
        """Not a detail: without ``stream_options.include_usage`` an
        OpenAI-compatible stream carries NO usage at all, and every streamed
        call would be unattributable by construction."""
        fake = FakeOpenAI([_text_stream(["hi"])])
        _client(OPENROUTER_SPEC, fake).complete(
            system="s", user_content="u", stream=True
        )
        sent = fake.kwargs[0]
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}

    def test_cost_record_is_emitted_for_a_streamed_call(self):
        records = []
        fake = FakeOpenAI([_text_stream(["hi"])])
        client = LLMClient(
            OPENROUTER_SPEC,
            callsite_id="director-plan",
            client_factory=lambda _spec, _key: fake,
            cost_sink=records.append,
        )
        result = client.complete(system="s", user_content="u", stream=True)

        assert len(records) == 1
        rec = records[0]
        assert rec["callsite_id"] == "director-plan"
        assert rec["input_tokens"] == 100 and rec["output_tokens"] == 50
        assert rec["cost_source"] == "price_card"
        assert rec["cost_usd"] is not None and rec["cost_usd"] > 0
        assert "usage_unknown" not in rec
        assert result.cost_emission_error is None

    def test_structured_streamed_emits_too(self):
        records = []
        fake = FakeOpenAI([_text_stream(['{"name": "a", "score": 1}'])])
        LLMClient(
            OPENROUTER_SPEC,
            callsite_id="director-plan",
            client_factory=lambda _spec, _key: fake,
            cost_sink=records.append,
        ).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True,
        )
        assert len(records) == 1
        assert records[0]["callsite_id"] == "director-plan"

    def test_absent_usage_is_unknown_never_zero(self, caplog):
        """A route that drops the usage chunk must not have its call priced at
        $0. The generation is still delivered — the LEDGER carries the gap."""
        records = []
        fake = FakeOpenAI([_text_stream(["hi"], usage=None)])
        client = LLMClient(
            OPENROUTER_SPEC,
            callsite_id="director-plan",
            client_factory=lambda _spec, _key: fake,
            cost_sink=records.append,
        )
        with caplog.at_level("ERROR"):
            result = client.complete(
                system="s", user_content="u", stream=True
            )

        assert result.text == "hi", "the generation is still returned"
        assert result.usage.usage_unknown is True
        rec = records[0]
        assert rec["cost_usd"] is None, "UNKNOWN, never 0"
        assert rec["cost_source"] == "usage_unreported"
        assert rec["usage_unknown"] is True

    def test_absent_usage_survives_a_later_reporting_attempt(self):
        """``raw_response`` is only the LAST attempt's. An attempt whose usage
        never arrived must not be forgotten because a later one reported."""
        fake = FakeOpenAI([
            _text_stream(["not json"], usage=None),
            _text_stream(['{"name": "a", "score": 1}']),
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True, attempts=2,
        )
        assert result.parsed == Probe(name="a", score=1)
        assert result.usage.usage_unknown is True


# ── honest degradation ────────────────────────────────────────────────────


class TestStreamingIsNeverSilentlyDropped:
    """An undeclared capability is not a capability. Falling back to a
    non-streaming request would return a valid completion carrying the exact
    request-deadline failure envelope streaming was asked for to remove — the
    config knob that quietly does nothing."""

    def test_a_route_that_does_not_declare_streaming_raises(self):
        spec = ModelSpec(
            "openrouter", "moonshotai/kimi-k2.6", supports_streaming=False
        )
        fake = FakeOpenAI([_text_stream(["hi"])])
        with pytest.raises(StreamingUnsupportedError, match="capabilities.streaming"):
            _client(spec, fake).complete(
                system="s", user_content="u", stream=True
            )
        assert fake.kwargs == [], "no request may reach the wire"

    def test_structured_refuses_the_same_way(self):
        spec = ModelSpec(
            "openrouter", "moonshotai/kimi-k2.6", supports_streaming=False
        )
        fake = FakeOpenAI([_text_stream(['{"name": "a", "score": 1}'])])
        with pytest.raises(StreamingUnsupportedError):
            _client(spec, fake).structured(
                system="s", user_content="u",
                schema=Probe, schema_name="Probe", stream=True,
            )
        assert fake.kwargs == []

    def test_a_hand_built_spec_may_assert_it(self):
        """Same split ``structured_outputs`` already has: the registry decides
        for a resolved route, the caller decides for an endpoint they chose."""
        assert ModelSpec("openrouter", "m").supports_streaming is True

    def test_non_streamed_calls_are_untouched(self):
        """The default path must not acquire streaming flags."""
        fake = FakeOpenAI([
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="hi"), finish_reason="stop"
                    )
                ],
                usage=_usage(),
                model="moonshotai/kimi-k2.6",
            )
        ])
        result = _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert "stream" not in fake.kwargs[0]
        assert "stream_options" not in fake.kwargs[0]
        assert result.streamed is False and result.stream_chunks == 0


# ── anthropic event doubles ───────────────────────────────────────────────


def _anthropic_start(model="claude-haiku-4-5", input_tokens=100):
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
            model=model,
            usage=SimpleNamespace(
                input_tokens=input_tokens,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cache_creation=None,
                server_tool_use=None,
            ),
        ),
    )


def _anthropic_text_events(pieces, *, stop_reason="end_turn", output_tokens=42):
    events = [
        _anthropic_start(),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text"),
        ),
    ]
    events += [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text=p),
        )
        for p in pieces
    ]
    events.append(
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason=stop_reason),
            usage=SimpleNamespace(output_tokens=output_tokens),
        )
    )
    events.append(SimpleNamespace(type="message_stop"))
    return events


def _anthropic_tool_events(tool_name, fragments, *, output_tokens=42):
    events = [
        _anthropic_start(),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(
                type="tool_use", name=tool_name, id="tu_1"
            ),
        ),
    ]
    events += [
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json=f),
        )
        for f in fragments
    ]
    events.append(
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=output_tokens),
        )
    )
    return events


class TestAnthropicStreamAccumulation:
    def test_an_unassemblable_tool_payload_does_not_read_as_an_empty_object(self):
        """``{}`` would read as "the model returned nothing"; ``None`` routes
        into the existing missing-tool-call classification."""
        from krepis.llm import _accumulate_anthropic_stream

        events = _anthropic_tool_events("Probe", ['{"name": "at'])
        msg = _accumulate_anthropic_stream(
            iter(events), idle_timeout=5, spec=ANTHROPIC_SPEC
        )
        assert msg.content[0].input is None

    def test_idle_timeout_carries_the_partial_anthropic_text(self):
        class _Stalling:
            def __init__(self):
                self.released = threading.Event()
                self.closed = False

            def __iter__(self):
                for event in _anthropic_text_events(["par", "tial"])[:4]:
                    yield event
                self.released.wait(30)

            def close(self):
                self.closed = True
                self.released.set()

        stream = _Stalling()
        with pytest.raises(StreamIdleTimeoutError) as excinfo:
            _client(ANTHROPIC_SPEC, FakeAnthropic([stream])).complete(
                system="s", user_content="u", stream=True, idle_timeout=0.25
            )
        assert excinfo.value.partial_text == "partial"
        assert stream.closed


# ── the escalation composes with streaming ────────────────────────────────


class TestAnExhaustedBudgetEscalatesOnTheStreamedPathToo:
    """A budget exhausted MID-STREAM escalates exactly as it does on the
    non-streamed path (``alpha-engine-config-I6917`` + I8164).

    The two changes met in ``structured()``, and the failure mode of getting
    the meeting wrong is quiet in both directions: a streamed exhaustion that
    escalated but should not have doubles the cost of a certain failure, and
    one that did not escalate when it should have hands back a truncated body
    dressed as a complete answer. What makes it compose rather than need its
    own copy is that a streamed response is re-assembled into the same
    ``ChatCompletion`` shape — so ``_reject_budget_exhausted`` reads the same
    ``finish_reason='length'`` on an empty body it always did, raises the same
    ``BudgetExhaustedError``, and the escalation wrapper above it never learns
    that the transport streamed. These tests pin that, so a future change to
    either half cannot quietly decouple them.
    """

    def _exhausted_stream(self):
        """An empty generation that stopped because it hit the ceiling.

        This is what an exhausted reasoning budget looks like ON THE WIRE: the
        stream is well-formed, chunks arrive, ``finish_reason`` is ``length``
        and not one content delta was ever produced.
        """
        usage = _usage(completion=50)
        usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=1024)
        return [
            _chunk(finish_reason="length"),
            _chunk(usage=usage),
        ]

    def test_it_escalates_once_and_the_re_issue_is_also_streamed(self):
        fake = FakeOpenAI([
            self._exhausted_stream(),
            _text_stream(['{"name": "a", "score": 1}']),
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True,
        )
        assert result.parsed == Probe(name="a", score=1)
        assert [k["max_tokens"] for k in fake.kwargs] == [1024, 2048], (
            "exactly one re-issue, at a RAISED ceiling"
        )
        assert all(k["stream"] is True for k in fake.kwargs), (
            "the escalated re-issue must not silently drop to a "
            "non-streaming request — that would hand back the request-"
            "deadline failure envelope on the retry that matters most"
        )
        assert result.streamed is True
        assert result.usage.budget_escalations == 1

    def test_a_truncated_body_is_never_returned_as_if_complete(self):
        """The whole point of the guard. An empty, length-capped stream is a
        budget fault, not a model that answered with nothing."""
        fake = FakeOpenAI([self._exhausted_stream()] * 6)
        with pytest.raises(BudgetExhaustedError) as excinfo:
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u",
                schema=Probe, schema_name="Probe", stream=True,
            )
        assert "max_tokens=2048" in str(excinfo.value), (
            "the error names the budget that was ACTUALLY too small"
        )
        assert "reasoning_tokens=1024" in str(excinfo.value)
        assert [k["max_tokens"] for k in fake.kwargs] == [1024, 2048], (
            "a second exhaustion at double the budget is a pathological ask; "
            "escalating forever is not the fix"
        )

    def test_the_exhausted_streams_spend_is_absorbed(self):
        fake = FakeOpenAI([
            self._exhausted_stream(),
            _text_stream(['{"name": "a", "score": 1}']),
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True,
        )
        assert result.usage.output_tokens == 100, "two billed streams"
        assert result.usage.reasoning_tokens == 1024, (
            "the exhausted attempt's reasoning draw is the whole diagnosis"
        )

    def test_an_unattributable_exhausted_stream_keeps_the_merged_call_unpriced(self):
        """``usage_unknown`` is STICKY across the escalation.

        The exhausted attempt's counters are absorbed into the escalated
        one's. If the flag did not travel with them the merged total would be
        priced as though every token in it had been reported — an
        understatement arriving in a complete-looking cost row.
        """
        records = []
        exhausted = [_chunk(finish_reason="length")]  # no usage chunk at all
        fake = FakeOpenAI([
            exhausted,
            _text_stream(['{"name": "a", "score": 1}']),
        ])
        result = LLMClient(
            OPENROUTER_SPEC,
            callsite_id="director-plan",
            client_factory=lambda _spec, _key: fake,
            cost_sink=records.append,
        ).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True,
        )
        assert result.parsed == Probe(name="a", score=1)
        assert result.usage.usage_unknown is True
        assert records[0]["cost_usd"] is None
        assert records[0]["cost_source"] == "usage_unreported"
        assert records[0]["budget_escalations"] == 1, (
            "an escalation nobody can count is indistinguishable from health"
        )

    def test_the_anthropic_stream_escalates_the_same_way(self):
        """Transport-independent: the escalation lives in ``structured()``,
        above the branch that chooses a wire."""
        exhausted = _anthropic_text_events(
            ["thinking..."], stop_reason="max_tokens"
        )
        fake = FakeAnthropic([
            exhausted,
            _anthropic_tool_events("Probe", ['{"name": "a", "score": 1}']),
        ])
        result = _client(ANTHROPIC_SPEC, fake).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True, attempts=1,
        )
        assert result.parsed == Probe(name="a", score=1)
        assert [p["max_tokens"] for p in fake.payloads] == [1024, 2048]
        assert all(p["stream"] is True for p in fake.payloads)
        assert result.usage.budget_escalations == 1

    def test_a_healthy_streamed_call_counts_zero_escalations(self):
        fake = FakeOpenAI([_text_stream(['{"name": "a", "score": 1}'])])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema=Probe, schema_name="Probe", stream=True,
        )
        assert result.usage.budget_escalations == 0
        assert len(fake.kwargs) == 1
