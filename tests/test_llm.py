"""Tests for ``krepis.llm.LLMClient`` — both transports via fake clients."""

import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from krepis.llm import (
    BudgetExhaustedError,
    LLMClient,
    LLMError,
    NullChoicesError,
    SearchOptions,
    _extract_json,
    _first_choice,
)
from krepis.llm_config import ROUTER_EDGE_PROVIDER, LLMConfigError, ModelSpec


# ── fixtures / fakes ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name, tool_input, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _search_use_block(query, block_id):
    return SimpleNamespace(
        type="server_tool_use", name="web_search", id=block_id,
        input={"query": query},
    )


def _search_result_block(tool_use_id, urls):
    return SimpleNamespace(
        type="web_search_tool_result",
        tool_use_id=tool_use_id,
        content=[SimpleNamespace(url=u, title=f"title:{u}") for u in urls],
    )


def _anthropic_usage(**kw):
    defaults = dict(
        input_tokens=100, output_tokens=50,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
        cache_creation=None, server_tool_use=None,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _anthropic_msg(content, usage=None, model="claude-haiku-4-5"):
    return SimpleNamespace(
        content=content, usage=usage or _anthropic_usage(), model=model
    )


class FakeAnthropic:
    """messages.create fake: pops queued responses, records payloads."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **payload):
        self.payloads.append(payload)
        return self._responses.pop(0)


def _openai_usage(
    prompt=100, completion=50, cached=0, cost=None, searches=None,
    nested_searches=None, nested_searches_obj=None, reasoning=None,
):
    u = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    if reasoning is not None:
        # `reasoning` may be an int (typed SDK shape) or a dict, because a
        # proxied provider delivers this as raw decoded JSON — both must read.
        u.completion_tokens_details = (
            reasoning if isinstance(reasoning, dict)
            else SimpleNamespace(reasoning_tokens=reasoning)
        )
    if cost is not None:
        u.cost = cost
    if searches is not None:
        u.web_search_requests = searches
    if nested_searches is not None:
        # The REAL OpenRouter shape (confirmed live 2026-07-06, corrected
        # after an initial getattr-based fix silently kept reading 0):
        # server_tool_use_details is an unmodeled Pydantic "extra" field,
        # so the SDK stores it as a plain dict — NOT an attribute-bearing
        # object — even though the equivalent Anthropic field IS a proper
        # nested object. Use a dict here to match reality.
        u.server_tool_use_details = {"web_search_requests": nested_searches}
    if nested_searches_obj is not None:
        # Belt-and-suspenders: some other OpenAI-compatible provider might
        # report this as a proper attribute-bearing object instead — the
        # extraction code supports both shapes.
        u.server_tool_use_details = SimpleNamespace(
            web_search_requests=nested_searches_obj
        )
    return u


def _openai_resp(
    content, usage=None, model="moonshotai/kimi-k2.6", annotations=None,
    finish_reason=None, tool_calls=None, served_provider=None,
):
    message = SimpleNamespace(content=content)
    if annotations is not None:
        message.annotations = annotations
    if tool_calls is not None:
        message.tool_calls = tool_calls
    choice = SimpleNamespace(message=message)
    if finish_reason is not None:
        choice.finish_reason = finish_reason
    resp = SimpleNamespace(
        choices=[choice],
        usage=usage or _openai_usage(),
        model=model,
    )
    if served_provider is not None:
        # Mirrors OpenRouter's non-standard top-level `provider` field
        # (confirmed live 2026-07-22) naming the routed upstream backend
        # (e.g. "DeepInfra") — distinct from the static transport name.
        resp.provider = served_provider
    return resp


class FakeOpenAI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        return self._responses.pop(0)


def _client(spec, fake):
    return LLMClient(
        spec,
        callsite_id="krepis-test",
        client_factory=lambda _spec, _key: fake,
    )


ANTHROPIC_SPEC = ModelSpec("anthropic", "claude-haiku-4-5", max_tokens=1024)
OPENROUTER_SPEC = ModelSpec("openrouter", "moonshotai/kimi-k2.6", max_tokens=1024)
OPENROUTER_LOOSE_SPEC = ModelSpec(
    "openrouter", "qwen/qwen3.7-plus:floor", max_tokens=1024,
    structured_outputs=False,
)


class Spec(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)


# ── complete ──────────────────────────────────────────────────────────────


class TestComplete:
    def test_anthropic(self):
        fake = FakeAnthropic([
            _anthropic_msg(
                [_text_block("hello"), _text_block("world")],
                usage=_anthropic_usage(
                    cache_read_input_tokens=40,
                    cache_creation_input_tokens=10,
                ),
            )
        ])
        result = _client(ANTHROPIC_SPEC, fake).complete(
            system="sys", user_content="hi"
        )
        assert result.text == "hello\n\nworld"
        assert result.provider == "anthropic"
        assert result.usage.input_tokens == 100
        assert result.usage.cache_read_tokens == 40
        assert result.usage.cache_create_tokens == 10
        payload = fake.payloads[0]
        assert payload["model"] == "claude-haiku-4-5"
        assert payload["max_tokens"] == 1024
        assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_anthropic_cache_system_off(self):
        fake = FakeAnthropic([_anthropic_msg([_text_block("x")])])
        _client(ANTHROPIC_SPEC, fake).complete(
            system="sys", user_content="hi", cache_system=False
        )
        assert "cache_control" not in fake.payloads[0]["system"][0]

    def test_anthropic_served_provider_is_none(self):
        # Single-backend transport — no routing ambiguity, no field to read.
        fake = FakeAnthropic([_anthropic_msg([_text_block("x")])])
        result = _client(ANTHROPIC_SPEC, fake).complete(
            system="sys", user_content="hi"
        )
        assert result.served_provider is None

    def test_openrouter_served_provider_captured(self):
        # config#3006 — jurisdiction/compliance checks read this field
        # instead of parsing raw_response themselves.
        fake = FakeOpenAI([_openai_resp("hey", served_provider="DeepInfra")])
        result = _client(OPENROUTER_SPEC, fake).complete(
            system="sys", user_content="hi"
        )
        assert result.served_provider == "DeepInfra"

    def test_served_provider_absent_when_not_reported(self):
        fake = FakeOpenAI([_openai_resp("hey")])
        result = _client(OPENROUTER_SPEC, fake).complete(
            system="sys", user_content="hi"
        )
        assert result.served_provider is None

    def test_openrouter_includes_usage_accounting(self):
        fake = FakeOpenAI([
            _openai_resp("hey", usage=_openai_usage(cached=25, cost=0.00123))
        ])
        result = _client(OPENROUTER_SPEC, fake).complete(
            system="sys", user_content="hi"
        )
        assert result.text == "hey"
        assert result.usage.cache_read_tokens == 25
        assert result.usage.provider_cost_usd == pytest.approx(0.00123)
        kwargs = fake.kwargs[0]
        assert kwargs["extra_body"] == {"usage": {"include": True}}
        assert kwargs["messages"][0] == {"role": "system", "content": "sys"}

    def test_plain_openai_no_extra_body(self):
        fake = FakeOpenAI([_openai_resp("hey")])
        _client(ModelSpec("openai", "gpt-x"), fake).complete(
            system="s", user_content="u"
        )
        assert "extra_body" not in fake.kwargs[0]

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY")
        client = LLMClient(OPENROUTER_SPEC, callsite_id="krepis-test")
        with pytest.raises(LLMConfigError, match="OPENROUTER_API_KEY"):
            client.complete(system="s", user_content="u")

    def test_reasoning_forwarded_on_openrouter(self):
        fake = FakeOpenAI([_openai_resp("hey")])
        spec = ModelSpec(
            "openrouter", "moonshotai/kimi-k2.6", max_tokens=1024,
            reasoning={"exclude": True},
        )
        _client(spec, fake).complete(system="s", user_content="u")
        assert fake.kwargs[0]["extra_body"] == {
            "usage": {"include": True}, "reasoning": {"exclude": True},
        }

    def test_reasoning_on_anthropic_raises(self):
        spec = ModelSpec(
            "anthropic", "claude-haiku-4-5", max_tokens=1024,
            reasoning={"effort": "low"},
        )
        client = _client(spec, FakeAnthropic([]))
        with pytest.raises(LLMConfigError, match="reasoning"):
            client.complete(system="s", user_content="u")


# ── structured ────────────────────────────────────────────────────────────


class TestStructuredAnthropic:
    def test_forced_tool_success(self):
        fake = FakeAnthropic([
            _anthropic_msg(
                [_tool_use_block("emit_spec", {"name": "a", "score": 90})]
            )
        ])
        result = _client(ANTHROPIC_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="emit_spec"
        )
        assert result.parsed == Spec(name="a", score=90)
        assert result.data == {"name": "a", "score": 90}
        payload = fake.payloads[0]
        assert payload["tool_choice"] == {"type": "tool", "name": "emit_spec"}
        assert payload["tools"][0]["input_schema"] == Spec.model_json_schema()

    def test_correction_retry_recovers(self):
        bad = _anthropic_msg(
            [_tool_use_block("emit_spec", {"name": "a", "score": 999})]
        )
        good = _anthropic_msg(
            [_tool_use_block("emit_spec", {"name": "a", "score": 50})]
        )
        fake = FakeAnthropic([bad, good])
        result = _client(ANTHROPIC_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="emit_spec"
        )
        assert result.parsed.score == 50
        # usage accumulated across BOTH attempts
        assert result.usage.input_tokens == 200
        # retry conversation carried the assistant turn + correction
        retry_messages = fake.payloads[1]["messages"]
        assert len(retry_messages) == 3
        assert retry_messages[1]["role"] == "assistant"
        assert "failed validation" in retry_messages[2]["content"]

    def test_exhaustion_raises_with_usage(self):
        bad = _anthropic_msg(
            [_tool_use_block("emit_spec", {"name": "a", "score": 999})]
        )
        fake = FakeAnthropic([bad, bad])
        with pytest.raises(LLMError) as exc_info:
            _client(ANTHROPIC_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec,
                schema_name="emit_spec",
            )
        assert exc_info.value.usage.input_tokens == 200

    def test_domain_validate_hook_feeds_retry(self):
        first = _anthropic_msg(
            [_tool_use_block("emit_spec", {"name": "ungrounded", "score": 10})]
        )
        second = _anthropic_msg(
            [_tool_use_block("emit_spec", {"name": "grounded", "score": 10})]
        )
        fake = FakeAnthropic([first, second])

        def check(spec):
            if spec.name != "grounded":
                raise ValueError("name must be grounded in the input")

        result = _client(ANTHROPIC_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec,
            schema_name="emit_spec", validate=check,
        )
        assert result.parsed.name == "grounded"
        assert "grounded in the input" in fake.payloads[1]["messages"][2]["content"]

    def test_missing_tool_block_retries_then_raises(self):
        no_tool = _anthropic_msg([_text_block("I refuse to use tools")])
        fake = FakeAnthropic([no_tool, no_tool])
        with pytest.raises(LLMError, match="no 'emit_spec' tool_use block"):
            _client(ANTHROPIC_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec,
                schema_name="emit_spec",
            )


class TestStructuredOpenAI:
    def test_strict_json_schema(self):
        fake = FakeOpenAI([_openai_resp('{"name": "a", "score": 5}')])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="emit_spec"
        )
        assert result.parsed == Spec(name="a", score=5)
        rf = fake.kwargs[0]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == Spec.model_json_schema()

    def test_no_strict_support_uses_json_instruction_and_fences(self):
        fake = FakeOpenAI([
            _openai_resp('```json\n{"name": "a", "score": 5}\n```')
        ])
        result = _client(OPENROUTER_LOOSE_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="emit_spec"
        )
        assert result.parsed.score == 5
        kwargs = fake.kwargs[0]
        assert "response_format" not in kwargs
        assert "JSON Schema" in kwargs["messages"][1]["content"]

    def test_raw_dict_schema(self):
        fake = FakeOpenAI([_openai_resp('{"anything": 1}')])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema={"type": "object"}, schema_name="blob",
        )
        assert result.data == {"anything": 1}
        assert result.parsed is None

    def test_served_provider_captured(self):
        fake = FakeOpenAI([
            _openai_resp('{"anything": 1}', served_provider="AtlasCloud")
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema={"type": "object"}, schema_name="blob",
        )
        assert result.served_provider == "AtlasCloud"

    def test_reasoning_forwarded(self):
        fake = FakeOpenAI([_openai_resp('{"name": "a", "score": 5}')])
        spec = ModelSpec(
            "openrouter", "moonshotai/kimi-k2.6", max_tokens=1024,
            reasoning={"max_tokens": 500},
        )
        _client(spec, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="emit_spec"
        )
        assert fake.kwargs[0]["extra_body"]["reasoning"] == {"max_tokens": 500}

    def test_reasoning_on_anthropic_raises(self):
        spec = ModelSpec(
            "anthropic", "claude-haiku-4-5", max_tokens=1024,
            reasoning={"effort": "low"},
        )
        client = _client(spec, FakeAnthropic([]))
        with pytest.raises(LLMConfigError, match="reasoning"):
            client.structured(
                system="s", user_content="u", schema=Spec, schema_name="emit_spec"
            )

    def test_exhaustion_raises(self):
        fake = FakeOpenAI([
            _openai_resp("not json at all"),
            _openai_resp("still not json"),
        ])
        with pytest.raises(LLMError):
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec,
                schema_name="emit_spec",
            )

    def test_non_json_transport_response_retries_then_succeeds(self):
        # Live incident 2026-07-20 (krepis#38): same transport-level
        # non-JSON-body failure as complete_grounded's guard — this call
        # site shares the identical unguarded ``.create()`` pattern.
        import json as _json

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _json.JSONDecodeError("Expecting value", "not json", 0)
            return _openai_resp('{"name": "a", "score": 5}')

        fake = FakeOpenAI([])
        fake.chat.completions.create = _create
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="emit_spec"
        )
        assert result.parsed == Spec(name="a", score=5)
        assert calls["n"] == 2

    def test_non_json_transport_response_raises_llmerror_after_exhaustion(self):
        import json as _json

        def _create(**kwargs):
            raise _json.JSONDecodeError("Expecting value", "not json", 0)

        fake = FakeOpenAI([])
        fake.chat.completions.create = _create
        with pytest.raises(LLMError, match="non-JSON response body"):
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec,
                schema_name="emit_spec",
            )


# ── complete_grounded ─────────────────────────────────────────────────────


class TestGrounded:
    def test_anthropic_search_events_and_text(self):
        msg = _anthropic_msg(
            [
                _text_block("Let me search."),
                _search_use_block("fed rates", "s1"),
                _search_result_block("s1", ["https://a.example", "https://b.example"]),
                _text_block("The final answer."),
            ],
            usage=_anthropic_usage(
                server_tool_use=SimpleNamespace(
                    web_search_requests=1, web_fetch_requests=0
                )
            ),
        )
        fake = FakeAnthropic([msg])
        result = _client(ANTHROPIC_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions(max_uses=7)
        )
        assert result.text == "The final answer."
        assert result.searches == [
            {
                "query": "fed rates",
                "urls": ["https://a.example", "https://b.example"],
                "result_count": 2,
                "error": None,
            }
        ]
        assert [c["url"] for c in result.citations] == [
            "https://a.example", "https://b.example",
        ]
        assert result.usage.web_search_requests == 1
        tools = fake.payloads[0]["tools"]
        assert tools[0]["type"].startswith("web_search_")
        assert tools[0]["max_uses"] == 7
        assert "tool_choice" not in fake.payloads[0]

    def test_anthropic_force_first_sets_tool_choice(self):
        fake = FakeAnthropic([_anthropic_msg([_text_block("t")])])
        _client(ANTHROPIC_SPEC, fake).complete_grounded(
            system="s", user_content="u",
            search=SearchOptions(force_first=True),
        )
        assert fake.payloads[0]["tool_choice"] == {
            "type": "tool", "name": "web_search",
        }

    def test_openrouter_web_tool_and_citations(self):
        annotations = [
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://news.example/x",
                    "title": "X happened",
                    "content": "excerpt",
                },
            }
        ]
        fake = FakeOpenAI([
            _openai_resp(
                "grounded answer",
                usage=_openai_usage(cost=0.002, searches=3),
                annotations=annotations,
            )
        ])
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u",
            search=SearchOptions(engine="exa", max_results=5),
        )
        assert result.text == "grounded answer"
        assert result.searches == []  # queries not exposed on this transport
        assert result.citations == [
            {"url": "https://news.example/x", "title": "X happened",
             "snippet": "excerpt"}
        ]
        assert result.usage.web_search_requests == 3
        extra_body = fake.kwargs[0]["extra_body"]
        assert extra_body["tools"] == [
            {
                "type": "openrouter:web_search",
                "parameters": {"engine": "exa", "max_results": 5},
            }
        ]

    def test_openrouter_reads_nested_server_tool_use_details(self):
        # The REAL response shape (confirmed live 2026-07-06, config#1659):
        # the search count lives under
        # usage.server_tool_use_details.web_search_requests, not a flat
        # usage.web_search_requests field. Before this fix, real grounded
        # OpenRouter calls always read web_search_requests as 0 regardless
        # of how much searching actually happened — silently breaking any
        # consumer's min-searches floor on this transport.
        fake = FakeOpenAI([
            _openai_resp(
                "grounded answer",
                usage=_openai_usage(nested_searches=5),
            )
        ])
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions()
        )
        assert result.usage.web_search_requests == 5

    def test_openrouter_nested_shape_takes_priority_over_flat(self):
        # If a provider somehow reports both, the real (nested) shape wins.
        fake = FakeOpenAI([
            _openai_resp(
                "grounded answer",
                usage=_openai_usage(searches=1, nested_searches=9),
            )
        ])
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions()
        )
        assert result.usage.web_search_requests == 9

    def test_openrouter_nested_shape_as_attribute_object_also_works(self):
        # Belt-and-suspenders: if some other OpenAI-compatible provider
        # reports server_tool_use_details as a proper attribute-bearing
        # object rather than a raw dict, that shape is read too.
        fake = FakeOpenAI([
            _openai_resp(
                "grounded answer",
                usage=_openai_usage(nested_searches_obj=7),
            )
        ])
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions()
        )
        assert result.usage.web_search_requests == 7

    def test_force_first_on_openrouter_raises(self):
        client = _client(OPENROUTER_SPEC, FakeOpenAI([]))
        with pytest.raises(LLMConfigError, match="force_first"):
            client.complete_grounded(
                system="s", user_content="u",
                search=SearchOptions(force_first=True),
            )

    def test_plain_openai_provider_raises(self):
        client = _client(ModelSpec("openai", "gpt-x"), FakeOpenAI([]))
        with pytest.raises(LLMConfigError, match="complete_grounded"):
            client.complete_grounded(
                system="s", user_content="u", search=SearchOptions()
            )

    def test_reasoning_forwarded_on_openrouter(self):
        # config#1659, 2026-07-06: without this, a reasoning-capable model
        # can spend its whole budget on chain-of-thought and return an
        # empty ``text`` even at a generous max_tokens (reproduced live
        # with Kimi K2.6). Verifies the override actually reaches the
        # wire.
        fake = FakeOpenAI([_openai_resp("grounded answer")])
        spec = ModelSpec(
            "openrouter", "moonshotai/kimi-k2.6", max_tokens=1024,
            reasoning={"exclude": True},
        )
        _client(spec, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions()
        )
        assert fake.kwargs[0]["extra_body"]["reasoning"] == {"exclude": True}

    _LEAKED_RESP = _openai_resp(
        "Welcome to Morning Signal. <|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.openrouter_web_search:4"
        "<|tool_call_argument_begin|>{\"query\": \"x\"}"
        "<|tool_call_end|><|tool_calls_section_end|>"
    )

    def test_openrouter_leaked_control_tokens_raises_after_exhausting_retries(self):
        # Live incident 2026-07-14: moonshotai/kimi-k2.6 via OpenRouter
        # emitted its own native tool-call token dialect straight into
        # ``message.content`` instead of the declared server-side
        # ``openrouter:web_search`` tool being resolved before the response
        # reached us. The 283-char result shipped as a live podcast episode
        # before this guard existed. Queues the leaked response for BOTH
        # attempts of the default ``attempts=2`` retry budget (see
        # test_openrouter_leak_recovers_on_retry for the same-provider
        # retry succeeding — the empirically-confirmed common case).
        fake = FakeOpenAI([self._LEAKED_RESP, self._LEAKED_RESP])
        with pytest.raises(LLMError, match="control-token"):
            _client(OPENROUTER_SPEC, fake).complete_grounded(
                system="s", user_content="u", search=SearchOptions()
            )
        assert len(fake.kwargs) == 2

    def test_openrouter_unresolved_tool_calls_field_raises_after_exhausting_retries(self):
        bad = _openai_resp(
            None,
            tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
                name="openrouter_web_search", arguments="{}"
            ))],
        )
        fake = FakeOpenAI([bad, bad])
        with pytest.raises(LLMError, match="unresolved tool call"):
            _client(OPENROUTER_SPEC, fake).complete_grounded(
                system="s", user_content="u", search=SearchOptions()
            )
        assert len(fake.kwargs) == 2

    def test_openrouter_finish_reason_tool_calls_raises_after_exhausting_retries(self):
        bad = _openai_resp("", finish_reason="tool_calls")
        fake = FakeOpenAI([bad, bad])
        with pytest.raises(LLMError, match="unresolved tool call"):
            _client(OPENROUTER_SPEC, fake).complete_grounded(
                system="s", user_content="u", search=SearchOptions()
            )
        assert len(fake.kwargs) == 2

    def test_openrouter_leak_recovers_on_retry(self):
        # Live incidents 2026-07-14/-16/-20 each confirmed a bare retry of
        # the SAME call (same provider/model) resolves the leak — this is
        # the empirically-common case the bounded retry exists for, so a
        # single transient leak must NOT escalate to the caller's
        # cross-provider fallback.
        fake = FakeOpenAI([self._LEAKED_RESP, _openai_resp("grounded answer")])
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions()
        )
        assert result.text == "grounded answer"
        assert len(fake.kwargs) == 2

    def test_attempts_below_one_raises(self):
        client = _client(OPENROUTER_SPEC, FakeOpenAI([]))
        with pytest.raises(ValueError, match="attempts must be >= 1"):
            client.complete_grounded(
                system="s", user_content="u", search=SearchOptions(),
                attempts=0,
            )

    def test_openrouter_leak_retry_budget_configurable(self):
        # attempts is a caller-tunable knob, not hardcoded.
        fake = FakeOpenAI([self._LEAKED_RESP, self._LEAKED_RESP, self._LEAKED_RESP])
        with pytest.raises(LLMError, match="control-token"):
            _client(OPENROUTER_SPEC, fake).complete_grounded(
                system="s", user_content="u", search=SearchOptions(),
                attempts=3,
            )
        assert len(fake.kwargs) == 3

    def test_openrouter_non_json_transport_response_retries_then_succeeds(self):
        # Live incident 2026-07-20 (krepis#38): OpenRouter returned a
        # malformed/non-JSON body on what the SDK treated as a successful
        # transaction. Invisible to the SDK's own max_retries (parsing
        # only fails after the response is already considered final) —
        # this must be retried as an ordinary attempt failure, not crash
        # the caller with a raw JSONDecodeError.
        import json as _json

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _json.JSONDecodeError("Expecting value", "not json", 0)
            return _openai_resp("grounded answer")

        fake = FakeOpenAI([])
        fake.chat.completions.create = _create
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions()
        )
        assert result.text == "grounded answer"
        assert calls["n"] == 2

    def test_openrouter_non_json_transport_response_raises_llmerror_after_exhaustion(self):
        import json as _json

        def _create(**kwargs):
            raise _json.JSONDecodeError("Expecting value", "not json", 0)

        fake = FakeOpenAI([])
        fake.chat.completions.create = _create
        with pytest.raises(LLMError, match="non-JSON response body"):
            _client(OPENROUTER_SPEC, fake).complete_grounded(
                system="s", user_content="u", search=SearchOptions()
            )

    def test_openrouter_clean_text_with_finish_reason_stop_still_works(self):
        # Regression guard: adding the finish_reason/tool_calls checks must
        # not break the ordinary happy path once a transport DOES populate
        # finish_reason="stop".
        fake = FakeOpenAI([
            _openai_resp("grounded answer", finish_reason="stop")
        ])
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions()
        )
        assert result.text == "grounded answer"

    def test_reasoning_on_anthropic_raises(self):
        spec = ModelSpec(
            "anthropic", "claude-haiku-4-5", max_tokens=1024,
            reasoning={"effort": "low"},
        )
        client = _client(spec, FakeAnthropic([]))
        with pytest.raises(LLMConfigError, match="reasoning"):
            client.complete_grounded(
                system="s", user_content="u", search=SearchOptions()
            )


# ── _extract_json ─────────────────────────────────────────────────────────


class TestExtractJson:
    def test_plain(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_preamble(self):
        assert _extract_json('Sure! Here you go: {"a": 1}') == {"a": 1}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            _extract_json("nothing here")


class TestCallsiteIdAndCostEmission:
    """Cost attribution at the LLMClient chokepoint (alpha-engine-config-I5206).

    The fleet ran 17 days with zero per-call cost telemetry because emission
    lived in a research-specific tracker that was retired with its graph, and
    nothing required attribution. These lock the properties that make that
    unrepeatable: attribution cannot be omitted, and emission cannot take
    down a call it observes.
    """

    def test_callsite_id_is_required(self):
        """Omitting it is a TypeError, not a default — an optional
        attribution field is one nobody fills."""
        with pytest.raises(TypeError):
            LLMClient(OPENROUTER_SPEC)

    @pytest.mark.parametrize("bad", ["", "   ", None, 123])
    def test_blank_or_non_string_callsite_id_rejected(self, bad):
        with pytest.raises(ValueError, match="non-empty callsite_id"):
            LLMClient(OPENROUTER_SPEC, callsite_id=bad)

    def test_no_sink_means_no_emission_and_no_error(self):
        """Default client emits nothing — public consumers pay nothing."""
        fake = FakeOpenAI([_openai_resp("hi")])
        client = _client(OPENROUTER_SPEC, fake)
        result = client.complete(system="s", user_content="u")
        assert result.text == "hi"
        assert result.cost_emission_error is None

    def test_sink_receives_record_stamped_with_callsite_id(self):
        records = []
        fake = FakeOpenAI([_openai_resp("hi")])
        client = LLMClient(
            OPENROUTER_SPEC,
            callsite_id="director-plan",
            client_factory=lambda _spec, _key: fake,
            cost_sink=records.append,
        )
        result = client.complete(system="s", user_content="u")

        assert len(records) == 1, "exactly one record per completed call"
        rec = records[0]
        assert rec["callsite_id"] == "director-plan"
        # The G6 input set must be present — this is what a cache-hit-rate
        # metric is computed from downstream.
        for field_name in (
            "input_tokens", "output_tokens", "cache_read_tokens",
            "prompt_cache_miss_tokens", "reasoning_tokens", "cost_usd",
        ):
            assert field_name in rec, f"{field_name} missing from cost record"
        assert result.cost_emission_error is None

    def test_sink_failure_does_not_break_the_call(self):
        """Emission runs AFTER the call succeeded. A telemetry fault must
        not become a production outage — but it must stay visible."""
        def _exploding_sink(_record):
            raise RuntimeError("s3 unavailable")

        fake = FakeOpenAI([_openai_resp("hi")])
        client = LLMClient(
            OPENROUTER_SPEC,
            callsite_id="director-plan",
            client_factory=lambda _spec, _key: fake,
            cost_sink=_exploding_sink,
        )
        result = client.complete(system="s", user_content="u")

        assert result.text == "hi", "the caller still gets its answer"
        assert result.cost_emission_error is not None, (
            "a swallowed emission failure must surface on the artifact, "
            "not only in a log line"
        )
        assert "s3 unavailable" in result.cost_emission_error

    def test_structured_calls_emit_too(self):
        """The decorator sits on the method, not on one return path — so
        every public entry point is covered, not just complete()."""
        records = []
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        fake = FakeOpenAI([_openai_resp('{"a": "b"}')])
        client = LLMClient(
            OPENROUTER_SPEC,
            callsite_id="evaljudge-sync",
            client_factory=lambda _spec, _key: fake,
            cost_sink=records.append,
        )
        client.structured(
            system="s", user_content="u",
            schema=schema, schema_name="Probe",
        )
        assert len(records) == 1
        assert records[0]["callsite_id"] == "evaljudge-sync"


# ── null-choices bodies (the 200 that carries the provider's error) ────────


def _null_choices_resp(choices=None, model="moonshotai/kimi-k2.6"):
    """OpenRouter's shape for an upstream provider failure: HTTP 200, no
    ``choices``, an ``error`` object in the body. The SDK builds this without
    complaint, so nothing is raised until something reads ``choices[0]``."""
    return SimpleNamespace(
        choices=choices,
        error={"message": "upstream provider error", "code": 502},
        id="gen-null-choices",
        model=model,
        usage=None,
    )


class TestNullChoices:
    """A null/empty ``choices`` body is a retryable transport failure.

    Regression guard: this was fixed locally in
    ``crucible-research/thinktank/client.py::_NullChoicesError`` because this
    chokepoint lacked it. Measured 2026-07-30 on
    crucible-research#530 (the migration onto this library): the migrated
    client raised ``TypeError: 'NoneType' object is not subscriptable`` —
    the fork's protection was being dropped by adopting the shared code.
    """

    @pytest.mark.parametrize("choices", [None, []])
    def test_first_choice_raises_typed_error_naming_the_provider_error(self, choices):
        with pytest.raises(NullChoicesError) as exc_info:
            _first_choice(_null_choices_resp(choices))
        message = str(exc_info.value)
        assert "upstream provider error" in message, (
            "the provider's own error message must reach the log — the "
            "original TypeError discarded it unread"
        )
        assert "gen-null-choices" in message

    @pytest.mark.parametrize("choices", [None, []])
    def test_structured_retries_a_null_choices_body(self, choices, monkeypatch):
        monkeypatch.setattr("krepis.llm._retry_backoff_sleep", lambda _a: None)
        fake = FakeOpenAI([
            _null_choices_resp(choices),
            _openai_resp('{"name": "ok", "score": 7}'),
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="Spec",
        )
        assert result.parsed.name == "ok"
        assert len(fake.kwargs) == 2, "the bad body must cost one attempt, not the call"

    def test_structured_gives_up_with_a_diagnosable_llm_error(self, monkeypatch):
        monkeypatch.setattr("krepis.llm._retry_backoff_sleep", lambda _a: None)
        fake = FakeOpenAI([_null_choices_resp(), _null_choices_resp()])
        with pytest.raises(LLMError) as exc_info:
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
            )
        assert "upstream provider error" in str(exc_info.value)

    def test_grounded_retries_a_null_choices_body(self, monkeypatch):
        monkeypatch.setattr("krepis.llm._retry_backoff_sleep", lambda _a: None)
        fake = FakeOpenAI([
            _null_choices_resp(),
            _openai_resp("grounded answer", finish_reason="stop"),
        ])
        result = _client(OPENROUTER_SPEC, fake).complete_grounded(
            system="s", user_content="u", search=SearchOptions(),
        )
        assert result.text == "grounded answer"
        assert len(fake.kwargs) == 2

    def test_complete_converts_it_to_llm_error_not_a_bare_exception(self):
        """``complete`` has no retry loop, so there is nothing to classify the
        failure into — the module contract says a failed call raises LLMError."""
        fake = FakeOpenAI([_null_choices_resp()])
        with pytest.raises(LLMError) as exc_info:
            _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert "upstream provider error" in str(exc_info.value)

    def test_a_healthy_response_is_untouched(self):
        fake = FakeOpenAI([_openai_resp('{"name": "fine", "score": 1}')])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="Spec",
        )
        assert result.parsed.name == "fine"
        assert len(fake.kwargs) == 1


class TestRetryBackoff:
    """Body-level retries back off. They used to run in a tight loop, which is
    the worst possible cadence against a briefly-unhealthy gateway."""

    def test_structured_sleeps_between_attempts_with_growing_bound(self, monkeypatch):
        slept: list[int] = []
        monkeypatch.setattr("krepis.llm._retry_backoff_sleep", slept.append)
        fake = FakeOpenAI([
            _null_choices_resp(),
            _openai_resp('{"name": "ok", "score": 2}'),
        ])
        _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="Spec",
        )
        assert slept == [0], "one failed attempt, one backoff, keyed by attempt index"

    def test_no_backoff_after_the_final_attempt(self, monkeypatch):
        slept: list[int] = []
        monkeypatch.setattr("krepis.llm._retry_backoff_sleep", slept.append)
        fake = FakeOpenAI([_null_choices_resp(), _null_choices_resp()])
        with pytest.raises(LLMError):
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
            )
        assert slept == [0], "never sleep after the attempt that gives up"

    def test_delay_is_bounded_and_jittered(self):
        from krepis.llm import _RETRY_DELAY_CAP_S, _retry_backoff_sleep

        seen: list[float] = []
        import krepis.llm as _mod

        real_sleep = _mod._time.sleep
        _mod._time.sleep = seen.append
        try:
            for attempt in range(8):
                _retry_backoff_sleep(attempt)
        finally:
            _mod._time.sleep = real_sleep
        assert all(0.0 <= d <= _RETRY_DELAY_CAP_S for d in seen), seen


# ── budget ownership + empty-content visibility (alpha-engine-config#6396) ──


class TestEffectiveMaxTokens:
    """`max_tokens` is registry-owned. A caller literal wins — say so.

    `structured`/`complete`/`structured_with_search` all resolve the budget as
    "the caller's if given, else the row's". Raising it is ordinary. LOWERING
    it silently reverses the registry, and on a reasoning model that is not a
    smaller answer — `max_tokens` bounds reasoning + content together, so the
    trace consumes the budget and `content` comes back `''`.

    Live 2026-08-04: the Director passed a literal 8000 against a row carrying
    65536. Two ~100s completions, both fully billed, both empty. Raising the
    ROW 16384 -> 65536 as the remediation changed nothing, because the literal
    was what the request carried, and nothing logged the wire value.
    """

    def test_the_ADDRESSED_registry_entry_survives_onto_the_result(self):
        """`model` is the upstream name the provider reports; `registry_id` is
        the entry we addressed. They are NOT interchangeable: three registry
        entries (`deepseek-v4-flash`, `-low`, `-max`) share one upstream model
        string while declaring `{exclude: true}`, `{effort: low}` and
        `{effort: max}`. Without this, three configurations collapse into one
        row and cost cannot be attributed. alpha-engine-config-I6908.
        """
        spec = ModelSpec(
            "openrouter", "deepseek-v4-flash", max_tokens=1024,
            reasoning={"effort": "max"}, registry_id="deepseek-v4-flash-max",
        )
        fake = FakeOpenAI([_openai_resp("hi", model="deepseek-v4-flash")])
        result = _client(spec, fake).complete(system="s", user_content="u")
        assert result.model == "deepseek-v4-flash", "the provider's own name"
        assert result.registry_id == "deepseek-v4-flash-max", "the entry addressed"
        assert result.model != result.registry_id, (
            "the whole point is that these differ — a test asserting them equal "
            "would pass on a spec where the distinction does not exist"
        )

    def test_a_hand_built_spec_records_no_registry_entry(self):
        """`None` means 'not resolved from the registry', which is a different
        statement from 'resolved and unknown'."""
        fake = FakeOpenAI([_openai_resp("hi")])
        result = _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert result.registry_id is None

    def test_the_reasoning_share_is_recorded_on_a_SUCCESSFUL_call(self):
        """Until now the draw was observable only when a call came back EMPTY
        (`_budget_exhausted_error`), i.e. once per outage. A budget floor
        cannot be derived from a quantity recorded only on failure — which is
        why all three instances of this class were remediated by a guess.
        """
        fake = FakeOpenAI([_openai_resp("OK", usage=_openai_usage(
            prompt=17, completion=111, reasoning=108))])
        result = _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert result.usage.reasoning_tokens == 108
        assert result.usage.output_tokens == 111, (
            "reasoning is a SUBSET of output tokens, not an addition"
        )

    def test_the_reasoning_share_reads_from_a_raw_dict_too(self):
        """`getattr` on a dict silently returns the default — how
        `server_tool_use_details` read 0 for weeks (config#1659)."""
        fake = FakeOpenAI([_openai_resp("OK", usage=_openai_usage(
            prompt=17, completion=111, reasoning={"reasoning_tokens": 108}))])
        result = _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert result.usage.reasoning_tokens == 108

    def test_a_non_reasoning_response_records_zero_not_an_error(self):
        fake = FakeOpenAI([_openai_resp("OK")])
        result = _client(OPENROUTER_SPEC, fake).complete(system="s", user_content="u")
        assert result.usage.reasoning_tokens == 0

    def test_none_uses_the_registry_budget(self):
        fake = FakeOpenAI([_openai_resp('{"name": "a", "score": 1}')])
        _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="Spec",
        )
        assert fake.kwargs[0]["max_tokens"] == 1024

    def test_a_lower_caller_value_reaches_the_wire_and_warns(self, caplog):
        fake = FakeOpenAI([_openai_resp('{"name": "a", "score": 1}')])
        with caplog.at_level(logging.WARNING, logger="krepis.llm"):
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
                max_tokens=8,
            )
        assert fake.kwargs[0]["max_tokens"] == 8, (
            "the caller's value is what the request carries — the library "
            "does not overrule it, it only refuses to hide it"
        )
        warned = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warned, "a caller shrinking the registry budget logged nothing"
        assert "1024" in warned[0].getMessage(), (
            "the warning must name the registry value being overridden, or an "
            "operator cannot tell a registry change is inert"
        )

    def test_a_higher_caller_value_does_not_warn(self, caplog):
        fake = FakeOpenAI([_openai_resp('{"name": "a", "score": 1}')])
        with caplog.at_level(logging.WARNING, logger="krepis.llm"):
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
                max_tokens=4096,
            )
        assert fake.kwargs[0]["max_tokens"] == 4096
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_complete_resolves_the_budget_the_same_way(self, caplog):
        fake = FakeOpenAI([_openai_resp("hello")])
        with caplog.at_level(logging.WARNING, logger="krepis.llm"):
            _client(OPENROUTER_SPEC, fake).complete(
                system="s", user_content="u", max_tokens=8,
            )
        assert fake.kwargs[0]["max_tokens"] == 8
        assert [r for r in caplog.records if r.levelno == logging.WARNING], (
            "the override is invisible on `complete` too — every budget site "
            "goes through the same resolver, or the guard covers some of them"
        )


class TestEmptyContentIsVisible:
    """An empty `message.content` on a 200 must not pass silently.

    The caller-facing symptom actively misdirects: a structured caller reports
    `no JSON object found in response: ''`, which reads as a model that
    answered in prose. Three diagnostic cycles were spent on that reading
    while the response was 30 KB of reasoning trace.
    """

    def _empty_reasoning_resp(self):
        usage = _openai_usage()
        usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=7998)
        resp = _openai_resp("", usage=usage, finish_reason="stop")
        resp.choices[0].message.reasoning_content = "x" * 30000
        return resp

    def test_empty_content_logs_the_reasoning_budget(self, caplog):
        fake = FakeOpenAI([self._empty_reasoning_resp()] * 2)
        with caplog.at_level(logging.ERROR, logger="krepis.llm"):
            with pytest.raises(LLMError):
                _client(OPENROUTER_SPEC, fake).structured(
                    system="s", user_content="u", schema=Spec,
                    schema_name="Spec",
                )
        errors = [r.getMessage() for r in caplog.records
                  if r.levelno == logging.ERROR]
        assert errors, "an empty content produced no ERROR line"
        msg = errors[0]
        assert "reasoning_tokens=7998" in msg, (
            "without the reasoning-token count, an exhausted budget and a lost "
            "response body are the same log line"
        )
        assert "finish_reason='stop'" in msg
        assert "reasoning_content" in msg, (
            "the populated sibling fields are what say WHERE the output went"
        )

    def test_provider_extra_fields_are_named(self, caplog):
        """The most diagnostic field lives in pydantic's extras, not `vars()`.

        Measured 2026-08-04 against the live edge: `vars(message)` reported
        `['role']` on a message carrying 29,877 chars of `reasoning_content`,
        naming none of what an operator needs.
        """
        from krepis.llm import _empty_content_diagnostics

        class _Msg:
            def __init__(self):
                self.role = "assistant"
                self.content = ""

            model_extra = {"reasoning_content": "x" * 100, "refusal": None}

        choice = SimpleNamespace(message=_Msg(), finish_reason="length")
        out = _empty_content_diagnostics(
            SimpleNamespace(choices=[choice]), choice
        )
        assert "reasoning_content" in out
        assert "refusal" not in out, "empty fields are noise, not signal"

    def test_non_empty_content_logs_nothing(self, caplog):
        fake = FakeOpenAI([_openai_resp('{"name": "a", "score": 1}')])
        with caplog.at_level(logging.ERROR, logger="krepis.llm"):
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
            )
        assert not [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_whitespace_only_content_counts_as_empty(self, caplog):
        fake = FakeOpenAI([_openai_resp("   \n  ")] * 2)
        with caplog.at_level(logging.ERROR, logger="krepis.llm"):
            with pytest.raises(LLMError):
                _client(OPENROUTER_SPEC, fake).structured(
                    system="s", user_content="u", schema=Spec,
                    schema_name="Spec",
                )
        assert [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_diagnostics_never_mask_the_fault(self):
        """A response the diagnostic cannot introspect must still return ''."""
        from krepis.llm import _choice_text

        class _Hostile:
            @property
            def message(self):
                return SimpleNamespace(content="")

            def __getattr__(self, name):
                raise RuntimeError("this response resists introspection")

        resp = SimpleNamespace(choices=[_Hostile()])
        assert _choice_text(resp) == ""


class TestBudgetExhaustedIsNotRetried:
    """An exhausted budget is a certainty about every remaining attempt.

    `no JSON object found in response: ''` says *the model returned something
    unparseable* — a prompt or model problem. The actual fault is *max_tokens
    was too small for this ask*, a one-line registry change. Three wrong
    hypotheses were chased against a live paid endpoint before anyone looked
    at `finish_reason` (alpha-engine-config#6391).

    Retrying does not merely fail to inform. Measured on the Director's weekly
    call: two attempts, ~100s of generation each, both fully billed, and the
    second was guaranteed to fail before the first returned.
    """

    def _length_capped_empty(self):
        usage = _openai_usage()
        usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=7993)
        return _openai_resp("", usage=usage, finish_reason="length")

    def test_it_raises_on_the_first_occurrence(self):
        fake = FakeOpenAI([self._length_capped_empty()] * 2)
        with pytest.raises(BudgetExhaustedError) as exc:
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
            )
        assert len(fake.kwargs) == 1, (
            "the second attempt re-issues the identical ask under the "
            "identical ceiling — it can only fail again, and it bills again"
        )
        msg = str(exc.value)
        assert "max_tokens=1024" in msg, "name the budget that was too small"
        assert "reasoning_tokens=7993" in msg
        assert "no JSON object" not in msg, (
            "the old message points at the prompt; this fault is the budget"
        )

    def test_it_is_an_llm_error_so_callers_still_catch_it(self):
        fake = FakeOpenAI([self._length_capped_empty()])
        with pytest.raises(LLMError):
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
            )

    def test_it_carries_the_usage_of_the_billed_attempt(self):
        fake = FakeOpenAI([self._length_capped_empty()])
        with pytest.raises(BudgetExhaustedError) as exc:
            _client(OPENROUTER_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
            )
        assert exc.value.usage is not None
        assert exc.value.usage.output_tokens > 0, (
            "the attempt was billed — a failed call still has spend to record"
        )

    def test_empty_with_finish_reason_stop_keeps_the_retry(self):
        """A DIFFERENT fault: a model that answered with nothing.

        A retry can fix that one, so it must keep the corrective-retry path.
        """
        fake = FakeOpenAI([
            _openai_resp("", finish_reason="stop"),
            _openai_resp('{"name": "a", "score": 1}', finish_reason="stop"),
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="Spec",
        )
        assert result.parsed.name == "a"
        assert len(fake.kwargs) == 2

    def test_length_capped_WITH_content_is_ordinary_truncation(self):
        """Truncation the caller may still parse — not this fault."""
        fake = FakeOpenAI([
            _openai_resp('{"name": "a", "score": 1}', finish_reason="length"),
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u", schema=Spec, schema_name="Spec",
        )
        assert result.parsed.score == 1

    def test_anthropic_max_tokens_before_the_tool_block(self):
        msg = _anthropic_msg([_text_block("thinking...")])
        msg.stop_reason = "max_tokens"
        fake = FakeAnthropic([msg, msg])
        with pytest.raises(BudgetExhaustedError) as exc:
            _client(ANTHROPIC_SPEC, fake).structured(
                system="s", user_content="u", schema=Spec, schema_name="Spec",
            )
        assert "stop_reason='max_tokens'" in str(exc.value)
        assert len(fake.payloads) == 1


# ── Router-edge credential resolution at CALL time (config-I6373) ─────────


class TestRouterEdgeCredentialAtCallTime:
    """The edge credential resolves on the full chain, not the environment.

    ``resolve_group_spec`` names a per-consumer credential and the supported
    home for its value is SSM — precisely so the secret never enters an
    environment, a Lambda config, a CloudWatch log, or an SSM command string.
    Route admission resolved it that way; this leg read ``os.environ`` alone.
    A consumer configured exactly as alpha-engine-config-I6373 intends was
    therefore ADMITTED to the route and then failed the call it had just been
    admitted for.

    Measured 2026-08-04: the Think Tank spot box aborted 5s into its daily run
    with 0 theses written and ``challenger_selection`` unwritten, and
    ``alpha-engine-research-runner`` failed identically — both with all six
    ``KREPIS_*`` variables set and ``/alpha-engine/ROUTER_CONSUMER_*`` present
    and readable. Both halves had tests; neither test could see the other half,
    which is why the class-level test below exercises the SEAM rather than
    either side of it.
    """

    ROUTER_SPEC = ModelSpec(
        provider="litellm_proxy",
        model="med",
        base_url="https://router.example.invalid:8443",
        api_key_env="ROUTER_CONSUMER_THINKTANK",
    )

    def test_credential_only_in_ssm_resolves(self, monkeypatch):
        """The live failure, reproduced: nothing in the environment, value in SSM."""
        monkeypatch.delenv("ROUTER_CONSUMER_THINKTANK", raising=False)
        monkeypatch.setattr(
            "krepis.router._litellm_master_key_from_ssm",
            lambda name="LITELLM_MASTER_KEY": (
                "sk-from-ssm" if name == "ROUTER_CONSUMER_THINKTANK" else None
            ),
        )
        client = LLMClient(spec=self.ROUTER_SPEC, callsite_id="t")
        assert client._resolve_api_key() == "sk-from-ssm"

    def test_ssm_lookup_uses_this_consumers_name_not_the_shared_key(self, monkeypatch):
        """The edge identifies a consumer BY its credential value, so resolving
        the shared name would collapse this consumer into the director."""
        seen = []
        monkeypatch.delenv("ROUTER_CONSUMER_THINKTANK", raising=False)
        monkeypatch.setattr(
            "krepis.router._litellm_master_key_from_ssm",
            lambda name="LITELLM_MASTER_KEY": seen.append(name) or "sk",
        )
        LLMClient(spec=self.ROUTER_SPEC, callsite_id="t")._resolve_api_key()
        assert seen == ["ROUTER_CONSUMER_THINKTANK"]
        assert "LITELLM_MASTER_KEY" not in seen

    def test_environment_still_wins_when_set(self, monkeypatch):
        monkeypatch.setenv("ROUTER_CONSUMER_THINKTANK", "sk-from-env")
        monkeypatch.setattr(
            "krepis.router._litellm_master_key_from_ssm",
            lambda name="LITELLM_MASTER_KEY": "sk-from-ssm",
        )
        client = LLMClient(spec=self.ROUTER_SPEC, callsite_id="t")
        assert client._resolve_api_key() == "sk-from-env"

    def test_unresolvable_edge_credential_names_the_ssm_parameter(self, monkeypatch):
        """Naming only the env var sends an operator to the wrong place on the
        one path whose supported source is SSM."""
        monkeypatch.delenv("ROUTER_CONSUMER_THINKTANK", raising=False)
        with pytest.raises(LLMConfigError) as exc:
            LLMClient(spec=self.ROUTER_SPEC, callsite_id="t")._resolve_api_key()
        message = str(exc.value)
        # All three sources, so the message matches the chain that was walked.
        # The pre-I6373 message named the environment variable ALONE, which is
        # the one source a correctly-configured consumer deliberately does not
        # use — it read as "you forgot to set it" to an operator who had not.
        assert "/alpha-engine/ROUTER_CONSUMER_THINKTANK" in message
        assert "environment variable" in message
        assert "api_key=" in message

    def test_non_router_providers_stay_environment_only(self, monkeypatch):
        """Every other provider authenticates from a key that is deliberately in
        the environment. This change must not quietly give them an SSM read."""
        called = []
        monkeypatch.setattr(
            "krepis.router._litellm_master_key_from_ssm",
            lambda name="LITELLM_MASTER_KEY": called.append(name) or "sk",
        )
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        spec = ModelSpec(provider="openrouter", model="m")
        with pytest.raises(LLMConfigError) as exc:
            LLMClient(spec=spec, callsite_id="t")._resolve_api_key()
        assert called == []
        assert "OPENROUTER_API_KEY environment variable" in str(exc.value)

    def test_explicit_api_key_still_short_circuits(self, monkeypatch):
        monkeypatch.delenv("ROUTER_CONSUMER_THINKTANK", raising=False)
        client = LLMClient(
            spec=self.ROUTER_SPEC, callsite_id="t", api_key="sk-explicit"
        )
        assert client._resolve_api_key() == "sk-explicit"

    def test_the_two_halves_agree_on_the_credential_name(self, monkeypatch):
        """THE SEAM. Route admission and the call must look up the same name.

        Both halves passed their own tests while disagreeing about which
        credential the consumer had — first about the NAME (I6414), then about
        where its VALUE may live (this change). Asserting them separately is
        what let that happen twice, so this walks the whole path: the name the
        admission check resolves is the name the client authenticates with.
        """
        import krepis.router as _router

        monkeypatch.setenv(
            _router.ROUTER_CREDENTIAL_SECRET_ENV, "ROUTER_CONSUMER_THINKTANK"
        )
        monkeypatch.delenv("ROUTER_CONSUMER_THINKTANK", raising=False)
        seen = []
        monkeypatch.setattr(
            "krepis.router._litellm_master_key_from_ssm",
            lambda name="LITELLM_MASTER_KEY": seen.append(name) or "sk-shared",
        )

        admission = _router._resolve_litellm_master_key()
        spec = ModelSpec(
            provider=_router.ROUTER_EDGE_PROVIDER,
            model="med",
            base_url="https://router.example.invalid:8443",
            api_key_env=_router.router_credential_secret_name(),
        )
        call = LLMClient(spec=spec, callsite_id="t")._resolve_api_key()

        assert admission == call == "sk-shared"
        assert seen == ["ROUTER_CONSUMER_THINKTANK", "ROUTER_CONSUMER_THINKTANK"]


# ── Group alias must never masquerade as a served model (config-I6543) ────


class TestGroupServedModelNeverAliasesToTheGroupName:
    """A group-addressed spec's ``model`` is a synthetic name ("low" / "med"
    / "high" / "ultra") that is never itself a billable model and carries no
    price card. When the router response's ``model`` field comes back empty
    or equal to the alias, the client must raise rather than report the
    alias as the served model — the prior behavior let "low" flow all the
    way into ``thinktank/client.py::_cost_for``, which correctly found no
    price card for it and aborted the run with 0 theses written (live
    2026-08-04, alpha-engine-config-I6543).
    """

    ROUTER_SPEC = ModelSpec(
        provider=ROUTER_EDGE_PROVIDER,
        model="low",
        base_url="https://router.example.invalid:8443",
        api_key_env="ROUTER_CONSUMER_THINKTANK",
    )

    @staticmethod
    def _router_client(fake):
        # The router edge resolves its credential on the full SSM chain
        # (I6373); an explicit api_key short-circuits that so these tests
        # exercise served-model resolution, not credential resolution.
        return LLMClient(
            TestGroupServedModelNeverAliasesToTheGroupName.ROUTER_SPEC,
            callsite_id="krepis-test",
            client_factory=lambda _spec, _key: fake,
            api_key="sk-router-test",
        )

    def test_structured_raises_when_response_model_is_the_alias(self):
        fake = FakeOpenAI([_openai_resp('{"anything": 1}', model="low")])
        with pytest.raises(LLMConfigError, match="group='low'"):
            self._router_client(fake).structured(
                system="s", user_content="u",
                schema={"type": "object"}, schema_name="blob",
            )
        # Not retried: an alias echo is a data-integrity problem, not a
        # transient validation failure a corrective retry could fix.
        assert len(fake.kwargs) == 1

    def test_structured_raises_when_response_model_is_empty(self):
        fake = FakeOpenAI([_openai_resp('{"anything": 1}', model="")])
        with pytest.raises(LLMConfigError, match="did not report a served model"):
            self._router_client(fake).structured(
                system="s", user_content="u",
                schema={"type": "object"}, schema_name="blob",
            )

    def test_structured_accepts_a_real_served_model(self):
        fake = FakeOpenAI([
            _openai_resp('{"anything": 1}', model="deepseek/deepseek-v4-flash")
        ])
        result = self._router_client(fake).structured(
            system="s", user_content="u",
            schema={"type": "object"}, schema_name="blob",
        )
        assert result.model == "deepseek/deepseek-v4-flash"

    def test_complete_raises_when_response_model_is_the_alias(self):
        fake = FakeOpenAI([_openai_resp("hello", model="low")])
        with pytest.raises(LLMConfigError, match="group='low'"):
            self._router_client(fake).complete(system="s", user_content="u")

    def test_complete_accepts_a_real_served_model(self):
        fake = FakeOpenAI([_openai_resp("hello", model="deepseek/deepseek-v4-flash")])
        result = self._router_client(fake).complete(
            system="s", user_content="u"
        )
        assert result.model == "deepseek/deepseek-v4-flash"

    def test_non_router_provider_is_unaffected_by_alias_matching_model(self):
        """A pinned (non-group-addressed) spec legitimately requests and
        receives the SAME model name back — that must never raise. The
        strict check is scoped to ``ROUTER_EDGE_PROVIDER`` only."""
        fake = FakeOpenAI([
            _openai_resp('{"anything": 1}', model="moonshotai/kimi-k2.6")
        ])
        result = _client(OPENROUTER_SPEC, fake).structured(
            system="s", user_content="u",
            schema={"type": "object"}, schema_name="blob",
        )
        assert result.model == "moonshotai/kimi-k2.6"

    # ── qualified {group}-{mid} deployment names (0.39.0 derivation) ──

    _STRIP_REGISTRY = """
model_groups:
  low:
    - deepseek-v4-flash
    - gpt-oss-120b

models:
  - id: deepseek-v4-flash
    provider: deepseek
    route: egress_proxy
    api_base: http://127.0.0.1:8972/v1
    model: deepseek-v4-flash
    status: active
  - id: gpt-oss-120b
    provider: openrouter
    route: openrouter
    model: openai/gpt-oss-120b
    status: active
"""

    @pytest.fixture
    def strip_registry(self, tmp_path, monkeypatch):
        reg = tmp_path / "LLM_MODEL_REGISTRY.yaml"
        reg.write_text(self._STRIP_REGISTRY)
        monkeypatch.setenv("LLM_MODEL_REGISTRY_PATH", str(reg))

    def test_qualified_primary_name_passes_the_guard_and_resolves(
        self, strip_registry
    ):
        """A wire response whose ``model`` is the qualified primary
        deployment name ``{group}-{mid}`` — what the 0.39.0 derivation
        names every deployment, primary included — must pass the guard and
        come back as the registry entry's upstream model, the identifier
        the price cards are keyed on (alpha-engine-config-I6543,
        2026-08-09 comment)."""
        fake = FakeOpenAI([_openai_resp("hello", model="low-deepseek-v4-flash")])
        result = self._router_client(fake).complete(system="s", user_content="u")
        assert result.model == "deepseek-v4-flash"

    def test_qualified_fallback_name_resolves_to_its_route_correct_slug(
        self, strip_registry
    ):
        fake = FakeOpenAI([
            _openai_resp('{"anything": 1}', model="low-gpt-oss-120b")
        ])
        result = self._router_client(fake).structured(
            system="s", user_content="u",
            schema={"type": "object"}, schema_name="blob",
        )
        assert result.model == "openai/gpt-oss-120b"

    # ── deployment-ADDRESSED calls (the wire shape since router.py emits
    # deployment_id = _qualified_primary on the litellm_proxy route) ──

    _DEPLOYMENT_SPEC = ModelSpec(
        provider=ROUTER_EDGE_PROVIDER,
        model="low-deepseek-v4-flash",
        base_url="https://router.example.invalid:8443",
        api_key_env="ROUTER_CONSUMER_THINKTANK",
    )

    def _deployment_client(self, fake):
        return LLMClient(
            self._DEPLOYMENT_SPEC,
            callsite_id="krepis-test",
            client_factory=lambda _spec, _key: fake,
            api_key="sk-router-test",
        )

    def test_echoed_qualified_deployment_name_is_not_masquerade(
        self, strip_registry
    ):
        """The consumer addressed a concrete deployment; LiteLLM stamps the
        model AS ADDRESSED back, so served == spec.model on every HEALTHY
        call. Rejecting that rejected every successful call: the Think Tank
        challenger arm aborted with 0 theses written and wrote no challenger
        selection between 2026-08-01 and 2026-08-10 (run b150c317eeef,
        `group='med-deepseek-v4-flash-max'`)."""
        fake = FakeOpenAI([_openai_resp("hello", model="low-deepseek-v4-flash")])
        result = self._deployment_client(fake).complete(system="s", user_content="u")
        assert result.model == "deepseek-v4-flash"

    def test_echoed_deployment_name_resolves_in_structured_too(
        self, strip_registry
    ):
        fake = FakeOpenAI([
            _openai_resp('{"anything": 1}', model="low-gpt-oss-120b")
        ])
        spec = ModelSpec(
            provider=ROUTER_EDGE_PROVIDER,
            model="low-gpt-oss-120b",
            base_url="https://router.example.invalid:8443",
            api_key_env="ROUTER_CONSUMER_THINKTANK",
        )
        client = LLMClient(
            spec, callsite_id="krepis-test",
            client_factory=lambda _spec, _key: fake, api_key="sk-router-test",
        )
        result = client.structured(
            system="s", user_content="u",
            schema={"type": "object"}, schema_name="blob",
        )
        assert result.model == "openai/gpt-oss-120b"

    def test_echoed_bare_group_raises_without_touching_the_registry(self):
        """No ``strip_registry`` fixture on purpose: a bare group echo must
        produce the precise masquerade error even where no registry exists
        on disk (this repo's own CI). Asking the registry about "low" turned
        that into a FileNotFoundError."""
        fake = FakeOpenAI([_openai_resp("hello", model="low")])
        with pytest.raises(LLMConfigError, match="did not report a served model"):
            self._router_client(fake).complete(system="s", user_content="u")

    def test_echoed_bare_group_still_raises(self, strip_registry):
        """The masquerade this guard exists for is UNCHANGED: a bare group
        name is not a derived deployment name, does not resolve through the
        registry, and must never be billed or recorded."""
        fake = FakeOpenAI([_openai_resp("hello", model="low")])
        with pytest.raises(LLMConfigError, match="group='low'"):
            self._router_client(fake).complete(system="s", user_content="u")

    def test_echoed_unresolvable_deployment_name_still_raises(
        self, strip_registry
    ):
        """An echoed name the registry cannot resolve is indistinguishable
        from masquerade — the registry, never the string's shape, is what
        licenses the pass."""
        spec = ModelSpec(
            provider=ROUTER_EDGE_PROVIDER,
            model="low-model-we-never-heard-of",
            base_url="https://router.example.invalid:8443",
            api_key_env="ROUTER_CONSUMER_THINKTANK",
        )
        fake = FakeOpenAI([_openai_resp("hello", model="low-model-we-never-heard-of")])
        client = LLMClient(
            spec, callsite_id="krepis-test",
            client_factory=lambda _spec, _key: fake, api_key="sk-router-test",
        )
        with pytest.raises(LLMConfigError, match="did not report a served model"):
            client.complete(system="s", user_content="u")

    def test_unresolvable_group_prefixed_name_raises(self, strip_registry):
        """A ``{group}-``-prefixed served model the local registry cannot
        resolve means the router and this consumer read different
        registries — fail loud, not at the price-card lookup downstream."""
        fake = FakeOpenAI([_openai_resp("hello", model="low-model-we-never-heard-of")])
        with pytest.raises(LLMConfigError, match="does not resolve through"):
            self._router_client(fake).complete(system="s", user_content="u")
