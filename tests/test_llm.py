"""Tests for ``krepis.llm.LLMClient`` — both transports via fake clients."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from krepis.llm import (
    LLMClient,
    LLMError,
    SearchOptions,
    _extract_json,
)
from krepis.llm_config import LLMConfigError, ModelSpec


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


def _openai_usage(prompt=100, completion=50, cached=0, cost=None, searches=None):
    u = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    if cost is not None:
        u.cost = cost
    if searches is not None:
        u.web_search_requests = searches
    return u


def _openai_resp(content, usage=None, model="moonshotai/kimi-k2.6", annotations=None):
    message = SimpleNamespace(content=content)
    if annotations is not None:
        message.annotations = annotations
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage or _openai_usage(),
        model=model,
    )


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
    return LLMClient(spec, client_factory=lambda _spec, _key: fake)


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
        client = LLMClient(OPENROUTER_SPEC)
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
