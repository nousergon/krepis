"""
Unit tests for ``krepis.judge``.

Pins the LLM-as-judge transport core lifted from
``crucible-research/evals/judge.py`` + ``evals/judge_models.py``
(nousergon/alpha-engine-config#1675, #2575).

The contract-test block (``TestByteIdenticalContract``) is the bar the
lift issue set: reproduce the pre-lift ``crucible-research`` behavior
for each lifted function against fixed inputs and assert field-identical
output. These pin the exact pre-refactor implementations inline (not
imported — crucible-research is a separate repo/dependency direction)
so a future edit to ``krepis.judge`` that silently changes wire-format
behavior (custom_id shape, tool spec shape, rubric rendering) fails
here first, before it reaches the consumer PR.
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from krepis.judge import (
    REASON_CONTROL_TOKEN_LEAK,
    REASON_TRUNCATED_BEFORE_TOOL_CALL,
    JudgeModelSpec,
    JudgeToolCallLeakError,
    ToolResultNotFoundError,
    build_structured_tool_spec,
    check_openai_tool_response_for_leak,
    decode_custom_id,
    encode_custom_id,
    parse_batch_tool_result,
    render_rubric,
    request_model_for,
    resolve,
)


# ── Fixture registry (mirrors crucible-research's judge_models.py HAIKU/SONNET) ──

HAIKU = JudgeModelSpec(
    logical_key="claude-haiku-4-5",
    request_model="claude-haiku-4-5-20251001",
    tag="h45",
    pinned=True,
    pin_note="pinned to dated snapshot",
)
SONNET = JudgeModelSpec(
    logical_key="claude-sonnet-4-6",
    request_model="claude-sonnet-4-6",
    tag="s46",
    pinned=False,
    pin_note="no dated snapshot published",
)
SPECS = (HAIKU, SONNET)
TAG_BY_LOGICAL = {s.logical_key: s.tag for s in SPECS}


# ── resolve / request_model_for ──────────────────────────────────────────


def test_resolve_by_logical_key():
    assert resolve("claude-haiku-4-5", SPECS) is HAIKU


def test_resolve_by_request_model():
    assert resolve("claude-haiku-4-5-20251001", SPECS) is HAIKU


def test_resolve_by_tag():
    assert resolve("s46", SPECS) is SONNET


def test_resolve_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown judge model"):
        resolve("gpt-5", SPECS)


def test_request_model_for_returns_pinned_snapshot():
    assert request_model_for("claude-haiku-4-5", SPECS) == "claude-haiku-4-5-20251001"


def test_request_model_for_unpinned_returns_alias():
    # SONNET has no dated snapshot — request_model equals the alias itself.
    assert request_model_for("claude-sonnet-4-6", SPECS) == "claude-sonnet-4-6"


# ── render_rubric ─────────────────────────────────────────────────────────


def test_render_rubric_substitutes_json_blocks():
    template = "INPUT:\n{agent_input}\n\nOUTPUT:\n{agent_output}"
    rendered = render_rubric(
        template,
        agent_input={"ticker": "AAPL", "score": 4},
        agent_output={"verdict": "buy"},
    )
    assert "INPUT:" in rendered and "OUTPUT:" in rendered
    assert json.dumps({"ticker": "AAPL", "score": 4}, indent=2) in rendered
    assert json.dumps({"verdict": "buy"}, indent=2) in rendered


def test_render_rubric_handles_non_json_types_via_default_str():
    from datetime import date

    template = "{agent_input} / {agent_output}"
    # Should not raise despite the non-JSON-serializable date object.
    rendered = render_rubric(
        template, agent_input={"as_of": date(2026, 7, 15)}, agent_output=None,
    )
    assert "2026-07-15" in rendered


# ── build_structured_tool_spec / parse_batch_tool_result ────────────────


class _DimScore(BaseModel):
    model_config = ConfigDict(extra="allow")
    dimension: str
    score: int = Field(ge=1, le=5)
    reasoning: str


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    dimension_scores: list[_DimScore]
    overall_reasoning: str


def test_build_structured_tool_spec_from_pydantic_model():
    spec = build_structured_tool_spec(
        _JudgeOutput, tool_name="RubricEvalLLMOutput", description="Emit the eval.",
    )
    assert spec["name"] == "RubricEvalLLMOutput"
    assert spec["description"] == "Emit the eval."
    assert spec["input_schema"] == _JudgeOutput.model_json_schema()


def test_build_structured_tool_spec_from_raw_dict():
    raw_schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    spec = build_structured_tool_spec(raw_schema, tool_name="Foo", description="d")
    assert spec["input_schema"] == raw_schema
    # Must be a copy, not the same object (mutation isolation).
    assert spec["input_schema"] is not raw_schema


def _batch_message_dict(tool_name: str, tool_input: dict) -> dict:
    return {
        "content": [
            {"type": "text", "text": "thinking..."},
            {"type": "tool_use", "id": "toolu_1", "name": tool_name, "input": tool_input},
        ],
    }


def test_parse_batch_tool_result_returns_validated_pydantic_instance():
    payload = {
        "dimension_scores": [{"dimension": "clarity", "score": 4, "reasoning": "ok"}],
        "overall_reasoning": "solid",
    }
    msg = _batch_message_dict("RubricEvalLLMOutput", payload)
    parsed = parse_batch_tool_result(
        msg, tool_name="RubricEvalLLMOutput", schema=_JudgeOutput,
    )
    assert isinstance(parsed, _JudgeOutput)
    assert parsed.overall_reasoning == "solid"
    assert parsed.dimension_scores[0].score == 4


def test_parse_batch_tool_result_returns_raw_dict_without_schema():
    payload = {"overall_reasoning": "x", "dimension_scores": []}
    msg = _batch_message_dict("RubricEvalLLMOutput", payload)
    parsed = parse_batch_tool_result(msg, tool_name="RubricEvalLLMOutput")
    assert parsed == payload


def test_parse_batch_tool_result_accepts_sdk_object_shape():
    class _Block:
        def __init__(self, type_, name=None, input=None):
            self.type = type_
            self.name = name
            self.input = input

    class _Message:
        def __init__(self, content):
            self.content = content

    msg = _Message([
        _Block("text"),
        _Block("tool_use", name="RubricEvalLLMOutput", input={"a": 1}),
    ])
    parsed = parse_batch_tool_result(msg, tool_name="RubricEvalLLMOutput")
    assert parsed == {"a": 1}


def test_parse_batch_tool_result_raises_when_tool_not_found():
    msg = _batch_message_dict("SomeOtherTool", {"x": 1})
    with pytest.raises(ToolResultNotFoundError, match="No tool_use block named"):
        parse_batch_tool_result(msg, tool_name="RubricEvalLLMOutput")


def test_tool_result_not_found_error_is_value_error():
    # Existing ``except ValueError`` callers must still catch this.
    assert issubclass(ToolResultNotFoundError, ValueError)


def test_parse_batch_tool_result_propagates_validation_error_distinctly():
    """The tool WAS called (unlike test_..._raises_when_tool_not_found)
    but its input fails schema validation — a caller must be able to
    tell this apart from "tool never called" (e.g. to build a different
    diagnostic message), which a bare ``except ValueError`` cannot do
    since pydantic.ValidationError is itself a ValueError subclass."""
    # score=99 violates the ge=1, le=5 constraint on _DimScore.score.
    payload = {
        "dimension_scores": [{"dimension": "clarity", "score": 99, "reasoning": "x"}],
        "overall_reasoning": "y",
    }
    msg = _batch_message_dict("RubricEvalLLMOutput", payload)
    with pytest.raises(ValidationError):
        parse_batch_tool_result(msg, tool_name="RubricEvalLLMOutput", schema=_JudgeOutput)
    # Confirm it is NOT a ToolResultNotFoundError (the two must stay distinct).
    try:
        parse_batch_tool_result(msg, tool_name="RubricEvalLLMOutput", schema=_JudgeOutput)
    except ToolResultNotFoundError:
        pytest.fail("schema validation failure must not raise ToolResultNotFoundError")
    except ValidationError:
        pass


# ── check_openai_tool_response_for_leak ──────────────────────────────────
#
# Fixture shapes below are trimmed reproductions of REAL OpenRouter
# responses captured live against ``moonshotai/kimi-k2.6`` and
# ``deepseek/deepseek-v4-flash`` while building this guard
# (alpha-engine-config#2575, 2026-07-18) — not hand-invented shapes. The
# truncation fixture reproduces a live ``finish_reason="length"`` response
# with ``content=None`` and no ``tool_calls`` (the model spent its entire
# budget on chain-of-thought reasoning before the forced tool call); the
# control-token-leak shape mirrors the documented 2026-07-14 incident
# already guarded in ``krepis.llm.complete_grounded``.


def _openai_choice_dict(*, finish_reason, content=None, tool_calls=None):
    return {
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
    }


class _FakeMessage:
    def __init__(self, *, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, *, finish_reason, content=None, tool_calls=None):
        self.finish_reason = finish_reason
        self.message = _FakeMessage(content=content, tool_calls=tool_calls)


def test_check_leak_passes_clean_tool_call():
    # Live-shape: deepseek/deepseek-v4-flash clean structured tool call
    # (finish_reason="tool_calls", content=None, one well-formed tool_calls
    # entry) — captured live 2026-07-18. Must NOT raise.
    choice = _openai_choice_dict(
        finish_reason="tool_calls",
        content=None,
        tool_calls=[{"id": "x", "type": "function",
                      "function": {"name": "RubricEvalLLMOutput", "arguments": "{}"}}],
    )
    check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")


def test_check_leak_passes_when_no_signature_present():
    choice = _openai_choice_dict(finish_reason="stop", content="ignored prose")
    check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")


def test_check_leak_raises_on_truncation_before_tool_call_dict_shape():
    # Live-shape: moonshotai/kimi-k2.6 with max_tokens=200, no reasoning
    # exclusion — finish_reason="length", content=None, no tool_calls.
    # Captured live 2026-07-18 (config#2575 item 3 validation).
    choice = _openai_choice_dict(finish_reason="length", content=None, tool_calls=None)
    with pytest.raises(JudgeToolCallLeakError) as exc_info:
        check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")
    assert exc_info.value.reason == REASON_TRUNCATED_BEFORE_TOOL_CALL
    assert exc_info.value.finish_reason == "length"


def test_check_leak_raises_on_truncation_sdk_object_shape():
    """Same as the dict-shape test but against an SDK-object-like input
    (attribute access, not dict subscription) — the OpenAI SDK's real
    response type, which callers pass directly without dict-ifying."""
    choice = _FakeChoice(finish_reason="length", content=None, tool_calls=None)
    with pytest.raises(JudgeToolCallLeakError) as exc_info:
        check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")
    assert exc_info.value.reason == REASON_TRUNCATED_BEFORE_TOOL_CALL


def test_check_leak_passes_truncation_with_empty_list_tool_calls():
    # Empty list (not None) must be treated the same as absent.
    choice = _openai_choice_dict(finish_reason="length", content=None, tool_calls=[])
    with pytest.raises(JudgeToolCallLeakError):
        check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")


def test_check_leak_does_not_flag_truncation_when_tool_calls_present():
    # finish_reason="length" but the tool call itself DID come through —
    # not the truncation-before-tool-call failure mode this guards.
    choice = _openai_choice_dict(
        finish_reason="length", content=None,
        tool_calls=[{"id": "x", "type": "function",
                      "function": {"name": "RubricEvalLLMOutput", "arguments": "{}"}}],
    )
    check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")


def test_check_leak_raises_on_control_token_leak_in_content():
    # Mirrors the krepis.llm.complete_grounded live incident (2026-07-14):
    # native tool-call dialect leaked into content as prose.
    choice = _openai_choice_dict(
        finish_reason="stop",
        content="<|tool_calls_section_begin|>some leaked garbage<|tool_call_end|>",
    )
    with pytest.raises(JudgeToolCallLeakError) as exc_info:
        check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")
    assert exc_info.value.reason == REASON_CONTROL_TOKEN_LEAK


def test_check_leak_raises_on_control_token_leak_even_with_tool_calls_present():
    # Live-confirmed co-occurrence pattern: a leak alongside a tool_calls
    # block is still untrustworthy, not just stray extra text.
    choice = _openai_choice_dict(
        finish_reason="stop",
        content="<|tool_call_begin|>junk",
        tool_calls=[{"id": "x", "type": "function",
                      "function": {"name": "RubricEvalLLMOutput", "arguments": "{}"}}],
    )
    with pytest.raises(JudgeToolCallLeakError) as exc_info:
        check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")
    assert exc_info.value.reason == REASON_CONTROL_TOKEN_LEAK


def test_judge_tool_call_leak_error_is_value_error():
    # Existing broad ``except ValueError`` retry loops must still catch it.
    assert issubclass(JudgeToolCallLeakError, ValueError)


def test_check_leak_handles_missing_message_gracefully():
    # Defensive: a malformed choice with no message at all must not crash
    # the guard itself with an AttributeError — treated as no-leak-signature
    # (ordinary downstream parsing will fail loudly on the missing message).
    choice = {"finish_reason": "stop"}
    check_openai_tool_response_for_leak(choice, tool_name="RubricEvalLLMOutput")


# ── encode_custom_id / decode_custom_id ──────────────────────────────────


_CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def test_encode_custom_id_matches_anthropic_charset():
    cid = encode_custom_id(
        subject_id="sector_quant:technology", run_id="2026-07-15",
        judge_model="claude-haiku-4-5", tag_by_logical=TAG_BY_LOGICAL,
    )
    assert _CUSTOM_ID_PATTERN.match(cid)
    assert len(cid) <= 64


def test_encode_decode_roundtrip():
    cid = encode_custom_id(
        subject_id="ic_cio", run_id="2026-07-15T12-00",
        judge_model="claude-sonnet-4-6", tag_by_logical=TAG_BY_LOGICAL,
    )
    subject, run, model = decode_custom_id(cid, tag_by_logical=TAG_BY_LOGICAL)
    assert subject == "ic_cio"
    assert run == "2026-07-15T12-00"
    assert model == "claude-sonnet-4-6"


def test_encode_custom_id_truncates_long_subject_id():
    long_subject = "thesis_update:" + ("x" * 100)
    cid = encode_custom_id(
        subject_id=long_subject, run_id="2026-07-15",
        judge_model="claude-haiku-4-5", tag_by_logical=TAG_BY_LOGICAL,
    )
    assert len(cid) <= 64
    assert _CUSTOM_ID_PATTERN.match(cid)


def test_encode_custom_id_unknown_model_falls_back_to_hash_tag():
    cid = encode_custom_id(
        subject_id="ic_cio", run_id="2026-07-15",
        judge_model="unregistered-model", tag_by_logical=TAG_BY_LOGICAL,
    )
    assert _CUSTOM_ID_PATTERN.match(cid)
    # Hash-stable: encoding the same unknown model twice yields the same tag.
    cid2 = encode_custom_id(
        subject_id="ic_cio", run_id="2026-07-15",
        judge_model="unregistered-model", tag_by_logical=TAG_BY_LOGICAL,
    )
    assert cid == cid2


def test_decode_custom_id_rejects_malformed_shape():
    with pytest.raises(ValueError, match="expected three"):
        decode_custom_id("not_a_valid_custom_id", tag_by_logical=TAG_BY_LOGICAL)


def test_encode_custom_id_sanitizes_disallowed_chars_in_caller_supplied_tag():
    # The subject_id/run_id segments are pre-sanitized, but a caller-owned
    # tag_by_logical map is trusted as-is — a tag containing charset-illegal
    # characters (space, "!") exercises the last-ditch whole-string sanitize
    # fallback rather than a per-segment one.
    cid = encode_custom_id(
        subject_id="a", run_id="b", judge_model="weird",
        tag_by_logical={"weird": "bad tag!"},
    )
    assert _CUSTOM_ID_PATTERN.match(cid)
    assert cid == "a__b__bad-tag-"


# ── Contract test: byte/field-identical vs pre-lift crucible-research ────
#
# Reproduces the exact pre-refactor crucible-research implementations
# inline (evals/judge.py as of alpha-engine-config#2575's anchor commit)
# and asserts the new krepis.judge functions produce identical output for
# fixed inputs. This is the "byte-identical contract test" the lift issue
# requires — it cannot import crucible-research directly (separate repo,
# and the whole point of the lift is that crucible-research will import
# FROM here after the consumer PR, not the reverse), so the pre-lift
# behavior is pinned inline as a golden reference.


class TestByteIdenticalContract:
    @staticmethod
    def _legacy_encode_custom_id(
        *, judged_agent_id: str, run_id: str, judge_model: str, tag_by_logical: dict,
    ) -> str:
        """Verbatim copy of crucible-research evals/judge.py::encode_custom_id
        (pre-lift, as of the anchor commit) for byte-identical comparison."""
        tag = tag_by_logical.get(judge_model)
        if tag is None:
            tag = f"x{abs(hash(judge_model)) % 10_000:04d}"
        safe_agent = re.sub(r"[^a-zA-Z0-9_-]", "-", judged_agent_id)
        safe_run = re.sub(r"[^a-zA-Z0-9_-]", "-", run_id)
        fixed_overhead = len(safe_run) + len(tag) + 4
        max_agent = max(8, 64 - fixed_overhead)
        if len(safe_agent) > max_agent:
            safe_agent = safe_agent[:max_agent]
        cid = f"{safe_agent}__{safe_run}__{tag}"
        if not _CUSTOM_ID_PATTERN.match(cid):
            cid = re.sub(r"[^a-zA-Z0-9_-]", "-", cid)[:64]
        return cid

    @staticmethod
    def _legacy_render_rubric(template: str, agent_input, agent_output) -> str:
        """Verbatim copy of crucible-research evals/judge.py::_render_rubric
        (modulo the DecisionArtifact→plain-value indirection, which is the
        exact seam the lift moves)."""
        return template.format(
            agent_input=json.dumps(agent_input, indent=2, default=str),
            agent_output=json.dumps(agent_output, indent=2, default=str),
        )

    @pytest.mark.parametrize(
        "judged_agent_id,run_id,judge_model",
        [
            ("sector_quant:technology", "2026-07-15", "claude-haiku-4-5"),
            ("ic_cio", "2026-07-15T09-30", "claude-sonnet-4-6"),
            ("thesis_update:financials:JPM", "2026-07-15", "claude-haiku-4-5"),
            ("thesis_update:" + "z" * 90, "2026-07-15", "claude-haiku-4-5"),
        ],
    )
    def test_encode_custom_id_byte_identical(self, judged_agent_id, run_id, judge_model):
        legacy = self._legacy_encode_custom_id(
            judged_agent_id=judged_agent_id, run_id=run_id,
            judge_model=judge_model, tag_by_logical=TAG_BY_LOGICAL,
        )
        lifted = encode_custom_id(
            subject_id=judged_agent_id, run_id=run_id,
            judge_model=judge_model, tag_by_logical=TAG_BY_LOGICAL,
        )
        assert lifted == legacy

    @pytest.mark.parametrize(
        "agent_input,agent_output",
        [
            ({"ticker": "AAPL"}, {"verdict": "buy", "conviction": 0.8}),
            ({}, {}),
            ({"nested": {"a": [1, 2, 3]}}, None),
        ],
    )
    def test_render_rubric_field_identical(self, agent_input, agent_output):
        template = "AGENT INPUT:\n{agent_input}\n\nAGENT OUTPUT:\n{agent_output}"
        legacy = self._legacy_render_rubric(template, agent_input, agent_output)
        lifted = render_rubric(
            template, agent_input=agent_input, agent_output=agent_output,
        )
        assert lifted == legacy

    def test_build_structured_tool_spec_matches_legacy_shape(self):
        """crucible-research's _build_rubric_tool_spec() builds
        {"name", "description", "input_schema"} from
        RubricEvalLLMOutput.model_json_schema() — same three keys, same
        schema-source contract, generalized to any Pydantic model."""
        spec = build_structured_tool_spec(
            _JudgeOutput,
            tool_name="RubricEvalLLMOutput",
            description=(
                "Emit the rubric eval as a structured tool call. Each "
                "rubric dimension produces one entry in dimension_scores "
                "with an integer score and short reasoning. "
                "overall_reasoning is a 1-2 sentence cross-dimension "
                "summary."
            ),
        )
        assert set(spec.keys()) == {"name", "description", "input_schema"}
        assert spec["name"] == "RubricEvalLLMOutput"
        assert spec["input_schema"] == _JudgeOutput.model_json_schema()
