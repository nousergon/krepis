"""
Unit tests for ``krepis.anthropic_payload``.

Pins the institutional-chokepoint contract for raw-Anthropic-SDK
payload construction. Surfaced as a lib lift after the 2026-05-26
morning-signal incident where the historical
``{role: "assistant", content: prefill}`` opener-pin was combined with
the ``web_search_20250305`` server tool, producing two consecutive
silent HTTP 400 cron-firing failures before the operator noticed.

* Validator MUST raise on (server-tool + trailing assistant message)
  for every server-tool prefix in ``SERVER_TOOL_PREFIXES``.
* Validator MUST NOT raise on (server-tool alone) or (prefill alone).
* ``build_messages_payload`` MUST return a payload that validates
  cleanly AND has the cached system block + the user message + the
  optional tools, in the exact shape ``messages.create()`` expects.
* ``build_web_search_tool`` MUST default to
  :data:`DEFAULT_WEB_SEARCH_MAX_USES` so consumers can't silently lose
  the runaway-cost cap.

See ``[[feedback_no_silent_fails]]`` + the alpha-engine SOTA
sub-sub-rule (second-adoption signal → lift to lib).
"""

from __future__ import annotations

import pytest

from krepis.anthropic_payload import (
    DEFAULT_WEB_SEARCH_MAX_USES,
    SERVER_TOOL_PREFIXES,
    PayloadInvariantError,
    build_batches_request_params,
    build_messages_payload,
    build_web_search_tool,
    validate_payload,
)


# ── validate_payload — server-tool ⊥ assistant-prefill ───────────────────────


@pytest.mark.parametrize(
    "tool_type",
    [
        "web_search_20250305",
        "computer_use_20250124",
        "bash_20250124",
        "text_editor_20250124",
    ],
)
def test_validate_rejects_server_tool_with_trailing_assistant(tool_type):
    """The 2026-05-26 regression class: any server-side tool combined
    with a trailing assistant message (prefill) returns HTTP 400. The
    validator catches it at the producer site so the failure can never
    reach a 5 AM cron firing."""
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "tools": [{"type": tool_type, "name": "t"}],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Welcome"},
        ],
    }
    with pytest.raises(PayloadInvariantError, match="server-side tools"):
        validate_payload(payload)


def test_payload_invariant_error_is_value_error():
    """Existing ``except ValueError`` callers MUST still catch payload
    bugs — institutional default that subclasses of ``ValueError``
    remain catchable as ValueError."""
    assert issubclass(PayloadInvariantError, ValueError)


def test_validate_allows_server_tool_without_prefill():
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    validate_payload(payload)


def test_validate_allows_prefill_without_server_tool():
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Y"},
        ],
    }
    validate_payload(payload)


def test_validate_allows_no_tools_no_prefill():
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    validate_payload(payload)


def test_validate_treats_empty_tools_list_as_no_server_tools():
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "tools": [],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Y"},
        ],
    }
    validate_payload(payload)


def test_validate_allows_non_server_tool_with_prefill():
    """Client-side tool definitions (no server-tool prefix) compose
    fine with a trailing assistant message; only Anthropic's
    server-side tool-use loop has the constraint."""
    payload = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 100,
        "tools": [{"type": "custom_thing", "name": "x"}],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Y"},
        ],
    }
    validate_payload(payload)


def test_server_tool_prefixes_is_immutable_tuple():
    """Constant MUST be a tuple, not a list — defends against
    consumers patching the prefix set at runtime, which would silently
    expand the validator's blast radius."""
    assert isinstance(SERVER_TOOL_PREFIXES, tuple)
    assert "web_search_" in SERVER_TOOL_PREFIXES
    assert "computer_use_" in SERVER_TOOL_PREFIXES


# ── build_web_search_tool ────────────────────────────────────────────────────


def test_build_web_search_tool_defaults():
    spec = build_web_search_tool()
    assert spec["type"] == "web_search_20250305"
    assert spec["name"] == "web_search"
    assert spec["max_uses"] == DEFAULT_WEB_SEARCH_MAX_USES == 20


def test_build_web_search_tool_max_uses_override():
    spec = build_web_search_tool(max_uses=5)
    assert spec["max_uses"] == 5


def test_build_web_search_tool_custom_name():
    spec = build_web_search_tool(name="custom_search")
    assert spec["name"] == "custom_search"


# ── build_messages_payload ───────────────────────────────────────────────────


def test_build_messages_payload_shape_with_tools():
    payload = build_messages_payload(
        model="claude-sonnet-4-5",
        system_prompt="static prompt",
        user_content="dynamic preamble",
        max_tokens=100,
        tools=[build_web_search_tool()],
    )
    assert payload["model"] == "claude-sonnet-4-5"
    assert payload["max_tokens"] == 100
    # system block cached by default
    assert payload["system"] == [
        {
            "type": "text",
            "text": "static prompt",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    # single user message; no assistant prefill (would conflict with web_search)
    assert payload["messages"] == [
        {"role": "user", "content": "dynamic preamble"}
    ]
    assert payload["tools"][0]["type"] == "web_search_20250305"
    assert payload["tools"][0]["max_uses"] == 20


def test_build_messages_payload_without_tools_omits_tools_key():
    """Anthropic SDK rejects ``tools=[]`` vs ``tools`` missing
    differently in some model snapshots; safer to omit the key entirely
    when there are no tools."""
    payload = build_messages_payload(
        model="claude-sonnet-4-5",
        system_prompt="p",
        user_content="u",
        max_tokens=10,
    )
    assert "tools" not in payload


def test_build_messages_payload_cache_system_false_omits_cache_control():
    payload = build_messages_payload(
        model="claude-sonnet-4-5",
        system_prompt="p",
        user_content="u",
        max_tokens=10,
        cache_system=False,
    )
    assert "cache_control" not in payload["system"][0]


def test_build_messages_payload_extra_kwargs_pass_through():
    payload = build_messages_payload(
        model="claude-sonnet-4-5",
        system_prompt="p",
        user_content="u",
        max_tokens=10,
        extra={"temperature": 0.7, "stop_sequences": ["\n\n"]},
    )
    assert payload["temperature"] == 0.7
    assert payload["stop_sequences"] == ["\n\n"]


def test_build_messages_payload_validates_extra_that_breaks_invariant():
    """Validation runs AFTER the extra-merge so an ``extra`` dict that
    smuggles in an assistant prefill alongside a server tool still
    trips the invariant. This is the load-bearing guarantee — callers
    cannot bypass the chokepoint by routing fields through ``extra``."""
    with pytest.raises(PayloadInvariantError):
        build_messages_payload(
            model="claude-sonnet-4-5",
            system_prompt="p",
            user_content="u",
            max_tokens=10,
            tools=[build_web_search_tool()],
            extra={
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "Y"},
                ]
            },
        )


def test_build_messages_payload_morning_signal_replication():
    """The exact production shape used by morning-signal post-fix.
    Pins the canonical raw-SDK consumer pattern so a future repo
    landing on this lib module gets a working template."""
    opener = "Welcome to Morning Signal."
    payload = build_messages_payload(
        model="claude-sonnet-4-5",
        system_prompt="# Morning Signal production prompt (~1.3K tokens of static text)",
        user_content=(
            "Today is Tuesday, May 26, 2026. This is the MORNING edition of Morning Signal. "
            "Generate today's morning episode per the system prompt.\n\n"
            f"Your response MUST begin verbatim with this exact line, "
            f"with no preamble or acknowledgement before it:\n\n{opener}"
        ),
        max_tokens=4096,
        tools=[build_web_search_tool(max_uses=20)],
    )
    # Validator already ran inside build_messages_payload — assert the
    # shape matches what messages.create() expects post-fix.
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["max_uses"] == 20
    assert len(payload["messages"]) == 1  # no assistant prefill
    assert opener in payload["messages"][0]["content"]


# ── build_batches_request_params ─────────────────────────────────────────────


_FORCE_TOOL_CHOICE = {"type": "tool", "name": "RubricEvalLLMOutput"}


def _custom_tool_spec():
    """A non-server-side tool — what the judge batch uses for structured output."""
    return {
        "name": "RubricEvalLLMOutput",
        "description": "Emit the rubric eval payload as structured JSON.",
        "input_schema": {
            "type": "object",
            "properties": {"score": {"type": "integer"}},
            "required": ["score"],
        },
    }


def test_build_batches_request_params_judge_shape():
    """Replicates the alpha-engine-research judge call shape: no system
    prompt, custom tool, forced tool_choice, no caching. Locks the
    minimal viable Batches request envelope the judge actually ships."""
    req = build_batches_request_params(
        custom_id="judge-abc-123",
        model="claude-haiku-4-5",
        max_tokens=2048,
        user_content="Rubric prompt body here…",
        tools=[_custom_tool_spec()],
        tool_choice=_FORCE_TOOL_CHOICE,
    )
    assert req["custom_id"] == "judge-abc-123"
    params = req["params"]
    assert params["model"] == "claude-haiku-4-5"
    assert params["max_tokens"] == 2048
    assert params["messages"] == [{"role": "user", "content": "Rubric prompt body here…"}]
    assert params["tools"] == [_custom_tool_spec()]
    assert params["tool_choice"] == _FORCE_TOOL_CHOICE
    # No system prompt by default — judge inlines rubric into user content.
    assert "system" not in params


def test_build_batches_request_params_with_system_prompt_no_cache_default():
    """When a system prompt IS provided, it lands as a one-element system
    array. Caching is OFF by default for batches per the docstring rationale."""
    req = build_batches_request_params(
        custom_id="x",
        model="claude-sonnet-4-6",
        max_tokens=256,
        user_content="u",
        system_prompt="You are a helpful assistant.",
    )
    sys_blocks = req["params"]["system"]
    assert sys_blocks == [{"type": "text", "text": "You are a helpful assistant."}]
    assert "cache_control" not in sys_blocks[0]


def test_build_batches_request_params_with_system_prompt_cache_opt_in():
    """``cache_system=True`` attaches ephemeral cache_control (the
    opt-in path for batches with large repeated system prompts)."""
    req = build_batches_request_params(
        custom_id="x",
        model="claude-sonnet-4-6",
        max_tokens=256,
        user_content="u",
        system_prompt="Large repeated system prompt.",
        cache_system=True,
    )
    assert req["params"]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_build_batches_request_params_validates_server_tool_prefill_invariant():
    """The Batches builder honors the same server-tool ⊥ assistant-prefill
    invariant as the sync builder — caught via ``extra`` smuggling."""
    with pytest.raises(PayloadInvariantError):
        build_batches_request_params(
            custom_id="x",
            model="claude-sonnet-4-6",
            max_tokens=256,
            user_content="u",
            tools=[build_web_search_tool()],
            extra={
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "Y"},
                ]
            },
        )


def test_build_batches_request_params_no_system_no_tools_minimal():
    """Minimal shape: only model + max_tokens + messages. Pins that
    optional fields don't leak ``None`` keys into the payload."""
    req = build_batches_request_params(
        custom_id="x",
        model="claude-haiku-4-5",
        max_tokens=64,
        user_content="ping",
    )
    params = req["params"]
    assert set(params.keys()) == {"model", "max_tokens", "messages"}


def test_build_batches_request_params_extra_merges_into_params():
    """``extra`` keys merge into ``params`` (e.g. metadata for batch-side
    observability). Validation still runs."""
    req = build_batches_request_params(
        custom_id="x",
        model="claude-haiku-4-5",
        max_tokens=64,
        user_content="u",
        extra={"metadata": {"user_id": "judge-v3"}},
    )
    assert req["params"]["metadata"] == {"user_id": "judge-v3"}
