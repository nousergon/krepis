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
    MAX_CACHE_BREAKPOINTS,
    SERVER_TOOL_PREFIXES,
    PayloadInvariantError,
    build_batches_request_params,
    build_messages_payload,
    build_web_search_tool,
    ensure_message_breakpoint_spacing,
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


# ── 4-breakpoint ceiling (G3, §3.7) ──────────────────────────────────────────


class TestCountCacheBreakpoints:
    """``_count_cache_breakpoints`` — count ``cache_control`` markers
    across system blocks and message content blocks."""

    def test_zero_breakpoints_empty_payload(self):
        from krepis.anthropic_payload import _count_cache_breakpoints
        assert _count_cache_breakpoints({}) == 0
        assert _count_cache_breakpoints({"system": [], "messages": []}) == 0

    def test_zero_breakpoints_no_cache_control(self):
        from krepis.anthropic_payload import _count_cache_breakpoints
        payload = {
            "system": [{"type": "text", "text": "hello"}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert _count_cache_breakpoints(payload) == 0

    def test_one_system_breakpoint(self):
        from krepis.anthropic_payload import _count_cache_breakpoints
        payload = {
            "system": [{"type": "text", "text": "prompt",
                         "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert _count_cache_breakpoints(payload) == 1

    def test_breakpoints_in_message_content_blocks(self):
        from krepis.anthropic_payload import _count_cache_breakpoints
        payload = {
            "system": [{"type": "text", "text": "prompt",
                         "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "part a",
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "part b"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "response",
                     "cache_control": {"type": "ephemeral"}},
                ]},
            ],
        }
        # system block (1) + user content block (1) + assistant content block (1) = 3
        assert _count_cache_breakpoints(payload) == 3


class TestValidate4BreakpointCeiling:
    """``validate_payload`` MUST reject payloads exceeding 4 breakpoints."""

    def test_allows_four_breakpoints(self):
        """Boundary: exactly 4 breakpoints is valid."""
        payload = {
            "system": [
                {"type": "text", "text": "s1",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "s2",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "s3",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "s4",
                 "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        validate_payload(payload)

    def test_rejects_five_breakpoints(self):
        payload = {
            "system": [
                {"type": "text", "text": "s1",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "s2",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "s3",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "s4",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "s5",
                 "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        with pytest.raises(PayloadInvariantError, match="4-breakpoint|4 cache_control|MAX_CACHE_BREAKPOINTS"):
            validate_payload(payload)

    def test_rejects_breakpoints_across_system_and_messages(self):
        """Breakpoints in both system AND message content blocks
        count toward the same ceiling — the API limit is per-request."""
        five_system_blocks = [
            {"type": "text", "text": f"s{i}",
             "cache_control": {"type": "ephemeral"}}
            for i in range(4)
        ]
        payload = {
            "system": five_system_blocks,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "extra",
                 "cache_control": {"type": "ephemeral"}},
            ]}],
        }
        with pytest.raises(PayloadInvariantError):
            validate_payload(payload)

    def test_existing_server_tool_check_still_fires(self):
        """Adding the breakpoint ceiling must not break the existing
        server-tool ⊥ assistant-prefill invariant; both checks coexist."""
        payload = {
            "system": [{"type": "text", "text": "p",
                         "cache_control": {"type": "ephemeral"}}],
            "tools": [{"type": "web_search_20250305", "name": "w"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Y"},
            ],
        }
        with pytest.raises(PayloadInvariantError, match="server-side tools"):
            validate_payload(payload)

    def test_max_breakpoints_constant_exported(self):
        """The constant is public so consumers can reference it."""
        assert MAX_CACHE_BREAKPOINTS == 4


# ── 20-block lookback (G3, §3.7) ──────────────────────────────────────────────


class TestEnsureMessageBreakpointSpacing:
    """``ensure_message_breakpoint_spacing`` — intermediate breakpoint
    placement in multi-turn messages.

    Both unenforced G3 rules are M1-only, and the fleet currently runs
    zero active M1 models. These tests verify the logic is correct so
    it fires when needed, and that the no-op path (M2/M4) is inert.
    """

    def test_noop_when_supports_explicit_breakpoints_false(self):
        """M2/M4 path: markers are never placed."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
        ]
        result = ensure_message_breakpoint_spacing(
            messages, supports_explicit_breakpoints=False,
        )
        assert result is messages
        assert "cache_control" not in result[0]["content"][0]

    def test_noop_for_short_conversation(self):
        """Below the 20-block lookback window: no markers needed."""
        blocks = [{"type": "text", "text": str(i)} for i in range(19)]
        messages = [{"role": "user", "content": blocks}]
        ensure_message_breakpoint_spacing(messages)
        assert all("cache_control" not in b for b in messages[0]["content"])

    def test_places_first_intermediate_at_20_blocks(self):
        """At exactly 20 consecutive blocks without a breakpoint,
        the next block gets one (blocks are 0-indexed: block 19 is
        the 20th)."""
        blocks = [{"type": "text", "text": str(i)} for i in range(21)]
        messages = [{"role": "user", "content": blocks}]
        ensure_message_breakpoint_spacing(messages)
        # block 0-18: no breakpoint (18 blocks)
        # block 19: 20th block, gets breakpoint
        for i in range(19):
            assert "cache_control" not in blocks[i], f"block {i} should not have cache_control"
        assert "cache_control" in blocks[19], "block 19 (20th block) should have intermediate breakpoint"
        # After placing one, the counter resets; block 20 is the first
        # block after the breakpoint (at offset 1, < 15 → no marker).
        assert "cache_control" not in blocks[20]

    def test_respects_existing_breakpoint_in_messages(self):
        """An existing ``cache_control`` marker resets the counter
        — blocks after it don't count toward the 20-block window."""
        blocks = [{"type": "text", "text": str(i)} for i in range(25)]
        # Place an existing breakpoint at block 10
        blocks[10]["cache_control"] = {"type": "ephemeral"}
        messages = [{"role": "user", "content": blocks}]
        ensure_message_breakpoint_spacing(messages)
        # After the breakpoint at block 10, counters reset.
        # Blocks 11-29 (19 more) have no new breakpoint.
        for i in range(11, 25):
            assert "cache_control" not in blocks[i], f"block {i} should not have new cache_control"

    def test_places_repeated_intermediates_at_15_block_interval(self):
        """After the 20-block lookback window triggers the first
        intermediate, further breakpoints are placed ~15 blocks
        apart (counter values 20, 35, 50, ... = (N-20)%15==0).
        The counter is NOT reset on placement so intervals stay
        approximately 15 rather than reverting to 20 each time."""
        blocks = [{"type": "text", "text": str(i)} for i in range(70)]
        messages = [{"role": "user", "content": blocks}]
        ensure_message_breakpoint_spacing(messages)
        # Block 19 = 20th block (0-indexed) → first intermediate
        assert "cache_control" in blocks[19]
        # After first placement, counter continues (not reset).
        # Counter=35 → (35-20)%15==0 → block 34 (15 blocks later)
        assert "cache_control" in blocks[34]
        # Counter=50 → (50-20)%15==0 → block 49 (15 blocks later)
        assert "cache_control" in blocks[49]
        # Blocks beyond: counter < 20 before next trigger (65) → none
        assert "cache_control" not in blocks[60]
        assert "cache_control" not in blocks[69]

    def test_single_string_content_no_markers_placed(self):
        """String content is a single content block; no intermediate
        breakpoints are needed since content is indivisible."""
        messages = [
            {"role": "user", "content": "a long string content"},
        ]
        ensure_message_breakpoint_spacing(messages)
        assert isinstance(messages[0]["content"], str)

    def test_mixed_string_and_list_messages(self):
        """A multi-turn conversation with mixed message shapes."""
        messages = [
            {"role": "user", "content": "short text"},
            {"role": "assistant",
             "content": [{"type": "text", "text": "response"}]},
        ]
        result = ensure_message_breakpoint_spacing(messages)
        assert result is messages
