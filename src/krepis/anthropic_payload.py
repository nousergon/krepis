"""
Anthropic ``messages.create()`` payload-construction chokepoint.

Consolidation substrate for the raw-Anthropic-SDK call shape that
multiple consumer repos now ship. First adopter is morning-signal
(``src/morning_signal/claude.py``); alpha-engine-research is the future
second raw-SDK adopter once the LangChain wrappers retire. Per the
``[[feedback_lift_invariants_to_chokepoint_after_second_recurrence]]``
discipline and the alpha-engine SOTA sub-sub-rule (mirror a pattern
across repos → lift to lib), this module bakes the known-good payload
shape + invariant validation into one place.

**Why this exists.** 2026-05-26 morning-signal incident: the 5/25-night
PR #33 (prompt caching + ``web_search max_uses`` cap) shipped on top
of the historical ``{role: "assistant", content: prefill}`` opener-pin.
The combination of ``web_search`` (any server-side tool) with a
trailing assistant message is rejected by the Anthropic API with HTTP
400::

    "This model does not support assistant message prefill.
     The conversation must end with a user message."

Two consecutive cron firings (5/25 PM at 00:00 UTC, 5/26 AM at 12:00
UTC) failed silently before the operator noticed. The producer-side
``_validate_request_payload`` chokepoint in morning-signal was the
local fix; this module is the lib lift so the next raw-SDK consumer
inherits the invariant without re-discovering it the hard way.

**Composes with:**

- :mod:`krepis.cost` — :func:`cost.metadata_from_anthropic_message`
  is the canonical adapter for converting a returned ``Message`` into
  a ``ModelMetadata`` cost-telemetry record. This module is the
  outbound counterpart (request side); ``cost`` is the inbound side
  (response side).

**Public surface:**

- :data:`SERVER_TOOL_PREFIXES` — type-prefix tuple for Anthropic
  server-side tool definitions that share the "tool loop ends on
  user message" constraint.
- :data:`DEFAULT_WEB_SEARCH_MAX_USES` — runaway-cost insurance cap
  default; lifted from morning-signal PR #33.
- :data:`MAX_CACHE_BREAKPOINTS` — hard API limit on
  ``cache_control`` markers per request (``4``).
- :func:`build_messages_payload` — construct the kwargs dict to splat
  into ``client.messages.create(**payload)``. Always validates before
  returning.
- :func:`validate_payload` — pure invariant check against a constructed
  payload. Raises :exc:`ValueError` on known-incompatible shapes.
- :func:`build_web_search_tool` — convenience builder for the
  ``web_search_20250305`` tool spec with the runaway-cost cap default.
- :func:`ensure_message_breakpoint_spacing` — place intermediate
  ``cache_control`` markers in multi-turn ``messages[]`` per §3.7.
- :exc:`PayloadInvariantError` — subclass of ``ValueError`` raised by
  :func:`validate_payload`. Distinct type so callers can catch payload
  bugs separately from other ValueErrors.

**Anti-pattern this module forbids:** combining any server-side tool
(``web_search_*``, ``computer_use_*``, ``bash_*``, ``text_editor_*``)
with a conversation whose final ``messages[-1].role == "assistant"``.
The tool-loop semantics require the conversation to alternate ending
on a user / tool_result turn so the model can decide whether to emit
another tool_use block before final text.
"""

from __future__ import annotations

from typing import Any


# Anthropic server-side tool type prefixes. Each of these tool types
# triggers Anthropic's server-side tool-use loop, which requires the
# conversation to end on a user (or tool_result) turn so the model can
# decide whether to emit another tool_use block before final text.
# Combining any of these with a trailing assistant message (prefill)
# returns HTTP 400 "This model does not support assistant message
# prefill." Verified against the 2026-05-26 morning-signal incident.
SERVER_TOOL_PREFIXES: tuple[str, ...] = (
    "web_search_",
    "computer_use_",
    "bash_",
    "text_editor_",
)

# Runaway-cost insurance on ``web_search_20250305``. Anthropic bills
# ``web_search`` at $10/1k requests; an uncapped spec lets a malformed
# prompt or model-loop bug rack up unbounded fees. 20 sits above
# morning-signal's empirical typical (~15 across the 9-segment briefing)
# so it functions as insurance not throttling. Lifted from
# morning-signal PR #33.
DEFAULT_WEB_SEARCH_MAX_USES: int = 20

# Hard API limit on the number of ``cache_control`` markers per request.
# The Anthropic Messages API rejects any request carrying more than 4
# ``cache_control`` breakpoints with an HTTP 400 error. This limit is
# provider-enforced and model-independent, so the check is unconditional.
MAX_CACHE_BREAKPOINTS: int = 4

# Content-block threshold for the 20-block lookback rule (§3.7 of the
# prompt-caching policy). When a multi-turn request has more than this
# many consecutive content blocks since the last ``cache_control``
# breakpoint, an intermediate breakpoint should be placed. The constant
# is exported so callers that construct multi-turn payloads can call
# :func:`ensure_message_breakpoint_spacing` after building their
# ``messages[]`` list but before passing it to the payload builder.
_LOOKBACK_WINDOW: int = 20
_INTERMEDIATE_BREAKPOINT_INTERVAL: int = 15


class PayloadInvariantError(ValueError):
    """Raised by :func:`validate_payload` on a known-incompatible
    Anthropic ``messages.create()`` request shape. Subclass of
    :class:`ValueError` so existing ``except ValueError`` callers still
    catch it; distinct type so a caller that cares specifically about
    payload bugs can catch this without swallowing other ValueErrors.
    """


def _has_server_tool(tools: list[dict] | None) -> bool:
    if not tools:
        return False
    return any(
        any(t.get("type", "").startswith(p) for p in SERVER_TOOL_PREFIXES)
        for t in tools
    )


def _count_cache_breakpoints(payload: dict[str, Any]) -> int:
    """Count explicit ``cache_control`` markers across *payload*.

    Checks the ``system`` field (a list of content blocks per the
    Anthropic Messages API) and the ``content`` lists inside each
    message in ``messages[]``.

    A ``cache_control`` marker placed on a segment below the model's
    ``cache_min_tokens`` still counts against the hard 4-breakpoint
    ceiling set by the Anthropic API — the provider returns HTTP 400
    when the total exceeds :data:`MAX_CACHE_BREAKPOINTS` regardless of
    whether individual markers are effective. So this function counts
    markers unconditionally; it does not check token lengths.
    """
    count = 0
    for block in (payload.get("system") or []):
        if isinstance(block, dict) and "cache_control" in block:
            count += 1
    for msg in (payload.get("messages") or []):
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    count += 1
        elif isinstance(content, dict) and "cache_control" in content:
            count += 1
    return count


def ensure_message_breakpoint_spacing(
    messages: list[dict[str, Any]],
    *,
    supports_explicit_breakpoints: bool = True,
) -> list[dict[str, Any]]:
    """Place intermediate ``cache_control`` breakpoints in a multi-turn
    ``messages[]`` list per the 20-block lookback rule (§3.7 of the
    prompt-caching policy).

    When *supports_explicit_breakpoints* is ``False`` (the model uses
    automatic prefix caching or none), the messages are returned
    unchanged — this is the no-op-on-M2/M4 path.

    When ``True``, the function walks each message's ``content`` list
    and counts consecutive content blocks since the last
    ``cache_control`` marker. If the gap reaches
    :data:`_LOOKBACK_WINDOW` (20) blocks, an intermediate breakpoint
    is placed on the next content block. Further breakpoints follow
    at roughly :data:`_INTERMEDIATE_BREAKPOINT_INTERVAL` (15) blocks.

    **Current code paths produce at most one system-block breakpoint
    and a single user message, so this function is a no-op on today's
    production traffic.** It exists as a forward-looking guard so that
    the next person who extends the payload builder to handle multi-turn
    conversations with many content blocks inherits the rule rather than
    rediscovering it.

    The function mutates content-block dicts **in place** and returns
    the same list reference, matching the pattern used by
    :func:`build_messages_payload` and
    :func:`build_batches_request_params`.

    Returns:
        The same *messages* list (mutated in place), for convenience.
    """
    if not supports_explicit_breakpoints:
        return messages

    blocks_since_breakpoint = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            # String content or dict content — single block, no
            # subdivision where intermediate breakpoints would go.
            # A dict content block COULD carry cache_control; count it.
            if isinstance(content, dict) and "cache_control" in content:
                blocks_since_breakpoint = 0
            elif isinstance(content, dict):
                blocks_since_breakpoint += 1
            elif isinstance(content, str):
                blocks_since_breakpoint += 1
            continue

        for block in content:
            if not isinstance(block, dict):
                blocks_since_breakpoint += 1
                continue
            if "cache_control" in block:
                blocks_since_breakpoint = 0
                continue
            blocks_since_breakpoint += 1
            if (blocks_since_breakpoint >= _LOOKBACK_WINDOW
                    and (blocks_since_breakpoint - _LOOKBACK_WINDOW)
                    % _INTERMEDIATE_BREAKPOINT_INTERVAL == 0):
                block["cache_control"] = {"type": "ephemeral"}

    return messages


def validate_payload(payload: dict[str, Any]) -> None:
    """Raise :exc:`PayloadInvariantError` on a known-incompatible
    Anthropic ``messages.create()`` payload shape.

    Currently enforced invariants:

    1. **Server-tool ⊥ assistant-prefill.** If ``payload["tools"]``
       contains any type with a :data:`SERVER_TOOL_PREFIXES` prefix
       AND ``payload["messages"][-1]["role"] == "assistant"``,
       Anthropic returns HTTP 400. Surfaced 2026-05-26.

    2. **4-breakpoint ceiling.** The Anthropic Messages API rejects
       requests with more than :data:`MAX_CACHE_BREAKPOINTS` (4)
       ``cache_control`` markers. Counted across both the ``system``
       field and message ``content`` blocks.

    The validator is a producer-side chokepoint: failing here at
    construction time means the bug class can't reach a production
    cron firing.
    """
    messages = payload.get("messages") or []
    tools = payload.get("tools") or []

    if _has_server_tool(tools):
        last_role = messages[-1]["role"] if messages else None
        if last_role == "assistant":
            raise PayloadInvariantError(
                "Anthropic payload invariant violated: server-side tools "
                "(types prefixed with any of "
                f"{SERVER_TOOL_PREFIXES}) cannot be combined with a "
                "trailing assistant message (prefill). The API rejects "
                "this with HTTP 400 'This model does not support "
                "assistant message prefill. The conversation must end "
                "with a user message.' Either drop the prefill or drop "
                "the server tool."
            )

    breakpoint_count = _count_cache_breakpoints(payload)
    if breakpoint_count > MAX_CACHE_BREAKPOINTS:
        raise PayloadInvariantError(
            f"Anthropic payload invariant violated: {breakpoint_count} "
            f"cache_control markers exceeds the hard API limit of "
            f"{MAX_CACHE_BREAKPOINTS}. The Anthropic Messages API "
            "rejects requests with more than 4 cache_control markers."
        )


def build_web_search_tool(
    *,
    max_uses: int = DEFAULT_WEB_SEARCH_MAX_USES,
    name: str = "web_search",
) -> dict[str, Any]:
    """Build the ``web_search_20250305`` tool spec with the runaway-cost
    cap. ``max_uses`` defaults to :data:`DEFAULT_WEB_SEARCH_MAX_USES`.
    """
    return {
        "type": "web_search_20250305",
        "name": name,
        "max_uses": max_uses,
    }


def build_messages_payload(
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    tools: list[dict] | None = None,
    cache_system: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a validated kwargs dict for ``client.messages.create()``.

    Returns a dict the caller splats into the SDK:

        payload = build_messages_payload(...)
        response = client.messages.create(**payload)

    Args:
        model: Anthropic model identifier (e.g. ``"claude-sonnet-4-5"``).
        system_prompt: The static system-prompt text. Sent as a single
            ``system`` block; when ``cache_system=True`` (default) the
            block carries ``cache_control: {"type": "ephemeral"}`` so
            the prefix is cached at the 0.1× cache-read rate on every
            tool-loop re-read within one ``messages.create()`` call.
        user_content: The dynamic per-call user-message content
            (typically date + edition + any per-call instructions).
            Lives in the user message rather than the cached system
            block so the static prefix stays per-call cacheable.
        max_tokens: ``max_tokens`` for the call.
        tools: Optional list of tool specs. May include server-side
            tools (``web_search_20250305`` etc.) — :func:`validate_payload`
            enforces the server-tool ⊥ prefill invariant.
        cache_system: When ``True`` (default) attach ephemeral
            ``cache_control`` to the ``system`` block. Pass ``False``
            for one-shot calls where caching has no return.
        extra: Optional dict merged into the result (e.g. ``stop_sequences``,
            ``temperature``, ``metadata``). Validation runs AFTER the
            merge so any extras that affect ``messages`` / ``tools``
            are checked too.

    Returns:
        Validated kwargs dict. Raises :exc:`PayloadInvariantError` on a
        known-incompatible shape.
    """
    system_block: dict[str, Any] = {"type": "text", "text": system_prompt}
    if cache_system:
        system_block["cache_control"] = {"type": "ephemeral"}

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [system_block],
        "messages": [{"role": "user", "content": user_content}],
    }
    if tools:
        payload["tools"] = list(tools)
    if extra:
        payload.update(extra)

    validate_payload(payload)
    return payload


def build_batches_request_params(
    *,
    custom_id: str,
    model: str,
    max_tokens: int,
    user_content: str,
    tools: list[dict] | None = None,
    tool_choice: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    cache_system: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct one entry of the ``messages.batches.create`` ``requests`` array.

    The Anthropic Batches API takes a list of ``{"custom_id", "params"}``
    dicts, where each ``params`` value is a kwargs dict for an underlying
    ``messages.create()`` call. This helper builds one such entry,
    validating the embedded payload via :func:`validate_payload`.

    Differs from :func:`build_messages_payload` along three axes the
    judge-batch path requires:

    1. **Optional system prompt.** Synchronous callers nearly always have
       a static system prompt (the lib default caches it); judge batches
       inject the entire rubric into the user message and have no system
       block. Pass ``system_prompt=None`` (the default) to emit no
       system block at all.
    2. **No cache_control by default.** The Batches API discounts every
       call 50% before prompt caching applies; the marginal value of
       caching is small enough that the existing judge path opts out.
       ``cache_system=False`` is the default for this reason; pass
       ``cache_system=True`` explicitly if the system prompt is large
       enough to benefit.
    3. **Explicit tool_choice.** Forced tool calls (
       ``{"type": "tool", "name": ...}``) are the dominant Batches use
       case (structured-output via a known schema). Pass ``tool_choice``
       directly rather than smuggling through ``extra``.

    All :func:`validate_payload` invariants run against the embedded
    ``params`` — including the server-tool ⊥ assistant-prefill check —
    so a future Batches caller that mixes ``web_search`` with a
    prefill won't reach Anthropic's HTTP 400.

    Args:
        custom_id: Per-request identifier returned in the batch result.
            Caller-owned; must be unique within a batch.
        model: Anthropic model identifier (e.g. ``"claude-haiku-4-5"``).
        max_tokens: ``max_tokens`` for the embedded call.
        user_content: The user-message content (typically the full
            rendered rubric / prompt body, since batch calls usually
            omit the system block).
        tools: Optional list of tool specs.
        tool_choice: Optional tool-choice spec (e.g.
            ``{"type": "tool", "name": "RubricEvalLLMOutput"}`` to force
            structured output via a specific tool).
        system_prompt: Optional system-prompt text. When ``None`` (the
            default), no ``system`` block is emitted.
        cache_system: When ``True``, attach ``cache_control: ephemeral``
            to the system block. Default ``False`` because Batches
            already discounts 50% and the marginal cache value is small.
            Ignored when ``system_prompt is None``.
        extra: Optional dict merged into ``params`` after construction
            (e.g. ``metadata``, ``stop_sequences``). Validation runs
            AFTER the merge.

    Returns:
        ``{"custom_id": custom_id, "params": <validated kwargs dict>}``,
        ready to splat into ``messages.batches.create(requests=[...])``.

    Raises :exc:`PayloadInvariantError` on a known-incompatible shape.
    """
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_content}],
    }
    if system_prompt is not None:
        system_block: dict[str, Any] = {"type": "text", "text": system_prompt}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}
        params["system"] = [system_block]
    if tools:
        params["tools"] = list(tools)
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    if extra:
        params.update(extra)

    validate_payload(params)
    return {"custom_id": custom_id, "params": params}
