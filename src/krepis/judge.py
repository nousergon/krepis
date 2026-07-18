"""
LLM-as-judge transport core — rubric rendering, structured-output tool
spec, batch custom_id codec, and the judge-model pin/re-anchor registry.

Lifted from ``crucible-research/evals/judge.py`` + ``evals/judge_models.py``
(nousergon/alpha-engine-config#1675, #2575). This module carries the
TRANSPORT/MECHANICS layer only — the parts that are identical no matter
which agent pipeline is being judged or which rubric is in play:

* :class:`JudgeModelSpec` / :func:`resolve` / :func:`request_model_for`
  — the three-identity judge-model registry (logical key / pinned
  request model / resolved model) and its re-anchor protocol. A
  model-swap is a regime break (ARCH doctrine): callers must log an
  explicit re-anchor marker when ``request_model`` changes for a given
  ``logical_key``.
* :func:`render_rubric` — render a rubric template against an
  input/output pair. Deliberately schema-agnostic (plain strings in,
  plain string out) so it composes with any prompt-loading mechanism a
  consumer repo uses.
* :func:`build_structured_tool_spec` / :func:`parse_batch_tool_result`
  — the Anthropic Batches-API structured-output shape: force a tool
  call via a JSON-schema-shaped tool spec, then pull the matching
  ``tool_use`` block back out of a batch result message. Schema-agnostic
  (accepts a Pydantic model class OR a raw JSON-schema dict) so it does
  not need to import any consumer-specific ``LLMOutput`` schema — see
  :func:`krepis.llm.LLMClient.structured` for the same schema-agnostic
  pattern on the synchronous path.
* :func:`encode_custom_id` / :func:`decode_custom_id` — round-trippable
  codec packing ``(subject_id, run_id, judge_model)`` into the Anthropic
  Batches API's 64-char ``custom_id`` charset
  (``^[a-zA-Z0-9_-]{1,64}$``), keyed off a caller-supplied tag map so
  the codec doesn't hardcode any particular judge-model registry.

**What deliberately stays in the consumer repo** (business logic, not
mechanics — moving it here would either require importing
consumer-specific schemas the wrong way across the dependency graph, or
would bake pipeline-specific judgment calls into a general-purpose
lib):

* Agent-to-rubric mapping (``resolve_rubric_for_agent``).
* Degenerate-input skip heuristics (``_is_degenerate_input``).
* Prompt-file loading (``load_prompt`` — reads from the
  alpha-engine-config repo layout).
* Cost-telemetry-wrapped LLM invocation (``track_llm_cost`` /
  ``get_cost_telemetry_callback``) and the langchain
  ``with_structured_output`` retry loop
  (``invoke_structured_with_validation_retry``) — these reach into
  pipeline-wide budget enforcement and CloudWatch telemetry that is
  research-pipeline-specific, not judge-specific.
* The persisted ``RubricEvalArtifact`` wrapper schema and its S3
  path/key conventions (``krepis.eval_artifacts`` / equivalent) — the
  storage contract is a consumer concern.

Composes with:

* :mod:`krepis.anthropic_payload` — :func:`build_batches_request_params`
  is the payload-construction chokepoint this module's tool spec feeds
  into.
* :mod:`krepis.llm` — :meth:`krepis.llm.LLMClient.structured` is the
  synchronous-path analogue (native SDK, no langchain) of the
  structured-output contract this module implements for the Batches
  API path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = [
    "JudgeModelSpec",
    "resolve",
    "request_model_for",
    "render_rubric",
    "build_structured_tool_spec",
    "parse_batch_tool_result",
    "ToolResultNotFoundError",
    "encode_custom_id",
    "decode_custom_id",
    "JudgeToolCallLeakError",
    "check_openai_tool_response_for_leak",
]


# ── Judge-model registry ─────────────────────────────────────────────────
#
# Separates three identities a judge pipeline must not collapse into a
# single string (nousergon/alpha-engine-config L4578(a)):
#
# * ``logical_key`` — the STABLE identity used for the persisted-artifact
#   path, any CloudWatch/metrics dimension, and the custom_id tag. Must
#   NOT change when the request model is repinned to a newer snapshot,
#   or any downstream time series keyed on it resets for a non-change.
# * ``request_model`` — the EXACT string sent to the provider API. Pinned
#   to an immutable dated snapshot wherever the provider publishes one,
#   so the same weights run on every call.
# * resolved model — the model identifier the provider's response
#   actually reports it ran. Not modeled here (it is a per-call
#   response fact, not a registry fact) — callers capture it from their
#   own response metadata and treat a change as the re-anchor trigger
#   below.
#
# **Re-anchor protocol (on judge upgrade).** When the resolved model
# changes for a given ``logical_key`` — the provider ships a new
# snapshot, or a caller deliberately bumps ``request_model`` — scores
# before and after are NOT comparable. Treat it as a regime break, not
# a quality regression: (1) bump ``request_model`` in the registry, (2)
# record the date + old→new resolved model in the consumer's experiment
# log, (3) reset the affected rolling-mean / control-band baselines so
# no alarm fires on the discontinuity.


@dataclass(frozen=True)
class JudgeModelSpec:
    """One judge model's registry entry (see module-level doctrine above)."""

    logical_key: str
    """Stable identity — persisted-artifact path / metrics dimension /
    custom_id tag. Never changes on a snapshot pin."""

    request_model: str
    """Exact string sent to the provider API. A dated snapshot when one
    exists (``pinned=True``), otherwise the alias (``pinned=False``)."""

    tag: str
    """Compact custom_id tag (keeps the Batches API custom_id under its
    64-char ceiling)."""

    pinned: bool
    """True iff ``request_model`` is an immutable dated snapshot. False
    means no snapshot is published and the alias is the canonical ID."""

    pin_note: str
    """Why this spec is (or isn't) pinned — auditable rationale."""


def resolve(
    model: str, specs: "tuple[JudgeModelSpec, ...]",
) -> JudgeModelSpec:
    """Resolve a logical key, request ID, or tag to its ``JudgeModelSpec``.

    Accepts any of the three identities so callers can pass whatever
    they hold — the persisted logical key, an explicit request ID, or a
    custom_id tag.

    ``specs`` is the caller-owned registry (a tuple of every
    :class:`JudgeModelSpec` the caller has registered) — this module
    does not own a global registry itself, since the judge-model roster
    is a per-consumer decision (Haiku/Sonnet today, an OpenRouter tier
    later).

    Raises ``KeyError`` for an unknown model: judge models are a closed,
    audited set, so an unrecognized string is a bug (a typo or an
    un-registered model), not something to paper over with a soft
    fallback.
    """
    by_logical = {s.logical_key: s for s in specs}
    by_request = {s.request_model: s for s in specs}
    by_tag = {s.tag: s for s in specs}
    spec = by_logical.get(model) or by_request.get(model) or by_tag.get(model)
    if spec is None:
        raise KeyError(
            f"Unknown judge model {model!r}; register it in the caller's "
            f"judge-model registry (known logical keys: "
            f"{sorted(by_logical)}). Judge models are a closed, audited "
            f"set — an unrecognized id is a bug, not a fallback."
        )
    return spec


def request_model_for(
    logical_key: str, specs: "tuple[JudgeModelSpec, ...]",
) -> str:
    """Exact API request string for a logical judge-model key.

    The one indirection every judge transport should route through so
    the pinned snapshot is applied in exactly one place.
    """
    return resolve(logical_key, specs).request_model


# ── Rubric rendering ──────────────────────────────────────────────────────


def render_rubric(
    template: str, *, agent_input: Any, agent_output: Any,
) -> str:
    """Render a rubric template against an (input, output) pair.

    ``template`` uses ``str.format``-style ``{agent_input}`` /
    ``{agent_output}`` placeholders. Both values are JSON-serialized
    with ``indent=2, default=str`` so stray non-JSON types (datetimes,
    Decimals, etc.) that snuck into a captured snapshot don't raise —
    they get a lossy but non-crashing ``str()`` representation instead.

    Deliberately takes plain values rather than a schema-specific
    "artifact" object — the caller (a pipeline that knows its own
    captured-artifact shape) does the JSON-field extraction; this
    function only knows how to render one JSON blob into a template
    twice.
    """
    return template.format(
        agent_input=json.dumps(agent_input, indent=2, default=str),
        agent_output=json.dumps(agent_output, indent=2, default=str),
    )


# ── Structured-output tool spec (Batches-API path) ──────────────────────
#
# The Anthropic Batches API has no LangChain-style
# ``with_structured_output(...)`` wrapper — callers synthesize the tool
# spec directly from their output schema and force the model to call it
# via ``tool_choice``. Schema-agnostic: accepts either a Pydantic
# ``BaseModel`` subclass (uses ``model_json_schema()`` for the tool spec
# and ``model_validate()`` to parse the result) or a raw JSON-schema
# dict (result stays a plain dict) — the same duck-typed pattern
# ``krepis.llm.LLMClient.structured`` uses on the synchronous path, so a
# consumer's judge-output Pydantic model never needs to be imported
# here.


def build_structured_tool_spec(
    schema: Any, *, tool_name: str, description: str,
) -> dict[str, Any]:
    """Synthesize an Anthropic tool-use spec forcing structured output.

    ``schema`` is either a Pydantic ``BaseModel`` subclass (its
    ``model_json_schema()`` becomes the tool's ``input_schema``) or a
    raw JSON-schema dict (used as-is). Pinning the input_schema to the
    live schema means a schema bump (e.g. adding a new field) flows
    into the batch tool automatically — no second source of truth.

    Pair with ``tool_choice={"type": "tool", "name": tool_name}`` on the
    request so the model cannot fall back to prose (which no downstream
    parser here understands).
    """
    schema_dict = (
        schema.model_json_schema() if hasattr(schema, "model_json_schema")
        else dict(schema)
    )
    return {
        "name": tool_name,
        "description": description,
        "input_schema": schema_dict,
    }


class ToolResultNotFoundError(ValueError):
    """Raised by :func:`parse_batch_tool_result` when no ``tool_use`` block
    named ``tool_name`` is found in the message.

    Subclass of :class:`ValueError` so existing ``except ValueError``
    callers still catch it, but a DISTINCT type from the schema
    validation failures :func:`parse_batch_tool_result` lets propagate
    (e.g. ``pydantic.ValidationError``, itself also a ``ValueError``
    subclass) — a caller that wants to recognize specifically "the tool
    was never called" (to build a diagnostic message pointing at the
    provider's retained raw result) needs to tell that apart from "the
    tool was called but its input failed schema validation", and
    catching bare ``ValueError`` cannot distinguish the two.
    """


def parse_batch_tool_result(
    message_payload: Any, *, tool_name: str, schema: Optional[Any] = None,
) -> Any:
    """Parse one Batches-API result's ``message`` block for a named tool call.

    Accepts either an SDK ``Message`` object or its dict equivalent (a
    Lambda/worker consumer often streams the raw dict to keep
    dependencies minimal). Locates the ``tool_use`` block named
    ``tool_name`` and returns its ``input``.

    ``schema`` is optional: pass a Pydantic ``BaseModel`` subclass to
    get back a validated instance (``schema.model_validate(input)``);
    omit it (or pass a raw JSON-schema dict / ``None``) to get the raw
    ``input`` dict back unvalidated, leaving validation to the caller.

    Raises :exc:`ToolResultNotFoundError` if no matching ``tool_use``
    block is found — the judge LLM did not emit the structured output
    via the forced tool, which the caller should treat the same as any
    other terminal-parse-failure (the batch result is preserved on the
    provider's side for the caller to re-pull and diagnose). Lets
    ``schema.model_validate``'s own ``pydantic.ValidationError`` (a
    DIFFERENT failure mode — the tool WAS called, but its input didn't
    match the schema) propagate unwrapped so callers can distinguish
    the two.
    """
    content = (
        message_payload["content"]
        if isinstance(message_payload, dict)
        else message_payload.content
    )
    for block in content:
        block_type = (
            block.get("type") if isinstance(block, dict) else block.type
        )
        block_name = (
            block.get("name") if isinstance(block, dict)
            else getattr(block, "name", None)
        )
        if block_type == "tool_use" and block_name == tool_name:
            tool_input = (
                block["input"] if isinstance(block, dict) else block.input
            )
            if schema is not None and hasattr(schema, "model_validate"):
                return schema.model_validate(tool_input)
            return tool_input
    raise ToolResultNotFoundError(
        f"No tool_use block named {tool_name!r} found in batch result "
        f"message; the judge LLM did not emit the structured output via "
        f"the forced tool — inspect the raw batch result on the "
        f"provider's side."
    )


# ── OpenAI-transport (OpenRouter) structured-output leak guard ──────────
#
# ``build_structured_tool_spec`` / ``parse_batch_tool_result`` above are
# the Anthropic-Batches-API shape. A judge tier running on an
# OpenAI-compatible transport (OpenRouter, per nousergon/alpha-engine-
# config#2575 item 2/3) goes through :meth:`krepis.llm.LLMClient.structured`
# instead, which already retries on schema-validation failure but does
# NOT check ``finish_reason``/``tool_calls`` before parsing — unlike
# :meth:`krepis.llm.LLMClient.complete_grounded`, which ships that guard
# (``krepis#22``, 2026-07-14) for the grounded-search path only.
#
# Two DISTINCT live failure modes were confirmed against a real
# OpenRouter judge call while building this guard (config#2575, 2026-07-18,
# ``moonshotai/kimi-k2.6`` and generalized here to any OpenAI-transport
# judge model — the failure family is provider-agnostic, not
# Kimi-specific):
#
# 1. **Reasoning-budget truncation.** A reasoning-capable model spends its
#    entire ``max_tokens`` budget on internal chain-of-thought before ever
#    emitting the forced tool call: ``finish_reason="length"``,
#    ``message.content`` is ``null``/empty, ``message.tool_calls`` is
#    empty/absent. Passing this ``None``/empty payload into
#    ``schema.model_validate(...)`` (or a bare ``json.loads``) either
#    raises an opaque "expected dict, got NoneType" error indistinguishable
#    from an ordinary schema-validation retry, or — worse — a caller that
#    doesn't check first may treat the empty string as "nothing to
#    parse yet" and silently skip. This is the OpenAI-transport analogue of
#    the ``ModelSpec.reasoning`` empty-response gotcha documented on
#    :class:`~krepis.llm_config.ModelSpec`.
# 2. **Native tool-call token-dialect leak.** The model emits its own
#    control-token dialect (e.g. Kimi's
#    ``<|tool_calls_section_begin|>...<|tool_call_begin|>...``) as literal
#    text in ``message.content`` instead of (or alongside) a structured
#    ``tool_calls`` entry — the same class of leak
#    :func:`krepis.llm.LLMClient.complete_grounded` already guards against
#    for the grounded-search path (live incident 2026-07-14).
#
# Both are silent-failure risks for a JUDGE specifically because the
# retry loop that already exists around every judge call
# (``MAX_JUDGE_RETRIES`` in the consumer repo) treats ANY parse failure
# identically — it cannot tell "the model produced garbage JSON" (an
# ordinary stochastic non-conformance, ~20%/call, recoverable by
# resampling) apart from "the model never produced a parseable payload at
# all because of a structural transport issue" (not recoverable by
# resampling alone — needs a reasoning-exclude / budget bump). This guard
# lets a caller raise a DISTINCT, named exception for the second class so
# it can be logged/metriced separately from an ordinary retry — closing
# the "near-miss invisibility" gap noted in alpha-engine-config#2575's
# own issue thread (2026-07-15/16 comments) for the sibling
# Anthropic-Batches structural-parse-failure case.


class JudgeToolCallLeakError(ValueError):
    """An OpenAI-transport judge response failed to deliver a usable
    structured tool call — either truncated before the tool call (a
    reasoning-budget exhaustion) or the model's own tool-call token
    dialect leaked into ``content`` instead of a structured call.

    Subclass of :class:`ValueError` so existing broad ``except
    ValueError`` retry loops keep working unchanged, but a DISTINCT type
    from ordinary Pydantic ``ValidationError`` (itself a ``ValueError``
    subclass) — a caller that wants to count/alert on leak/truncation
    near-misses separately from ordinary stochastic schema
    non-conformance needs to tell the two apart. See the module-level
    comment above for the two confirmed failure shapes.
    """

    def __init__(self, message: str, *, reason: str, finish_reason: Optional[str]):
        super().__init__(message)
        self.reason = reason
        """Machine-stable failure-mode tag: ``"truncated_before_tool_call"``
        or ``"control_token_leak"`` — the two constants below."""
        self.finish_reason = finish_reason
        """The provider's raw ``finish_reason`` for this attempt, when
        available. Included so a caller's near-miss metric can carry it as
        a dimension without re-parsing the raw response."""


REASON_TRUNCATED_BEFORE_TOOL_CALL = "truncated_before_tool_call"
REASON_CONTROL_TOKEN_LEAK = "control_token_leak"

_JUDGE_CONTROL_TOKEN_RE = re.compile(r"<\|[a-zA-Z0-9_]{1,60}\|>")
"""Same pattern as ``krepis.llm._CONTROL_TOKEN_RE`` (kept as a separate
module-level compile rather than importing the private name across
modules — both mirror the Anthropic Batches API's
``^[a-zA-Z0-9_-]{1,64}$`` custom_id charset convention of "generalize the
class of token, not one vendor's exact spelling")."""


def check_openai_tool_response_for_leak(
    choice: Any, *, tool_name: str,
) -> None:
    """Raise :exc:`JudgeToolCallLeakError` if ``choice`` shows either
    known leak/truncation signature; return ``None`` (no raise) otherwise.

    ``choice`` is one ``response.choices[0]`` from an OpenAI-compatible
    chat-completions response (SDK object OR its dict equivalent — same
    duck-typed accept-both convention as :func:`parse_batch_tool_result`).

    Call this BEFORE attempting to parse ``message.tool_calls`` /
    ``message.content`` as the judge's structured output — it is a
    pre-parse gate, not a replacement for schema validation. A response
    that passes this check may still fail ordinary Pydantic validation
    (garbage-but-well-formed arguments); that is the existing, expected
    stochastic-non-conformance retry path and is NOT this guard's
    concern.

    Checks, in order:

    1. **Truncation before the tool call.** ``finish_reason == "length"``
       AND the message carries no ``tool_calls`` (or an empty list) —
       the model ran out of budget before emitting the forced call.
       Raises with ``reason=REASON_TRUNCATED_BEFORE_TOOL_CALL``.
    2. **Control-token leak.** ``message.content`` (when present) matches
       the ``<|...|>`` control-token pattern — the model's native
       tool-call dialect leaked into the text channel instead of (or in
       addition to) a structured call. Raises with
       ``reason=REASON_CONTROL_TOKEN_LEAK`` regardless of whether
       ``tool_calls`` is also present, since a leaking model's
       accompanying ``tool_calls`` block (if any) is not trustworthy
       either — live-confirmed pattern (2026-07-14 incident) is the leak
       co-occurring with a malformed or absent structured call, not a
       clean structured call plus stray text.

    Does not itself inspect ``tool_calls[*].function.name`` against
    ``tool_name`` — that's the caller's job once this pre-check passes
    (mirrors :func:`parse_batch_tool_result`'s tool-name matching for the
    Anthropic transport).
    """
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    finish_reason = (
        choice.get("finish_reason") if isinstance(choice, dict)
        else getattr(choice, "finish_reason", None)
    )
    tool_calls = (
        (message or {}).get("tool_calls") if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    content = (
        (message or {}).get("content") if isinstance(message, dict)
        else getattr(message, "content", None)
    ) or ""

    if finish_reason == "length" and not tool_calls:
        raise JudgeToolCallLeakError(
            f"tool={tool_name!r}: response truncated before emitting the "
            f"forced tool call (finish_reason='length', no tool_calls) — "
            f"almost certainly a reasoning-capable model exhausting its "
            f"token budget on internal chain-of-thought before the tool "
            f"call. Raise max_tokens and/or set ModelSpec.reasoning="
            f"{{'exclude': True}} (see krepis.llm_config.ModelSpec.reasoning "
            f"docstring) rather than treating this as an ordinary parse "
            f"retry.",
            reason=REASON_TRUNCATED_BEFORE_TOOL_CALL,
            finish_reason=finish_reason,
        )

    leak_match = _JUDGE_CONTROL_TOKEN_RE.search(content)
    if leak_match:
        raise JudgeToolCallLeakError(
            f"tool={tool_name!r}: judge response leaked raw control-token "
            f"syntax into content ({leak_match.group()!r}) — the model's "
            f"native tool-call dialect was not resolved into a structured "
            f"tool_calls entry by the gateway. Never parse `content` as "
            f"the judge's structured output unconditionally; this is a "
            f"transport/gateway failure, not a scoring input.",
            reason=REASON_CONTROL_TOKEN_LEAK,
            finish_reason=finish_reason,
        )


# ── Batch custom_id codec ─────────────────────────────────────────────────
#
# The Anthropic Batches API returns results keyed by an opaque
# ``custom_id`` (1-64 chars, ``^[a-zA-Z0-9_-]{1,64}$``). Callers encode a
# ``(subject_id, run_id, judge_model)`` triple into the custom_id on
# submission and decode it on the way out so a result can be persisted
# under the same identity it was submitted with, without depending on
# an in-flight plan manifest for correctness — the manifest (if a
# caller keeps one) is a convenience for ops visibility, not a
# load-bearing dependency for this codec.


_CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
"""Anthropic batch custom_id regex (per the Message Batches API docs)."""


def encode_custom_id(
    *,
    subject_id: str,
    run_id: str,
    judge_model: str,
    tag_by_logical: Mapping[str, str],
) -> str:
    """Encode ``(subject_id, run_id, judge_model)`` into a batch custom_id.

    ``tag_by_logical`` maps a judge model's logical key to its compact
    custom_id tag (caller-owned — e.g. derived from a tuple of
    :class:`JudgeModelSpec`). Replaces ``:``/``/`` separators with ``-``
    since the Anthropic custom_id charset only allows alphanumerics,
    ``-``, and ``_``. Truncates the ``subject_id`` segment if needed so
    the final string fits the 64-char ceiling. Round-trippable via
    :func:`decode_custom_id`.
    """
    tag = tag_by_logical.get(judge_model)
    if tag is None:
        # Unknown judge model — fall back to a hash-stable suffix so the
        # codec never raises on an unregistered model; the reverse
        # mapping just won't recover the original judge_model exactly.
        tag = f"x{abs(hash(judge_model)) % 10_000:04d}"
    safe_subject = re.sub(r"[^a-zA-Z0-9_-]", "-", subject_id)
    safe_run = re.sub(r"[^a-zA-Z0-9_-]", "-", run_id)
    # Reserve 4 chars for "__" separators + 3-char model tag.
    fixed_overhead = len(safe_run) + len(tag) + 4
    max_subject = max(8, 64 - fixed_overhead)
    if len(safe_subject) > max_subject:
        safe_subject = safe_subject[:max_subject]
    cid = f"{safe_subject}__{safe_run}__{tag}"
    if not _CUSTOM_ID_PATTERN.match(cid):
        # Last-ditch sanitize — strip anything that snuck through and
        # trim to the cap. The decode side just needs the model tag at
        # the tail; subject_id round-trip is best-effort once truncated.
        cid = re.sub(r"[^a-zA-Z0-9_-]", "-", cid)[:64]
    return cid


def decode_custom_id(
    custom_id: str, *, tag_by_logical: Mapping[str, str],
) -> tuple[str, str, str]:
    """Inverse of :func:`encode_custom_id`.

    Returns ``(subject_id, run_id, judge_model)``. The subject_id
    reconstruction maps ``-`` back to ``:`` only for the reverse tag
    lookup's judge_model segment; other ``-`` characters in the
    original subject_id would be lost if it was truncated on encode
    (acceptable — a caller with its own plan manifest carries the
    canonical subject_id and can override this best-effort decode).

    Raises ``ValueError`` if the custom_id doesn't match the expected
    triple-segment shape.
    """
    tag_reverse = {v: k for k, v in tag_by_logical.items()}
    parts = custom_id.split("__")
    if len(parts) != 3:
        raise ValueError(
            f"Cannot decode batch custom_id={custom_id!r}: expected "
            f"three '__'-separated segments, got {len(parts)}."
        )
    safe_subject, safe_run, tag = parts
    judge_model = tag_reverse.get(tag, tag)
    return safe_subject, safe_run, judge_model
