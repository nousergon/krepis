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
    "encode_custom_id",
    "decode_custom_id",
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

    Raises ``ValueError`` if no matching ``tool_use`` block is found —
    the judge LLM did not emit the structured output via the forced
    tool, which the caller should treat the same as any other
    terminal-parse-failure (the batch result is preserved on the
    provider's side for the caller to re-pull and diagnose).
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
    raise ValueError(
        f"No tool_use block named {tool_name!r} found in batch result "
        f"message; the judge LLM did not emit the structured output via "
        f"the forced tool — inspect the raw batch result on the "
        f"provider's side."
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
