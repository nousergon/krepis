"""SFT (supervised-fine-tuning) trace capture for :mod:`krepis.llm` calls.

Every production LLM call is potential distillation training data: the
rendered ``(prompt → completion)`` pair, the invocation params, usage,
and cost. This module captures that from an :class:`krepis.llm.LLMResult`
into the fleet's canonical **SFT v3 record envelope**, so rows emitted by
krepis consumers (morning-signal, vires, metron, mnemon, ...) are
directly ingestible by the same downstream curate/fine-tune/eval pipeline
as the Crucible producers.

**Schema provenance.** The v3 envelope is owned by ``nousergon_lib.sft``
(the Crucible fleet's chokepoint; provenance added in config#1539). That
library is AGPL and MIT products deliberately do not depend on it, so the
envelope — a cross-repo data CONTRACT, not strategy — is mirrored here
field-for-field. Any schema change must land in BOTH libraries in
lockstep; readers branch on ``schema_version``.

**Gating.** Capture is opt-in via environment flag:
``LLM_SFT_CAPTURE_ENABLED`` (neutral, product-facing name) or
``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` (the fleet-wide operator switch
the Crucible producers already honor) — either being truthy enables it.

**Sink.** Like :mod:`krepis.cost`, this module does the mapping; the
caller picks the sink. :func:`append_sft_jsonl` is the local-JSONL
convenience; S3 consumers put the same ``to_jsonl_bytes``-style payload
wherever their corpus prefix lives. Write failures RAISE
(:exc:`SftCaptureWriteError`) — capture an operator explicitly enabled
must not silently degrade (``feedback_no_silent_fails``); callers whose
product contract demands publish-over-capture wrap the call and log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union, cast

from krepis.llm_trace import (
    CAPTURE_FULL,
    EXTENSION_NAMESPACE,
    TRACE_SCHEMA_VERSION,
    TraceBlob,
    sha256_hex,
)

logger = logging.getLogger(__name__)

# Mirror of nousergon_lib.sft.SFT_SCHEMA_VERSION — see module docstring.
SFT_SCHEMA_VERSION = 3

CAPTURE_ENV_VARS = (
    "LLM_SFT_CAPTURE_ENABLED",
    "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED",
)

_TRUTHY = ("1", "true")


class SftCaptureWriteError(RuntimeError):
    """An enabled SFT capture failed to persist — loud, never swallowed."""


class SftProjectionError(ValueError):
    """A trace row that cannot be projected into an SFT record.

    Raised rather than returning a partial record: an SFT row with a missing
    prompt or completion is a wrong training example, not a smaller one.
    """


def capture_enabled() -> bool:
    """True when either capture flag is set truthy (``1`` / ``true``)."""
    return any(
        os.environ.get(var, "").lower() in _TRUTHY for var in CAPTURE_ENV_VARS
    )


def content_hash(input_messages: Any) -> str:
    """Stable SHA-256 over the model INPUT — the corpus dedup key.

    Byte-identical implementation to ``nousergon_lib.sft.content_hash``
    (canonicalized JSON, sorted keys, non-ASCII kept) so a krepis-captured
    trace and a fleet-captured replay of the same call collapse under the
    corpus ``dedup``.
    """
    canon = json.dumps(input_messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _jsonable(obj: Any) -> Any:
    """Best-effort JSON-friendly view of an SDK object.

    Pydantic SDK objects (anthropic / openai responses) expose
    ``model_dump``; plain containers pass through; anything else falls
    back to ``str`` at serialization time via ``default=str``.
    """
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # noqa: BLE001 — capture must not break the call
            return str(obj)
    return obj


def _normalized_input_messages(raw_request: dict) -> List[dict]:
    """The full rendered prompt as one message list.

    The anthropic wire format carries the system prompt in a separate
    ``system`` field; distillation needs the COMPLETE input, so it is
    normalized to a leading ``{"role": "system"}`` message. The openai
    wire format already includes it in ``messages``.
    """
    messages = list(raw_request.get("messages") or [])
    system = raw_request.get("system")
    if system:
        if isinstance(system, list):
            text = "\n\n".join(
                blk.get("text", "") if isinstance(blk, dict) else str(blk)
                for blk in system
            )
        else:
            text = str(system)
        messages = [{"role": "system", "content": text}] + messages
    return messages


def build_sft_record(
    result: Any,
    *,
    producer: str,
    meta: Optional[dict] = None,
    cost_usd: Optional[float] = None,
    call_seq: Optional[int] = None,
    source: str = "live",
    captured_at: Optional[str] = None,
) -> dict:
    """Map an :class:`krepis.llm.LLMResult` → canonical SFT v3 record dict.

    Parameters
    ----------
    result
        :class:`~krepis.llm.LLMResult` (or ``StructuredResult`` /
        ``GroundedResult`` — their extra payloads are captured too:
        ``data`` becomes ``structured_output``; grounded ``searches`` /
        ``citations`` land in ``meta``).
    producer
        Corpus producer identifier (e.g. ``"morning_signal"``,
        ``"vires_coach"``, ``"metron_advisor"``, ``"mnemon_judge"``).
        Must be non-empty.
    meta
        Producer-specific dimensions (edition, tenant, prompt version,
        ...). Merged with any grounded-search payload.
    cost_usd
        The priced cost the consumer already computed (typically the
        ``cost_usd`` field of :func:`krepis.cost.record_llm_call`'s
        record). Optional.
    source
        Provenance source: ``live`` (default) / ``replay`` / ``synthetic``.
    captured_at
        ISO-8601 override; defaults to now (UTC).
    """
    if not str(producer).strip():
        raise ValueError("producer must be a non-empty identifier")
    if source not in ("live", "replay", "synthetic"):
        raise ValueError(f"source must be live|replay|synthetic, got {source!r}")

    raw_request = getattr(result, "raw_request", None) or {}
    input_messages = _normalized_input_messages(raw_request)
    invocation_params = {
        k: _jsonable(v)
        for k, v in raw_request.items()
        if k not in ("messages", "system")
    }

    record_meta = dict(meta or {})
    record_meta.setdefault("provider", getattr(result, "provider", None))
    searches = getattr(result, "searches", None)
    citations = getattr(result, "citations", None)
    if searches:
        record_meta["searches"] = searches
    if citations:
        record_meta["citations"] = citations

    usage = getattr(result, "usage", None)
    usage_dict = asdict(usage) if usage is not None else None

    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "producer": producer,
        "captured_at": captured_at
        or datetime.now(timezone.utc).isoformat(),
        "model": getattr(result, "model", None),
        "call_seq": call_seq,
        "input_messages": input_messages,
        "invocation_params": invocation_params,
        "output_message": _jsonable(getattr(result, "raw_response", None)),
        "output_text": getattr(result, "text", None),
        "structured_output": getattr(result, "data", None),
        "usage": usage_dict,
        "cost_usd": cost_usd,
        "meta": record_meta,
        "provenance": {
            "source": source,
            "content_hash": content_hash(input_messages)
            if input_messages
            else None,
        },
    }


def append_sft_jsonl(path: Any, records: List[dict]) -> int:
    """Append records to a local JSONL sink (parents created). Returns the
    number written. Raises :exc:`SftCaptureWriteError` on failure."""
    if not records:
        return 0
    out_path = Path(path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
    except OSError as exc:
        raise SftCaptureWriteError(
            f"SFT capture append failed for {out_path}: {exc}"
        ) from exc
    return len(records)


def capture_llm_call(
    result: Any,
    *,
    producer: str,
    sink_path: Any,
    meta: Optional[dict] = None,
    cost_usd: Optional[float] = None,
    call_seq: Optional[int] = None,
    source: str = "live",
) -> bool:
    """One-call capture: gate → build → append.

    Returns ``False`` (no-op) when neither capture flag is enabled;
    ``True`` after a successful append. Persist failures raise
    :exc:`SftCaptureWriteError` — an operator who enabled capture gets a
    loud failure, not silently-missing training data.
    """
    if not capture_enabled():
        return False
    record = build_sft_record(
        result,
        producer=producer,
        meta=meta,
        cost_usd=cost_usd,
        call_seq=call_seq,
        source=source,
    )
    append_sft_jsonl(sink_path, [record])
    return True


# ── projection from an llm_trace.v1 row ────────────────────────────────────

#: The ``LLMUsage`` fields an ``llm_trace.v1`` row carries, and where. The
#: trace does not keep ``prompt_cache_miss_tokens``,
#: ``reasoning_tokens_max_attempt``, ``web_fetch_requests`` or
#: ``provider_cost_usd``, so a projected ``usage`` omits them rather than
#: reporting a zero the trace never measured.
_TRACE_USAGE_GEN_AI = (
    ("input_tokens", "usage.input_tokens"),
    ("output_tokens", "usage.output_tokens"),
    ("cache_read_tokens", "usage.cache_read.input_tokens"),
    ("cache_create_tokens", "usage.cache_creation.input_tokens"),
)
_TRACE_USAGE_EXTENSION = (
    "reasoning_tokens",
    "budget_escalations",
    "attempts",
    "web_search_requests",
    "usage_unknown",
)


def _as_mapping(value: object) -> Mapping[str, object]:
    """``value`` as a string-keyed mapping, or empty when it is not one."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return {}


def _blob_json(
    blobs: Mapping[str, Union[TraceBlob, bytes]], sha: object, what: str
) -> Mapping[str, object]:
    """Load the JSON object stored at ``sha``, checking it is that address."""
    if not isinstance(sha, str) or not sha:
        raise SftProjectionError(f"trace row has no {what} hash")
    blob = blobs.get(sha)
    if blob is None:
        raise SftProjectionError(f"{what} blob {sha} is not in the blobs given")
    data = blob.data if isinstance(blob, TraceBlob) else bytes(blob)
    if sha256_hex(data) != sha:
        raise SftProjectionError(
            f"{what} blob does not hash to its address {sha}"
        )
    try:
        loaded: object = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SftProjectionError(f"{what} blob {sha} is not JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SftProjectionError(f"{what} blob {sha} is not a JSON object")
    return cast("dict[str, object]", loaded)


def sft_record_from_trace(
    row: Mapping[str, object],
    blobs: Mapping[str, Union[TraceBlob, bytes]],
    *,
    producer: str,
    meta: Optional[Mapping[str, object]] = None,
    cost_usd: Optional[float] = None,
    call_seq: Optional[int] = None,
    source: str = "live",
    captured_at: Optional[str] = None,
) -> Dict[str, object]:
    """Project one ``llm_trace.v1`` row (plus its blobs) → SFT v3 record dict.

    This is the trace-first path to the same envelope :func:`build_sft_record`
    builds from a live result (alpha-engine-config-I12312, audit §6.2: "the
    existing ``build_sft_record`` becomes a projection of the trace record"),
    so a distillation corpus can be cut from ``llm_traces/v1/`` after the fact
    instead of needing its own capture flag at call time.

    Parameters
    ----------
    row
        One trace row as :func:`krepis.llm_trace.build_trace_record` builds it
        (or as read back from ``llm_traces/v1/rows/``).
    blobs
        sha256 → body, as :class:`~krepis.llm_trace.TraceBlob` or the
        decompressed bytes read from ``llm_traces/v1/blobs/``. Each body is
        checked against its address before it is used.
    producer, meta, cost_usd, call_seq, source
        As for :func:`build_sft_record`.
    captured_at
        ISO-8601 override; defaults to the row's ``ended_at``, so projecting
        the same row twice yields the same record.

    **Field by field.** ``input_messages`` and ``invocation_params`` come from
    the input blob (the exact request the transport sent) by the same
    normalization :func:`build_sft_record` applies, so their
    ``content_hash`` matches a live capture of the same call. ``output_text``
    and ``structured_output`` come from the output blob. ``output_message`` is
    that output body (``text``, plus ``data`` / ``web_citations`` when
    present): the trace keeps no raw provider response, and the envelope types
    ``output_message`` as producer-serialized opaque JSON. ``usage`` carries
    only the counters the trace records (see ``_TRACE_USAGE_GEN_AI``).
    ``meta.trace`` names the trace and span, so a projected row joins back to
    the trace it came from.

    **Refused, loudly** (:exc:`SftProjectionError`): a row that is not
    ``llm_trace.v1``; a row whose content capture is not ``full`` — a
    ``hash_only`` row promised no body, and none is reconstructed from
    anything else; a call that raised (it has no completion); and a body that
    is missing, not JSON, or does not hash to its address.
    """
    if not str(producer).strip():
        raise ValueError("producer must be a non-empty identifier")
    if source not in ("live", "replay", "synthetic"):
        raise ValueError(f"source must be live|replay|synthetic, got {source!r}")
    if row.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise SftProjectionError(
            f"not an {TRACE_SCHEMA_VERSION} row: {row.get('schema_version')!r}"
        )
    if row.get("error"):
        raise SftProjectionError("the traced call raised; it has no completion")
    ext = _as_mapping(row.get(EXTENSION_NAMESPACE))
    content = _as_mapping(ext.get("content"))
    if content.get("capture") != CAPTURE_FULL:
        raise SftProjectionError(
            f"content capture is {content.get('capture')!r}, not "
            f"{CAPTURE_FULL!r}: the row carries no bodies to project"
        )

    raw_request = _blob_json(blobs, content.get("input_sha256"), "input")
    output_body = _blob_json(blobs, content.get("output_sha256"), "output")

    input_messages = cast(
        "List[Dict[str, object]]", _normalized_input_messages(dict(raw_request))
    )
    invocation_params = {
        k: _jsonable(v)
        for k, v in raw_request.items()
        if k not in ("messages", "system")
    }

    gen_ai = _as_mapping(row.get("gen_ai"))
    usage_ext = _as_mapping(ext.get("usage"))
    usage: Dict[str, object] = {
        name: gen_ai.get(key) for name, key in _TRACE_USAGE_GEN_AI
    }
    usage.update({name: usage_ext.get(name) for name in _TRACE_USAGE_EXTENSION})

    record_meta: Dict[str, object] = dict(meta or {})
    record_meta.setdefault("provider", gen_ai.get("provider.name"))
    web = output_body.get("web_citations")
    if web:
        record_meta["citations"] = web
    record_meta["trace"] = {
        "trace_id": row.get("trace_id"),
        "span_id": row.get("span_id"),
        "callsite_id": ext.get("callsite_id"),
        "run_id": ext.get("run_id"),
    }

    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "producer": producer,
        "captured_at": captured_at or row.get("ended_at"),
        "model": gen_ai.get("response.model"),
        "call_seq": call_seq,
        "input_messages": input_messages,
        "invocation_params": invocation_params,
        "output_message": dict(output_body),
        "output_text": output_body.get("text"),
        "structured_output": output_body.get("data"),
        "usage": usage,
        "cost_usd": cost_usd,
        "meta": record_meta,
        "provenance": {
            "source": source,
            "content_hash": content_hash(input_messages)
            if input_messages
            else None,
        },
    }
