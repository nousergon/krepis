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
from typing import Any, List, Optional

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
