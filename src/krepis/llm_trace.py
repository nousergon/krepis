"""The per-call LLM trace record (``llm_trace.v1``) and its source citations.

**Why this exists.** :mod:`krepis.cost` gives every call a *cost* record:
tokens, model and USD. That answers "what did it cost". It cannot answer
"what did the model see, what did it say, and which sources did it rely on",
and an evaluation needs exactly those three things. This module builds the
record that can: one JSONL row per logical call, carrying

- OpenTelemetry GenAI semantic-convention attribute names (``gen_ai.*``), so
  an OTLP exporter can be added later without touching a single caller;
- W3C trace-context ids (32-hex ``trace_id``, 16-hex ``span_id``), so a run
  that makes many calls reconstructs as one trace;
- latency, the provider response id and the route facts the adapter
  already knows (served deployment, fallback, attempts, streaming);
- content as **sha256 references** to content-addressed blobs (the exact
  request that was sent, and the response), never inline;
- ``citations[]``: every source injected into the prompt, each with a stable
  source id, the sha256 of the exact injected text, and its character
  offsets in the rendered prompt.

**Citations are computed, not inferred.** :class:`SourceBlock` and
:func:`render_sources` put sources into the prompt with stable delimiters
and record each source's ``content_sha256`` and offsets *while rendering*.
Whether the output cited a source is a deterministic parse of ``[cid]``
markers (:func:`cited_cids`), never a model judgment. Two runs over the same
inputs produce the same citation records byte for byte.

**Nothing fleet-specific is baked in.** Where the rows go is
:mod:`krepis.trace_sink`'s business, configured by the consumer.

The builder is pure: given the same inputs it returns the same row. The
non-deterministic inputs (clock, span id) are parameters, which is what lets
the golden-record tests pin the shape exactly.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from krepis.llm_config import ROUTER_EDGE_PROVIDER

logger = logging.getLogger(__name__)

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "EXTENSION_NAMESPACE",
    "row_extension",
    "TRACE_ID_ENV_VAR",
    "TRACEPARENT_ENV_VAR",
    "CAPTURE_ENV_VAR",
    "CAPTURE_FULL",
    "CAPTURE_HASH_ONLY",
    "SOURCE_KINDS",
    "CITATION_INSTRUCTION",
    "TraceBlob",
    "TraceContext",
    "SourceBlock",
    "RenderedSource",
    "RenderedSources",
    "render_sources",
    "bind_sources",
    "cited_cids",
    "canonical_json_bytes",
    "sha256_hex",
    "new_span_id",
    "derive_trace_id",
    "resolve_trace_context",
    "resolve_capture_mode",
    "build_trace_record",
    "isoformat_utc",
    "emission_failures",
    "record_emission_failure",
]

#: Version tag carried on every row. Readers dispatch on it; a change to the
#: row's meaning is a new version, never an edit to this one.
TRACE_SCHEMA_VERSION = "llm_trace.v1"

#: The namespace every fleet-specific field lives under, so nothing outside
#: OpenTelemetry's own vocabulary sits beside ``gen_ai.*``.
EXTENSION_NAMESPACE = "krepis"

#: An orchestrator that already has a W3C trace id exports it here, so every
#: call in its run joins the same trace.
TRACE_ID_ENV_VAR = "KREPIS_TRACE_ID"
#: The W3C ``traceparent`` carried as an environment variable (the
#: OpenTelemetry environment-carrier convention). Wins over
#: :data:`TRACE_ID_ENV_VAR` because it also names the parent span.
TRACEPARENT_ENV_VAR = "TRACEPARENT"
#: ``full`` (default) or ``hash_only``. ``hash_only`` keeps every hash and
#: every citation record but writes no content blob — for a consumer whose
#: content may not be retained (a multi-tenant product without per-tenant
#: consent, for example).
CAPTURE_ENV_VAR = "KREPIS_TRACE_CAPTURE"

CAPTURE_FULL = "full"
CAPTURE_HASH_ONLY = "hash_only"
_CAPTURE_MODES = frozenset({CAPTURE_FULL, CAPTURE_HASH_ONLY})

#: What a cited source IS. Closed on purpose: a free-text kind is a join key
#: nobody can rely on.
SOURCE_KINDS = frozenset(
    {"rag_chunk", "web", "s3_object", "artifact", "metric", "doc"}
)

#: An instruction a caller MAY append to its prompt so the model's citations
#: come back in the form :func:`cited_cids` parses. Not injected
#: automatically: the prompt belongs to the caller.
CITATION_INSTRUCTION = (
    "When a statement relies on a source, cite it inline by its cid in "
    "square brackets, for example [c1] or [c1, c3]."
)

#: The normalized message list a span indexes into is ``[system, user]`` —
#: the OpenTelemetry ``gen_ai.input.messages`` order with the system
#: instructions first. Sources are always rendered into the user turn.
USER_MESSAGE_INDEX = 1

_CID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_CITE_MARKER_RE = re.compile(
    r"\[\s*([A-Za-z0-9_.\-]{1,64}(?:\s*,\s*[A-Za-z0-9_.\-]{1,64})*)\s*\]"
)
_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_TRACEPARENT_RE = re.compile(
    r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)

# ── emission-failure counter ───────────────────────────────────────────────

_FAILURES = 0
_FAILURES_LOCK = threading.Lock()


def record_emission_failure() -> int:
    """Count one trace-emission failure. Returns the new process total."""
    global _FAILURES
    with _FAILURES_LOCK:
        _FAILURES += 1
        return _FAILURES


def emission_failures() -> int:
    """Trace records this process failed to build or hand to its sink.

    The counter a caller (or a run-end summary) asserts on. Emission never
    raises into the LLM call, so this, the ERROR log line and
    ``LLMResult.trace_emission_error`` are the surfaces a loss shows up on.
    """
    with _FAILURES_LOCK:
        return _FAILURES


# ── hashing and ids ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TraceBlob:
    """One content-addressed body: the bytes whose sha256 is its address."""

    data: bytes
    media_type: str


def canonical_json_bytes(obj: Any) -> bytes:
    """The one serialization every content hash is taken over.

    Sorted keys, no insignificant whitespace, UTF-8 without ASCII escaping.
    Non-JSON values fall back to ``str()`` so a request carrying an SDK
    object still hashes, deterministically, rather than failing the trace.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def sha256_hex(data: Any) -> str:
    """sha256 of ``data`` (``str`` is UTF-8 encoded), as lowercase hex."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def new_span_id(rand: Any = os.urandom) -> str:
    """A fresh W3C span id: 16 lowercase hex, never all zeros."""
    while True:
        value = rand(8).hex()
        if value != "0" * 16:
            return value


def derive_trace_id(run_id: str) -> str:
    """A W3C trace id derived from ``run_id``, stable across processes.

    Every call in a run therefore shares one trace id without anyone having
    to propagate it — the run id is already exported to every stage
    (``KREPIS_RUN_ID``), and deriving rather than minting is what keeps two
    processes of the same run on the same trace.
    """
    value = hashlib.sha256(f"krepis-trace:{run_id}".encode("utf-8")).hexdigest()[:32]
    return value if value != "0" * 32 else "0" * 31 + "1"


@dataclass(frozen=True)
class TraceContext:
    """Where one call sits: its trace, its parent span, and its run."""

    trace_id: str
    parent_span_id: Optional[str] = None
    run_id: Optional[str] = None
    #: How ``trace_id`` was obtained: ``traceparent``, ``env``, ``run_id``
    #: or ``random``. Recorded so a reader can tell a deliberately
    #: propagated trace from a per-call one.
    source: str = "random"


def resolve_trace_context(
    run_id: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    rand: Any = os.urandom,
) -> TraceContext:
    """Resolve the trace context for one call.

    Order: a valid ``TRACEPARENT`` (trace id and parent span); a valid
    ``KREPIS_TRACE_ID``; a trace id derived from ``run_id``; a random one.
    A malformed value is logged and ignored, as the W3C specification
    requires of an invalid ``traceparent`` — it never fails the call.
    """
    env = os.environ if environ is None else environ
    traceparent = (env.get(TRACEPARENT_ENV_VAR) or "").strip().lower()
    if traceparent:
        match = _TRACEPARENT_RE.match(traceparent)
        if (
            match
            and match.group(1) != "ff"
            and match.group(2) != "0" * 32
            and match.group(3) != "0" * 16
        ):
            return TraceContext(
                trace_id=match.group(2),
                parent_span_id=match.group(3),
                run_id=run_id,
                source="traceparent",
            )
        logger.warning(
            "llm trace: ignoring malformed %s=%r (W3C traceparent expected)",
            TRACEPARENT_ENV_VAR, traceparent,
        )
    explicit = (env.get(TRACE_ID_ENV_VAR) or "").strip().lower()
    if explicit:
        if _HEX32_RE.match(explicit) and explicit != "0" * 32:
            return TraceContext(trace_id=explicit, run_id=run_id, source="env")
        logger.warning(
            "llm trace: ignoring malformed %s=%r (32 lowercase hex expected)",
            TRACE_ID_ENV_VAR, explicit,
        )
    if run_id:
        return TraceContext(
            trace_id=derive_trace_id(run_id), run_id=run_id, source="run_id"
        )
    while True:
        value = rand(16).hex()
        if value != "0" * 32:
            return TraceContext(trace_id=value, run_id=None, source="random")


def resolve_capture_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """The process capture mode from :data:`CAPTURE_ENV_VAR`.

    An unknown value resolves to ``hash_only`` — the conservative reading,
    because the variable exists to restrict what is retained, and guessing
    ``full`` on a typo would retain exactly what someone tried to exclude.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(CAPTURE_ENV_VAR) or "").strip().lower()
    if not raw:
        return CAPTURE_FULL
    if raw in _CAPTURE_MODES:
        return raw
    logger.warning(
        "llm trace: unknown %s=%r — using %r (the restrictive mode)",
        CAPTURE_ENV_VAR, raw, CAPTURE_HASH_ONLY,
    )
    return CAPTURE_HASH_ONLY


def isoformat_utc(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SS.mmmZ`` — one timestamp format on every row."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


# ── sources ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SourceBlock:
    """One source a caller injects into a prompt.

    Args:
        cid: The citation id the model sees and cites (``c1``). Unique per
            call; letters, digits, ``_ . -``; at most 64 characters.
        kind: One of :data:`SOURCE_KINDS`.
        source_id: The stable identity of the source in its own store — a
            chunk id, an S3 key, a URL. This is the join key evals use.
        text: The exact text injected. Its sha256 is recorded, so pass what
            the model will read, not a longer original.
        document_id: The parent document, when the source is a part of one.
        uri: Where a human can find it.
        version: The source's version (an S3 VersionId, a filing date).
        retrieval: How it was retrieved — ``method``, ``rank``, ``score``,
            ``rerank_score``, ``query_sha256``, ``retrieved_at``.
        meta: Anything else worth keeping. Must be JSON-serializable.
    """

    cid: str
    kind: str
    source_id: str
    text: str
    document_id: Optional[str] = None
    uri: Optional[str] = None
    version: Optional[str] = None
    retrieval: Optional[Mapping[str, Any]] = None
    meta: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.cid, str) or not _CID_RE.match(self.cid):
            raise ValueError(
                f"SourceBlock.cid must match {_CID_RE.pattern}; got {self.cid!r}"
            )
        if self.kind not in SOURCE_KINDS:
            raise ValueError(
                f"SourceBlock.kind must be one of {sorted(SOURCE_KINDS)}; "
                f"got {self.kind!r}"
            )
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("SourceBlock.source_id must be a non-empty string")
        if not isinstance(self.text, str):
            raise ValueError("SourceBlock.text must be a string")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceBlock":
        """Build from a plain mapping — the shape a library without a krepis
        dependency can return (``RetrievalResult.as_source()``, say)."""
        known = {
            "cid", "kind", "source_id", "text", "document_id", "uri",
            "version", "retrieval", "meta",
        }
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"SourceBlock.from_mapping: unknown keys {unknown}")
        return cls(**dict(value))


@dataclass(frozen=True)
class RenderedSource:
    """One source as rendered: its block and where its text landed."""

    block: SourceBlock
    content_sha256: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class RenderedSources:
    """The rendered source section and each source's offsets within it."""

    text: str
    sources: Tuple[RenderedSource, ...] = field(default_factory=tuple)


def _open_tag(block: SourceBlock) -> str:
    return f'<source cid="{html.escape(block.cid, quote=True)}">\n'


_CLOSE_TAG = "\n</source>"


def render_sources(
    blocks: Iterable[Any], *, separator: str = "\n\n"
) -> RenderedSources:
    """Render sources into one prompt section, recording hashes and offsets.

    Each source becomes ``<source cid="c1">\\n{text}\\n</source>``. The text
    is inserted unchanged — its sha256 is of exactly these characters — and
    its ``[char_start, char_end)`` offsets are Python string indices
    (Unicode code points) into :attr:`RenderedSources.text`.

    Accepts :class:`SourceBlock` or mappings (see
    :meth:`SourceBlock.from_mapping`). A repeated ``cid`` raises, because two
    sources answering to one citation id make every citation of it ambiguous.
    """
    parts: List[str] = []
    rendered: List[RenderedSource] = []
    seen: set = set()
    cursor = 0
    for index, raw in enumerate(blocks):
        block = raw if isinstance(raw, SourceBlock) else SourceBlock.from_mapping(raw)
        if block.cid in seen:
            raise ValueError(f"render_sources: duplicate cid {block.cid!r}")
        seen.add(block.cid)
        if index:
            parts.append(separator)
            cursor += len(separator)
        head = _open_tag(block)
        start = cursor + len(head)
        end = start + len(block.text)
        parts.append(head + block.text + _CLOSE_TAG)
        cursor = end + len(_CLOSE_TAG)
        rendered.append(
            RenderedSource(
                block=block,
                content_sha256=sha256_hex(block.text),
                char_start=start,
                char_end=end,
            )
        )
    return RenderedSources(text="".join(parts), sources=tuple(rendered))


def bind_sources(
    user_content: str, sources: Any
) -> Tuple[str, Tuple[RenderedSource, ...]]:
    """Place ``sources`` in ``user_content`` and return absolute offsets.

    - ``None`` or empty: ``user_content`` unchanged, no sources.
    - A sequence of :class:`SourceBlock` (or mappings): rendered with
      :func:`render_sources` and PREPENDED to ``user_content``, separated by
      a blank line.
    - A :class:`RenderedSources` the caller already embedded in its own
      template: located in ``user_content``. It must occur exactly once —
      zero or several occurrences raise ``ValueError`` before any spend,
      because an offset into the wrong place is a citation to the wrong text.

    Offsets in the result are relative to the returned user content.
    """
    if sources is None:
        return user_content, ()
    if isinstance(sources, RenderedSources):
        if not sources.sources:
            return user_content, ()
        count = user_content.count(sources.text)
        if count != 1:
            raise ValueError(
                f"sources: the rendered source section occurs {count} times in "
                f"user_content; it must occur exactly once for its offsets to "
                f"be exact. Embed RenderedSources.text in the prompt verbatim, "
                f"once."
            )
        base = user_content.index(sources.text)
        return user_content, tuple(
            RenderedSource(
                block=s.block,
                content_sha256=s.content_sha256,
                char_start=base + s.char_start,
                char_end=base + s.char_end,
            )
            for s in sources.sources
        )
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise TypeError(
            "sources must be None, a sequence of SourceBlock, or RenderedSources"
        )
    if not sources:
        return user_content, ()
    rendered = render_sources(sources)
    prefix = rendered.text + "\n\n"
    return prefix + user_content, rendered.sources


def cited_cids(text: str, known: Iterable[str]) -> List[str]:
    """The ``known`` cids that ``text`` cites with ``[cid]`` markers.

    Deterministic and order-preserving (first citation first). A marker
    naming an unknown cid is ignored: ``[1]`` in a numbered list is not a
    citation of anything this call injected.
    """
    known_set = set(known)
    found: List[str] = []
    for match in _CITE_MARKER_RE.finditer(text or ""):
        for cid in match.group(1).split(","):
            cid = cid.strip()
            if cid in known_set and cid not in found:
                found.append(cid)
    return found


# ── the record ─────────────────────────────────────────────────────────────


def _usage_fields(usage: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def _get(name: str) -> int:
        try:
            return int(getattr(usage, name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    gen_ai = {
        "usage.input_tokens": _get("input_tokens"),
        "usage.output_tokens": _get("output_tokens"),
        "usage.cache_read.input_tokens": _get("cache_read_tokens"),
        "usage.cache_creation.input_tokens": _get("cache_create_tokens"),
    }
    extension = {
        "reasoning_tokens": _get("reasoning_tokens"),
        "attempts": _get("attempts"),
        "budget_escalations": _get("budget_escalations"),
        "web_search_requests": _get("web_search_requests"),
        "usage_unknown": bool(getattr(usage, "usage_unknown", False)),
    }
    return gen_ai, extension


def _output_body(result: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {"text": getattr(result, "text", "")}
    data = getattr(result, "data", None)
    if data is not None and hasattr(result, "structured_output_rung"):
        body["data"] = data
    web = getattr(result, "citations", None)
    if web:
        body["web_citations"] = [dict(c) for c in web]
    return body


def _source_entry(
    rendered: RenderedSource, cited: Iterable[str]
) -> Dict[str, Any]:
    block = rendered.block
    return {
        "cid": block.cid,
        "kind": block.kind,
        "source_id": block.source_id,
        "document_id": block.document_id,
        "uri": block.uri,
        "version": block.version,
        "content_sha256": rendered.content_sha256,
        "span": {
            "message_index": USER_MESSAGE_INDEX,
            "char_start": rendered.char_start,
            "char_end": rendered.char_end,
        },
        "retrieval": dict(block.retrieval) if block.retrieval else None,
        "meta": dict(block.meta) if block.meta else None,
        "cited_in_output": block.cid in set(cited),
    }


def build_trace_record(
    result: Any,
    *,
    callsite_id: str,
    context: TraceContext,
    span_id: str,
    started_at: datetime,
    ended_at: datetime,
    duration_ms: float,
    request_model: str,
    sources: Sequence[RenderedSource] = (),
    system: Optional[str] = None,
    subject: Optional[Mapping[str, Any]] = None,
    prompt_id: Optional[str] = None,
    prompt_version: Optional[str] = None,
    capture: str = CAPTURE_FULL,
    capture_reason: Optional[str] = None,
    library_version: Optional[str] = None,
    error: Optional[BaseException] = None,
) -> Tuple[Dict[str, Any], Dict[str, TraceBlob]]:
    """Build one ``llm_trace.v1`` row and the blobs it references.

    Pure: every non-deterministic input (clock, span id) is a parameter.

    **Shape** (llm-platform-policy § Call trace record). The top level holds
    what OpenTelemetry and W3C trace context define: ``trace_id``,
    ``span_id``, ``parent_span_id``, timing, ``gen_ai`` (GenAI semantic
    convention attribute names, verbatim) and ``error`` (``error.type``).
    Everything fleet-specific — call site, run, subject, route facts, content
    hashes, citations, DLP — sits under the ``krepis`` extension namespace
    and never renames a ``gen_ai.*`` field.

    **A call that raised is traced too** (``error`` set): ``result`` is then
    a stand-in carrying the request that reached the transport, if any, and
    there is no output. ``error.message`` is omitted whenever the DLP scan
    blocked or errored, so a matched value can never reach a trace, and
    whenever the record is ``hash_only`` (declared, or downgraded from
    ``full``): an exception's text can quote the model's output or the
    prompt (a structured-output ``LLMError`` names the invalid payload, a
    provider 400 can echo the request), and ``hash_only`` promises no body
    text of either. ``error.type`` is always kept.

    **Content is the post-DLP body.** ``raw_request`` is the exact payload
    :meth:`krepis.llm.LLMClient._dlp_scan_request` scanned and the transport
    sent; ``krepis.session_dlp`` blocks a body, it never rewrites one, so
    ``dlp.redactions`` from krepis is always empty. A redaction made further
    along the wire (a localhost egress proxy) is invisible here and is
    recorded as ``proxy_redaction: "unknown"`` rather than assumed absent.

    **Bodies are kept only for a body that was scanned and passed.** A
    result whose ``dlp_verdict`` is anything but ``"ok"`` (DLP disabled, or
    a path that did not scan) is downgraded to ``hash_only``: the hashes and
    citation records stay, no body is written. Fail-closed for content,
    never for the call.

    Returns ``(row, blobs)`` where ``blobs`` maps sha256 → :class:`TraceBlob`
    and is empty under ``hash_only``.
    """
    if not _HEX16_RE.match(span_id or ""):
        raise ValueError(f"span_id must be 16 lowercase hex; got {span_id!r}")
    if capture not in _CAPTURE_MODES:
        raise ValueError(f"capture must be one of {sorted(_CAPTURE_MODES)}")

    dlp_verdict = getattr(result, "dlp_verdict", None)
    if capture == CAPTURE_FULL and dlp_verdict != "ok":
        capture = CAPTURE_HASH_ONLY
        capture_reason = f"dlp_verdict={dlp_verdict}"
    elif capture == CAPTURE_HASH_ONLY and not capture_reason:
        capture_reason = "configured"

    raw_request = getattr(result, "raw_request", None) or {}
    request_bytes = canonical_json_bytes(raw_request)
    # A call that raised before reaching the transport (a DLP block, a
    # config error) sent nothing; it has no input to hash.
    input_sha = sha256_hex(request_bytes) if (raw_request or error is None) else None
    output_bytes = canonical_json_bytes(_output_body(result)) if error is None else None
    output_sha = sha256_hex(output_bytes) if output_bytes is not None else None
    system_sha = sha256_hex(system) if isinstance(system, str) else None

    blobs: Dict[str, TraceBlob] = {}
    if capture == CAPTURE_FULL:
        if input_sha is not None:
            blobs[input_sha] = TraceBlob(request_bytes, "application/json")
        if output_sha is not None and output_bytes is not None:
            blobs[output_sha] = TraceBlob(output_bytes, "application/json")
        for rendered in sources:
            blobs[rendered.content_sha256] = TraceBlob(
                rendered.block.text.encode("utf-8"), "text/plain; charset=utf-8"
            )

    known = [s.block.cid for s in sources]
    scan_text = getattr(result, "text", "") or ""
    data = getattr(result, "data", None)
    if data:
        scan_text += "\n" + canonical_json_bytes(data).decode("utf-8")
    cited = cited_cids(scan_text, known)
    citations: List[Dict[str, Any]] = [_source_entry(s, cited) for s in sources]

    # Web citations a grounded call RETURNED. They were not injected, so they
    # carry no span; the snippet is what is hashed (and kept, under `full`).
    for index, web in enumerate(getattr(result, "citations", None) or [], start=1):
        url = (web or {}).get("url") or ""
        snippet = (web or {}).get("snippet")
        snippet_sha = sha256_hex(snippet) if isinstance(snippet, str) else None
        if isinstance(snippet, str) and snippet_sha and capture == CAPTURE_FULL:
            blobs[snippet_sha] = TraceBlob(
                snippet.encode("utf-8"), "text/plain; charset=utf-8"
            )
        citations.append(
            {
                "cid": f"web{index}",
                "kind": "web",
                "source_id": url,
                "document_id": None,
                "uri": url or None,
                "version": None,
                "content_sha256": snippet_sha,
                "span": None,
                "retrieval": {"method": "web_search"},
                "meta": {"title": (web or {}).get("title")},
                "cited_in_output": True,
            }
        )

    gen_usage, usage_ext = _usage_fields(getattr(result, "usage", None))
    finish = getattr(result, "finish_reason", None)
    max_tokens = raw_request.get("max_tokens") if isinstance(raw_request, dict) else None
    response_id = getattr(result, "response_id", None)
    provider = getattr(result, "provider", None)
    gen_ai: Dict[str, Any] = {
        "operation.name": "chat",
        "provider.name": provider,
        "request.model": request_model,
        "request.max_tokens": max_tokens,
        "response.model": getattr(result, "model", None),
        "response.id": response_id,
        "response.finish_reasons": [finish] if finish else [],
    }
    gen_ai.update(gen_usage)

    error_entry = None
    if error is not None:
        message: Optional[str] = str(error)[:500]
        if dlp_verdict in ("dlp_block", "scan_error"):
            message = None  # a DLP reason may quote what it matched
        elif capture == CAPTURE_HASH_ONLY:
            # An exception's text can quote the output or the prompt
            # (alpha-engine-config-I12312, found building metron-ops-PR367);
            # hash_only writes no body text, so it writes no message either.
            message = None
        error_entry = {"type": type(error).__name__, "message": message}

    extension: Dict[str, Any] = {
        "callsite_id": callsite_id,
        "run_id": context.run_id,
        "trace_id_source": context.source,
        "subject": dict(subject) if subject else None,
        "usage": usage_ext,
        "route": {
            "registry_id": getattr(result, "registry_id", None),
            "served_deployment": getattr(result, "served_deployment", None),
            "served_provider": getattr(result, "served_provider", None),
            "fallback_used": bool(getattr(result, "fallback_used", False)),
            # LiteLLM's spend log keys its row on the response id, so on the
            # router edge this IS the join to the edge's spend export.
            "edge_request_id": (
                response_id if provider == ROUTER_EDGE_PROVIDER else None
            ),
            "streamed": bool(getattr(result, "streamed", False)),
            "stream_chunks": int(getattr(result, "stream_chunks", 0) or 0),
            "dropped_params": list(getattr(result, "dropped_params", None) or []),
            "structured_output_rung": getattr(result, "structured_output_rung", None),
        },
        "prompt": {
            "template_id": prompt_id,
            "template_version": prompt_version,
            "system_sha256": system_sha,
        },
        "content": {
            "capture": capture,
            "capture_reason": capture_reason,
            "input_sha256": input_sha,
            "output_sha256": output_sha,
            "media_type": "application/json",
        },
        "citations": citations,
        "dlp": {
            "verdict": dlp_verdict,
            "redactions": [],
            "proxy_redaction": "unknown",
        },
        "code": {"krepis": library_version},
    }
    row: Dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": context.trace_id,
        "span_id": span_id,
        "parent_span_id": context.parent_span_id,
        "started_at": isoformat_utc(started_at),
        "ended_at": isoformat_utc(ended_at),
        "duration_ms": round(float(duration_ms), 3),
        "gen_ai": gen_ai,
        "error": error_entry,
        EXTENSION_NAMESPACE: extension,
    }
    return row, blobs


def row_extension(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``krepis`` extension object of a row (empty if absent)."""
    ext = row.get(EXTENSION_NAMESPACE)
    return ext if isinstance(ext, Mapping) else {}
