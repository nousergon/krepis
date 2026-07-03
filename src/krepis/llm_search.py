"""Normalized search-grounding events for provider-agnostic LLM calls.

Two providers can ground a generation in live web search, with different
response shapes:

- **Anthropic** server-side ``web_search`` tool: the response ``content``
  interleaves ``server_tool_use`` blocks (carrying the QUERY the model
  issued) with ``web_search_tool_result`` blocks (carrying the returned
  URLs), paired by ``tool_use_id``.
- **OpenRouter** ``openrouter:web_search`` server tool: the response
  message carries ``url_citation`` annotations (URL + title + excerpt) —
  citations only; the queries the model issued are NOT exposed.

This module normalizes both into two event shapes:

- :class:`SearchEvent` — one issued search (query + returned URLs).
  Anthropic-only today; the schema is exactly what morning-signal's
  ``search_telemetry`` JSONL sink has always written, so existing
  telemetry consumers keep working unchanged.
- :class:`Citation` — one cited source (url/title/snippet). Available on
  both providers.

Extraction errors are never swallowed — telemetry that silently degrades
has no value (``feedback_no_silent_fails``).
"""

from __future__ import annotations

from typing import Any, List, Optional

try:  # TypedDict with total=False on 3.9 lives in typing
    from typing import TypedDict
except ImportError:  # pragma: no cover - py<3.8 unsupported anyway
    from typing_extensions import TypedDict  # type: ignore


class SearchEvent(TypedDict):
    """One issued web search: what the model asked and what came back."""

    query: str
    urls: List[str]
    result_count: int
    error: Optional[str]


class Citation(TypedDict, total=False):
    """One cited source in a grounded response."""

    url: str
    title: Optional[str]
    snippet: Optional[str]


# ── Anthropic (server-side web_search tool) ──────────────────────────────


def extract_anthropic_search_events(msg: Any) -> List[SearchEvent]:
    """Pair ``server_tool_use`` blocks with their ``web_search_tool_result``
    blocks and return one :class:`SearchEvent` per search, in issue order.

    Lifted verbatim (logic-wise) from morning-signal
    ``search_telemetry.extract_searches`` — the schema is that JSONL sink's
    long-standing on-disk contract. A ``server_tool_use`` with no matching
    result block (or an error result) yields ``urls=[]`` with ``error``
    populated.
    """
    tool_uses: dict = {}
    results_by_id: dict = {}
    errors_by_id: dict = {}
    order: List[str] = []

    for block in getattr(msg, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "server_tool_use" and getattr(block, "name", None) == "web_search":
            block_id = getattr(block, "id", None)
            if block_id is None:
                continue
            inp = getattr(block, "input", None) or {}
            query = inp.get("query", "") if isinstance(inp, dict) else ""
            tool_uses[block_id] = {"query": query}
            order.append(block_id)
        elif btype == "web_search_tool_result":
            tool_use_id = getattr(block, "tool_use_id", None)
            if tool_use_id is None:
                continue
            content = getattr(block, "content", None)
            if isinstance(content, list):
                results_by_id[tool_use_id] = content
            else:
                err_code = getattr(content, "error_code", None)
                errors_by_id[tool_use_id] = err_code or str(content)

    out: List[SearchEvent] = []
    for block_id in order:
        info = tool_uses[block_id]
        results = results_by_id.get(block_id, [])
        urls: List[str] = []
        for r in results:
            url = getattr(r, "url", None)
            if isinstance(url, str):
                urls.append(url)
        out.append(
            SearchEvent(
                query=info["query"],
                urls=urls,
                result_count=len(urls),
                error=errors_by_id.get(block_id),
            )
        )
    return out


def extract_anthropic_citations(msg: Any) -> List[Citation]:
    """Flatten every returned search result into a :class:`Citation` list.

    The Anthropic ``web_search_tool_result`` content items carry ``url`` +
    ``title``; this is the cross-provider citation view of the same data
    :func:`extract_anthropic_search_events` reports per-search.
    """
    citations: List[Citation] = []
    for block in getattr(msg, "content", None) or []:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        content = getattr(block, "content", None)
        if not isinstance(content, list):
            continue
        for r in content:
            url = getattr(r, "url", None)
            if not isinstance(url, str):
                continue
            citations.append(
                Citation(url=url, title=getattr(r, "title", None), snippet=None)
            )
    return citations


def final_text_after_last_tool(content: Any) -> str:
    """Return only the text the model wrote AFTER its last tool use.

    With a server-side search tool the response interleaves ``text`` /
    tool-use / tool-result blocks; the model narrates its plan in the text
    emitted before and between searches and writes the real answer in the
    text run after the final search. Joining every text block drags that
    narration into the output (morning-signal 2026-05-30 incident); keeping
    only the post-final-tool text removes it positionally.

    If the model never used a tool, all text is kept. If there is no text
    after the final tool block, falls back to all text blocks so the whole
    answer is never silently dropped — the caller's empty-output check
    still fires.
    """
    blocks = list(content or [])
    last_tool_idx = -1
    for i, block in enumerate(blocks):
        if getattr(block, "type", None) != "text":
            last_tool_idx = i
    tail = "\n\n".join(
        getattr(b, "text", "")
        for b in blocks[last_tool_idx + 1 :]
        if getattr(b, "type", None) == "text"
    ).strip()
    if tail:
        return tail
    return "\n\n".join(
        getattr(b, "text", "")
        for b in blocks
        if getattr(b, "type", None) == "text"
    ).strip()


# ── OpenRouter (openrouter:web_search server tool) ────────────────────────


def extract_openrouter_citations(completion: Any) -> List[Citation]:
    """Extract ``url_citation`` annotations from an OpenRouter chat
    completion into a :class:`Citation` list.

    OpenRouter's web-search server tool attaches annotations to the
    assistant message: each ``{"type": "url_citation", "url_citation":
    {"url", "title", "content", ...}}``. Both attribute-style (SDK
    objects) and dict-style annotations are accepted.

    Note the asymmetry vs Anthropic: OpenRouter exposes citations only —
    the queries the model issued are not in the response, so there is no
    OpenRouter counterpart to :func:`extract_anthropic_search_events`.
    """
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    annotations = getattr(message, "annotations", None) or []

    citations: List[Citation] = []
    for ann in annotations:
        if isinstance(ann, dict):
            ann_type = ann.get("type")
            payload = ann.get("url_citation") or {}
            get = payload.get
        else:
            ann_type = getattr(ann, "type", None)
            payload = getattr(ann, "url_citation", None)
            if payload is None:
                continue
            def get(key, _payload=payload):  # noqa: ANN001 — tiny local shim
                return getattr(_payload, key, None)
        if ann_type != "url_citation":
            continue
        url = get("url")
        if not isinstance(url, str):
            continue
        citations.append(
            Citation(url=url, title=get("title"), snippet=get("content"))
        )
    return citations
