"""Tests for ``krepis.llm_search`` — normalized search events + citations."""

from types import SimpleNamespace

from krepis.llm_search import (
    extract_anthropic_citations,
    extract_anthropic_search_events,
    extract_openrouter_citations,
    final_text_after_last_tool,
)


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _use(query, block_id):
    return SimpleNamespace(
        type="server_tool_use", name="web_search", id=block_id,
        input={"query": query},
    )


def _result(tool_use_id, urls):
    return SimpleNamespace(
        type="web_search_tool_result", tool_use_id=tool_use_id,
        content=[SimpleNamespace(url=u, title=f"t:{u}") for u in urls],
    )


def _error_result(tool_use_id, code):
    return SimpleNamespace(
        type="web_search_tool_result", tool_use_id=tool_use_id,
        content=SimpleNamespace(error_code=code),
    )


def _msg(content):
    return SimpleNamespace(content=content)


class TestAnthropicSearchEvents:
    def test_pairing_in_issue_order(self):
        msg = _msg([
            _use("q1", "a"), _result("a", ["u1"]),
            _text("mid"),
            _use("q2", "b"), _result("b", ["u2", "u3"]),
        ])
        events = extract_anthropic_search_events(msg)
        assert [e["query"] for e in events] == ["q1", "q2"]
        assert events[1]["urls"] == ["u2", "u3"]
        assert events[1]["result_count"] == 2
        assert events[0]["error"] is None

    def test_error_result(self):
        msg = _msg([_use("q", "a"), _error_result("a", "max_uses_exceeded")])
        events = extract_anthropic_search_events(msg)
        assert events == [
            {"query": "q", "urls": [], "result_count": 0,
             "error": "max_uses_exceeded"}
        ]

    def test_use_without_result(self):
        events = extract_anthropic_search_events(_msg([_use("q", "a")]))
        assert events[0]["urls"] == [] and events[0]["result_count"] == 0

    def test_no_searches(self):
        assert extract_anthropic_search_events(_msg([_text("x")])) == []

    def test_citations_flatten_results(self):
        msg = _msg([_use("q", "a"), _result("a", ["u1", "u2"])])
        cites = extract_anthropic_citations(msg)
        assert [c["url"] for c in cites] == ["u1", "u2"]
        assert cites[0]["title"] == "t:u1"


class TestFinalTextAfterLastTool:
    def test_keeps_only_post_tool_text(self):
        content = [
            _text("I'll search now."),
            _use("q", "a"), _result("a", ["u"]),
            _text("Real answer part 1."), _text("Part 2."),
        ]
        assert final_text_after_last_tool(content) == (
            "Real answer part 1.\n\nPart 2."
        )

    def test_no_tool_keeps_everything(self):
        assert final_text_after_last_tool([_text("a"), _text("b")]) == "a\n\nb"

    def test_no_tail_falls_back_to_all_text(self):
        content = [_text("everything before"), _use("q", "a"), _result("a", [])]
        assert final_text_after_last_tool(content) == "everything before"

    def test_empty(self):
        assert final_text_after_last_tool([]) == ""
        assert final_text_after_last_tool(None) == ""


class TestOpenRouterCitations:
    def test_dict_annotations(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(annotations=[
                {"type": "url_citation",
                 "url_citation": {"url": "https://x", "title": "X",
                                  "content": "snippet"}},
                {"type": "other", "url_citation": {"url": "https://skip"}},
            ])
        )])
        cites = extract_openrouter_citations(completion)
        assert cites == [{"url": "https://x", "title": "X",
                          "snippet": "snippet"}]

    def test_attr_annotations(self):
        ann = SimpleNamespace(
            type="url_citation",
            url_citation=SimpleNamespace(
                url="https://y", title="Y", content=None
            ),
        )
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(annotations=[ann])
        )])
        cites = extract_openrouter_citations(completion)
        assert cites[0]["url"] == "https://y"
        assert cites[0]["title"] == "Y"

    def test_no_annotations(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="hi")
        )])
        assert extract_openrouter_citations(completion) == []

    def test_no_choices(self):
        assert extract_openrouter_citations(SimpleNamespace(choices=[])) == []
