"""core/rag_retrieval.py: sub-question queries, junk filter, per-source cap, interleaving, kill switch."""
from __future__ import annotations

from types import SimpleNamespace

from core import rag_retrieval as rr


def _doc(src, text, page=1):
    return SimpleNamespace(page_content=f"[Document: {src} | Section: s | PDF page {page}]\n{text}",
                           metadata={"source_file": src, "page": page})


BODY = "Measure the rms in emission-free regions of the image before primary-beam correction. " * 3


class _Svc:
    def __init__(self, by_query):
        self.by_query, self.calls = by_query, []

    def search_with_diagnostics(self, q, **kw):
        self.calls.append(q)
        return self.by_query.get(q, []), {"year_conflict": None, "q": q}


def test_sub_queries_split_multi_part_questions_only():
    q = "What is a spectral window? How do I get the channel width; and why does it matter for my line?"
    assert rr.sub_queries(q) == ["What is a spectral window?", "How do I get the channel width;",
                                 "and why does it matter for my line?"]
    assert rr.sub_queries("How do I measure the noise in an ALMA image?") == []


def test_junk_chunks_are_dropped_and_base64_is_stripped():
    assert rr.clean_chunk_text("[Document: X | p]\n**6**")[1] is True
    text, junk = rr.clean_chunk_text("[Document: X]\n" + "A" * 400 + " " + BODY)
    assert junk is False and "AAAAAAAAAA" not in text and "binary data removed" in text
    assert rr.clean_chunk_text("[Document: X]\n" + "QUJD" * 200)[1] is True


def test_retrieve_interleaves_caps_sources_and_dedupes(monkeypatch):
    monkeypatch.setenv("QUASAR_RAG_MULTI_QUERY", "1")
    monkeypatch.setenv("QUASAR_RAG_QUERY_EXPANSION", "0")
    q = "What does the archive list as sensitivity? How do I measure the rms myself?"
    primer = [_doc("primer.pdf", BODY + str(i), page=i) for i in range(5)]
    handbook = [_doc("handbook.pdf", BODY + "hb", page=9), _doc("stub.pdf", "**6**")]
    svc = _Svc({q: primer, "What does the archive list as sensitivity?": primer[:2],
                "How do I measure the rms myself?": handbook})
    docs, diag = rr.retrieve(svc, q)
    sources = [d.metadata["source_file"] for d in docs]
    assert sources.count("primer.pdf") == 2, "per-source cap"
    assert "handbook.pdf" in sources[:3], "the sub-question's best chunk comes before a third primer chunk"
    assert "stub.pdf" not in sources and diag["multi_query"]["dropped_junk"] == 1
    assert len(svc.calls) == 3 and diag["q"] == q, "diagnostics come from the whole-question search"


def test_kill_switch_restores_the_single_top3_search(monkeypatch):
    monkeypatch.setenv("QUASAR_RAG_MULTI_QUERY", "0")
    q = "What is X? How do I compute Y for my data?"
    svc = _Svc({q: [_doc(f"s{i}.pdf", BODY) for i in range(5)]})
    docs, _ = rr.retrieve(svc, q)
    assert len(docs) == 3 and svc.calls == [q]


def test_model_written_queries_join_the_search_and_failures_fall_back(monkeypatch):
    monkeypatch.setenv("QUASAR_RAG_MULTI_QUERY", "1")
    monkeypatch.setenv("QUASAR_RAG_QUERY_EXPANSION", "1")
    q = "How do I estimate the noise in my ALMA image?"
    svc = _Svc({q: [_doc("primer.pdf", BODY)], "primary beam correction raises noise toward edges": [_doc("handbook.pdf", BODY)]})
    docs, diag = rr.retrieve(svc, q, expand_call=lambda prompt: "1. primary beam correction raises noise toward edges\nok\n")
    assert [d.metadata["source_file"] for d in docs] == ["primer.pdf", "handbook.pdf"]
    assert diag["multi_query"]["expanded"] == ["primary beam correction raises noise toward edges"]

    def boom(prompt):
        raise RuntimeError("model down")
    docs, diag = rr.retrieve(svc, q, expand_call=boom)
    assert [d.metadata["source_file"] for d in docs] == ["primer.pdf"] and diag["multi_query"]["expanded"] == []
    assert rr.parse_expansion("- a b c d\n2) e f g h\n" + "x " * 40) == ["a b c d", "e f g h"]


def test_cx02_expansion_and_searches_carry_the_request_context(monkeypatch):
    from core import llm_client

    monkeypatch.setenv("QUASAR_RAG_MULTI_QUERY", "1")
    monkeypatch.setenv("QUASAR_RAG_QUERY_EXPANSION", "1")
    seen = {}
    marker = object()
    monkeypatch.setattr(llm_client._request_tls, "context", marker, raising=False)

    def expand(prompt):
        seen["expand"] = llm_client.get_llm_request_context()
        return "a query about primary beam noise\n"
    svc = _Svc({"q one two three four": [_doc("p.pdf", BODY)]})
    rr.retrieve(svc, "q one two three four", expand_call=expand)
    assert seen["expand"] is marker


def test_cx03_cx04_stuck_optional_work_cannot_block_retrieval(monkeypatch):
    import threading
    import time

    monkeypatch.setenv("QUASAR_RAG_MULTI_QUERY", "1")
    monkeypatch.setenv("QUASAR_RAG_RETRIEVAL_BUDGET", "0.5")
    monkeypatch.setenv("QUASAR_DOC_QUERY_TIMEOUT", "0.2")
    gate = threading.Event()
    q = "How do I estimate the rms? Where is the QA2 report kept?"

    class Slow(_Svc):
        def search_with_diagnostics(self, query, **kw):
            if query != q:
                gate.wait(5)            # a stuck optional search
            return super().search_with_diagnostics(query, **kw)
    svc = Slow({q: [_doc("p.pdf", BODY)]})

    def slow_expand(prompt):
        gate.wait(5)
        return ""
    t0 = time.monotonic()
    docs, diag = rr.retrieve(svc, q, expand_call=slow_expand)
    assert time.monotonic() - t0 < 2.0 and [d.metadata["source_file"] for d in docs] == ["p.pdf"]
    assert diag["multi_query"]["late_subqueries_dropped"] == 2
    # The in-flight cap: while that expansion is still running, further calls skip expansion instead of piling up.
    rr.expand_queries("x y z w", call=slow_expand, timeout=0.05)   # second stuck call: both slots now busy
    called = []
    assert rr.expand_queries("x y z w", call=lambda p: called.append(1) or "a b c d", timeout=0.1) == []
    assert called == [], "no third expansion while two are still in flight"
    assert all(t.daemon for t in threading.enumerate() if t.name.startswith("rag-"))
    gate.set()
    time.sleep(0.3)
    assert rr.expand_queries("x y z w", call=lambda p: "a b c d", timeout=1.0) == ["a b c d"], "slots freed"
