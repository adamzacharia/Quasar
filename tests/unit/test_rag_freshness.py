import sys
import types

import pytest


try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    langchain_core = types.ModuleType("langchain_core")
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = Document
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.documents"] = documents_module

    pymupdf_module = types.ModuleType("langchain_pymupdf4llm")
    pymupdf_module.PyMuPDF4LLMLoader = object
    sys.modules["langchain_pymupdf4llm"] = pymupdf_module

    community_module = types.ModuleType("langchain_community")
    loaders_module = types.ModuleType("langchain_community.document_loaders")
    text_loader_module = types.ModuleType("langchain_community.document_loaders.text")
    text_loader_module.TextLoader = object
    sys.modules["langchain_community"] = community_module
    sys.modules["langchain_community.document_loaders"] = loaders_module
    sys.modules["langchain_community.document_loaders.text"] = text_loader_module

    splitters_module = types.ModuleType("langchain_text_splitters")
    splitters_module.RecursiveCharacterTextSplitter = object
    sys.modules["langchain_text_splitters"] = splitters_module

    openai_module = types.ModuleType("langchain_openai")
    openai_module.OpenAIEmbeddings = object
    sys.modules["langchain_openai"] = openai_module

try:
    from qdrant_client.models import PayloadSchemaType  # noqa: F401
except (ImportError, ModuleNotFoundError):
    class _PayloadSchemaType:
        INTEGER = "integer"
        KEYWORD = "keyword"

    qdrant_module = types.ModuleType("qdrant_client")
    qdrant_models_module = types.ModuleType("qdrant_client.models")
    qdrant_models_module.PayloadSchemaType = _PayloadSchemaType
    sys.modules["qdrant_client"] = qdrant_module
    sys.modules["qdrant_client.models"] = qdrant_models_module

try:
    import services.vector_db  # noqa: F401
except (ImportError, ModuleNotFoundError):
    vector_db_module = types.ModuleType("services.vector_db")
    vector_db_module.upsert_vectors = lambda *args, **kwargs: None
    vector_db_module.search_vectors = lambda *args, **kwargs: []
    vector_db_module.scroll_all = lambda *args, **kwargs: []
    vector_db_module.delete_by_filter = lambda *args, **kwargs: False
    vector_db_module.delete_collection = lambda *args, **kwargs: False
    vector_db_module.create_payload_index = lambda *args, **kwargs: None
    vector_db_module.collection_count = lambda *args, **kwargs: 0
    vector_db_module.ensure_collection = lambda *args, **kwargs: None
    sys.modules["services.vector_db"] = vector_db_module

import services.rag_service as rag_service
from services.rag_service import RAGService, _build_filter_conditions, detect_year_conflicts


pytestmark = pytest.mark.unit

_MISSING = object()


def _doc(year=_MISSING, *, score=0.9, text="ALMA documentation chunk", cycle=None):
    metadata = {"_semantic_score": score}
    if year is not _MISSING:
        metadata["doc_year"] = year
    if cycle:
        metadata["alma_cycle"] = cycle
    return Document(page_content=text, metadata=metadata)


class _FakeEmbeddings:
    def embed_query(self, query):
        return [0.0]


def _fake_rag_service():
    service = RAGService.__new__(RAGService)
    service.embeddings = _FakeEmbeddings()
    service.general_collection = "alma_general"
    service.personal_collection = None
    return service


def test_build_filter_conditions_exact_year_overrides_range_and_exact_filters():
    filters, ranges = _build_filter_conditions(
        year=2025,
        min_year=2020,
        max_year=2026,
        category="technical_handbook",
        source_file="alma-handbook.pdf",
    )

    assert filters == {
        "doc_category": "technical_handbook",
        "source_file": "alma-handbook.pdf",
    }
    assert ranges == {"doc_year": {"gte": 2025, "lte": 2025}}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"min_year": 2021, "max_year": 2024}, {"doc_year": {"gte": 2021, "lte": 2024}}),
        ({"min_year": 2022}, {"doc_year": {"gte": 2022}}),
        ({"max_year": 2023}, {"doc_year": {"lte": 2023}}),
        ({}, {}),
    ],
)
def test_build_filter_conditions_year_ranges(kwargs, expected):
    filters, ranges = _build_filter_conditions(**kwargs)

    assert filters == {}
    assert ranges == expected


def test_hybrid_rerank_recency_breaks_near_semantic_tie_without_bm25(monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    older = _doc(2021, score=0.900, text="same query text")
    newer = _doc(2025, score=0.899, text="same query text")

    ranked = RAGService._hybrid_rerank("query", [older, newer], k=2)

    assert ranked[0].metadata["doc_year"] == 2025
    assert ranked[0].metadata["_score"] > ranked[1].metadata["_score"]


def test_hybrid_rerank_clear_semantic_gap_beats_newer_recency_without_bm25(monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    older_better = _doc(2021, score=0.95, text="strong semantic match")
    newer_weaker = _doc(2025, score=0.50, text="weak semantic match")

    ranked = RAGService._hybrid_rerank("query", [older_better, newer_weaker], k=2)

    assert ranked[0].metadata["doc_year"] == 2021


def test_hybrid_rerank_recency_cannot_override_relevance_across_many_years(monkeypatch):
    # Regression guard for the original additive-recency bug: with many
    # candidates spanning a wide year range, recency's range can exceed the gap
    # between adjacent relevance ranks. The clearly-best (but oldest) chunk must
    # still rank #1, and ordering must follow relevance, not year.
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    best_old = _doc(2014, score=0.97, text="definitive answer")
    candidates = [best_old] + [
        _doc(year, score=score, text="weaker answer")
        for year, score in [
            (2024, 0.80),
            (2025, 0.70),
            (2023, 0.60),
            (2026, 0.50),
        ]
    ]

    ranked = RAGService._hybrid_rerank("query", candidates, k=len(candidates))

    # The oldest-but-strongest chunk wins despite four newer competitors.
    assert ranked[0].metadata["doc_year"] == 2014
    # Scores are strictly non-increasing (relevance dominates the ordering).
    scores = [doc.metadata["_score"] for doc in ranked]
    assert scores == sorted(scores, reverse=True)


def test_detect_year_conflicts_reports_span_and_ignores_missing_years():
    conflict = detect_year_conflicts(
        [
            _doc(2021),
            _doc("2024"),
            _doc(),
            _doc("unknown"),
        ],
        span_threshold=2,
    )

    assert conflict is not None
    assert conflict["min_year"] == 2021
    assert conflict["max_year"] == 2024
    assert conflict["span"] == 3
    assert conflict["years"] == [2021, 2024]
    assert "Cycle 9 (2021)" in conflict["message"]
    assert "Cycle 12 (2024)" in conflict["message"]


def test_detect_year_conflicts_none_when_span_small_or_years_missing():
    assert detect_year_conflicts([_doc(2024), _doc(2025)], span_threshold=2) is None
    assert detect_year_conflicts([_doc(), _doc("unknown")], span_threshold=2) is None


def test_search_with_diagnostics_attaches_conflict_and_search_returns_docs(monkeypatch):
    hits = [
        {
            "score": 0.91,
            "payload": {"text": "Cycle 12 handbook", "doc_year": 2024},
        },
        {
            "score": 0.90,
            "payload": {"text": "Cycle 8 handbook", "doc_year": 2020},
        },
    ]

    def fake_search_vectors(*args, **kwargs):
        return hits

    monkeypatch.setattr(rag_service, "search_vectors", fake_search_vectors)
    service = _fake_rag_service()

    docs, diagnostics = service.search_with_diagnostics(
        "ALMA handbook",
        k=2,
        include_personal=False,
    )

    assert len(docs) == 2
    assert diagnostics["year_conflict"]["span"] == 4
    assert all(doc.metadata["_year_conflict"] == diagnostics["year_conflict"] for doc in docs)

    docs_only = service.search("ALMA handbook", k=2, include_personal=False)
    assert isinstance(docs_only, list)
    assert all(isinstance(doc, Document) for doc in docs_only)


def test_search_diagnostics_use_final_min_score_filtered_docs(monkeypatch):
    hits = [
        {
            "score": 0.91,
            "payload": {"text": "current handbook", "doc_year": 2024},
        },
        {
            "score": 0.20,
            "payload": {"text": "stale handbook", "doc_year": 2020},
        },
    ]

    def fake_search_vectors(*args, **kwargs):
        return hits

    monkeypatch.setattr(rag_service, "search_vectors", fake_search_vectors)
    service = _fake_rag_service()

    docs, diagnostics = service.search_with_diagnostics(
        "ALMA handbook",
        k=2,
        include_personal=False,
        min_score=0.5,
    )

    assert [doc.metadata["doc_year"] for doc in docs] == [2024]
    assert diagnostics["year_conflict"] is None
    assert "_year_conflict" not in docs[0].metadata
