"""services/rag_enrich.py: section paths, TOC detection, printed page labels,
PDF date normalisation, content kind and the per-chunk header; plus the
notebook source identity and scripts/rag_migrate.py copy_collection
(guard task-dd87861-28714)."""
import os
import sys

import pytest

from services import rag_enrich as E

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

pytestmark = pytest.mark.unit


def test_section_tracker_builds_paths_and_carries_across_pages():
    pages = [
        "# ALMA Users' Policies\n\nintro\n\n## 9 Data access\n\ntext\n\n### **9.4 Proprietary periods**\n\nmore",
        "continuation of 9.4 on the next page\n\n### 9.5 Calibration data\n\ncal",
        "## 10 Publication\n\npub",
    ]
    t = E.SectionTracker(pages)
    assert t.section_at(0, 0) == "ALMA Users' Policies"
    # numbered headings drop the unnumbered document title
    assert t.section_at(0, pages[0].index("more")) == "9 Data access > 9.4 Proprietary periods"
    assert t.section_at(1, 0) == "9 Data access > 9.4 Proprietary periods"  # carried across the page break
    assert t.section_at(1, pages[1].index("cal")) == "9 Data access > 9.5 Calibration data"
    assert t.section_at(2, 50) == "10 Publication"  # sibling chapter pops 9.x
    assert t.section_at(9, 0) == ""


def test_numbered_headings_keep_only_true_ancestors():
    pages = [
        "# ALMA Cycle 13 Proposer's Guide\n\n## Revision History\n\nv1\n\n### 6 Post-proposal activities\n\n#### 6.1 Proprietary data\n\ntext",
        "# 9.4.1 Proprietary period and QA2 access\n\nbody",
    ]
    t = E.SectionTracker(pages)
    assert t.section_at(0, pages[0].index("text")) == "6 Post-proposal activities > 6.1 Proprietary data"
    # 9.4.1 is not under 6 / 6.1: unrelated numbered sections are dropped
    assert t.section_at(1, 30) == "9.4.1 Proprietary period and QA2 access"
    assert E.heading_number("9.4.1 Proprietary") == "9.4.1"
    assert E.heading_number_depth("A.2 Band 2") == 2
    assert E.heading_number_depth("A User's Guide") == 0


def test_headings_inside_code_fences_are_ignored():
    text = "## Real section\n\n```python\n## example comment\nprint(1)\n```\n\nafter"
    assert [h[2] for h in E.headings_in(text)] == ["Real section"]
    t = E.SectionTracker([text])
    assert t.section_at(0, text.index("after")) == "Real section"


def test_clean_heading_strips_markup():
    assert E.clean_heading("**9.5 Calibration data**") == "9.5 Calibration data"
    assert E.clean_heading("[Data Lab](https://x) *FAQ*") == "Data Lab FAQ"


@pytest.mark.parametrize(("text", "toc"), [
    ("8.1 Purpose and Scope ........................ 13\n8.2 Classification ........ 13\n8.3 Process ....... 14", True),
    ("10.4.4 Antenna Pointing . . . . . . . . . 166\n10.4.5 Antenna Position . . . . . . . 166\n10.5 Other . . . . . 167", True),
    ("9.4 OBSERVATIONAL DATA ACCESS ........................ 17\n"
     "_9.4.1 Proprietary period and QA2 access .............. 17_\n"
     "_9.4.2 QA0 raw data access ................. 17_\n"
     "**1 WHAT'S NEW ................................ 3**", True),
    ("The proprietary period is 12 months.\nIt starts when data are delivered.\nSee Section 9.4.", False),
    # CX-05: a real table with a few dotted rows among prose rows is kept
    ("Band 3 sensitivity ........ 12\nBand 6 sensitivity ........ 14\nBand 7 sensitivity ........ 16\n"
     "These values assume 1 hour on source.\nPWV of 1 mm.\nElevation 60 deg.\nDual polarization.", False),
])
def test_toc_detection(text, toc):
    assert E.is_toc_chunk(text) is toc


def test_content_kind():
    assert E.content_kind("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |") == "table"
    assert E.content_kind("```\nprint(1)\n```") == "code"
    assert E.content_kind("Plain prose about ALMA.") == "text"


@pytest.mark.parametrize(("page_text", "label"), [
    ("10\nproposal deadline. Any proposed duplicate...\nlast line", "10"),
    ("Figure 3 caption\n...text...\n21", "21"),
    ("CONTENTS\niv\n1 Intro . . . 3", "iv"),
    ("Page 7 of 29\nbody", "7"),
    ("No number here\njust text", ""),
])
def test_printed_page_label(page_text, label):
    assert E.printed_page_label(page_text) == label


@pytest.mark.parametrize(("raw", "iso"), [
    ("D:20260226135411Z00'00'", "2026-02-26T13:54:11Z"),
    ("D:20251104205239+05'30'", "2025-11-04T15:22:39Z"),  # CX-06: offset converted to UTC
    ("D:20250101000000-08'00'", "2025-01-01T08:00:00Z"),
    ("2026-09-25T13:17:11+00:00", "2026-09-25T13:17:11Z"),
    ("2026-09-25T18:47:11+05:30", "2026-09-25T13:17:11Z"),
    ("D:2024", ""),
    ("", ""),
])
def test_normalize_pdf_date(raw, iso):
    assert E.normalize_pdf_date(raw) == iso


def test_pdf_document_facts_drops_empty_and_normalises():
    facts = E.pdf_document_facts({"creationDate": "D:20260226135411Z00'00'", "modDate": "", "producer": "Quartz",
                                  "author": "", "title": "", "format": "PDF 1.3"})
    assert facts == {"pdf_created": "2026-02-26T13:54:11Z", "pdf_producer": "Quartz", "pdf_format": "PDF 1.3"}


def test_iso_doc_date_and_header():
    assert E.iso_doc_date(2026, 3, None) == "2026-03"
    assert E.iso_doc_date(2026, 3, 19) == "2026-03-19"
    assert E.iso_doc_date(None, None, None) == ""
    h = E.chunk_header("ALMA Cycle 13 Users' Policies (Doc 13.16)", "6 Selection > 6.4 Descoping", 12, "10")
    assert h == "[Document: ALMA Cycle 13 Users' Policies (Doc 13.16) | Section: 6 Selection > 6.4 Descoping | PDF page 12 (printed page 10)]"
    assert E.chunk_header("T", "", 5, "5") == "[Document: T | PDF page 5]"
    assert E.chunk_header("T", "", None, "") == "[Document: T]"


def test_loader_keys_to_drop_cover_local_paths():
    assert {"source", "file_path", "creationDate", "moddate"} <= E.LOADER_KEYS_TO_DROP


# ── ingest_notebooks: repo-relative identity (CX-01) ─────────────────────

def test_notebook_sources_use_repo_relative_names(tmp_path):
    import ingest_notebooks as nb
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "README.md").write_text("# Title\n\n" + "Useful documentation text. " * 20, encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "plot.pdf").write_bytes(b"%PDF-1.4" + b"x" * 2000)
    calls = []

    class FakeRag:
        def ingest_document(self, path, **kw):
            calls.append(kw)
            return {"success": True, "chunks": 1}

    nb.ingest_directory(str(tmp_path), "repo-x", {"category": "tutorial"}, FakeRag())
    names = sorted(c["original_filename"] for c in calls)
    assert names == ["repo-x/a/README.md", "repo-x/b/README.md"]  # distinct identities, report PDF skipped
    assert all(c["replace_existing"] for c in calls)


# ── rag_migrate.copy_collection (CX-07/08/09) ─────────────────────────────

def _store(path, n=0, name="c"):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    cl = QdrantClient(path=str(path))
    if n is not None:
        cl.create_collection(name, vectors_config=VectorParams(size=2, distance=Distance.COSINE))
        if n:
            cl.upsert(name, points=[PointStruct(id=i + 1, vector=[1.0, float(i)], payload={"i": i}) for i in range(n)])
    return cl


def test_copy_collection_paginates_to_the_last_page(tmp_path):
    import rag_migrate as rm
    src = _store(tmp_path / "src", n=301)
    dst = _store(tmp_path / "dst", n=None)
    r = rm.copy_collection(src, dst, "c", batch=100, require_indexes=False, log=lambda *a: None)
    assert r == {"source": 301, "target": 301}
    got = {p.id: p.payload["i"] for p in dst.scroll("c", limit=500)[0]}
    assert got == {i + 1: i for i in range(301)}


def test_copy_collection_refuses_non_empty_target(tmp_path):
    import rag_migrate as rm
    src = _store(tmp_path / "src", n=3)
    dst = _store(tmp_path / "dst", n=2)
    with pytest.raises(rm.MigrationError, match="not empty"):
        rm.copy_collection(src, dst, "c", require_indexes=False, log=lambda *a: None)
    assert dst.count("c", exact=True).count == 2  # nothing deleted


def test_copy_collection_retries_are_idempotent_and_missing_indexes_fail(tmp_path, monkeypatch):
    import rag_migrate as rm
    src = _store(tmp_path / "src", n=5)
    real = _store(tmp_path / "dst", n=None)

    class Flaky:
        """Target whose first upsert fails after writing; index creation is a no-op."""
        def __init__(self, inner):
            self.inner, self.failed = inner, False

        def __getattr__(self, k):
            return getattr(self.inner, k)

        def upsert(self, **kw):
            self.inner.upsert(**kw)
            if not self.failed:
                self.failed = True
                raise ConnectionError("reset after write")

        def create_payload_index(self, **kw):
            return None

    monkeypatch.setattr(rm.time, "sleep", lambda s: None)
    with pytest.raises(rm.MigrationError, match="indexes missing"):
        rm.copy_collection(src, Flaky(real), "c", batch=10, require_indexes=True, log=lambda *a: None)
    assert real.count("c", exact=True).count == 5  # retried batch did not duplicate


def test_copy_collection_resume_finishes_an_interrupted_copy(tmp_path):
    import rag_migrate as rm
    from qdrant_client.models import PointStruct
    src = _store(tmp_path / "src", n=7)
    dst = _store(tmp_path / "dst", n=None)
    from qdrant_client.models import Distance, VectorParams
    dst.create_collection("c", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    dst.upsert("c", points=[PointStruct(id=i + 1, vector=[1.0, float(i)], payload={"i": i}) for i in range(3)])  # partial
    with pytest.raises(rm.MigrationError, match="resume"):
        rm.copy_collection(src, dst, "c", require_indexes=False, log=lambda *a: None)
    r = rm.copy_collection(src, dst, "c", require_indexes=False, log=lambda *a: None, resume=True)
    assert r == {"source": 7, "target": 7}
    # a target holding points the source does not have fails the count check
    dst.upsert("c", points=[PointStruct(id=999, vector=[0.0, 1.0], payload={})])
    with pytest.raises(rm.MigrationError, match="count mismatch"):
        rm.copy_collection(src, dst, "c", require_indexes=False, log=lambda *a: None, resume=True)


def test_copy_collection_named_vectors(tmp_path):
    import rag_migrate as rm
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    src = QdrantClient(path=str(tmp_path / "src"))
    src.create_collection("n", vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)})
    src.upsert("n", points=[PointStruct(id=i + 1, vector={"dense": [1.0, float(i)]}, payload={"i": i}) for i in range(4)])
    dst = QdrantClient(path=str(tmp_path / "dst"))
    assert rm.copy_collection(src, dst, "n", require_indexes=False, log=lambda *a: None) == {"source": 4, "target": 4}
    p = dst.retrieve("n", ids=[2], with_vectors=True)[0]
    assert set(p.vector) == {"dense"} and len(p.vector["dense"]) == 2


def test_classification_ignores_folder_names_in_identity():
    import services.rag_service as rs
    assert rs._classify_category("jwst-notes/readme.md") == "jwst"  # raw string would match the folder
    meta = rs.extract_document_metadata(__file__, ["x"], original_filename="jwst-tutorials/sub/readme.md")
    assert meta["doc_category"] == "general"  # classified by basename only (CX-10)



# ── guard task-dd87861-1630 follow-ups (CX-17, CX-18) ──

@pytest.mark.parametrize("text", [
    "## Real\n\n```text\n~~~\n## code comment\n```\n\nafter",      # tilde line inside a backtick fence
    "## Real\n\n~~~\n```\n## code comment\n~~~\n\nafter",          # backticks inside a tilde fence
    "## Real\n\n````\n```\n## code comment\n````\n\nafter",        # shorter fence cannot close a longer one
    "## Real\n\n```python\n## code comment\n``` not a close\n```\n\nafter",  # closer must be bare
])
def test_g1630_cx17_fences_close_only_on_a_matching_marker(text):
    assert [h[2] for h in E.headings_in(text)] == ["Real"]
    assert E.SectionTracker([text]).section_at(0, text.index("after")) == "Real"


def test_g1630_cx17_heading_after_a_closed_fence_still_counts():
    text = "## A\n\n~~~\ncode\n~~~\n\n## B\n\n```\n## no\n```\n## C"
    assert [h[2] for h in E.headings_in(text)] == ["A", "B", "C"]


@pytest.mark.parametrize(("raw", "iso"), [
    ("2026-99-99T12:00:00Z", ""),
    ("2026-02-30", ""),
    ("2026-13-01T00:00:00", ""),
    ("2026-02-26Tgarbage", "2026-02-26"),   # valid date, unparseable time: keep the date
])
def test_g1630_cx18_invalid_iso_dates_are_empty(raw, iso):
    assert E.normalize_pdf_date(raw) == iso
