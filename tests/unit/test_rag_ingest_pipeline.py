"""RAG corpus refresh (2026-09-25): HTML->Markdown converter, manifest metadata
and --verify, Knowledgebase parsing, manifest-driven ingest (override metadata,
per-chunk document header, replace-on-reingest), category mapping, the local
persistent Qdrant mode, and the covered-facility documentation route."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import download_alma_docs as dl  # noqa: E402
import ingest_alma_kb as kb  # noqa: E402
import ingest_html_pages as hp  # noqa: E402

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────────────
# HTML -> Markdown
# ──────────────────────────────────────────────────────────────────

PAGE = """
<html><head><title>Cycle 6 Key Policies - JWST User Documentation</title>
<meta property="og:site_name" content="JWST User Documentation">
<meta property="article:modified_time" content="2026-07-20T10:00:00Z">
<script>var x = "should vanish";</script><style>.a{}</style></head>
<body>
<header><h1>JWST User Documentation</h1><nav><a href="/">Home</a></nav></header>
<div class="breadcrumb">Home &gt; Policies</div>
<main>
  <h1>JWST Key Policies</h1>
  <p>Proposals are due <strong>30 September 2026</strong>.
     See the <a href="/jwst-call-for-proposals">call</a> and <a href="#top">top</a>.</p>
  <h2>Exclusive access</h2>
  <ul>
    <li>Default is <em>12 months</em>
      <ul><li>zero for some programs</li></ul>
    </li>
    <li>Survey programs have none</li>
  </ul>
  <ol><li>first</li><li>second</li></ol>
  <table>
    <tr><th>Program</th><th>EAP</th></tr>
    <tr><td>GO</td><td>12 | months</td></tr>
  </table>
  <pre>SELECT TOP 5 * FROM ivoa.obscore</pre>
  <dl><dt>EAP</dt><dd>exclusive access period</dd></dl>
  <!-- a comment -->
</main>
<footer>Copyright STScI</footer>
</body></html>
"""


def test_html_to_markdown_keeps_main_content_and_drops_chrome():
    title, md = hp.html_to_markdown(PAGE, base_url="https://jwst-docs.stsci.edu/jwst-opportunities/x")
    assert title == "JWST Key Policies"  # site-name <h1> skipped
    assert md.startswith("# JWST Key Policies")
    for gone in ("should vanish", "Home", "Copyright", "a comment", "breadcrumb"):
        assert gone not in md
    assert "**30 September 2026**" in md
    assert "[call](https://jwst-docs.stsci.edu/jwst-call-for-proposals)" in md  # made absolute
    assert "[top]" not in md and "top" in md  # fragment links keep text only
    assert "## Exclusive access" in md


def test_html_to_markdown_lists_tables_code_and_definitions():
    _, md = hp.html_to_markdown(PAGE, base_url="https://example.org/")
    assert "- Default is *12 months*" in md
    assert "  - zero for some programs" in md  # nested list indented
    assert "1. first" in md and "2. second" in md
    assert "| Program | EAP |" in md and "|---|---|" in md
    assert "| GO | 12 \\| months |" in md  # pipes in cells escaped
    assert "```\nSELECT TOP 5 * FROM ivoa.obscore\n```" in md
    assert "**EAP**" in md and "exclusive access period" in md


def test_html_to_markdown_single_column_table_and_fallback_root():
    html = "<html><body><table><tr><td>only</td></tr><tr><td>cells</td></tr></table></body></html>"
    title, md = hp.html_to_markdown(html)
    assert title == ""
    assert md == "only\ncells"


def test_content_nested_in_malformed_link_or_meta_is_kept():
    """The ESO data access policy page wraps its text in an unclosed <link>."""
    html = ('<html><body><div id="main"><div class="leftnavi navigation"><ul><li><a href="/x">Menu</a></li></ul></div>'
            '<div class="richtext"><link rel="stylesheet" href="a.css"><p>' + "Raw data have a one-year proprietary period. " * 6 +
            '</p></div></div></body></html>')
    _, md = hp.html_to_markdown(html)
    assert "one-year proprietary period" in md
    assert "Menu" not in md  # ESO CMS left navigation dropped


def test_page_title_strips_sphinx_pilcrow_and_uses_title_suffix_as_site_name():
    html = "<html><head><title>2. FAQs ¶ - Data Lab documentation</title></head><body><h1>2. FAQs¶</h1></body></html>"
    title, _ = hp.html_to_markdown(html)
    assert title == "2. FAQs"


def test_page_date_reads_standard_meta_tags():
    from bs4 import BeautifulSoup
    assert hp.page_date(BeautifulSoup(PAGE, "html.parser")) == "2026-07-20"
    assert hp.page_date(BeautifulSoup("<html></html>", "html.parser")) is None
    assert hp.page_date(BeautifulSoup('<time datetime="2025-12-22">x</time>', "html.parser")) == "2025-12-22"


def test_front_matter_round_trip():
    meta = {"title": "T", "source_url": "https://x.org/p", "fetched_at": "2026-09-25", "category": "jwst"}
    text = hp.with_front_matter(meta, "# Body\n\ntext")
    back, body = hp.split_front_matter(text)
    assert back == meta
    assert body == "# Body\n\ntext\n"
    assert hp.split_front_matter("no front matter") == ({}, "no front matter")


def test_page_metadata_prefers_manifest_date_then_page_date_then_fetch_date():
    page = {"filename": "x.md", "url": "https://almascience.nrao.edu/news/x", "category": "news", "cycle": "Cycle 13"}
    m = hp.page_metadata({**page, "doc_date": "2025-12-22"}, "T", "2026-09-25", "2026-01-01")
    assert (m["doc_year"], m["doc_month"], m["doc_day"]) == (2025, 12, 22)
    assert "dated 2025-12-22" in m["doc_label"] and "almascience.nrao.edu" in m["doc_label"]
    m = hp.page_metadata(page, "T", "2026-09-25", "2026-01-05")
    assert (m["doc_year"], m["doc_month"]) == (2026, 1)
    m = hp.page_metadata(page, "T", "2026-09-25", None)
    assert (m["doc_year"], m["doc_month"], m["doc_day"]) == (2026, 9, 25)
    assert "dated" not in m["doc_label"]
    assert m["source_url"] == page["url"] and m["fetched_at"] == "2026-09-25"
    assert m["doc_category"] == "news" and m["alma_cycle"] == "Cycle 13" and m["doc_status"] == "live-page"


def test_page_manifest_is_consistent():
    names = [p["filename"] for p in hp.PAGES]
    assert len(names) == len(set(names))
    for p in hp.PAGES:
        assert p["filename"].endswith(".md") and p["url"].startswith("https://") and p["category"]


def test_fetch_rejects_cloudflare_challenge(monkeypatch):
    class R:
        headers = {"Content-Type": "text/html"}
        encoding = "utf-8"
        text = '<div class="cf-turnstile"></div>'

        def raise_for_status(self):
            pass

    monkeypatch.setattr(hp, "polite_get", lambda url, **k: R())
    with pytest.raises(ValueError, match="Cloudflare"):
        hp.fetch_and_convert({"url": "https://casaguides.nrao.edu/x", "filename": "x.md", "category": "casa_guides"}, "2026-09-25")


# ──────────────────────────────────────────────────────────────────
# PDF manifest + --verify
# ──────────────────────────────────────────────────────────────────

def test_manifest_has_unique_filenames_and_no_retired_overlap():
    names = [d["filename"] for d in dl.MANIFEST + dl.LOCAL_REFERENCE_DOCS]
    assert len(names) == len(set(names))
    assert not set(names) & set(dl.RETIRED_FILES)
    for d in dl.MANIFEST:
        assert d["url"].startswith("https://") and d["category"] and d["title"] and d["status"]
    # The Cycle 12 Java-OT manuals and the duplicate handbook must stay retired.
    for f in ("alma-ot-usermanual.pdf", "alma-ot-refmanual.pdf", "alma-technical-handbook.pdf",
              "alma-proposers-guide.pdf", "RLM.pdf", "mem0.pdf"):
        assert f in dl.RETIRED_FILES


def test_manifest_metadata_maps_dates_status_and_label():
    m = dl.manifest_metadata("alma-user-policies-cycle13.pdf")
    assert m["doc_category"] == "user_policies" and m["alma_cycle"] == "Cycle 13"
    assert (m["doc_year"], m["doc_month"]) == (2026, 3) and "doc_day" not in m
    assert m["doc_number"] == "13.16" and m["doc_status"] == "current"
    assert m["source_url"].endswith("/cycle13/alma-user-policies")
    assert m["doc_label"].startswith("ALMA Cycle 13 Users' Policies (Doc 13.16, v1.0, 2026-03)")
    old = dl.manifest_metadata("alma-scheduling-blocks-guide-cycle4.pdf")
    assert old["alma_cycle"] == "Cycle 4" and old["doc_status"] == "current-old"
    assert "still the one ALMA links for Cycle 13" in old["doc_label"]
    assert dl.manifest_metadata("not-in-manifest.pdf") is None
    ref = dl.manifest_metadata("alma_tap_columns.txt")
    assert ref["source_url"] == "" and ref["doc_category"] == "archive"


@pytest.mark.parametrize(("slug", "stem"), [
    ("alma-qa2-data-products-for-cycle-12", "alma-qa2-data-products"),
    ("alma-qa2-data-products-for-cycle-13", "alma-qa2-data-products"),
    ("snoopi-user-manual-march2026version", "snoopi-user-manual"),
    ("alma-user-policies/view", "alma-user-policies"),
])
def test_slug_stem_normalises_cycle_and_version_suffixes(slug, stem):
    assert dl._slug_stem(slug) == stem


class _Resp:
    def __init__(self, status=206, size=None, ctype="application/pdf", text=""):
        self.status_code = status
        self.headers = {"Content-Type": ctype}
        if size is not None:
            self.headers["Content-Range"] = f"bytes 0-0/{size}"
        self.text = text

    def close(self):
        pass


def _fake_get(routes):
    def get(url, **kwargs):
        if url in routes:
            r = routes[url]
            if isinstance(r, Exception):
                raise r
            return r
        return _Resp(status=404, ctype="application/json")
    return get


def _entry(name, url, slug="", cycle="Cycle 13"):
    return {"filename": name, "url": url, "slug": slug, "cycle": cycle, "category": "x", "title": name, "status": "current"}


def test_verify_manifest_reports_every_status(tmp_path):
    B = dl.ALMA_BASE
    (tmp_path / "ok.pdf").write_bytes(b"x" * 10)
    (tmp_path / "changed.pdf").write_bytes(b"x" * 10)
    (tmp_path / "newer.pdf").write_bytes(b"x" * 10)
    (tmp_path / "probe.pdf").write_bytes(b"x" * 10)
    (tmp_path / "old-retired.pdf").write_bytes(b"x")
    (tmp_path / "stray.pdf").write_bytes(b"x")
    manifest = [
        _entry("ok.pdf", f"{B}/cycle13/ok-doc", "ok-doc"),
        _entry("changed.pdf", f"{B}/cycle13/changed-doc", "changed-doc"),
        _entry("missing.pdf", f"{B}/cycle13/missing-doc", "missing-doc"),
        _entry("dead.pdf", f"{B}/cycle13/dead-doc", "dead-doc"),
        _entry("html.pdf", f"{B}/cycle13/html-doc", "html-doc"),
        _entry("newer.pdf", f"{B}/cycle13/newer-doc", "newer-doc"),
        _entry("probe.pdf", f"{B}/cycle13/probe-doc", "probe-doc"),
    ]
    routes = {
        B: _Resp(200, text=f'<a href="{B}/cycle14/newer-doc/view">new</a> <a href="{B}/cycle13/ok-doc/view">ok</a>'),
        f"{B}/cycle13/ok-doc": _Resp(size=10),
        f"{B}/cycle13/changed-doc": _Resp(size=99),
        f"{B}/cycle13/missing-doc": _Resp(size=5),
        f"{B}/cycle13/dead-doc": ConnectionError("reset"),
        f"{B}/cycle13/html-doc": _Resp(200, ctype="text/html"),
        f"{B}/cycle13/newer-doc": _Resp(size=10),
        f"{B}/cycle13/probe-doc": _Resp(size=10),
        f"{B}/cycle14/probe-doc": _Resp(size=12),
    }
    rows = dl.verify_manifest(str(tmp_path), manifest=manifest, retired={"old-retired.pdf": "superseded"},
                              get=_fake_get(routes), index_url=B)
    status = {r["filename"]: r["status"] for r in rows}
    assert status == {
        "ok.pdf": "OK",
        "changed.pdf": "CHANGED",
        "missing.pdf": "MISSING_LOCAL",
        "dead.pdf": "UNREACHABLE",
        "html.pdf": "NOT_PDF",
        "newer.pdf": "NEWER_CYCLE",
        "probe.pdf": "NEWER_CYCLE",
        "old-retired.pdf": "RETIRED_ON_DISK",
        "stray.pdf": "LOCAL_ONLY",
    }
    detail = {r["filename"]: r["detail"] for r in rows}
    assert "cycle14" in detail["newer.pdf"] and "cycle14/probe-doc" in detail["probe.pdf"]
    assert "99" in detail["changed.pdf"]


def test_verify_manifest_survives_unreachable_index(tmp_path):
    B = dl.ALMA_BASE
    (tmp_path / "ok.pdf").write_bytes(b"x" * 10)
    routes = {B: ConnectionError("down"), f"{B}/cycle13/ok-doc": _Resp(size=10)}
    rows = dl.verify_manifest(str(tmp_path), manifest=[_entry("ok.pdf", f"{B}/cycle13/ok-doc", "ok-doc")],
                              retired={}, get=_fake_get(routes), index_url=B)
    assert {r["filename"]: r["status"] for r in rows} == {"(portal index)": "UNREACHABLE", "ok.pdf": "OK"}


# ──────────────────────────────────────────────────────────────────
# Knowledgebase
# ──────────────────────────────────────────────────────────────────

def test_kb_category_page_parsing_dedupes_and_skips_pdf_links():
    html = """
      <a href="/kb/articles/how-do-i-download-data">How do I download   data?</a>
      <a href="/kb/articles/how-do-i-download-data">dup</a>
      <a href="/kb/articles/pdf/how-do-i-download-data">PDF</a>
      <a href="/kb/archive-data-retrieval">category</a>
      <a href="/kb/articles/what-cycle-13-proposal-issues">What Cycle 13 proposal issues?</a>
      <a href="/kb/articles/empty"></a>"""
    got = kb.parse_category_page(html)
    assert got == [
        {"slug": "how-do-i-download-data", "title": "How do I download data?"},
        {"slug": "what-cycle-13-proposal-issues", "title": "What Cycle 13 proposal issues?"},
    ]


def test_kb_byline_and_metadata():
    text = "ALMA Science\nKnowledgebase > General > What Cycle 13 ...\nSarah Bagley - 2026-04-22 - General\nThis article..."
    by = kb.parse_byline(text)
    assert by == {"author": "Sarah Bagley", "date": "2026-04-22", "kb_category": "General"}
    assert kb.parse_byline("no byline here") is None
    art = {"slug": "what-cycle-13-proposal-issues", "title": "What Cycle 13 proposal issues?", "category": "general"}
    m = kb.kb_metadata(art, by, "2026-09-25")
    assert m["doc_category"] == "knowledgebase" and m["alma_cycle"] == "Cycle 13"
    assert (m["doc_year"], m["doc_month"], m["doc_day"]) == (2026, 4, 22)
    assert m["source_url"] == "https://help.almascience.org/kb/articles/what-cycle-13-proposal-issues"
    assert "last updated 2026-04-22" in m["doc_label"]
    m2 = kb.kb_metadata({**art, "title": "How do I reset my password?"}, None, "2026-09-25")
    assert m2["alma_cycle"] == "" and m2["doc_year"] == 2026 and m2["kb_category"] == "general"
    assert kb.kb_filename("abc") == "kb-abc.pdf"
    assert "historical-articles" in kb.SKIP_BY_DEFAULT


# ──────────────────────────────────────────────────────────────────
# RAG service: category map + manifest-driven ingest
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("fname", "cat"), [
    ("kb-how-can-i-rerun-the-pipeline-on-my-data.pdf", "knowledgebase"),  # not "pipeline"
    ("alma-science-archive-manual-cycle13.pdf", "archive_manual"),
    ("alma-archive-primer-cycle13.pdf", "archive"),
    ("alma-qa2-data-products-cycle12.pdf", "qa2_products"),
    ("alma-science-primer-cycle13.pdf", "primer"),
    ("alma-na-arcguide-cycle13.pdf", "arc_guide"),
    ("datalab-manual-sql-gotchas.md", "data_lab"),
    ("jwst-cycle6-key-policies-jdox.md", "jwst"),
    ("eso-data-access-policy.md", "eso"),
    ("vla-oss-2027-proposing.md", "vla"),
    ("ivoa-tap-1.1.pdf", "ivoa"),
    ("news-gaia-dr4-release-page.md", "news"),
    ("alma-technical-handbook-cycle13.pdf", "technical_handbook"),
    ("alma_pipeline_users_guide_2025.pdf", "pipeline"),
])
def test_category_map_specific_patterns_win(fname, cat):
    from services.rag_service import _classify_category
    assert _classify_category(fname) == cat


class _Emb:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


def _service(monkeypatch, calls):
    import services.rag_service as rs
    from langchain_core.documents import Document

    svc = rs.RAGService.__new__(rs.RAGService)
    svc.embeddings = _Emb()
    svc.general_collection = "alma_general"
    svc.personal_collection = None
    svc.user_id = None
    svc._load_document = lambda path: [Document(page_content="Proprietary period is 12 months. " * 3, metadata={"page": 4})]
    monkeypatch.setattr(rs, "upsert_vectors", lambda coll, ids, vecs, pays: calls.append(("upsert", coll, pays)))
    monkeypatch.setattr(rs, "delete_by_filter",
                        lambda coll, flt, exclude_conditions=None: calls.append(("delete", coll, flt)) or True)
    monkeypatch.setattr(rs, "delete_older_runs",
                        lambda coll, sf, run_id, run_ts: calls.append(("delete", coll, {"source_file": sf}, run_id, run_ts)) or True)
    return svc


def test_ingest_document_override_metadata_header_and_replace(monkeypatch, tmp_path):
    calls = []
    svc = _service(monkeypatch, calls)
    f = tmp_path / "alma-user-policies-cycle13.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    meta = dl.manifest_metadata("alma-user-policies-cycle13.pdf")
    res = svc.ingest_document(str(f), override_metadata=meta, replace_existing=True)
    assert res["success"], res
    kinds = [c[0] for c in calls]
    assert kinds == ["upsert", "delete"]  # old chunks removed only after the new ones are stored
    assert calls[1][2] == {"source_file": "alma-user-policies-cycle13.pdf"}
    pay = calls[0][2][0]
    assert pay["doc_year"] == 2026 and pay["doc_month"] == 3 and pay["doc_day"] is None
    assert pay["doc_category"] == "user_policies" and pay["alma_cycle"] == "Cycle 13"
    assert pay["doc_title"] == "ALMA Cycle 13 Users' Policies"
    for k in ("doc_number", "doc_version", "doc_status", "source_url", "fetched_at", "doc_label"):
        assert pay[k] == meta[k]
    assert pay["text"].startswith("[Document: ALMA Cycle 13 Users' Policies (Doc 13.16")
    assert pay["page"] == 5 and pay["pdf_page"] == 5  # loader page 4 is 0-based
    assert "source" not in pay and "file_path" not in pay  # no local paths in payloads


def test_ingest_document_without_override_keeps_legacy_behaviour(monkeypatch, tmp_path):
    calls = []
    svc = _service(monkeypatch, calls)
    f = tmp_path / "my-notes.txt"
    f.write_text("x")
    res = svc.ingest_document(str(f), extra_metadata={"doc_category": "ignored", "custom": 1})
    assert res["success"]
    assert [c[0] for c in calls] == ["upsert"]  # no delete without replace_existing
    pay = calls[0][2][0]
    assert pay["doc_category"] == "general"  # extra_metadata never overrides
    assert pay["custom"] == 1
    assert not pay["text"].startswith("[Document:")
    assert "doc_label" not in pay and "source_url" not in pay


# ──────────────────────────────────────────────────────────────────
# vector_db: persistent local mode
# ──────────────────────────────────────────────────────────────────

def test_qdrant_path_mode_persists_across_processes(tmp_path):
    store = str(tmp_path / "qdrant")
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {ROOT!r})
        os.environ["QDRANT_PATH"] = {store!r}
        os.environ["QDRANT_URL"] = "https://unreachable.example"
        os.environ["QDRANT_API_KEY"] = "k"
        from services import vector_db as v
        assert v.backend_label() == "local:" + {store!r}, v.backend_label()
        assert not v.is_using_cloud()
        if sys.argv[1] == "write":
            v.upsert_vectors("t", ["00000000-0000-0000-0000-000000000001"], [[1.0, 0.0]], [{{"source_file": "a.pdf"}}])
        else:
            assert v.collection_count("t") == 1
            hits = v.search_vectors("t", [1.0, 0.0], limit=1)
            assert hits[0]["payload"]["source_file"] == "a.pdf"
            assert v.get_collection_info("t")["points_count"] == 1
            v.delete_by_filter("t", {{"source_file": "a.pdf"}})
            assert v.collection_count("t") == 0
        print("OK")
    """)
    for step in ("write", "read"):
        out = subprocess.run([sys.executable, "-c", script, step], capture_output=True, text=True, timeout=120)
        assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-2000:]


# ──────────────────────────────────────────────────────────────────
# Routing: covered-facility documentation questions
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "When is the JWST Cycle 6 proposal deadline?",
    "Is LS DR11 available in Data Lab?",
    "Which Legacy Surveys data releases does Data Lab host?",
    "What is the ESO proprietary period for raw data?",
    "What is the VLA 2027B proposal deadline?",
    "When is Gaia DR4 released?",
    "What are the known issues with DESI DR1?",
    "How do I write an ADQL query with a TAP service?",
])
def test_covered_facility_knowledge_questions_reach_documentation(q):
    from core.rag_routing import documentation_rag_override, facility_documentation_question
    assert facility_documentation_question(q) is True
    assert documentation_rag_override(q) is True


@pytest.mark.parametrize("q", [
    "Plot the redshift distribution of DESI DR1 LRGs",
    "Are there public JWST NIRSpec spectra of the z = 6.3 quasar SDSS J0100+2802? If so, which observing modes or gratings were used and under which program?",
    "Which JWST program first imaged the Pillars of Creation in M16 with NIRCam, and when was that observation taken?",
    "Show me a color image of the center of M31 from the DECam Legacy Surveys.",
    "How do I download HST FITS files from MAST?",  # MAST docs are not in the corpus
    "Cone search Gaia DR3 around RA = 10, Dec = 20 and get the parallaxes",
])
def test_facility_data_requests_do_not_reach_documentation(q):
    from core.rag_routing import documentation_rag_override, facility_documentation_question
    assert facility_documentation_question(q) is False
    assert documentation_rag_override(q) is False


# ──────────────────────────────────────────────────────────────────
# Guard task-dd87861-25750 regressions (CX-NN)
# ──────────────────────────────────────────────────────────────────

def test_cx01_sync_never_reingests_retired_files_left_on_disk(tmp_path):
    import ingest_docs_folder as idf
    for f in ("alma-ot-usermanual.pdf", "alma-ot-usermanual-cycle13.pdf", "notes.md", "x.docx"):
        (tmp_path / f).write_bytes(b"x")
    files, skipped = idf.select_files(str(tmp_path), [], dl.RETIRED_FILES)
    assert files == ["alma-ot-usermanual-cycle13.pdf", "notes.md"]
    assert skipped == ["alma-ot-usermanual.pdf"]


def test_cx02_replace_upserts_before_deleting_the_previous_run(monkeypatch, tmp_path):
    import services.rag_service as rs
    calls = []
    svc = _service(monkeypatch, calls)
    f = tmp_path / "alma-user-policies-cycle13.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    assert svc.ingest_document(str(f), override_metadata=dl.manifest_metadata(f.name), replace_existing=True)["success"]
    assert [c[0] for c in calls] == ["upsert", "delete"]
    run, ts = calls[0][2][0]["ingest_run"], calls[0][2][0]["ingest_ts"]
    assert all(p["ingest_run"] == run and p["ingest_ts"] == ts for p in calls[0][2])
    assert calls[1][2] == {"source_file": f.name} and calls[1][3:] == (run, ts)


def test_cx02_failed_upsert_keeps_old_chunks(monkeypatch, tmp_path):
    import services.rag_service as rs
    calls = []
    svc = _service(monkeypatch, calls)

    def boom(*a, **k):
        raise RuntimeError("qdrant write timeout")

    monkeypatch.setattr(rs, "upsert_vectors", boom)
    f = tmp_path / "alma-user-policies-cycle13.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    res = svc.ingest_document(str(f), override_metadata=dl.manifest_metadata(f.name), replace_existing=True)
    assert res["success"] is False
    # the only delete is the rollback of the failed run, never the old document
    assert all("ingest_run" in c[2] for c in calls if c[0] == "delete")


def _row_status(rows):
    return {r["filename"]: r["status"] for r in rows}


def test_cx03_cx04_cx05_verify_edge_cases(tmp_path):
    B = dl.ALMA_BASE
    (tmp_path / "nosize.pdf").write_bytes(b"x" * 10)
    (tmp_path / "redir.pdf").write_bytes(b"x" * 10)

    class Redirected(_Resp):
        url = f"{B}/cycle13/redir-doc"  # cycle14 path redirected back to the current edition

    manifest = [_entry("nosize.pdf", f"{B}/cycle13/nosize-doc", "nosize-doc"),
                _entry("redir.pdf", f"{B}/cycle13/redir-doc", "redir-doc")]
    routes = {
        B: _Resp(404, ctype="text/html", text=f'<a href="{B}/cycle14/nosize-doc">would mislead</a>'),
        f"{B}/cycle13/nosize-doc": _Resp(200),  # PDF but no Content-Range / Content-Length
        f"{B}/cycle13/redir-doc": _Resp(size=10),
        f"{B}/cycle14/redir-doc": Redirected(size=10),
    }
    rows = dl.verify_manifest(str(tmp_path), manifest=manifest, retired={}, get=_fake_get(routes), index_url=B)
    assert _row_status(rows) == {"(portal index)": "UNREACHABLE", "nosize.pdf": "NO_SIZE", "redir.pdf": "OK"}


def test_cx06_subset_verify_does_not_flag_other_manifest_files(tmp_path):
    B = dl.ALMA_BASE
    (tmp_path / "a.pdf").write_bytes(b"x" * 10)
    (tmp_path / "b.pdf").write_bytes(b"x" * 10)
    routes = {f"{B}/cycle13/a-doc": _Resp(size=10)}
    rows = dl.verify_manifest(str(tmp_path), manifest=[_entry("a.pdf", f"{B}/cycle13/a-doc")], retired={},
                              get=_fake_get(routes), index_url=None, known_filenames={"a.pdf", "b.pdf"})
    assert _row_status(rows) == {"a.pdf": "OK"}


def test_cx07_current_old_entry_not_linked_by_portal(tmp_path):
    B = dl.ALMA_BASE
    (tmp_path / "old.pdf").write_bytes(b"x" * 10)
    old = {**_entry("old.pdf", f"{B}/cycle4/old-guide", "old-guide", cycle="Cycle 4"), "status": "current-old"}
    for index_html, expect in ((f'<a href="{B}/cycle4/old-guide">x</a>', "OK"), ("<a>nothing</a>", "NOT_LINKED")):
        routes = {B: _Resp(200, text=index_html), f"{B}/cycle4/old-guide": _Resp(size=10)}
        rows = dl.verify_manifest(str(tmp_path), manifest=[old], retired={}, get=_fake_get(routes), index_url=B)
        assert _row_status(rows)["old.pdf"] == expect


def test_cx08_fetched_at_is_the_local_download_date(tmp_path):
    f = tmp_path / "alma-user-policies-cycle13.pdf"
    f.write_bytes(b"x")
    os.utime(f, (1735689600, 1735689600))  # 2025-01-01 UTC
    assert dl.manifest_metadata(f.name, path=str(f))["fetched_at"] in ("2024-12-31", "2025-01-01")
    assert dl.manifest_metadata(f.name)["fetched_at"] == dl.FETCH_DATE


def test_cx09_short_main_beats_whole_body():
    html = ("<html><body><div class='promo'>" + "Popular stories newsletter signup. " * 20 + "</div>"
            "<main><h1>Deadline</h1><p>Proposals due 30 September.</p></main></body></html>")
    _, md = hp.html_to_markdown(html)
    assert "Proposals due 30 September." in md and "newsletter" not in md


def test_cx10_block_children_in_table_cells_are_separated():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td><p>First.</p><p>Second.</p></td><td>x</td></tr></table>"
    _, md = hp.html_to_markdown(html)
    assert "First. Second." in md


def test_cx11_saved_challenge_page_rejected(tmp_path):
    saved = tmp_path / "saved.html"
    saved.write_text('<html><div class="cf-turnstile"></div></html>', encoding="utf-8")
    with pytest.raises(ValueError, match="Cloudflare"):
        hp.fetch_and_convert({"url": "https://casaguides.nrao.edu/x", "filename": "x.md", "category": "casa_guides",
                              "saved_html": str(saved)}, "2026-09-25")


def _run_html_main(monkeypatch, tmp_path, argv, body):
    page = {"filename": "p.md", "url": "https://example.org/p", "category": "news"}
    monkeypatch.setattr(hp, "PAGES", [page])
    monkeypatch.setattr(hp, "PAGES_DIR", str(tmp_path))
    monkeypatch.setattr(hp, "fetch_and_convert", lambda p, d: ("T", body, None))
    monkeypatch.setattr(sys, "argv", ["ingest_html_pages.py"] + argv)
    hp.main()


def test_cx12_estimate_writes_no_markdown(monkeypatch, tmp_path):
    _run_html_main(monkeypatch, tmp_path, ["--estimate"], "long enough body " * 40)
    assert not any(tmp_path.rglob("*.md"))


def test_cx17_html_cli_exits_nonzero_on_failed_pages(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run_html_main(monkeypatch, tmp_path, ["--no-ingest"], "too short")
    assert exc.value.code == 1


def test_cx13_cx16_date_source_and_kb_category_reach_the_payload(monkeypatch, tmp_path):
    calls = []
    svc = _service(monkeypatch, calls)
    f = tmp_path / "kb-x.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    meta = kb.kb_metadata({"slug": "x", "title": "T", "category": "general"},
                          {"author": "A", "date": "2026-04-22", "kb_category": "General"}, "2026-09-25")
    assert svc.ingest_document(str(f), override_metadata=meta)["success"]
    pay = calls[-1][2][0]
    assert pay["kb_category"] == "General" and pay["doc_date_source"] == "kb-byline"
    page = hp.page_metadata({"filename": "p.md", "url": "https://e.org/p", "category": "news"}, "T", "2026-09-25", None)
    assert page["doc_date_source"] == "fetched"


@pytest.mark.parametrize(("q", "expected"), [
    ("When is the JWST flux of M51 highest?", False),          # CX-18
    ("Show me the JWST Cycle 6 proposal deadline", True),      # CX-19
    ("Get the ESO data access policy", True),                  # CX-19
    ("Show me Legacy Surveys DR11 images of M31", False),
    ("When is Gaia DR4 released?", True),
])
def test_cx18_cx19_routing_phrasings(q, expected):
    from core.rag_routing import documentation_rag_override
    assert documentation_rag_override(q) is expected


def test_cx22_inventory_diff_reports_metadata_changes_with_same_chunk_count():
    import rag_inventory as ri
    src = {"chunks": 5, "category": "news", "cycle": "", "year": 2026, "month": 3, "status": "live-page",
           "source_url": "u", "ingested_max": "2026-09-25T10:00"}
    before = {"collections": {"c": {"points": 5, "sources": {"a.md": src}}}}
    after = {"collections": {"c": {"points": 5, "sources": {"a.md": {**src, "month": 9, "ingested_max": "2026-09-26T10:00"}}}}}
    out = ri.diff(before, after)
    assert "CHANGED a.md" in out and "month 3 -> 9" in out
    assert "CHANGED" not in ri.diff(before, before)


# ── verify round 1 reopens (CX-09, CX-14, CX-17, CX-20) and new CX-23/CX-24 ──

def test_cx09_short_main_nested_in_long_content_wins():
    html = ("<html><body><div id='content'><div class='promo'>" + "Popular stories newsletter signup. " * 20 +
            "</div><main><h1>Deadline</h1><p>Proposals due 30 September.</p></main></div></body></html>")
    _, md = hp.html_to_markdown(html)
    assert "Proposals due 30 September." in md and "newsletter" not in md


def test_cx14_cached_kb_pdfs_are_refreshed_after_max_age(tmp_path):
    f = tmp_path / "kb-x.pdf"
    assert kb.needs_download(str(f), force=False, max_age_days=30) is True  # missing
    f.write_bytes(b"%PDF-")
    now = os.path.getmtime(f)
    assert kb.needs_download(str(f), force=False, max_age_days=30, now=now + 5 * 86400) is False
    assert kb.needs_download(str(f), force=False, max_age_days=30, now=now + 31 * 86400) is True
    assert kb.needs_download(str(f), force=True, max_age_days=30, now=now) is True


def test_cx17_kb_estimate_exits_nonzero_when_downloads_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(kb, "KB_DIR", str(tmp_path))
    monkeypatch.setattr(kb, "crawl", lambda cats: [{"slug": "a", "title": "A", "category": "general"}])

    def dead(url, **k):
        raise ConnectionError("reset")

    monkeypatch.setattr(kb, "polite_get", dead)
    monkeypatch.setattr(sys, "argv", ["ingest_alma_kb.py", "--estimate"])
    with pytest.raises(SystemExit) as exc:
        kb.main()
    assert exc.value.code == 1


def test_cx20_relative_qdrant_path_is_the_same_store_from_any_cwd(tmp_path):
    seen = []
    for cwd in (ROOT, os.path.join(ROOT, "ui-pro"), str(tmp_path)):
        script = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {ROOT!r})
            os.chdir({cwd!r})
            os.environ["QDRANT_PATH"] = "cache/qdrant_rel_test"
            from services import vector_db as v
            print(v.QDRANT_PATH)
        """)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-2000:]
        seen.append(out.stdout.strip().splitlines()[-1])
    assert len(set(seen)) == 1
    assert seen[0] == os.path.normpath(os.path.join(ROOT, "cache", "qdrant_rel_test"))


def test_cx23_partial_upsert_failure_rolls_back_the_new_run(monkeypatch, tmp_path):
    import services.rag_service as rs
    calls = []
    svc = _service(monkeypatch, calls)

    def half_then_fail(coll, ids, vecs, pays):
        calls.append(("upsert-partial", coll, pays))
        raise RuntimeError("batch 2 timed out")

    monkeypatch.setattr(rs, "upsert_vectors", half_then_fail)
    monkeypatch.setattr(rs, "delete_by_filter",
                        lambda coll, flt, exclude_conditions=None: calls.append(("delete", coll, flt, exclude_conditions)) or True)
    f = tmp_path / "alma-user-policies-cycle13.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    res = svc.ingest_document(str(f), override_metadata=dl.manifest_metadata(f.name), replace_existing=True)
    assert res["success"] is False
    run = calls[0][2][0]["ingest_run"]
    deletes = [c for c in calls if c[0] == "delete"]
    # only the rollback of THIS run; the old document is never deleted
    assert deletes == [("delete", "alma_general", {"source_file": f.name, "ingest_run": run}, None)]


@pytest.mark.parametrize(("q", "expected"), [
    ("Fetch ESO proprietary data", False),                       # CX-24
    ("Fetch ESO proprietary data under the access policy", False),
    ("Fetch ESO proprietary reduced products under the access policy", False),
    ("Fetch ESO proprietary rows under the access policy", False),
    ("Fetch ESO proprietary reduced exposures under the access policy", False),
    ("Get the JWST exclusive access policy", True),
    ("Show me the VLA 2027B proposal deadline", True),
    ("Retrieve JWST exclusive access data for program 1234", False),
    ("Show me the JWST Cycle 6 proposal deadline", True),
    ("Get the ESO data access policy", True),
    ("List the known issues for DESI DR1", True),
])
def test_cx24_request_verbs_need_a_documentation_object(q, expected):
    from core.rag_routing import documentation_rag_override
    assert documentation_rag_override(q) is expected


def test_cx25_concurrent_replacements_converge_on_the_newest_run(tmp_path):
    """Real embedded Qdrant: runs A (older) and B (newer) both upsert before
    either cleans up; legacy chunks without ingest_ts also exist. After both
    cleanups exactly B's chunks remain."""
    store = str(tmp_path / "q")
    script = textwrap.dedent(f"""
        import os, sys, uuid
        sys.path.insert(0, {ROOT!r})
        os.environ["QDRANT_PATH"] = {store!r}
        from services import vector_db as v
        def pts(tag, n, run=None, ts=None):
            ids = [str(uuid.uuid4()) for _ in range(n)]
            pay = [{{"source_file": "doc.pdf", "tag": tag}} for _ in range(n)]
            for p in pay:
                if run: p["ingest_run"] = run
                if ts is not None: p["ingest_ts"] = ts
            v.upsert_vectors("c", ids, [[1.0, 0.0]] * n, pay)
        pts("legacy", 3)
        pts("other", 2)
        v.upsert_vectors("c", [str(uuid.uuid4())], [[0.0, 1.0]], [{{"source_file": "keep.pdf", "tag": "keep"}}])
        pts("A", 4, "run-a", 100.0)
        pts("B", 5, "run-b", 200.0)
        v.delete_older_runs("c", "doc.pdf", "run-a", 100.0)   # A cleans up after B wrote
        v.delete_older_runs("c", "doc.pdf", "run-b", 200.0)
        tags = sorted(p["payload"]["tag"] for p in v.scroll_all("c", limit=100))
        print(tags)
        assert tags == ["B"] * 5 + ["keep"], tags
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "OK" in out.stdout, (out.stdout + out.stderr)[-2000:]


def test_cx25_late_older_run_leaves_a_duplicate_never_a_loss(tmp_path):
    """Round 4: B (newer) uploads and cleans up first, then an older run A
    uploads and cleans up. Self-deletion was removed (it could empty the
    source when the newer run was still partial), so both runs remain:
    a duplicate until the next replacement, never a loss."""
    store = str(tmp_path / "q")
    script = textwrap.dedent(f"""
        import os, sys, uuid
        sys.path.insert(0, {ROOT!r})
        os.environ["QDRANT_PATH"] = {store!r}
        from services import vector_db as v
        def pts(tag, n, run, ts):
            ids = [str(uuid.uuid4()) for _ in range(n)]
            pay = [{{"source_file": "doc.pdf", "tag": tag, "ingest_run": run, "ingest_ts": ts}} for _ in range(n)]
            v.upsert_vectors("c", ids, [[1.0, 0.0]] * n, pay)
        pts("B", 5, "run-b", 200)
        v.delete_older_runs("c", "doc.pdf", "run-b", 200)
        pts("A", 4, "run-a", 100)
        v.delete_older_runs("c", "doc.pdf", "run-a", 100)
        tags = sorted(p["payload"]["tag"] for p in v.scroll_all("c", limit=100))
        assert tags == ["A"] * 4 + ["B"] * 5, tags
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "OK" in out.stdout, (out.stdout + out.stderr)[-2000:]


def test_cx02_cx23_round4_partial_newer_run_then_rollback_never_empties_the_source(tmp_path):
    """Codex round-4 sequence: legacy chunks + complete run A; a newer run B
    is mid-upload (partial) when A cleans up; B then fails and rolls back.
    A's chunks must survive."""
    store = str(tmp_path / "q")
    script = textwrap.dedent(f"""
        import os, sys, uuid
        sys.path.insert(0, {ROOT!r})
        os.environ["QDRANT_PATH"] = {store!r}
        from services import vector_db as v
        def pts(tag, n, run=None, ts=None):
            ids = [str(uuid.uuid4()) for _ in range(n)]
            pay = [{{"source_file": "doc.pdf", "tag": tag}} for _ in range(n)]
            for p in pay:
                if run: p["ingest_run"] = run
                if ts is not None: p["ingest_ts"] = ts
            v.upsert_vectors("c", ids, [[1.0, 0.0]] * n, pay)
        pts("legacy", 3)
        pts("A", 4, "run-a", 100)
        pts("B-partial", 2, "run-b", 200)                       # B's first batch only
        v.delete_older_runs("c", "doc.pdf", "run-a", 100)        # A cleans up
        v.delete_by_filter("c", {{"source_file": "doc.pdf", "ingest_run": "run-b"}})  # B rolls back
        tags = sorted(p["payload"]["tag"] for p in v.scroll_all("c", limit=100))
        assert tags == ["A"] * 4, tags
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "OK" in out.stdout, (out.stdout + out.stderr)[-2000:]



# ──────────────────────────────────────────────────────────────────
# Guard task-dd87861-1630 follow-ups (CX-13..CX-16, CX-20, CX-21)
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("only", "files", "retired", "flags", "refused"), [
    (["typo.pdf"], ["a.pdf"], [], dict(sync=False, no_wipe=False, estimate=False), "not found"),
    (["typo.pdf"], ["a.pdf"], [], dict(sync=True, no_wipe=False, estimate=False), "not found"),
    (["typo.pdf"], ["a.pdf"], [], dict(sync=False, no_wipe=False, estimate=True), "not found"),
    (["a.pdf"], ["a.pdf"], [], dict(sync=False, no_wipe=False, estimate=False), "requires --sync"),
    (["a.pdf"], ["a.pdf"], [], dict(sync=False, no_wipe=True, estimate=False), "requires --sync"),
    (["old.pdf"], [], ["old.pdf"], dict(sync=True, no_wipe=False, estimate=False), "only retired"),
    ([], [], [], dict(sync=False, no_wipe=False, estimate=False), "refusing to wipe"),
    (["a.pdf"], ["a.pdf"], [], dict(sync=True, no_wipe=False, estimate=False), None),
    (["a.pdf"], ["a.pdf"], [], dict(sync=False, no_wipe=False, estimate=True), None),
    ([], ["a.pdf"], [], dict(sync=False, no_wipe=False, estimate=False), None),   # legacy full re-ingest
    ([], [], [], dict(sync=True, no_wipe=False, estimate=False), None),
])
def test_g1630_cx21_selection_error(only, files, retired, flags, refused):
    import ingest_docs_folder as idf
    err = idf.selection_error(only, files, retired, **flags)
    assert (err is None) if refused is None else (refused in err)


@pytest.mark.parametrize("argv", [
    ["--only", "alma-user-policies-cycle31.pdf"],            # unmatched, default (wipe) mode
    ["--only", "alma-user-policies-cycle31.pdf", "--sync"],  # unmatched, sync mode
    ["--only", "notes.md"],                                  # matched but no --sync
])
def test_g1630_cx21_only_never_reaches_the_collection_wipe(monkeypatch, tmp_path, argv):
    import ingest_docs_folder as idf
    import services.rag_service as rs
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")

    class _NoRag:
        def __init__(self, *a, **k):
            pytest.fail("RAGService must not be constructed (the default mode wipes alma_general)")

    monkeypatch.setattr(rs, "RAGService", _NoRag)
    monkeypatch.setattr(sys, "argv", ["ingest_docs_folder.py", "--dir", str(tmp_path)] + argv)
    with pytest.raises(SystemExit) as exc:
        idf.main()
    assert exc.value.code == 2


def test_g1630_cx13_empty_kb_crawl_keeps_the_index_and_fails(monkeypatch, tmp_path):
    import json
    index = tmp_path / "_index.json"
    old = [{"slug": "a", "title": "A", "category": "general"}]
    index.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(kb, "KB_DIR", str(tmp_path))
    monkeypatch.setattr(kb, "crawl", lambda cats: [])
    monkeypatch.setattr(sys, "argv", ["ingest_alma_kb.py", "--list"])
    with pytest.raises(SystemExit) as exc:
        kb.main()
    assert exc.value.code == 1
    assert json.loads(index.read_text(encoding="utf-8")) == old


def test_g1630_cx14_corrupt_cached_kb_pdf_is_one_failure_not_an_abort(monkeypatch, tmp_path, capsys):
    import fitz
    monkeypatch.setattr(kb, "KB_DIR", str(tmp_path))
    monkeypatch.setattr(kb, "crawl", lambda cats: [{"slug": "bad", "title": "B", "category": "general"},
                                                    {"slug": "good", "title": "G", "category": "general"}])
    monkeypatch.setattr(kb, "polite_get", lambda url, **k: pytest.fail("fresh cache must not be re-downloaded"))
    (tmp_path / kb.kb_filename("bad")).write_bytes(b"%PDF-1.4 truncated garbage")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Knowledgebase answer text")
    doc.save(str(tmp_path / kb.kb_filename("good")))
    doc.close()
    monkeypatch.setattr(sys, "argv", ["ingest_alma_kb.py", "--estimate"])
    with pytest.raises(SystemExit) as exc:
        kb.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL bad: unreadable cached PDF" in out and "1 article PDFs" in out


def test_g1630_cx15_article_header_content_is_kept_site_header_dropped():
    html = ("<html><body><header><h1>Observatory News</h1><p>Site tagline here</p></header>"
            "<main><article><header><h1>Call for Proposals</h1>"
            "<p>Applications close 30 September.</p></header>"
            "<p>" + "Details of the call follow. " * 5 + "</p></article></main>"
            "<footer>Copyright</footer></body></html>")
    title, md = hp.html_to_markdown(html)
    assert title == "Call for Proposals"
    assert "# Call for Proposals" in md and "Applications close 30 September." in md
    assert "Site tagline" not in md and "Observatory News" not in md and "Copyright" not in md


def test_g1630_cx16_no_fetch_short_saved_page_never_replaces_the_indexed_page(monkeypatch, tmp_path):
    import services.rag_service as rs
    saved = tmp_path / "p.md"
    saved.write_text(hp.with_front_matter({"title": "T", "source_url": "https://example.org/p",
                                           "fetched_at": "2026-09-25", "category": "news"}, "truncated"),
                     encoding="utf-8")
    ingested = []

    class _Rag:
        def ingest_document(self, *a, **k):
            ingested.append(a)
            return {"success": True, "chunks": 1}

    monkeypatch.setattr(rs, "RAGService", _Rag)
    monkeypatch.setattr(hp, "page_path", lambda page, pages_dir=None: str(saved))
    with pytest.raises(SystemExit) as exc:
        _run_html_main(monkeypatch, tmp_path, ["--no-fetch"], "unused")
    assert exc.value.code == 1 and ingested == []


def test_g1630_cx20_failed_rollback_is_reported_not_swallowed(monkeypatch, tmp_path):
    import services.rag_service as rs
    calls = []
    svc = _service(monkeypatch, calls)
    monkeypatch.setattr("time.sleep", lambda s: None)

    def half_then_fail(coll, ids, vecs, pays):
        calls.append(("upsert-partial", coll, pays))
        raise RuntimeError("batch 2 timed out")

    def rollback_down(coll, flt, exclude_conditions=None):
        calls.append(("delete", coll, flt))
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(rs, "upsert_vectors", half_then_fail)
    monkeypatch.setattr(rs, "delete_by_filter", rollback_down)
    f = tmp_path / "alma-user-policies-cycle13.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    res = svc.ingest_document(str(f), override_metadata=dl.manifest_metadata(f.name), replace_existing=True)
    run = calls[0][2][0]["ingest_run"]
    assert res["success"] is False
    assert "batch 2 timed out" in res["error"] and "rollback" in res["error"] and run in res["error"]
    assert "remain searchable" in res["error"]
    assert len([c for c in calls if c[0] == "delete"]) == 3  # retried before giving up


def test_g1630_cx20_transient_rollback_failure_is_retried(monkeypatch, tmp_path):
    import services.rag_service as rs
    calls = []
    svc = _service(monkeypatch, calls)
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = []

    def fail_upsert(coll, ids, vecs, pays):
        raise RuntimeError("batch 2 timed out")

    def flaky(coll, flt, exclude_conditions=None):
        attempts.append(flt)
        if len(attempts) == 1:
            raise ConnectionError("blip")
        return True

    monkeypatch.setattr(rs, "upsert_vectors", fail_upsert)
    monkeypatch.setattr(rs, "delete_by_filter", flaky)
    f = tmp_path / "alma-user-policies-cycle13.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    res = svc.ingest_document(str(f), override_metadata=dl.manifest_metadata(f.name), replace_existing=True)
    assert res["success"] is False and res["error"] == "batch 2 timed out"
    assert len(attempts) == 2
