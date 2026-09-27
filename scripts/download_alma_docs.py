"""
Documentation PDF manifest + downloader for the RAG corpus (docs/pdfs/).

The manifest below is the single source of truth for what the ``alma_general``
collection should contain from docs/pdfs: URL, local filename, category, cycle,
document number/version/date and status. ``scripts/ingest_docs_folder.py``
reads it (``manifest_metadata``) so every chunk carries the authoritative
metadata instead of whatever the PDF ModDate says.

Usage:
    python scripts/download_alma_docs.py            # download missing files
    python scripts/download_alma_docs.py --list     # list manifest, no download
    python scripts/download_alma_docs.py --force    # re-download everything
    python scripts/download_alma_docs.py --verify   # report stale / changed entries
    python scripts/download_alma_docs.py --only alma-user-policies-cycle13.pdf

--verify compares the portal against the local files without downloading:
  * each entry's URL (1-byte range request) -> size vs the local file
  * the ALMA "current documents" index page -> is a newer cycle path linked
    for the same document?
  * a probe of the next cycle's path (cycleN+1/<slug>) for ALMA entries
and prints one status per entry (OK, CHANGED, NEWER_CYCLE, MISSING_LOCAL,
UNREACHABLE, NOT_PDF, NO_SIZE, NOT_LINKED, LOCAL_ONLY) plus RETIRED files still
on disk.
Exit status 1 when anything is not OK.
"""
import os
import re
import sys
import argparse
from typing import Callable, Dict, List, Optional

import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Target download directory ──
DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "pdfs"
)

ALMA_BASE = "https://almascience.nrao.edu/documents-and-tools"
ALMA_INDEX_URL = ALMA_BASE  # lists the current document for every class
FETCH_DATE = "2026-09-25"

# status values:
#   current      latest edition, published for the current cycle
#   current-old  old edition that the current cycle's documents page still links
#   reference    non-ALMA standard / external reference
# "slug" is the ALMA portal document id used by --verify's next-cycle probe.
ALMA_DOCS: List[Dict[str, str]] = [
    # ── Cycle 13 document set (portal index, seen 2026-09-25) ──
    {"filename": "alma-proposers-guide-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-proposers-guide",
     "slug": "alma-proposers-guide", "category": "proposers_guide", "cycle": "Cycle 13",
     "doc_number": "13.2", "version": "1.0", "doc_date": "2026-03", "status": "current",
     "title": "ALMA Cycle 13 Proposer's Guide"},
    {"filename": "alma-technical-handbook-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-technical-handbook",
     "slug": "alma-technical-handbook", "category": "technical_handbook", "cycle": "Cycle 13",
     "doc_number": "13.3", "version": "1.0", "doc_date": "2026-03-01", "status": "current",
     "title": "ALMA Cycle 13 Technical Handbook"},
    {"filename": "alma-user-policies-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-user-policies",
     "slug": "alma-user-policies", "category": "user_policies", "cycle": "Cycle 13",
     "doc_number": "13.16", "version": "1.0", "doc_date": "2026-03", "status": "current",
     "title": "ALMA Cycle 13 Users' Policies"},
    {"filename": "alma-ot-usermanual-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-ot-usermanual",
     "slug": "alma-ot-usermanual", "category": "observing_tool", "cycle": "Cycle 13",
     "doc_number": "13.5", "version": "1.0", "doc_date": "2026-02", "status": "current",
     "title": "ALMA Cycle 13 Observing Tool User Manual (web-based OT)"},
    {"filename": "alma-ot-refmanual-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-ot-refmanual",
     "slug": "alma-ot-refmanual", "category": "observing_tool", "cycle": "Cycle 13",
     "doc_number": "13.6", "version": "1.0", "doc_date": "2026-02", "status": "current",
     "title": "ALMA Cycle 13 Observing Tool Reference Manual (web-based OT)"},
    {"filename": "alma-archive-primer-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/archive-primer",
     "slug": "archive-primer", "category": "archive", "cycle": "Cycle 13",
     "doc_number": "13.22", "version": "1.0", "doc_date": "2026-01", "status": "current",
     "title": "ALMA Cycle 13 Archive Primer"},
    {"filename": "alma-science-archive-manual-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/science-archive-manual",
     "slug": "science-archive-manual", "category": "archive_manual", "cycle": "Cycle 13",
     "doc_number": "12.15", "version": "1.0", "doc_date": "2026-03", "status": "current",
     "title": "ALMA Science Archive Manual"},
    {"filename": "alma-snoopi-user-manual-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/snoopi-user-manual-march2026version",
     "slug": "snoopi-user-manual", "category": "snoopi", "cycle": "Cycle 13",
     "doc_number": "13.11", "version": "1.0", "doc_date": "2026-03", "status": "current",
     "title": "ALMA SnooPI User Manual (Cycle 13)"},
    {"filename": "alma-large-program-data-products-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-large-program-data-products",
     "slug": "alma-large-program-data-products", "category": "large_programs", "cycle": "Cycle 13",
     "doc_number": "11.23", "version": "2.1", "doc_date": "2026-02-03", "status": "current",
     "title": "ALMA Large Program Data Products Standard"},
    {"filename": "alma-science-primer-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-science-primer",
     "slug": "alma-science-primer", "category": "primer", "cycle": "Cycle 13",
     "doc_number": "13.1", "version": "1.0", "doc_date": "2026-02-25", "status": "current",
     "title": "Observing with ALMA: A Primer (Cycle 13)"},
    {"filename": "alma-na-arcguide-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-na-arcguide",
     "slug": "alma-na-arcguide", "category": "arc_guide", "cycle": "Cycle 13",
     "doc_number": "13.9", "version": "1.0", "doc_date": "2026-03", "status": "current",
     "title": "Guide to the North American ALMA Regional Center and the NAASC (Cycle 13)"},
    {"filename": "alma-eu-arcguide-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-eu-arcguide",
     "slug": "alma-eu-arcguide", "category": "arc_guide", "cycle": "Cycle 13",
     "doc_number": "13.8", "version": "1.0", "doc_date": "2026-03", "status": "current",
     "title": "Guide to the European ALMA Regional Centre (Cycle 13)"},
    {"filename": "alma-ea-arcguide-cycle13.pdf", "url": f"{ALMA_BASE}/cycle13/alma-ea-arcguide",
     "slug": "alma-ea-arcguide", "category": "arc_guide", "cycle": "Cycle 13",
     "doc_number": "13.7", "version": "1.0", "doc_date": "2026-02-13", "status": "current",
     "title": "Guide to the East Asian ALMA Regional Center (Cycle 13)"},
    # ── Current-cycle documents that are still Cycle 12 editions ──
    {"filename": "alma-qa2-data-products-cycle12.pdf", "url": f"{ALMA_BASE}/cycle12/alma-qa2-data-products-for-cycle-12",
     "slug": "alma-qa2-data-products-for-cycle-13", "category": "qa2_products", "cycle": "Cycle 12",
     "doc_number": "12.12", "version": "1.0", "doc_date": "2025-10", "status": "current",
     "title": "ALMA QA2 Data Products for Cycle 12",
     "notes": "Cycle 13 edition expected around Oct 2026 (unverified); re-check after 1 Oct 2026."},
    {"filename": "alma_pipeline_users_guide_2025.pdf",
     "url": "https://almascience.org/processing/documents-and-tools/cycle12/alma_pipeline_users_guide_2025",
     "slug": "", "category": "pipeline", "cycle": "Cycle 12",
     "doc_number": "", "version": "1.1 (Pipeline 2025.1.0.35, CASA 6.6.6-17)", "doc_date": "2025-10", "status": "current",
     "title": "ALMA Science Pipeline User's Guide 2025.1.0.35",
     "notes": "Current release is 2025.1.0.37 on CASA 6.6.6-18 (see pipeline version note)."},
    {"filename": "alma-pipeline-reference-manual-2025.pdf",
     "url": "https://almascience.org/processing/documents-and-tools/cycle12/reference-manual-2025",
     "slug": "", "category": "pipeline", "cycle": "Cycle 12",
     "doc_number": "", "version": "Release 2025.1.0", "doc_date": "2025-09-29", "status": "current",
     "title": "ALMA Pipeline Tasks Reference Manual, release 2025.1.0"},
    # ── Old editions that the Cycle 13 documents page still links ──
    {"filename": "phase2_quickstartguide_cycle10-1.pdf", "url": f"{ALMA_BASE}/phase2_quickstartguide_cycle10-1.pdf",
     "slug": "", "category": "phase2", "cycle": "Cycle 10",
     "doc_number": "10.20", "version": "1.0", "doc_date": "2023-09", "status": "current-old",
     "title": "ALMA Phase 2 Quickstart Guide (Cycle 10 edition, still linked for Cycle 13)"},
    {"filename": "alma-scheduling-blocks-guide-cycle4.pdf", "url": f"{ALMA_BASE}/cycle4/users-guide-to-alma-scheduling-blocks",
     "slug": "users-guide-to-alma-scheduling-blocks", "category": "scheduling", "cycle": "Cycle 4",
     "doc_number": "4.19", "version": "1", "doc_date": "2016-08-02", "status": "current-old",
     "title": "A User's Guide to ALMA Scheduling Blocks (Cycle 4 edition, still linked for Cycle 13)"},
    {"filename": "alma-principles-review-process-cycle9.pdf", "url": f"{ALMA_BASE}/cycle9/principles-review-process",
     "slug": "principles-review-process", "category": "review_process", "cycle": "Cycle 9",
     "doc_number": "AEDM 2021-047-O Rev.3", "version": "", "doc_date": "2022-03-21", "status": "current-old",
     "title": "Principles of the ALMA Proposal Review Process (Cycle 9 edition, still linked for Cycle 13)"},
]

# Non-ALMA reference PDFs (other archives and IVOA standards).
EXTERNAL_DOCS: List[Dict[str, str]] = [
    {"filename": "ztf-explanatory-supplement.pdf",
     "url": "https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_explanatory_supplement.pdf",
     "category": "ztf", "cycle": "", "doc_number": "", "version": "5.0",
     "doc_date": "2020-06-10", "status": "reference", "title": "ZTF Science Data System Explanatory Supplement"},
    {"filename": "eso-phase3-science-data-products-standard.pdf",
     "url": "https://www.eso.org/sci/observing/phase3/p3sdpstd.pdf",
     "category": "eso", "cycle": "", "doc_number": "ESO-044286", "version": "8",
     "doc_date": "2022-03-15", "status": "reference", "title": "ESO Science Data Products Standard (Phase 3)"},
    {"filename": "eso-reduced-data-products-description.pdf",
     "url": "https://archive.eso.org/cms/eso-data/phase3/ESO_reduced_data_products_description.pdf",
     "category": "eso", "cycle": "", "doc_number": "", "version": "",
     "doc_date": "2020-06", "status": "reference", "title": "ESO Reduced Data Products Description"},
    {"filename": "ivoa-tap-1.1.pdf", "url": "https://www.ivoa.net/documents/TAP/20190927/REC-TAP-1.1.pdf",
     "category": "ivoa", "cycle": "", "doc_number": "REC-TAP-1.1", "version": "1.1",
     "doc_date": "2019-09-27", "status": "reference", "title": "IVOA Table Access Protocol 1.1"},
    {"filename": "ivoa-adql-2.1.pdf", "url": "https://www.ivoa.net/documents/ADQL/20231215/REC-ADQL-2.1.pdf",
     "category": "ivoa", "cycle": "", "doc_number": "REC-ADQL-2.1", "version": "2.1",
     "doc_date": "2023-12-15", "status": "reference", "title": "IVOA Astronomical Data Query Language 2.1"},
    {"filename": "ivoa-obscore-1.1.pdf", "url": "https://www.ivoa.net/documents/ObsCore/20170509/REC-ObsCore-v1.1-20170509.pdf",
     "category": "ivoa", "cycle": "", "doc_number": "REC-ObsCore-1.1", "version": "1.1",
     "doc_date": "2017-05-09", "status": "reference", "title": "IVOA Observation Data Model Core Components (ObsCore) 1.1"},
    {"filename": "ivoa-datalink-1.1.pdf", "url": "https://www.ivoa.net/documents/DataLink/20231215/REC-DataLink-1.1.pdf",
     "category": "ivoa", "cycle": "", "doc_number": "REC-DataLink-1.1", "version": "1.1",
     "doc_date": "2023-12-15", "status": "reference", "title": "IVOA DataLink 1.1"},
    {"filename": "ivoa-sia-2.0.pdf", "url": "https://www.ivoa.net/documents/SIA/20151223/REC-SIA-2.0-20151223.pdf",
     "category": "ivoa", "cycle": "", "doc_number": "REC-SIA-2.0", "version": "2.0",
     "doc_date": "2015-12-23", "status": "reference", "title": "IVOA Simple Image Access 2.0"},
    {"filename": "alma-memo-621-wsu.pdf", "url": "https://arxiv.org/pdf/2211.00195",
     "category": "wsu", "cycle": "", "doc_number": "ALMA Memo 621", "version": "",
     "doc_date": "2022-11-02", "status": "reference",
     "title": "ALMA Memo 621: The ALMA Wideband Sensitivity Upgrade (arXiv:2211.00195)"},
]

# Files that must NOT be in the collection any more (kept out of docs/pdfs;
# ingest_docs_folder --sync deletes their chunks).
RETIRED_FILES: Dict[str, str] = {
    "alma-proposers-guide.pdf": "Cycle 12 Proposer's Guide (Doc 12.2), superseded by Doc 13.2",
    "alma-technical-handbook.pdf": "byte-identical duplicate of alma-technical-handbook-cycle13.pdf",
    "alma-user-policies.pdf": "Cycle 12 Users' Policies (Doc 12.16), superseded by Doc 13.16",
    "alma-ot-usermanual.pdf": "Cycle 12 OT User Manual (Doc 12.5, retired Java OT)",
    "alma-ot-refmanual.pdf": "Cycle 12 OT Reference Manual (Doc 12.6, retired Java OT)",
    "alma-ot-quickstart.pdf": "Cycle 12 OT Quickstart (Doc 12.10, retired Java OT; no Cycle 13 edition)",
    "archive-primer.pdf": "Cycle 12 Archive Primer (Doc 12.22), superseded by Doc 13.22",
    "snoopi-user-manual.pdf": "Cycle 12 SnooPI manual (Doc 12.11), superseded by Doc 13.11",
    "alma-large-program-data-products.pdf": "Doc 7.23 (2020), superseded by Doc 11.23 v2.1",
    "reference-manual-2025.pdf": "renamed to alma-pipeline-reference-manual-2025.pdf",
    "users-guide-to-alma-scheduling-blocks.pdf": "renamed to alma-scheduling-blocks-guide-cycle4.pdf",
    "principles-review-process.pdf": "renamed to alma-principles-review-process-cycle9.pdf",
    "RLM.pdf": "LLM research paper, off-topic for the astronomy corpus",
    "mem0.pdf": "LLM research paper, off-topic for the astronomy corpus",
    "Downloading_Data_from_Alma.pdf": "image-only 2-page slide; content superseded by the Science Archive Manual",
}

MANIFEST: List[Dict[str, str]] = ALMA_DOCS + EXTERNAL_DOCS

# Local-only reference files (docs/references/): no download URL, ingested
# with the same metadata contract. Not probed by --verify.
LOCAL_REFERENCE_DOCS: List[Dict[str, str]] = [
    {"filename": "alma_tap_columns.txt", "url": "",
     "category": "archive", "cycle": "", "doc_number": "", "version": "",
     "doc_date": "2025-12", "status": "reference",
     "title": "ALMA TAP ivoa.obscore column reference (73 columns, from the ALMA TAP service)"},
    {"filename": "ALMA Pipeline Known Issues.txt", "url": "https://casaguides.nrao.edu/index.php/ALMA_Pipeline_Known_Issues",
     "category": "pipeline", "cycle": "", "doc_number": "", "version": "snapshot, PL2025 section at 2025.1.0.34",
     "doc_date": "2025-12", "status": "reference",
     "title": "ALMA Pipeline Known Issues (CASA Guides page snapshot, Dec 2025; the live page may list later fixes)"},
    {"filename": "alma-cycle14-status-note-2026-09.md",
     "url": "https://almascience.nrao.edu/news/web-based-ot-for-cycle-13-call-and-expected-capabilities-during-cycle-14",
     "category": "news", "cycle": "Cycle 14", "doc_number": "", "version": "",
     "doc_date": "2026-09-25", "status": "version-note",
     "title": "ALMA Cycle 14 status note (deadline not yet announced; capabilities unchanged; recheck at the Cycle 14 pre-announcement)"},
    {"filename": "alma-pipeline-version-note-2026-09.md", "url": "https://almascience.nrao.edu/processing/science-pipeline",
     "category": "pipeline", "cycle": "Cycle 12", "doc_number": "", "version": "Pipeline 2025.1.0.37 / CASA 6.6.6-18",
     "doc_date": "2026-09-25", "status": "version-note",
     "title": "ALMA pipeline and CASA version note (current release 2025.1.0.37 on CASA 6.6.6-18; recheck after 1 Oct 2026)"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Quasar-RAG-Downloader/1.1"
}


def manifest_entry(filename: str) -> Optional[Dict[str, str]]:
    """Return the manifest entry for a local filename, or None."""
    for doc in MANIFEST + LOCAL_REFERENCE_DOCS:
        if doc["filename"] == filename:
            return doc
    return None


def _parse_doc_date(doc_date: str) -> Dict[str, Optional[int]]:
    parts = [p for p in (doc_date or "").split("-") if p]
    out: Dict[str, Optional[int]] = {"doc_year": None, "doc_month": None, "doc_day": None}
    for key, part in zip(("doc_year", "doc_month", "doc_day"), parts):
        try:
            out[key] = int(part)
        except ValueError:
            break
    return out


def file_fetch_date(path: Optional[str]) -> str:
    """Date the local copy was downloaded (file mtime), else FETCH_DATE."""
    try:
        if path and os.path.exists(path):
            from datetime import datetime
            return datetime.fromtimestamp(os.path.getmtime(path)).date().isoformat()
    except OSError:
        pass
    return FETCH_DATE


def manifest_metadata(filename: str, path: Optional[str] = None) -> Optional[Dict[str, object]]:
    """Authoritative chunk metadata for a manifest file (None if unknown).

    Keys map onto the RAG payload schema: doc_* fields override what
    extract_document_metadata guesses from the PDF; the rest are added.
    ``fetched_at`` is the local file's download date when ``path`` is given.
    """
    doc = manifest_entry(filename)
    if doc is None:
        return None
    meta: Dict[str, object] = {
        "doc_category": doc["category"],
        "doc_title": doc["title"],
        "alma_cycle": doc.get("cycle", ""),
        "doc_number": doc.get("doc_number", ""),
        "doc_version": doc.get("version", ""),
        "doc_status": doc.get("status", ""),
        "source_url": doc.get("url") or "",
        "fetched_at": file_fetch_date(path),
    }
    for key, value in _parse_doc_date(doc.get("doc_date", "")).items():
        if value is not None:
            meta[key] = value
    meta["doc_label"] = doc_label(doc)
    return meta


def doc_label(doc: Dict[str, str]) -> str:
    """One-line document header prepended to every chunk (visible to the model)."""
    bits = [doc["title"]]
    ident = []
    if doc.get("doc_number"):
        ident.append(f"Doc {doc['doc_number']}" if doc["doc_number"][0].isdigit() else doc["doc_number"])
    if doc.get("version"):
        ident.append(f"v{doc['version']}" if doc["version"][0].isdigit() else doc["version"])
    if doc.get("doc_date"):
        ident.append(doc["doc_date"])
    if ident:
        bits.append("(" + ", ".join(ident) + ")")
    status = doc.get("status")
    if status == "current-old":
        bits.append("[older edition, still the one ALMA links for Cycle 13]")
    return " ".join(bits)


def download_pdf(url: str, dest_path: str, description: str) -> bool:
    """Download a PDF from a URL. Returns True on success."""
    try:
        print(f"  Downloading: {description}")
        print(f"     URL: {url}")
        response = requests.get(url, headers=HEADERS, stream=True, timeout=120, allow_redirects=True)
        response.raise_for_status()
        tmp_path = dest_path + ".part"
        first_bytes = b""
        total_size = 0
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if not first_bytes:
                    first_bytes = chunk[:5]
                f.write(chunk)
                total_size += len(chunk)
        if first_bytes[:5] != b"%PDF-":
            print(f"     Not a PDF (starts with {first_bytes[:20]!r}); discarded")
            os.remove(tmp_path)
            return False
        os.replace(tmp_path, dest_path)
        print(f"     Saved: {os.path.basename(dest_path)} ({total_size / 1048576:.1f} MB)")
        return True
    except Exception as e:  # network errors are reported, never fatal
        print(f"     Download failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────
# --verify
# ──────────────────────────────────────────────────────────────────

_CYCLE_PATH_RE = re.compile(r"documents-and-tools/cycle(\d{1,2})/([A-Za-z0-9_.-]+)")


def _remote_probe(url: str, get: Callable = requests.get) -> Dict[str, object]:
    """1-byte range request: returns {ok, status, size, content_type}."""
    try:
        r = get(url, headers={**HEADERS, "Range": "bytes=0-0"}, timeout=60, allow_redirects=True, stream=True)
        size = None
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            try:
                size = int(cr.rsplit("/", 1)[1])
            except ValueError:
                size = None
        if size is None and r.status_code == 200:
            try:
                size = int(r.headers.get("Content-Length", ""))
            except ValueError:
                size = None
        ctype = r.headers.get("Content-Type", "")
        final_url = str(getattr(r, "url", "") or url)
        r.close()
        return {"ok": r.status_code in (200, 206), "status": r.status_code, "size": size,
                "content_type": ctype, "final_url": final_url}
    except Exception as e:
        return {"ok": False, "status": None, "size": None, "content_type": "", "final_url": url, "error": str(e)}


def _entry_cycle_number(doc: Dict[str, str]) -> Optional[int]:
    m = re.search(r"cycle(\d{1,2})/", doc.get("url", ""))
    if m:
        return int(m.group(1))
    m = re.search(r"Cycle\s+(\d{1,2})", doc.get("cycle", ""))
    return int(m.group(1)) if m else None


def _slug_stem(slug: str) -> str:
    """Normalise a portal slug so 'snoopi-user-manual-march2026version' and
    'alma-qa2-data-products-for-cycle-12' compare by document identity."""
    s = slug.lower().replace("/view", "")
    s = re.sub(r"-(?:for-)?cycle-?\d+$", "", s)
    s = re.sub(r"-(?:[a-z]+)?20\d\dversion$", "", s)
    return s


def verify_manifest(
    docs_dir: str = DOCS_DIR,
    manifest: Optional[List[Dict[str, str]]] = None,
    retired: Optional[Dict[str, str]] = None,
    get: Callable = requests.get,
    index_url: Optional[str] = ALMA_INDEX_URL,
    known_filenames: Optional[set] = None,
) -> List[Dict[str, object]]:
    """Compare manifest entries with the portal and the local files.

    Returns one row per entry: {filename, status, detail}. ``get`` is
    injectable for tests (same signature as requests.get). ``manifest`` may be
    a subset (--only); ``known_filenames`` (default: the full manifest) decides
    which local PDFs count as LOCAL_ONLY, so a subset run does not flag the rest.

    Statuses: OK, CHANGED (size differs), NO_SIZE (remote gave no size, so
    unverified), NEWER_CYCLE, NOT_LINKED (a current-old entry the portal index
    no longer links), MISSING_LOCAL, UNREACHABLE, NOT_PDF, RETIRED_ON_DISK,
    LOCAL_ONLY.
    """
    manifest = MANIFEST if manifest is None else manifest
    retired = RETIRED_FILES if retired is None else retired
    known = known_filenames if known_filenames is not None else {d["filename"] for d in MANIFEST + list(manifest)}
    rows: List[Dict[str, object]] = []

    linked: Dict[str, int] = {}
    index_text: Optional[str] = None
    if index_url:
        try:
            r = get(index_url, headers=HEADERS, timeout=60)
            if getattr(r, "status_code", 200) != 200:
                raise ValueError(f"HTTP {r.status_code}")
            index_text = r.text or ""
            for cyc, slug in _CYCLE_PATH_RE.findall(index_text):
                stem = _slug_stem(slug)
                linked[stem] = max(linked.get(stem, 0), int(cyc))
        except Exception as e:
            rows.append({"filename": "(portal index)", "status": "UNREACHABLE", "detail": str(e)})

    for doc in manifest:
        path = os.path.join(docs_dir, doc["filename"])
        local_size = os.path.getsize(path) if os.path.exists(path) else None
        probe = _remote_probe(doc["url"], get=get)
        status, detail = "OK", ""
        if not probe["ok"]:
            status = "UNREACHABLE"
            detail = f"HTTP {probe.get('status')} {probe.get('error', '')}".strip()
        elif "pdf" not in str(probe["content_type"]).lower():
            status = "NOT_PDF"
            detail = f"content-type {probe['content_type']}"
        elif local_size is None:
            status = "MISSING_LOCAL"
            detail = f"remote {probe['size']} bytes"
        elif probe["size"] is None:
            status = "NO_SIZE"
            detail = "remote sent no Content-Range/Content-Length; size not compared"
        elif probe["size"] != local_size:
            status = "CHANGED"
            detail = f"remote {probe['size']} bytes vs local {local_size}"

        cyc = _entry_cycle_number(doc)
        slug = doc.get("slug") or ""
        if status in ("OK", "CHANGED", "MISSING_LOCAL", "NO_SIZE"):
            newest_linked = linked.get(_slug_stem(slug)) if slug else None
            if cyc is not None and newest_linked and newest_linked > cyc:
                status, detail = "NEWER_CYCLE", f"portal index links cycle{newest_linked}/ for {slug}"
            elif doc.get("status") == "current-old":
                # Old editions are current only while the portal still links them.
                marker = doc["url"].split("documents-and-tools/", 1)[-1]
                if index_text is not None and marker not in index_text:
                    status, detail = "NOT_LINKED", f"portal index no longer links {marker}"
            elif cyc is not None and slug:
                nxt = f"{ALMA_BASE}/cycle{cyc + 1}/{slug}"
                p2 = _remote_probe(nxt, get=get)
                # A redirect back to the current edition is not a newer one.
                if (p2["ok"] and "pdf" in str(p2["content_type"]).lower()
                        and f"cycle{cyc + 1}/" in str(p2.get("final_url", nxt))):
                    status, detail = "NEWER_CYCLE", f"{nxt} exists"
            elif not slug and status == "OK":
                detail = "size check only (no portal slug)"
        rows.append({"filename": doc["filename"], "status": status, "detail": detail})

    if os.path.isdir(docs_dir):
        for fn in sorted(os.listdir(docs_dir)):
            if not fn.lower().endswith(".pdf"):
                continue
            if fn in retired:
                rows.append({"filename": fn, "status": "RETIRED_ON_DISK", "detail": retired[fn]})
            elif fn not in known:
                rows.append({"filename": fn, "status": "LOCAL_ONLY", "detail": "not in manifest"})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Download / verify documentation PDFs for the RAG corpus")
    parser.add_argument("--list", action="store_true", help="List documents without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--verify", action="store_true", help="Report stale/changed entries; no download")
    parser.add_argument("--only", action="append", default=[], help="Limit to these filenames")
    args = parser.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)
    docs = [d for d in MANIFEST if not args.only or d["filename"] in args.only]

    if args.verify:
        rows = verify_manifest(manifest=docs, known_filenames={d["filename"] for d in MANIFEST})
        bad = 0
        for row in rows:
            if row["status"] != "OK":
                bad += 1
            print(f"  {row['status']:<16} {row['filename']:<50} {row['detail']}")
        print(f"\n{len(rows) - bad} OK, {bad} need attention")
        sys.exit(1 if bad else 0)

    existing = set(os.listdir(DOCS_DIR))
    if args.list:
        for doc in docs:
            state = "EXISTS" if doc["filename"] in existing else "NEW"
            print(f"  [{state}] {doc['filename']}  ({doc['category']}, {doc.get('cycle') or '-'}, {doc['status']})")
            print(f"         {doc['url']}")
        return

    results = {"success": 0, "skipped": 0, "failed": 0}
    for doc in docs:
        dest_path = os.path.join(DOCS_DIR, doc["filename"])
        if doc["filename"] in existing and not args.force:
            results["skipped"] += 1
            continue
        if download_pdf(doc["url"], dest_path, doc["title"]):
            results["success"] += 1
        else:
            results["failed"] += 1
    print(f"\nDownloaded {results['success']}, skipped {results['skipped']}, failed {results['failed']}")
    print("Next: python scripts/ingest_docs_folder.py --sync")


if __name__ == "__main__":
    main()
