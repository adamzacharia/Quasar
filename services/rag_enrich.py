# services/rag_enrich.py
"""
Per-chunk enrichment for the documentation RAG corpus (2026-09-26 rebuild).

Pure functions used by RAGService.ingest_document:

  * page numbers   -- PyMuPDF4LLM's ``page`` metadata is 0-based; citations
                      must show the 1-based physical page (``pdf_page``), plus
                      the printed page number when a header/footer shows one.
  * sections       -- the Markdown heading path in force at each chunk
                      ("9 Data access > 9.4 Proprietary periods"), carried
                      across page boundaries.
  * PDF metadata   -- normalised creation/modification dates, producer,
                      author, title (the loader's raw duplicated keys and local
                      file paths are dropped from payloads).
  * chunk facts    -- content kind (text / table / code), size in characters
                      and tokens, table-of-contents detection, facility.
"""
from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

# Loader metadata that must never reach a payload: absolute local paths and the
# raw, duplicated PDF info keys (replaced by the normalised pdf_* fields).
LOADER_KEYS_TO_DROP = {
    "source", "file_path", "producer", "creator", "creationdate", "creationDate",
    "moddate", "modDate", "format", "title", "author", "subject", "keywords", "trapped",
    "total_pages",
}

# doc_category -> facility (lets retrieval filter or boost by facility later).
FACILITY_BY_CATEGORY = {
    "jwst": "JWST", "eso": "ESO", "vla": "VLA", "data_lab": "NOIRLab Data Lab",
    "desi": "DESI", "legacy_surveys": "Legacy Surveys", "ivoa": "IVOA", "ztf": "ZTF",
    "gaia": "Gaia",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)
# "9", "9.4", "9.4.1", "A.2" at the start of a heading: its depth is more
# reliable than the converter's font-size-based Markdown level.
_NUMBERED_RE = re.compile(r"^(?:(?:chapter|section|appendix)\s+)?((?:\d{1,2}|[A-Z](?=\.\d))(?:\.\d{1,2})*)\.?\s+\S", re.I)


def heading_number(title: str) -> str:
    """"9.4.1" for "9.4.1 Proprietary period ...", "" if unnumbered."""
    m = _NUMBERED_RE.match(title or "")
    return m.group(1).upper() if m else ""


def heading_number_depth(title: str) -> int:
    """1 for "9 Data access", 2 for "9.4 ...", 3 for "9.4.1 ..."; 0 if unnumbered."""
    num = heading_number(title)
    return num.count(".") + 1 if num else 0
_TOC_LINE_RE = re.compile(r"(?:\.\s?){4,}\s*\d{1,3}\s*$|\.{4,}\s*\d{1,3}\s*$")
_PDF_DATE_RE = re.compile(r"(?:D:)?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(Z|[+-]\d{2}'?\d{2}'?)?")


def clean_heading(text: str) -> str:
    """Strip Markdown emphasis / links from a heading line."""
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    t = t.replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"(?<!\w)[*_](?!\s)|(?<!\s)[*_](?!\w)", "", t)
    return re.sub(r"\s+", " ", t).strip(" :-")


_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")


def fenced_spans(text: str) -> List[Tuple[int, int]]:
    """(start, end) character spans of fenced code blocks. A fence closes only
    on the same marker character, at least as long as the opener, with nothing
    after it; an unclosed fence runs to the end of the text."""
    spans, opener, start, pos = [], "", 0, 0
    for line in (text or "").splitlines(keepends=True):
        m = _FENCE_RE.match(line.rstrip("\r\n"))
        if m:
            marker, rest = m.group(1), m.group(2)
            if not opener:
                opener, start = marker, pos
            elif marker[0] == opener[0] and len(marker) >= len(opener) and not rest.strip():
                spans.append((start, pos + len(line)))
                opener = ""
        pos += len(line)
    if opener:
        spans.append((start, pos))
    return spans


def headings_in(text: str) -> List[Tuple[int, int, str]]:
    """(char_offset, level, title) for every Markdown heading in ``text``."""
    out = []
    spans = fenced_spans(text)
    for m in _HEADING_RE.finditer(text or ""):
        # a "## example" line inside a fenced code block is code, not a heading (CX-03)
        if any(s <= m.start() < e for s, e in spans):
            continue
        title = clean_heading(m.group(2))
        if title and len(title) <= 200:
            out.append((m.start(), len(m.group(1)), title))
    return out


class SectionTracker:
    """Heading-path lookup for (page_index, char_offset) across a document.

    Built once per document from the per-page texts; a chunk's section is the
    heading stack in force at its start offset, inheriting the previous pages'
    stack when a page starts without a heading."""

    def __init__(self, pages: Sequence[str], max_depth: int = 3):
        self.max_depth = max_depth
        self._pages: list = []
        # stack entries: (level, title, number) -- number "" if unnumbered
        stack: Tuple[Tuple[int, str, str], ...] = ()
        for text in pages:
            start_stack = stack
            offsets, stacks = [], []
            for off, level, title in headings_in(text):
                num = heading_number(title)
                if num:
                    # A numbered heading keeps only its true numbered ancestors
                    # ("9", "9.4" for "9.4.1"), dropping unnumbered headings
                    # (document title, "Revision History", captions) and
                    # unrelated sections left on the stack.
                    kept = tuple(h for h in stack if h[2] and num.startswith(h[2] + "."))
                    stack = kept + ((num.count(".") + 1, title, num),)
                else:
                    kept = tuple(h for h in stack if h[0] < level)
                    stack = kept + ((level, title, ""),)
                offsets.append(off)
                stacks.append(stack)
            self._pages.append((offsets, stacks, start_stack))

    def section_at(self, page_index: int, offset: int) -> str:
        if not (0 <= page_index < len(self._pages)):
            return ""
        offsets, stacks, start_stack = self._pages[page_index]
        i = bisect_right(offsets, max(offset, 0)) - 1
        stack = stacks[i] if i >= 0 else start_stack
        return " > ".join(h[1] for h in stack[-self.max_depth:])


def is_toc_chunk(text: str) -> bool:
    """True for table-of-contents / index chunks (dotted leaders to page numbers)."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    # PyMuPDF4LLM wraps TOC lines in emphasis ("... 17_", "**1 WHAT'S NEW .... 3**").
    hits = sum(1 for ln in lines if _TOC_LINE_RE.search(ln.strip().rstrip("*_` ")))
    # A real table/list can have a few dotted rows; a TOC is mostly them, or
    # says so (guard CX-05).
    says_toc = bool(re.search(r"\b(?:table\s+of\s+)?contents\b", text, re.I))
    return hits >= 3 and hits / len(lines) >= (0.3 if says_toc else 0.5)


def content_kind(text: str) -> str:
    """'table' | 'code' | 'text' for a chunk (Markdown-aware)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "text"
    if "```" in text or sum(1 for ln in lines if ln.startswith((">>>", "$ ", "import ", "from ")) ) >= 3:
        return "code"
    if sum(1 for ln in lines if ln.startswith("|") and ln.count("|") >= 3) >= max(3, len(lines) // 3):
        return "table"
    return "text"


def printed_page_label(page_text: str) -> str:
    """The page number printed in a header/footer, when there is a bare one."""
    lines = [ln.strip() for ln in (page_text or "").splitlines() if ln.strip()]
    for ln in (lines[-3:][::-1] + lines[:2]):
        m = re.fullmatch(r"(?:page\s+)?(\d{1,4}|[ivxlcdm]{1,7})(?:\s+of\s+\d{1,4})?", ln, re.I)
        if m:
            return m.group(1)
    return ""


def normalize_pdf_date(value: object) -> str:
    """'D:20260226135411Z00'00'' / '2026-02-26T13:54:11+00:00' -> ISO date(time), or ''."""
    s = str(value or "").strip()
    if not s:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            # keep a valid date with an unparseable time; drop impossible dates
            try:
                return date.fromisoformat(s[:10]).isoformat()
            except ValueError:
                return ""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    m = _PDF_DATE_RE.match(s)
    if not m or not m.group(2):
        return ""
    y, mo, d, hh, mm, ss, tz = (m.group(i) for i in range(1, 8))
    offset = timedelta(0)
    if tz and tz != "Z":
        digits = tz[1:].replace("'", "")
        try:
            offset = timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0))
        except ValueError:
            offset = timedelta(0)
        if tz[0] == "-":
            offset = -offset
    try:
        local = datetime(int(y), int(mo), int(d or 1), int(hh or 0), int(mm or 0), int(ss or 0),
                         tzinfo=timezone(offset))
    except ValueError:
        return ""
    # Always UTC with an explicit Z, so offsets are never silently dropped (CX-06).
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pdf_document_facts(loader_meta: Dict[str, object]) -> Dict[str, str]:
    """Normalised PDF info fields from PyMuPDF loader metadata (empty values dropped)."""
    facts = {
        "pdf_created": normalize_pdf_date(loader_meta.get("creationDate") or loader_meta.get("creationdate")),
        "pdf_modified": normalize_pdf_date(loader_meta.get("modDate") or loader_meta.get("moddate")),
        "pdf_producer": str(loader_meta.get("producer") or "").strip(),
        "pdf_author": str(loader_meta.get("author") or "").strip(),
        "pdf_title": str(loader_meta.get("title") or "").strip(),
        "pdf_format": str(loader_meta.get("format") or "").strip(),
    }
    return {k: v for k, v in facts.items() if v}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


_ENC = None


def token_count(text: str) -> int:
    """cl100k_base token count (the embedding model's tokenizer); 0 if unavailable."""
    global _ENC
    try:
        if _ENC is None:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        return len(_ENC.encode_ordinary(text or ""))
    except Exception:
        return 0


def iso_doc_date(year: Optional[int], month: Optional[int], day: Optional[int]) -> str:
    if not year:
        return ""
    if month and day:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    if month:
        return f"{int(year):04d}-{int(month):02d}"
    return f"{int(year):04d}"


def chunk_header(label: str, section: str, page: Optional[int], page_label: str) -> str:
    """One-line header prepended to each chunk (the model sees it; CITE_AS
    built in core/runner.py only carries file, page and date)."""
    parts = [label] if label else []
    if section:
        parts.append(f"Section: {section}")
    if page:
        p = f"PDF page {page}"
        if page_label and page_label != str(page):
            p += f" (printed page {page_label})"
        parts.append(p)
    return "[Document: " + " | ".join(parts) + "]" if parts else ""
