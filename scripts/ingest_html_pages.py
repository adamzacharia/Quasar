"""
HTML documentation pages -> Markdown -> RAG corpus (alma_general).

ingest_docs_folder.py only accepts .pdf/.txt/.md, so web documentation (ALMA
news and OT FAQ pages, Data Lab manual, JDox, ESO policy, VLA OSS, DESI docs,
...) goes through this script:

  1. fetch each page in PAGES (polite: per-host delay, browser-like UA), or
     read a browser-saved copy (``saved_html``) for hosts that block scripts;
  2. convert the main content to Markdown (``html_to_markdown``, dependency
     free: BeautifulSoup only);
  3. save docs/references/pages/<category>/<filename> with a front-matter
     block (title, source_url, fetched_at, category, page_date);
  4. ingest the Markdown body with authoritative metadata (category, cycle,
     source_url, fetched_at, doc date, a one-line document header on each
     chunk), replacing that page's previous chunks.

Usage:
    python scripts/ingest_html_pages.py --estimate          # fetch + convert, token estimate, no writes
    python scripts/ingest_html_pages.py                     # fetch, convert, ingest all pages
    python scripts/ingest_html_pages.py --only news-alma-cycle13-early-planning.md
    python scripts/ingest_html_pages.py --no-fetch          # re-ingest the saved Markdown only
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from datetime import date
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(PROJECT_ROOT, "docs", "references", "pages")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Quasar-RAG-Downloader/1.1"}
HOST_DELAY_S = 2.0  # help.almascience.org asks crawl-delay 2; use it everywhere
MIN_BODY_CHARS = 200  # shorter extractions are treated as failed pages

# ──────────────────────────────────────────────────────────────────
# HTML -> Markdown
# ──────────────────────────────────────────────────────────────────

_DROP_TAGS = {
    "script", "style", "noscript", "nav", "footer", "aside", "form",
    "iframe", "svg", "button", "select", "input", "textarea", "template", "img",
    "picture", "video", "audio", "canvas",
}
# <link>/<meta> are void elements and normally empty, but malformed pages make
# html.parser nest real content inside them (the ESO data access policy page
# wraps its whole text in a <link>), so they are rendered transparently.
_DROP_SELECTORS = [
    "[role=navigation]", "[role=banner]", "[role=contentinfo]", "[role=search]",
    "[aria-hidden=true]", ".breadcrumb", ".breadcrumbs", "#portal-breadcrumbs",
    ".sphinxsidebar", "div.related", ".headerlink", "#viewlet-below-content",
    "#viewlet-above-content", ".documentByLine", "#portal-footer", ".skip-link",
    ".sr-only", ".visually-hidden", ".cookie-banner", "#cookie-banner", ".toc-macro",
    ".sidebar", "#sidebar", ".social-share", ".print-only",
    ".navigation", ".leftnavi", ".smenu",  # ESO CMS left menu inside #main
]
_MAIN_SELECTORS = [
    "[role=main]", "main", "article", "#content-core", "#content", "div.document",
    "div.body", "#main-content", "#main", ".main-content",
]
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "ul", "ol", "li", "pre", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "dl", "dt", "dd", "hr",
    "figure", "figcaption", "details", "summary", "tbody", "thead", "tr", "center",
    "header",
}
_WS_RE = re.compile(r"\s+")


def _is_site_header(node: Tag) -> bool:
    """A <header> outside any article/section/main is the site or page banner
    and is dropped; a header inside one (an article's title, byline or
    deadline) is content and is kept."""
    return (
        node.name == "header"
        and node.find_parent(["article", "section", "main"]) is None
        and node.find_parent(attrs={"role": "main"}) is None
    )


def _clean_inline(text: str) -> str:
    return _WS_RE.sub(" ", text)


def _abs_href(href: str, base_url: str) -> str:
    href = (href or "").strip()
    if not href or href.startswith(("#", "javascript:", "mailto:")):
        return ""
    return urljoin(base_url, href) if base_url else href


def _inline(node, base_url: str) -> str:
    """Render inline content of a node to a single Markdown line."""
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return _clean_inline(str(node))
    if not isinstance(node, Tag):
        return ""
    name = node.name
    if name in _DROP_TAGS or _is_site_header(node):
        return ""
    if name == "br":
        return " "
    inner = "".join(_inline(c, base_url) for c in node.children)
    if name in ("strong", "b"):
        s = inner.strip()
        return f" **{s}** " if s else ""
    if name in ("em", "i"):
        s = inner.strip()
        return f" *{s}* " if s else ""
    if name == "code":
        s = inner.strip()
        return f"`{s}`" if s else ""
    if name == "a":
        s = inner.strip()
        href = _abs_href(node.get("href", ""), base_url)
        if not s:
            return ""
        if href and href.startswith("http") and href != s:
            return f"[{s}]({href})"
        return s
    if name in _BLOCK_TAGS:
        # Block children rendered inline (paragraphs inside a table cell)
        # must not run together: "<p>First.</p><p>Second.</p>" (CX-10).
        return f" {inner} "
    return inner


def _table(node: Tag, base_url: str) -> str:
    rows: List[List[str]] = []
    for tr in node.find_all("tr"):
        if tr.find_parent("table") is not node:
            continue  # nested table rows belong to the inner table
        cells = [
            _clean_inline(_inline(c, base_url)).strip().replace("|", "\\|")
            for c in tr.find_all(["th", "td"], recursive=False)
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if width == 1:  # layout table: plain lines
        return "\n".join(r[0] for r in rows if r[0])
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _blocks(node, base_url: str, depth: int = 0) -> List[str]:
    """Render a node's children to a list of Markdown blocks."""
    out: List[str] = []
    buf: List[str] = []

    def flush():
        text = _clean_inline("".join(buf)).strip()
        buf.clear()
        if text:
            out.append(text)

    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            buf.append(str(child))
            continue
        if not isinstance(child, Tag) or child.name in _DROP_TAGS or _is_site_header(child):
            continue
        name = child.name
        if name not in _BLOCK_TAGS:
            buf.append(_inline(child, base_url))
            continue
        flush()
        if re.fullmatch(r"h[1-6]", name):
            text = _clean_inline(_inline(child, base_url)).strip()
            if text:
                out.append("#" * int(name[1]) + " " + text)
        elif name in ("ul", "ol"):
            n = 0
            for li in child.find_all("li", recursive=False):
                n += 1
                marker = f"{n}." if name == "ol" else "-"
                sub = _blocks(li, base_url, depth + 1)
                if not sub:
                    continue
                indent = "  " * depth
                first, rest = sub[0], sub[1:]
                item = [f"{indent}{marker} {first}"]
                item += [s if s.lstrip().startswith(("-", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
                         else f"{indent}  {s}" for s in rest]
                out.append("\n".join(item))
        elif name == "pre":
            code = child.get_text().strip("\n")
            if code.strip():
                out.append(f"```\n{code}\n```")
        elif name == "table":
            t = _table(child, base_url)
            if t:
                out.append(t)
        elif name == "dt":
            text = _clean_inline(_inline(child, base_url)).strip()
            if text:
                out.append(f"**{text}**")
        elif name == "blockquote":
            inner = _blocks(child, base_url, depth)
            if inner:
                out.append("\n".join("> " + line for b in inner for line in b.split("\n")))
        elif name == "hr":
            continue
        else:  # p, div, section, li, dd, figure, details, ...
            out.extend(_blocks(child, base_url, depth))
    flush()
    return out


_TITLE_SEP_RE = re.compile("\\s[-|\u2013\u2014]\\s")


def _tidy_title(t: str) -> str:
    return _clean_inline(t).replace("\u00b6", "").strip(" -|\u2013\u2014")


def page_title(soup: BeautifulSoup, root: Optional[Tag] = None) -> str:
    """The page's own heading first (og:title is often the site name,
    e.g. JDox 'JWST User Documentation'), then meta titles, then <title>.
    An <h1> equal to the site name (from og:site_name or the <title> suffix)
    is skipped."""
    site = ""
    m = soup.select_one("meta[property='og:site_name']")
    if m and m.get("content"):
        site = _tidy_title(m["content"])
    elif soup.title and soup.title.string and _TITLE_SEP_RE.search(soup.title.string):
        site = _tidy_title(_TITLE_SEP_RE.split(soup.title.string)[-1])
    for scope in ([root] if root is not None else []) + [soup]:
        for h1 in scope.find_all("h1"):
            text = _tidy_title(h1.get_text(" "))
            if text and text.lower() != site.lower():
                return text
    for sel in ("meta[property='og:title']", "meta[name='DC.title']"):
        m = soup.select_one(sel)
        if m and m.get("content"):
            return _tidy_title(m["content"])
    if soup.title and soup.title.string:
        return _tidy_title(soup.title.string)
    return ""


_DATE_META = (
    "article:modified_time", "DC.date.modified", "dcterms.modified", "DC.date",
    "article:published_time", "dcterms.created", "date", "last-modified",
)


def page_date(soup: BeautifulSoup) -> Optional[str]:
    """Best-effort page date (YYYY-MM-DD) from standard meta tags."""
    for key in _DATE_META:
        m = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if m and m.get("content"):
            hit = re.search(r"(20\d\d)-(\d\d)-(\d\d)", m["content"])
            if hit:
                return hit.group(0)
    t = soup.find("time", attrs={"datetime": True})
    if t:
        hit = re.search(r"(20\d\d)-(\d\d)-(\d\d)", t["datetime"])
        if hit:
            return hit.group(0)
    return None


def select_main(soup: BeautifulSoup, selector: Optional[str] = None) -> Tag:
    """The element holding the page's main content."""
    if selector:
        el = soup.select_one(selector)
        if el is not None:
            return el
    # The first NON-EMPTY element in priority order wins, however short: a
    # one-line <main> must beat an enclosing #content or the whole <body>,
    # which would pull in newsletters and "popular stories" (CX-09). Empty
    # anchors such as ESO's <a id="content"> are skipped.
    for sel in _MAIN_SELECTORS:
        el = soup.select_one(sel)
        if el is not None and el.get_text(" ", strip=True):
            return el
    return soup.body or soup


def html_to_markdown(html: str, base_url: str = "", selector: Optional[str] = None) -> Tuple[str, str]:
    """Convert an HTML page to (title, markdown) keeping only the main content.

    Drops scripts, navigation, headers/footers, breadcrumbs and sidebars;
    renders headings, paragraphs, (nested) lists, tables, code blocks,
    definition lists and links (made absolute against ``base_url``).
    """
    soup = BeautifulSoup(html or "", "html.parser")
    for sel in _DROP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    root = select_main(soup, selector)
    title = page_title(soup, root)
    blocks = _blocks(root, base_url)
    md = "\n\n".join(b for b in blocks if b.strip())
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return title, md


# ──────────────────────────────────────────────────────────────────
# Front matter
# ──────────────────────────────────────────────────────────────────

_FM_KEYS = ("title", "source_url", "fetched_at", "page_date", "category", "cycle", "http_last_modified")

# Last-Modified header per fetched URL (informational only: CMS pages often
# send the fetch time, so it is recorded but never used as the page date).
HTTP_LAST_MODIFIED: Dict[str, str] = {}


def _http_date(value: str) -> str:
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(value).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return ""


def with_front_matter(meta: Dict[str, str], body: str) -> str:
    lines = ["---"] + [f"{k}: {str(meta.get(k, '')).strip()}" for k in _FM_KEYS if meta.get(k)] + ["---", ""]
    return "\n".join(lines) + body.strip() + "\n"


def split_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    meta: Dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[end + 5:].lstrip("\n")


# ──────────────────────────────────────────────────────────────────
# Page manifest
# ──────────────────────────────────────────────────────────────────
# filename: the source_file shown in CITE_AS; keep it descriptive and put the
#   cycle/semester in it so a citation alone says which edition it is.
# doc_date: the page's own date when known (news post date etc.); otherwise
#   page_date() from meta tags, else the fetch date.
# selector: optional CSS selector for the content root.
_A = "https://almascience.nrao.edu"
_DL = "https://datalab.noirlab.edu/docs/manual"

PAGES: List[Dict[str, str]] = [
    # ── ALMA: Cycle 13 call, news, OT FAQ, pipeline status, statistics, WSU ──
    {"filename": "news-alma-cycle13-early-planning-2025-12.md", "url": f"{_A}/news/announcement-for-early-proposal-planning-for-cycle-13",
     "category": "news", "cycle": "Cycle 13", "doc_date": "2025-12-22"},
    {"filename": "news-alma-cycle13-band2-update.md", "url": f"{_A}/news/update-on-band-2-for-early-proposal-planning-for-cycle-13",
     "category": "news", "cycle": "Cycle 13"},
    {"filename": "news-alma-web-ot-cycle13-and-cycle14-capabilities-2026-03.md",
     "url": f"{_A}/news/web-based-ot-for-cycle-13-call-and-expected-capabilities-during-cycle-14",
     "category": "news", "cycle": "Cycle 13", "doc_date": "2026-03-02"},
    {"filename": "alma-cycle13-call-for-proposals-page.md", "url": f"{_A}/proposing/call-for-proposals",
     "category": "news", "cycle": "Cycle 13"},
    {"filename": "alma-ot-faq-and-known-issues-cycle13.md", "url": f"{_A}/proposing/observing-tool/faq-and-known-issues",
     "category": "observing_tool", "cycle": "Cycle 13"},
    {"filename": "alma-science-pipeline-overview-page.md", "url": f"{_A}/processing/science-pipeline",
     "category": "pipeline", "cycle": ""},
    {"filename": "alma-proposal-statistics-page.md", "url": f"{_A}/documents-and-tools/alma-proposal-statistics",
     "category": "proposal_statistics", "cycle": ""},
    {"filename": "nrao-alma-wsu-page.md", "url": "https://science.nrao.edu/facilities/alma/science_sustainability/wideband-sensitivity-upgrade",
     "category": "wsu", "cycle": ""},
    {"filename": "alma-observatory-wsu-program-page.md", "url": "https://www.almaobservatory.org/en/scientists/alma-2030-wsu/wsu-program/",
     "category": "wsu", "cycle": ""},
    {"filename": "nrao-alma-ambassadors-cycle14.md", "url": "https://science.nrao.edu/facilities/alma/ambassadors-program",
     "category": "news", "cycle": "Cycle 14"},
    # ALMA Science Archive notebooks missing locally (nb7's local .ipynb is
    # truncated; nb8/nb9 were never cloned). nb9 replaces the image-only
    # Downloading_Data_from_Alma.pdf, which was a browser print of it.
    {"filename": "alma-archive-notebook-nb7-query-by-sensitivity.md",
     "url": f"{_A}/alma-data/archive/archive-notebooks/nb7_ALMA_Query_by_sensitivity.html", "category": "tutorial"},
    {"filename": "alma-archive-notebook-nb8-query-using-astroquery.md",
     "url": f"{_A}/alma-data/archive/archive-notebooks/nb8_ALMA_Query_using_astroquery.html", "category": "tutorial"},
    {"filename": "alma-archive-notebook-nb9-download-data.md",
     "url": f"{_A}/alma-data/archive/archive-notebooks/nb9_ALMA_Download_data.html", "category": "tutorial"},
    # ── NOIRLab Astro Data Lab ──
    {"filename": "datalab-data-surveys-index.md", "url": "https://datalab.noirlab.edu/data", "category": "data_lab"},
    {"filename": "datalab-dataset-legacy-surveys-ls-dr10-dr11.md", "url": "https://datalab.noirlab.edu/data/legacy-surveys", "category": "data_lab"},
    {"filename": "datalab-dataset-desi-edr-dr1.md", "url": "https://datalab.noirlab.edu/data/desi", "category": "data_lab"},
    {"filename": "datalab-manual-index.md", "url": f"{_DL}/", "category": "data_lab"},
    {"filename": "datalab-manual-faqs.md", "url": f"{_DL}/FAQs/FAQs.html", "category": "data_lab"},
    {"filename": "datalab-manual-introduction.md", "url": f"{_DL}/UsingAstroDataLab/Introduction/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-data-access-interfaces.md", "url": f"{_DL}/UsingAstroDataLab/DataAccessInterfaces/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-guidance-constructing-queries.md", "url": f"{_DL}/UsingAstroDataLab/GuidanceConstructingQueries/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-sql-gotchas.md", "url": f"{_DL}/UsingAstroDataLab/SQLGotchas/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-known-issues.md", "url": f"{_DL}/UsingAstroDataLab/KnownIssues/KnownIssues/KnownIssues.html", "category": "data_lab"},
    {"filename": "datalab-manual-client-interfaces.md", "url": f"{_DL}/UsingAstroDataLab/ClientInterfaces/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-service-interfaces.md", "url": f"{_DL}/UsingAstroDataLab/ServiceInterfaces/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-web-interfaces.md", "url": f"{_DL}/UsingAstroDataLab/WebInterfaces/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-web-portal.md", "url": f"{_DL}/UsingAstroDataLab/WebPortal/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-compute-processing.md", "url": f"{_DL}/UsingAstroDataLab/ComputeProcessing/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-upload-external-data.md", "url": f"{_DL}/UsingAstroDataLab/UploadExternalData/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-jupyter-notebooks.md", "url": f"{_DL}/UsingAstroDataLab/JupyterNotebooks/JupyterNotebooks.html", "category": "data_lab"},
    {"filename": "datalab-manual-command-line-tools.md", "url": f"{_DL}/UsingAstroDataLab/CommandLineTools/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-install.md", "url": f"{_DL}/UsingAstroDataLab/InstallDataLab/index.html", "category": "data_lab"},
    {"filename": "datalab-manual-example-queries.md", "url": f"{_DL}/Appendices/ExampleQueries/ExampleQueries.html", "category": "data_lab"},
    {"filename": "datalab-manual-glossary.md", "url": f"{_DL}/Appendices/Glossary/Glossary.html", "category": "data_lab"},
    # ── Legacy Surveys / DESI ──
    {"filename": "legacysurvey-dr11-index.md", "url": "https://www.legacysurvey.org/dr11/", "category": "legacy_surveys"},
    {"filename": "legacysurvey-dr11-description.md", "url": "https://www.legacysurvey.org/dr11/description/", "category": "legacy_surveys"},
    {"filename": "desi-dr1-release-docs.md", "url": "https://data.desi.lbl.gov/doc/releases/dr1/", "category": "desi"},
    {"filename": "desi-dr1-known-issues.md", "url": "https://data.desi.lbl.gov/doc/releases/dr1/known-issues/", "category": "desi"},
    # ── JWST (JDox Cycle 6) ──
    {"filename": "jwst-cycle6-call-for-proposals-jdox.md",
     "url": "https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-call-for-proposals-for-cycle-6",
     "category": "jwst", "cycle": "JWST Cycle 6"},
    {"filename": "jwst-general-science-policies-jdox.md",
     "url": "https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-general-science-policies",
     "category": "jwst", "cycle": ""},
    {"filename": "jwst-cycle6-key-policies-jdox.md",
     "url": "https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-call-for-proposals-for-cycle-6/jwst-key-policies",
     "category": "jwst", "cycle": "JWST Cycle 6"},
    {"filename": "jwst-cycle6-new-features-and-updates-jdox.md",
     "url": "https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-call-for-proposals-for-cycle-6/jwst-new-features-and-updates",
     "category": "jwst", "cycle": "JWST Cycle 6"},
    {"filename": "jwst-cycle6-how-your-proposal-is-evaluated-jdox.md",
     "url": "https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-call-for-proposals-for-cycle-6/jwst-how-your-proposal-is-evaluated",
     "category": "jwst", "cycle": "JWST Cycle 6"},
    {"filename": "jwst-policy-data-rights-and-exclusive-access-jdox.md",
     "url": "https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-general-science-policies/nasa-smd-policies-and-guidelines-for-the-operations-of-jwst-at-stsci/policy-2-data-rights-and-data-dissemination",
     "category": "jwst", "cycle": ""},
    {"filename": "jwst-duplicate-observations-policy-jdox.md",
     "url": "https://jwst-docs.stsci.edu/jwst-opportunities-and-policies/jwst-general-science-policies/jwst-duplicate-observations-policy",
     "category": "jwst", "cycle": ""},
    {"filename": "jwst-cycle6-timeline-stsci-news.md",
     "url": "https://www.stsci.edu/contents/news/jwst/2026/stsci-announces-the-timeline-for-the-jwst-cycle-6-call-for-proposals",
     "category": "jwst", "cycle": "JWST Cycle 6"},
    # ── ESO ──
    {"filename": "eso-data-access-policy.md", "url": "https://archive.eso.org/cms/eso-data-access-policy.html", "category": "eso"},
    {"filename": "eso-archive-faq.md", "url": "https://archive.eso.org/cms/faq.html", "category": "eso"},
    # ── NRAO VLA ──
    {"filename": "vla-oss-2027-introduction.md", "url": "https://science.nrao.edu/facilities/vla/docs/manuals/oss", "category": "vla"},
    {"filename": "vla-oss-2027-proposing.md", "url": "https://science.nrao.edu/facilities/vla/docs/manuals/oss/proposing", "category": "vla"},
    {"filename": "vla-oss-2027-performance.md", "url": "https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance", "category": "vla"},
    {"filename": "vla-oss-2027-widar.md", "url": "https://science.nrao.edu/facilities/vla/docs/manuals/oss/widar", "category": "vla"},
    {"filename": "vla-config-plans-and-proposal-deadlines.md",
     "url": "https://science.nrao.edu/facilities/vla/proposing/configpropdeadlines", "category": "vla"},
    # ── Release timelines (other missions) ──
    {"filename": "news-gaia-dr4-release-page.md", "url": "https://www.cosmos.esa.int/web/gaia/data-release-4", "category": "news"},
    {"filename": "news-euclid-dr1-timeline-page.md", "url": "https://www.cosmos.esa.int/web/euclid/dr1-timeline", "category": "news"},
]


def page_entry(filename: str) -> Optional[Dict[str, str]]:
    for p in PAGES:
        if p["filename"] == filename:
            return p
    return None


def page_path(page: Dict[str, str], pages_dir: str = PAGES_DIR) -> str:
    return os.path.join(pages_dir, page["category"], page["filename"])


def page_metadata(page: Dict[str, str], title: str, fetched_at: str, found_date: Optional[str]) -> Dict[str, object]:
    """Authoritative payload metadata for one converted page."""
    doc_date = page.get("doc_date") or found_date or fetched_at
    y, m, d = (doc_date.split("-") + [None, None])[:3]
    host = urlparse(page["url"]).netloc
    label = f"{title} (web page {host}, fetched {fetched_at}"
    label += f", dated {doc_date})" if (page.get("doc_date") or found_date) else ")"
    meta: Dict[str, object] = {
        "doc_category": page["category"],
        "doc_title": title,
        "alma_cycle": page.get("cycle", ""),
        "doc_status": "live-page",
        # CX-13: an undated live page is dated by its fetch (it describes the
        # current state as of that day); record which date source was used.
        "doc_date_source": "manifest" if page.get("doc_date") else ("page-meta" if found_date else "fetched"),
        "source_url": page["url"],
        "fetched_at": fetched_at,
        "doc_label": label,
        "doc_year": int(y),
    }
    if m:
        meta["doc_month"] = int(m)
    if d:
        meta["doc_day"] = int(d)
    return meta


# ──────────────────────────────────────────────────────────────────
# Fetch / ingest
# ──────────────────────────────────────────────────────────────────

_last_hit: Dict[str, float] = {}


def polite_get(url: str, delay: float = HOST_DELAY_S, timeout: int = 60):
    import requests
    host = urlparse(url).netloc
    wait = _last_hit.get(host, 0) + delay - time.time()
    if wait > 0:
        time.sleep(wait)
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    finally:
        _last_hit[host] = time.time()


def fetch_and_convert(page: Dict[str, str], fetched_at: str) -> Tuple[str, str, Optional[str]]:
    """Returns (title, markdown, found_date). Raises on HTTP failure."""
    if page.get("saved_html"):
        with open(page["saved_html"], encoding="utf-8", errors="replace") as f:
            html = f.read()
        if "cf-turnstile" in html or "challenge-platform" in html:
            raise ValueError("saved copy is a Cloudflare challenge page, not the document")
    else:
        r = polite_get(page["url"])
        r.raise_for_status()
        # Servers that omit the charset make requests assume ISO-8859-1 and
        # mangle UTF-8 (Sphinx pages: "FAQs Â¶").
        if not r.encoding or r.encoding.lower() in ("iso-8859-1", "latin-1"):
            r.encoding = "utf-8"
        if "html" not in r.headers.get("Content-Type", "html"):
            raise ValueError(f"not HTML: {r.headers.get('Content-Type')}")
        html = r.text
        if "cf-turnstile" in html or "challenge-platform" in html:
            raise ValueError("Cloudflare challenge page; save it from a browser and set saved_html")
        lm = _http_date(r.headers.get("Last-Modified", ""))
        if lm:
            HTTP_LAST_MODIFIED[page["url"]] = lm
    soup = BeautifulSoup(html, "html.parser")
    found_date = page_date(soup)
    title, md = html_to_markdown(html, base_url=page["url"], selector=page.get("selector"))
    return page.get("title") or title, md, found_date


def main():
    parser = argparse.ArgumentParser(description="Convert HTML documentation pages and ingest them")
    parser.add_argument("--only", action="append", default=[], help="Limit to these filenames")
    parser.add_argument("--estimate", action="store_true", help="Fetch + convert + token estimate; no ingest")
    parser.add_argument("--no-fetch", action="store_true", help="Ingest the saved Markdown files only")
    parser.add_argument("--no-ingest", action="store_true", help="Fetch + save Markdown only")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from dotenv import load_dotenv
    load_dotenv()
    fetched_at = date.today().isoformat()
    pages = [p for p in PAGES if not args.only or p["filename"] in args.only]

    converted: List[Tuple[Dict[str, str], str, Dict[str, object]]] = []
    failures: List[Tuple[str, str]] = []
    for page in pages:
        path = page_path(page)
        try:
            if args.no_fetch:
                with open(path, encoding="utf-8") as f:
                    fm, body = split_front_matter(f.read())
                title, found = fm.get("title", page["filename"]), fm.get("page_date") or None
                fetched = fm.get("fetched_at", fetched_at)
                if fm.get("http_last_modified"):
                    HTTP_LAST_MODIFIED[page["url"]] = fm["http_last_modified"]
            else:
                title, body, found = fetch_and_convert(page, fetched_at)
                fetched = fetched_at
            # Applies to saved Markdown too: a short or truncated body must
            # never replace the full indexed page.
            if len(body.strip()) < MIN_BODY_CHARS:
                raise ValueError(f"only {len(body.strip())} chars of content")
            if not args.no_fetch and not args.estimate:  # --estimate writes nothing
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(with_front_matter({
                        "title": title, "source_url": page["url"], "fetched_at": fetched,
                        "page_date": page.get("doc_date") or found or "", "category": page["category"],
                        "cycle": page.get("cycle", ""),
                        "http_last_modified": HTTP_LAST_MODIFIED.get(page["url"], ""),
                    }, body))
            meta = page_metadata(page, title, fetched, found)
            if HTTP_LAST_MODIFIED.get(page["url"]):
                meta["http_last_modified"] = HTTP_LAST_MODIFIED[page["url"]]
            converted.append((page, body, meta))
            print(f"  ok   {page['filename']:<62} {len(body):>7,} chars  {meta.get('doc_year')}-{meta.get('doc_month', '')}")
        except Exception as e:
            failures.append((page["filename"], str(e)))
            print(f"  FAIL {page['filename']:<62} {e}")

    if args.estimate or args.no_ingest:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = sum(int(len(enc.encode_ordinary(b)) * 1.25) for _, b, _ in converted)
        print(f"\n{len(converted)} pages, ~{tokens:,} embedding tokens, ~USD {tokens / 1e6 * 0.10:.4f}")
    if args.estimate or args.no_ingest:
        if failures:
            sys.exit(1)
        return

    from services.rag_service import RAGService
    rag = RAGService()
    total = 0
    for page, body, meta in converted:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
            tmp.write(body)
            tmp_path = tmp.name
        try:
            res = rag.ingest_document(tmp_path, original_filename=page["filename"],
                                      override_metadata=meta, replace_existing=True)
        finally:
            os.remove(tmp_path)
        if res.get("success"):
            total += res.get("chunks", 0)
        else:
            failures.append((page["filename"], res.get("error", "ingest failed")))
    print(f"\nIngested {len(converted)} pages, {total} chunks; {len(failures)} failures")
    for fn, err in failures:
        print(f"  FAILED {fn}: {err}")
    if failures:
        sys.exit(1)  # partial refresh must be visible to automation


if __name__ == "__main__":
    main()
