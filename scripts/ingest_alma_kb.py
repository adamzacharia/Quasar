"""
ALMA Helpdesk Knowledgebase -> RAG corpus (alma_general, category "knowledgebase").

Crawls https://help.almascience.org/kb category pages for article slugs and
downloads each article through its PDF endpoint
(/kb/articles/pdf/<slug>), so the normal PDF loader handles it. Honors the
site's robots.txt: crawl-delay 2 s between every request, never /api or
/agent. "Historical Articles" is skipped by default (retired systems).

Each article is stored as docs/references/kb/kb-<slug>.pdf and ingested with:
    doc_category=knowledgebase, doc_title="ALMA Helpdesk KB: <title>",
    kb_category, doc date = the article's own date (the "Author - YYYY-MM-DD -
    Category" line printed at the top of every article PDF), source_url,
    fetched_at, alma_cycle when the title names a cycle.

Usage:
    python scripts/ingest_alma_kb.py --list        # crawl categories, print article list
    python scripts/ingest_alma_kb.py --estimate    # download PDFs, token/cost estimate, no ingest
    python scripts/ingest_alma_kb.py               # download + ingest
    python scripts/ingest_alma_kb.py --no-fetch    # ingest already-downloaded PDFs
    (cached PDFs older than --max-age-days, default 30, are re-downloaded;
     --force re-downloads all)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest_html_pages import polite_get  # noqa: E402  (same polite fetcher, 2 s per host)

KB_BASE = "https://help.almascience.org"
KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "references", "kb")
CATEGORIES = [
    "general", "general-alma-queries", "alma-observing-tool-ot", "proposal-handling",
    "archive-data-retrieval", "offline-data-reduction-and-or-casa", "project-planning",
    "resources-observer-support", "historical-articles",
]
SKIP_BY_DEFAULT = {"historical-articles"}
_ARTICLE_HREF = re.compile(r"^/kb/articles/(?!pdf/)([a-z0-9-]+)$")
# "Sarah Bagley - 2026-04-22 - General"
_BYLINE = re.compile(r"^\s*(.+?)\s+-\s+(20\d\d-\d\d-\d\d)\s+-\s+(.+?)\s*$", re.M)


def parse_category_page(html: str) -> List[Dict[str, str]]:
    """Article {slug, title} pairs linked from a KB category page."""
    soup = BeautifulSoup(html, "html.parser")
    seen, out = set(), []
    for a in soup.find_all("a", href=True):
        m = _ARTICLE_HREF.match(a["href"].split("?")[0])
        if not m or m.group(1) in seen:
            continue
        title = " ".join(a.get_text(" ").split())
        if not title:
            continue
        seen.add(m.group(1))
        out.append({"slug": m.group(1), "title": title})
    return out


def parse_byline(text: str) -> Optional[Dict[str, str]]:
    """{author, date, kb_category} from an article PDF's first page text."""
    m = _BYLINE.search(text or "")
    if not m:
        return None
    return {"author": m.group(1), "date": m.group(2), "kb_category": m.group(3)}


def detect_cycle(title: str) -> str:
    m = re.search(r"\bCycle\s+(\d{1,2})\b", title or "", re.I)
    return f"Cycle {m.group(1)}" if m else ""


def kb_metadata(article: Dict[str, str], byline: Optional[Dict[str, str]], fetched_at: str) -> Dict[str, object]:
    title = article["title"]
    art_date = (byline or {}).get("date") or fetched_at
    y, mth, d = art_date.split("-")
    label = f"ALMA Helpdesk Knowledgebase article: {title} (last updated {art_date})"
    return {
        "doc_category": "knowledgebase",
        "doc_title": f"ALMA Helpdesk KB: {title}",
        "alma_cycle": detect_cycle(title),
        "doc_status": "live-page",
        "doc_date_source": "kb-byline" if (byline or {}).get("date") else "fetched",
        "kb_category": (byline or {}).get("kb_category", article.get("category", "")),
        "doc_author": (byline or {}).get("author", ""),
        "source_url": f"{KB_BASE}/kb/articles/{article['slug']}",
        "fetched_at": fetched_at,
        "doc_label": label,
        "doc_year": int(y), "doc_month": int(mth), "doc_day": int(d),
    }


def needs_download(path: str, force: bool, max_age_days: float, now: Optional[float] = None) -> bool:
    """Download when forced, missing, or older than max_age_days (CX-14:
    a cached copy must not hide an edited article forever)."""
    import time as _time
    if force or not os.path.exists(path):
        return True
    age_days = ((now if now is not None else _time.time()) - os.path.getmtime(path)) / 86400.0
    return age_days > max_age_days


def kb_filename(slug: str) -> str:
    return f"kb-{slug}.pdf"


def crawl(categories: List[str]) -> List[Dict[str, str]]:
    articles: Dict[str, Dict[str, str]] = {}
    for cat in categories:
        r = polite_get(f"{KB_BASE}/kb/{cat}")
        r.raise_for_status()
        found = parse_category_page(r.text)
        print(f"  {cat:<40} {len(found)} articles")
        for a in found:
            articles.setdefault(a["slug"], {**a, "category": cat})
    return list(articles.values())


def main():
    parser = argparse.ArgumentParser(description="Crawl the ALMA Helpdesk Knowledgebase into the RAG corpus")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download PDFs that exist")
    parser.add_argument("--max-age-days", type=float, default=30.0,
                        help="Re-download cached PDFs older than this (default 30) so edited articles refresh")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from dotenv import load_dotenv
    load_dotenv()

    os.makedirs(KB_DIR, exist_ok=True)
    index_path = os.path.join(KB_DIR, "_index.json")
    fetched_at = date.today().isoformat()
    cats = [c for c in CATEGORIES if args.include_historical or c not in SKIP_BY_DEFAULT]

    if args.no_fetch:
        with open(index_path, encoding="utf-8") as f:
            articles = json.load(f)
    else:
        articles = crawl(cats)
    if not articles:
        # Changed markup or a challenge page parses to zero articles; keep the
        # saved index and fail loudly instead of recording an empty corpus.
        print("No KB articles parsed; keeping the existing index. Check the KB page markup.")
        sys.exit(1)
    if not args.no_fetch:
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=1)
    print(f"{len(articles)} unique articles")
    if args.list:
        for a in articles:
            print(f"  [{a['category']}] {a['title']}")
        return

    import fitz
    failures = []
    texts: Dict[str, str] = {}
    for i, a in enumerate(articles, 1):
        path = os.path.join(KB_DIR, kb_filename(a["slug"]))
        if not args.no_fetch and needs_download(path, args.force, args.max_age_days):
            try:
                r = polite_get(f"{KB_BASE}/kb/articles/pdf/{a['slug']}")
                r.raise_for_status()
                if not r.content.startswith(b"%PDF-"):
                    raise ValueError(f"not a PDF ({r.headers.get('Content-Type')})")
                tmp_path = path + ".part"
                with open(tmp_path, "wb") as f:
                    f.write(r.content)
                with fitz.open(tmp_path) as _check:  # truncated/corrupt -> exception, old file kept
                    _check.page_count
                os.replace(tmp_path, path)
            except Exception as e:
                failures.append((a["slug"], str(e)))
                print(f"  FAIL {a['slug']}: {e}")
                continue
        if os.path.exists(path):
            try:
                with fitz.open(path) as d:
                    texts[a["slug"]] = "".join(p.get_text() for p in d)
            except Exception as e:  # corrupt cached PDF: record it, keep refreshing the rest
                failures.append((a["slug"], f"unreadable cached PDF: {e}"))
                print(f"  FAIL {a['slug']}: unreadable cached PDF: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(articles)} articles")

    if args.estimate:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tok = sum(int(len(enc.encode_ordinary(t)) * 1.25) for t in texts.values())
        print(f"{len(texts)} article PDFs, ~{tok:,} embedding tokens, ~USD {tok / 1e6 * 0.10:.4f}")
        if failures:
            print(f"{len(failures)} downloads failed; the estimate is incomplete")
            sys.exit(1)
        return

    from services.rag_service import RAGService
    rag = RAGService()
    chunks = 0
    for a in articles:
        path = os.path.join(KB_DIR, kb_filename(a["slug"]))
        if a["slug"] not in texts:
            continue
        # fetched_at = when this PDF was actually downloaded (a default run
        # reuses cached PDFs; --force refreshes them) (CX-14).
        downloaded = date.fromtimestamp(os.path.getmtime(path)).isoformat()
        meta = kb_metadata(a, parse_byline(texts[a["slug"]]), downloaded)
        res = rag.ingest_document(path, override_metadata=meta, replace_existing=True)
        if res.get("success"):
            chunks += res.get("chunks", 0)
        else:
            failures.append((a["slug"], res.get("error")))
    print(f"Ingested {len(texts)} KB articles, {chunks} chunks; {len(failures)} failures")
    for slug, err in failures:
        print(f"  FAILED {slug}: {err}")
    if failures:
        sys.exit(1)  # partial refresh must be visible to automation


if __name__ == "__main__":
    main()
