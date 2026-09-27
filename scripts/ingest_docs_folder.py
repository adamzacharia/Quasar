"""
Ingest documentation files (docs/pdfs/ by default) into Qdrant with rich metadata.

Files listed in the manifest (scripts/download_alma_docs.py) get authoritative
metadata (category, cycle, doc number, version, publication date, status,
source URL, fetch date, and a one-line document header on every chunk).
Unknown files fall back to the filename / PDF-metadata guesses.

Usage:
    python scripts/ingest_docs_folder.py --sync        # non-destructive refresh (recommended)
    python scripts/ingest_docs_folder.py --estimate    # token + cost estimate only, no writes
    python scripts/ingest_docs_folder.py --only alma-user-policies-cycle13.pdf --sync
    python scripts/ingest_docs_folder.py               # LEGACY: wipe alma_general, re-ingest folder
    python scripts/ingest_docs_folder.py --no-wipe     # LEGACY: append without replacing

--only requires --sync outside --estimate, and every --only name must exist
in the folder: the legacy default mode wipes the whole collection first.

--sync never wipes the collection (notebooks, KB articles and HTML pages share
it). It deletes the chunks of every RETIRED_FILES entry, then re-ingests each
file with replace_existing=True, so re-running it is idempotent.
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Python 3.13 'cgi' module shim for pyvo ---
import sys as _sys
if "cgi" not in _sys.modules:
    import types
    import email.message
    cgi = types.ModuleType("cgi")
    def parse_header(line):
        m = email.message.Message()
        m['content-type'] = line
        return m.get_content_type(), m.get_params() or {}
    cgi.parse_header = parse_header
    _sys.modules["cgi"] = cgi
# ---------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

SUPPORTED_EXT = {".pdf", ".txt", ".md"}
# text-embedding-ada-002 list price, USD per 1M input tokens.
EMBED_USD_PER_MTOK = 0.10


def discover(docs_dir: str, only=None):
    files = sorted(
        f for f in os.listdir(docs_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    )
    if only:
        files = [f for f in files if f in set(only)]
    return files


def select_files(docs_dir: str, only, retired):
    """(files to ingest, retired files skipped). Retired files still on disk
    must never be re-ingested after --sync deleted their chunks (CX-01)."""
    found = discover(docs_dir, only)
    return [f for f in found if f not in retired], [f for f in found if f in retired]


def selection_error(only, doc_files, skipped_retired, *, sync, no_wipe, estimate):
    """Reason to refuse the run before anything is written, or None.

    The legacy default mode wipes the whole shared collection before ingesting,
    so a subset (--only) or an empty selection there would delete everything
    and re-ingest little or nothing."""
    if only:
        found = set(doc_files) | set(skipped_retired)
        missing = [f for f in only if f not in found]
        if missing:
            return f"--only names not found in the docs folder: {', '.join(missing)}"
        if not doc_files:
            return "--only selected only retired files; nothing to ingest"
    if estimate:
        return None
    if only and not sync:
        return "--only requires --sync (the default mode wipes the whole collection)"
    if not doc_files and not sync and not no_wipe:
        return "no documents found; refusing to wipe the collection"
    return None


def extract_text_for_estimate(path: str) -> str:
    """Plain text used only for the token estimate (fast, no markdown pass)."""
    if path.lower().endswith(".pdf"):
        import fitz
        with fitz.open(path) as d:
            return "".join(page.get_text() for page in d)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def estimate_tokens(paths) -> dict:
    """Embedding tokens for the given files, counting the 200-char chunk
    overlap (chunk 1000 / overlap 200 re-embeds about 25 % of the text)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    per_file = {}
    for p in paths:
        text = extract_text_for_estimate(p)
        per_file[os.path.basename(p)] = int(len(enc.encode_ordinary(text)) * 1.25)
    total = sum(per_file.values())
    return {"per_file": per_file, "total_tokens": total,
            "usd": round(total / 1e6 * EMBED_USD_PER_MTOK, 4)}


def main():
    parser = argparse.ArgumentParser(description="Ingest docs into Qdrant with rich metadata")
    parser.add_argument("--no-wipe", action="store_true",
                        help="LEGACY: skip wiping the collection (append; may duplicate)")
    parser.add_argument("--sync", action="store_true",
                        help="Non-destructive refresh: drop retired sources, replace each file's chunks")
    parser.add_argument("--estimate", action="store_true", help="Print token/cost estimate and exit")
    parser.add_argument("--only", action="append", default=[], help="Limit to these filenames")
    parser.add_argument("--dir", type=str, default=None, help="Override docs directory path")
    args = parser.parse_args()

    from download_alma_docs import RETIRED_FILES, manifest_metadata  # scripts/ is on sys.path

    docs_dir = args.dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "pdfs"
    )
    if not os.path.exists(docs_dir):
        print(f"Directory not found: {docs_dir}")
        return

    doc_files, skipped_retired = select_files(docs_dir, args.only, RETIRED_FILES)
    print(f"Found {len(doc_files)} documents in {docs_dir}")
    for f in skipped_retired:
        print(f"  skipping retired file still on disk: {f} ({RETIRED_FILES[f]})")

    err = selection_error(args.only, doc_files, skipped_retired,
                          sync=args.sync, no_wipe=args.no_wipe, estimate=args.estimate)
    if err:
        print(f"Refusing to run: {err}")
        sys.exit(2)

    if args.estimate:
        est = estimate_tokens([os.path.join(docs_dir, f) for f in doc_files])
        for fn, tok in est["per_file"].items():
            print(f"  {fn:<55} {tok:>10,} tokens")
        print(f"  TOTAL {est['total_tokens']:,} tokens ~ USD {est['usd']}")
        return

    from services.rag_service import RAGService
    from services.vector_db import backend_label
    print(f"Vector store backend: {backend_label()}")
    rag = RAGService()
    print(f"Pre-ingestion stats: {rag.get_collection_stats()}")

    if args.sync:
        for fn, why in RETIRED_FILES.items():
            rag.delete_source(fn)
        print(f"Removed chunks of {len(RETIRED_FILES)} retired source files (if present)")
    elif not args.no_wipe:
        print("Wiping alma_general for clean re-ingestion (legacy mode; use --sync to keep other sources)")
        rag.wipe_general_collection()

    results = {}
    total_chunks = 0
    for i, doc_file in enumerate(doc_files, 1):
        file_path = os.path.join(docs_dir, doc_file)
        meta = manifest_metadata(doc_file, path=file_path)
        tag = "manifest" if meta else "UNLISTED (guessed metadata)"
        print(f"\n[{i}/{len(doc_files)}] {doc_file}  [{tag}]")
        try:
            result = rag.ingest_document(
                file_path,
                personal=False,
                override_metadata=meta,
                replace_existing=args.sync,
            )
        except Exception as e:  # keep going; report at the end
            result = {"success": False, "error": str(e)}
        results[doc_file] = result
        if result.get("success"):
            m = result.get("metadata", {})
            total_chunks += result.get("chunks", 0)
            print(f"    {result.get('chunks', 0)} chunks | {m.get('doc_year')}/{m.get('doc_month')} | "
                  f"{m.get('doc_category')} | {m.get('alma_cycle') or '-'} | {m.get('doc_title')}")
        else:
            print(f"    FAILED: {result.get('error', 'unknown error')}")

    rag.create_metadata_indexes()
    ok = sum(1 for v in results.values() if v.get("success"))
    print(f"\nIngested {ok}/{len(results)} files, {total_chunks} chunks")
    for fn, r in results.items():
        if not r.get("success"):
            print(f"  FAILED {fn}: {r.get('error')}")
    print(f"Post-ingestion stats: {rag.get_collection_stats()}")
    if ok < len(results):
        sys.exit(1)  # partial refresh must be visible to automation


if __name__ == "__main__":
    main()
