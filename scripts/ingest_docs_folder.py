"""
Script to ingest all ALMA documentation from docs/pdfs/ into Qdrant
with rich metadata (year, category, cycle, title, etc.).

Usage:
    conda run -n quasar python scripts/ingest_docs_folder.py
    conda run -n quasar python scripts/ingest_docs_folder.py --no-wipe  # Keep existing data
"""
import os
import sys
import argparse
from pathlib import Path

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

from dotenv import load_dotenv
load_dotenv()

from services.rag_service import RAGService


def main():
    parser = argparse.ArgumentParser(description="Ingest docs/pdfs/ into Qdrant with rich metadata")
    parser.add_argument("--no-wipe", action="store_true",
                        help="Skip wiping the existing collection (default: wipe first)")
    parser.add_argument("--dir", type=str, default=None,
                        help="Override docs directory path")
    args = parser.parse_args()

    print("=" * 70)
    print("  ALMA Documentation RAG Ingestion (with Metadata)")
    print("=" * 70)

    rag = RAGService()

    # Determine docs directory
    if args.dir:
        docs_dir = args.dir
    else:
        docs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs", "pdfs"
        )

    if not os.path.exists(docs_dir):
        print(f"✗ Directory not found: {docs_dir}")
        return

    # Discover files
    supported_ext = {".pdf", ".txt", ".md"}
    doc_files = sorted([
        f for f in os.listdir(docs_dir)
        if os.path.splitext(f)[1].lower() in supported_ext
    ])
    print(f"\nFound {len(doc_files)} documents in {docs_dir}")
    for f in doc_files:
        size_kb = os.path.getsize(os.path.join(docs_dir, f)) // 1024
        print(f"  • {f}  ({size_kb:,} KB)")

    # Pre-ingestion stats
    stats_before = rag.get_collection_stats()
    print(f"\nPre-ingestion stats: {stats_before}")

    # Wipe collection if requested (default: wipe)
    if not args.no_wipe:
        print("\n⚠  Wiping existing alma_general collection for clean re-ingestion...")
        rag.wipe_general_collection()
        print("   ✓ Collection wiped and recreated")
    else:
        print("\n📌 Keeping existing data (--no-wipe mode)")

    # Ingest each document
    print("\n" + "─" * 70)
    results = {}
    total_chunks = 0
    for i, doc_file in enumerate(doc_files, 1):
        file_path = os.path.join(docs_dir, doc_file)
        print(f"\n[{i}/{len(doc_files)}] Ingesting {doc_file}...")

        try:
            result = rag.ingest_document(
                file_path,
                personal=False,
                progress_callback=lambda msg, pct: print(f"    {msg} ({pct}%)"),
            )
            results[doc_file] = result

            if result.get("success"):
                meta = result.get("metadata", {})
                chunks = result.get("chunks", 0)
                total_chunks += chunks
                month_str = f"/{meta['doc_month']:02d}" if meta.get('doc_month') else ""
                print(f"    ✓ {chunks} chunks | Date: {meta.get('doc_year', '?')}{month_str} | "
                      f"Category: {meta.get('doc_category', '?')} | "
                      f"Cycle: {meta.get('alma_cycle', 'N/A')} | "
                      f"Title: {meta.get('doc_title', '?')}")
            else:
                print(f"    ✗ FAILED: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"    ✗ Exception: {e}")
            results[doc_file] = {"success": False, "error": str(e)}

    # Create payload indexes for fast filtered search
    print("\n" + "─" * 70)
    print("Creating payload indexes for filtered search...")
    rag.create_metadata_indexes()

    # Summary
    print("\n" + "=" * 70)
    print("  INGESTION SUMMARY")
    print("=" * 70)
    success_count = sum(1 for v in results.values() if v.get("success"))
    fail_count = len(results) - success_count
    print(f"\n  ✓ Successful: {success_count}/{len(results)}")
    if fail_count:
        print(f"  ✗ Failed:     {fail_count}/{len(results)}")
    print(f"  📦 Total chunks: {total_chunks}")

    print("\n  Per-document details:")
    print(f"  {'Document':<50} {'Status':<8} {'Chunks':<8} {'Year':<6} {'Category':<20}")
    print("  " + "─" * 92)
    for doc_file, result in results.items():
        if result.get("success"):
            meta = result.get("metadata", {})
            print(f"  {doc_file:<50} {'✓':<8} {result.get('chunks', 0):<8} "
                  f"{meta.get('doc_year', '?'):<6} {meta.get('doc_category', '?'):<20}")
        else:
            print(f"  {doc_file:<50} {'✗':<8} {'–':<8} {'–':<6} {'FAILED':<20}")

    # Post-ingestion stats
    stats_after = rag.get_collection_stats()
    print(f"\n  Post-ingestion stats: {stats_after}")

    # Quick verification: test a filtered search
    print("\n" + "=" * 70)
    print("  VERIFICATION: Test Search")
    print("=" * 70)
    test_queries = [
        ("ALMA Band 6 receiver", None, "Unfiltered"),
        ("ALMA pipeline calibration", 2025, "Year ≥ 2025"),
    ]
    for query, min_year, label in test_queries:
        print(f"\n  Query: '{query}' [{label}]")
        test_results = rag.search(query, k=2, min_year=min_year, include_personal=False)
        if test_results:
            for j, doc in enumerate(test_results, 1):
                src = doc.metadata.get("source_file", "?")
                year = doc.metadata.get("doc_year", "?")
                score = doc.metadata.get("_score", "?")
                page = doc.metadata.get("page", "?")
                cat = doc.metadata.get("doc_category", "?")
                print(f"    [{j}] {src} | Page {page} | Year: {year} | "
                      f"Score: {score} | Category: {cat}")
                print(f"        {doc.page_content[:120]}...")
        else:
            print("    (no results)")

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
