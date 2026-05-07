"""
Script to download official ALMA documentation PDFs into docs/pdfs/.
Handles cycle-aware naming to avoid conflicts with existing files.

Usage:
    conda run -n quasar python scripts/download_alma_docs.py
    conda run -n quasar python scripts/download_alma_docs.py --list   # Just list URLs, don't download
"""
import os
import sys
import re
import argparse
import requests
from pathlib import Path
from urllib.parse import urlparse, unquote

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Target download directory ──
DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "pdfs"
)

# ── Official ALMA Documentation URLs ──
# These are the direct PDF download links from the ALMA Science Portal.
# When new cycles are released, add entries here.
ALMA_DOCS = [
    # ── Cycle 13 (2026) ──
    {
        "url": "https://almascience.nrao.edu/documents-and-tools/cycle13/alma-proposers-guide",
        "filename": "alma-proposers-guide-cycle13.pdf",
        "category": "proposers_guide",
        "cycle": "Cycle 13",
        "description": "Cycle 13 Proposer's Guide",
    },
    {
        "url": "https://almascience.nrao.edu/documents-and-tools/cycle13/alma-technical-handbook",
        "filename": "alma-technical-handbook-cycle13.pdf",
        "category": "technical_handbook",
        "cycle": "Cycle 13",
        "description": "Cycle 13 Technical Handbook",
    },
    # ── Latest (cycle-agnostic) ──
    {
        "url": "https://almascience.nrao.edu/documents-and-tools/latest/alma-users-policies",
        "filename": "alma-users-policies-latest.pdf",
        "category": "user_policies",
        "cycle": "",
        "description": "ALMA Users' Policies (latest)",
    },
    {
        "url": "https://almascience.nrao.edu/documents-and-tools/latest/alma-primer",
        "filename": "alma-primer-latest.pdf",
        "category": "primer",
        "cycle": "",
        "description": "ALMA Primer — Getting Started",
    },
    {
        "url": "https://almascience.nrao.edu/documents-and-tools/latest/alma-ot-reference-manual",
        "filename": "alma-ot-refmanual-latest.pdf",
        "category": "observing_tool",
        "cycle": "",
        "description": "ALMA OT Reference Manual (latest)",
    },
    {
        "url": "https://almascience.nrao.edu/documents-and-tools/latest/alma-ot-quickstart-guide",
        "filename": "alma-ot-quickstart-latest.pdf",
        "category": "observing_tool",
        "cycle": "",
        "description": "ALMA OT Quick Start Guide (latest)",
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Quasar-RAG-Downloader/1.0"
}


def download_pdf(url: str, dest_path: str, description: str) -> bool:
    """Download a PDF from a URL. Returns True on success."""
    try:
        print(f"  ⬇  Downloading: {description}")
        print(f"     URL: {url}")

        response = requests.get(url, headers=HEADERS, stream=True, timeout=60, allow_redirects=True)

        # Check if the response is actually a PDF
        content_type = response.headers.get("Content-Type", "")

        # Some ALMA pages return HTML (the page, not the PDF). Try to find
        # the actual PDF link in the page.
        if "text/html" in content_type:
            # Try appending /at_download/file for Plone-based ALMA portal
            pdf_url = url.rstrip("/") + "/at_download/file"
            print(f"     HTML page detected, trying direct download: {pdf_url}")
            response = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=60, allow_redirects=True)
            content_type = response.headers.get("Content-Type", "")

            if "text/html" in content_type:
                # Second fallback: try with ?format=pdf
                pdf_url2 = url + "?format=pdf"
                print(f"     Still HTML, trying: {pdf_url2}")
                response = requests.get(pdf_url2, headers=HEADERS, stream=True, timeout=60, allow_redirects=True)
                content_type = response.headers.get("Content-Type", "")

        response.raise_for_status()

        # Verify it looks like a PDF
        first_bytes = b""
        total_size = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if not first_bytes:
                    first_bytes = chunk[:5]
                f.write(chunk)
                total_size += len(chunk)

        if first_bytes[:5] != b"%PDF-":
            print(f"     ⚠ Warning: Downloaded file may not be a valid PDF (starts with: {first_bytes[:20]})")
            os.remove(dest_path)
            return False

        size_mb = total_size / (1024 * 1024)
        print(f"     ✓ Saved: {os.path.basename(dest_path)} ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"     ✗ HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"     ✗ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download official ALMA documentation PDFs")
    parser.add_argument("--list", action="store_true", help="List documents without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    print("=" * 70)
    print("  ALMA Documentation Downloader")
    print("=" * 70)

    # Ensure docs directory exists
    os.makedirs(DOCS_DIR, exist_ok=True)

    # List existing files
    existing = set(os.listdir(DOCS_DIR))
    print(f"\nTarget directory: {DOCS_DIR}")
    print(f"Existing files: {len(existing)}")

    if args.list:
        print("\nDocuments to download:")
        for doc in ALMA_DOCS:
            status = "EXISTS" if doc["filename"] in existing else "NEW"
            print(f"  [{status}] {doc['filename']}")
            print(f"         {doc['description']}")
            print(f"         {doc['url']}")
        return

    # Download
    print(f"\nDownloading {len(ALMA_DOCS)} documents...\n")
    results = {"success": 0, "skipped": 0, "failed": 0}

    for doc in ALMA_DOCS:
        dest_path = os.path.join(DOCS_DIR, doc["filename"])

        if doc["filename"] in existing and not args.force:
            print(f"  ⏭  Skipping (exists): {doc['filename']}")
            results["skipped"] += 1
            continue

        ok = download_pdf(doc["url"], dest_path, doc["description"])
        if ok:
            results["success"] += 1
        else:
            results["failed"] += 1

    # Summary
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  ✓ Downloaded: {results['success']}")
    print(f"  ⏭  Skipped:   {results['skipped']}")
    if results["failed"]:
        print(f"  ✗ Failed:     {results['failed']}")
    print(f"\nTotal files in {DOCS_DIR}: {len(os.listdir(DOCS_DIR))}")

    if results["failed"]:
        print("\n⚠  Some downloads failed. This is usually because the ALMA portal")
        print("   uses JavaScript redirects. You can manually download these PDFs")
        print("   from the URLs listed above and place them in docs/pdfs/.")

    print("\n✅ Done! Run 'python scripts/ingest_docs_folder.py' to ingest into Qdrant.")


if __name__ == "__main__":
    main()
