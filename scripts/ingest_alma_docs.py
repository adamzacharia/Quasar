"""
Legacy entry point, kept so old instructions still work.

The original version resolved paths relative to scripts/ (so every file was
"not found") and listed Cycle 12 documents that are now retired. It now runs
the manifest-driven, non-destructive refresh for both document folders:

    python scripts/ingest_docs_folder.py --sync                      # docs/pdfs
    python scripts/ingest_docs_folder.py --sync --dir docs/references \\
        --only alma_tap_columns.txt --only "ALMA Pipeline Known Issues.txt" \\
        --only alma-pipeline-version-note-2026-09.md

HTML pages and the Helpdesk Knowledgebase have their own scripts
(ingest_html_pages.py, ingest_alma_kb.py).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    from download_alma_docs import LOCAL_REFERENCE_DOCS  # scripts/ is sys.path[0]

    script = os.path.join(HERE, "ingest_docs_folder.py")
    rc = subprocess.call([sys.executable, script, "--sync"])
    only = []
    for doc in LOCAL_REFERENCE_DOCS:
        only += ["--only", doc["filename"]]
    rc |= subprocess.call([sys.executable, script, "--sync", "--dir", os.path.join(ROOT, "docs", "references")] + only)
    sys.exit(rc)


if __name__ == "__main__":
    main()
