"""
Script to ingest Jupyter notebooks and documentation from GitHub repositories
into Qdrant for RAG. Supports cloning repos, extracting markdown cells from
notebooks, and ingesting PDFs/text files.

Strategy B: Extract only markdown/text cells from notebooks (skip noisy outputs),
combine with accompanying docs, and ingest with proper metadata tags.

Usage:
    conda run -n quasar python scripts/ingest_notebooks.py
    conda run -n quasar python scripts/ingest_notebooks.py --clone    # Clone repos first
    conda run -n quasar python scripts/ingest_notebooks.py --list     # Just list repos
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

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

# ── Configuration ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE_DIR = os.path.join(PROJECT_ROOT, "cloned_repos")
LOCAL_NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "alma-science-archive-notebooks")

# ── Repositories to clone and ingest ──
REPOS = [
    {
        "name": "casangi-examples",
        "url": "https://github.com/casangi/examples.git",
        "description": "Official CASA Jupyter examples",
        "category": "tutorial",
        "topics": ["CASA", "imaging", "calibration"],
    },
    {
        "name": "radio-astro-tools-tutorials",
        "url": "https://github.com/radio-astro-tools/tutorials.git",
        "description": "Spectral cube analysis, moment maps, PV diagrams",
        "category": "tutorial",
        "topics": ["spectral-cube", "moment-maps", "analysis"],
    },
    {
        "name": "ALminer",
        "url": "https://github.com/emerge-erc/ALminer.git",
        "description": "ALminer tutorial notebooks for ALMA archive mining",
        "category": "tutorial",
        "topics": ["ALminer", "archive", "data-mining"],
    },
    {
        "name": "casa-alma-sis14",
        "url": "https://github.com/akleroy/casa_alma_sis14.git",
        "description": "NRAO Synthesis Imaging School ALMA tutorials",
        "category": "tutorial",
        "topics": ["synthesis-imaging", "calibration", "CASA"],
    },
    {
        "name": "simulation-in-casa",
        "url": "https://github.com/urvashirau/Simulation-in-CASA.git",
        "description": "CASA simulation and advanced imaging demonstrations",
        "category": "tutorial",
        "topics": ["simulation", "imaging", "CASA"],
    },
]


def clone_repos(force: bool = False) -> Dict[str, str]:
    """Clone all configured repos into cloned_repos/. Returns {name: path}."""
    os.makedirs(CLONE_DIR, exist_ok=True)
    paths = {}

    for repo in REPOS:
        dest = os.path.join(CLONE_DIR, repo["name"])
        if os.path.exists(dest):
            if force:
                print(f"  ♻  Re-cloning {repo['name']}...")
                import shutil
                shutil.rmtree(dest)
            else:
                print(f"  ⏭  Already cloned: {repo['name']}")
                paths[repo["name"]] = dest
                continue

        print(f"  ⬇  Cloning {repo['name']} from {repo['url']}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo["url"], dest],
                check=True, capture_output=True, text=True
            )
            print(f"     ✓ Cloned to {dest}")
            paths[repo["name"]] = dest
        except subprocess.CalledProcessError as e:
            print(f"     ✗ Clone failed: {e.stderr}")
        except FileNotFoundError:
            print(f"     ✗ git not found. Please install git.")
            break

    return paths


def extract_notebook_markdown(notebook_path: str) -> str:
    """Extract only markdown and code cells from a Jupyter notebook.

    Strategy B: Keep markdown cells fully, keep code cells (source only,
    no outputs), skip output cells entirely to avoid noise.
    """
    try:
        with open(notebook_path, "r", encoding="utf-8", errors="replace") as f:
            nb = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"     ⚠ Could not parse {notebook_path}: {e}")
        return ""

    cells = nb.get("cells", [])
    if not cells:
        return ""

    parts = []
    for cell in cells:
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if not source.strip():
            continue

        if cell_type == "markdown":
            parts.append(source)
        elif cell_type == "code":
            # Include code source but clearly delimited
            parts.append(f"```python\n{source}\n```")
        # Skip 'raw' and other cell types

    return "\n\n".join(parts)


def discover_ingestible_files(directory: str) -> Dict[str, List[str]]:
    """Walk a directory and find all ingestible files, grouped by type."""
    found = {"notebooks": [], "pdfs": [], "markdown": [], "text": []}
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden dirs, __pycache__, .git, etc.
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for fname in files:
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext == ".ipynb":
                found["notebooks"].append(fpath)
            elif ext == ".pdf":
                found["pdfs"].append(fpath)
            elif ext == ".md":
                found["markdown"].append(fpath)
            elif ext in (".txt", ".rst"):
                found["text"].append(fpath)

    return found


def ingest_directory(
    directory: str,
    repo_name: str,
    repo_config: dict,
    rag_service,
    progress_callback=None,
) -> Dict[str, int]:
    """Ingest all supported files from a directory into Qdrant."""
    from langchain_core.documents import Document

    files = discover_ingestible_files(directory)
    stats = {"notebooks": 0, "pdfs": 0, "docs": 0, "chunks": 0, "failed": 0}

    total_files = sum(len(v) for v in files.values())
    print(f"\n  Found: {len(files['notebooks'])} notebooks, {len(files['pdfs'])} PDFs, "
          f"{len(files['markdown'])} markdown, {len(files['text'])} text files")

    # ── Ingest Notebooks (extract markdown cells) ──
    for nb_path in files["notebooks"]:
        rel_path = os.path.relpath(nb_path, directory)
        print(f"    📓 {rel_path}...", end=" ")

        content = extract_notebook_markdown(nb_path)
        if not content or len(content.strip()) < 50:
            print("(empty/too short, skipped)")
            continue

        try:
            # Create a temporary markdown file for the RAG pipeline
            temp_md = nb_path + ".extracted.md"
            # Add header with provenance
            header = (
                f"# {os.path.basename(nb_path)}\n"
                f"Source: {repo_config.get('url', repo_name)}\n"
                f"Category: {repo_config.get('category', 'tutorial')}\n"
                f"Topics: {', '.join(repo_config.get('topics', []))}\n\n---\n\n"
            )
            with open(temp_md, "w", encoding="utf-8") as f:
                f.write(header + content)

            result = rag_service.ingest_document(
                temp_md,
                personal=False,
                progress_callback=lambda msg, pct: None,
            )

            # Cleanup temp file
            os.remove(temp_md)

            if result.get("success"):
                chunks = result.get("chunks", 0)
                stats["notebooks"] += 1
                stats["chunks"] += chunks
                print(f"✓ {chunks} chunks")
            else:
                stats["failed"] += 1
                print(f"✗ {result.get('error', '?')}")

        except Exception as e:
            stats["failed"] += 1
            print(f"✗ {e}")

    # ── Ingest PDFs ──
    for pdf_path in files["pdfs"]:
        rel_path = os.path.relpath(pdf_path, directory)
        # Skip very small PDFs (likely placeholders)
        if os.path.getsize(pdf_path) < 1024:
            continue

        print(f"    📄 {rel_path}...", end=" ")
        try:
            result = rag_service.ingest_document(
                pdf_path,
                personal=False,
                progress_callback=lambda msg, pct: None,
            )
            if result.get("success"):
                chunks = result.get("chunks", 0)
                stats["pdfs"] += 1
                stats["chunks"] += chunks
                print(f"✓ {chunks} chunks")
            else:
                stats["failed"] += 1
                print(f"✗ {result.get('error', '?')}")
        except Exception as e:
            stats["failed"] += 1
            print(f"✗ {e}")

    # ── Ingest Markdown and Text files ──
    for doc_path in files["markdown"] + files["text"]:
        rel_path = os.path.relpath(doc_path, directory)
        # Skip very small files
        if os.path.getsize(doc_path) < 100:
            continue
        # Skip common non-content files
        basename = os.path.basename(doc_path).lower()
        if basename in ("license", "license.md", "contributing.md", "changelog.md"):
            continue

        print(f"    📝 {rel_path}...", end=" ")
        try:
            result = rag_service.ingest_document(
                doc_path,
                personal=False,
                progress_callback=lambda msg, pct: None,
            )
            if result.get("success"):
                chunks = result.get("chunks", 0)
                stats["docs"] += 1
                stats["chunks"] += chunks
                print(f"✓ {chunks} chunks")
            else:
                stats["failed"] += 1
                print(f"✗ {result.get('error', '?')}")
        except Exception as e:
            stats["failed"] += 1
            print(f"✗ {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Jupyter notebooks and docs from GitHub repos into Qdrant"
    )
    parser.add_argument("--clone", action="store_true",
                        help="Clone/update GitHub repos before ingesting")
    parser.add_argument("--force-clone", action="store_true",
                        help="Re-clone repos even if they exist")
    parser.add_argument("--list", action="store_true",
                        help="List repos and local notebooks without ingesting")
    parser.add_argument("--local-only", action="store_true",
                        help="Only ingest local alma-science-archive-notebooks")
    parser.add_argument("--no-wipe", action="store_true",
                        help="Don't wipe existing collection (append mode)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Notebook & Repository RAG Ingestion")
    print("=" * 70)

    # ── List mode ──
    if args.list:
        print("\nConfigured GitHub repositories:")
        for repo in REPOS:
            exists = os.path.exists(os.path.join(CLONE_DIR, repo["name"]))
            status = "CLONED" if exists else "NOT CLONED"
            print(f"  [{status}] {repo['name']}")
            print(f"           {repo['description']}")
            print(f"           {repo['url']}")

        local_exists = os.path.exists(LOCAL_NOTEBOOKS_DIR)
        print(f"\nLocal ALMA notebooks: {'✓ Found' if local_exists else '✗ Not found'}")
        if local_exists:
            files = discover_ingestible_files(LOCAL_NOTEBOOKS_DIR)
            print(f"  Notebooks: {len(files['notebooks'])}")
        return

    # ── Clone repos if requested ──
    if args.clone or args.force_clone:
        print("\nCloning repositories...")
        clone_repos(force=args.force_clone)

    # ── Initialize RAG service ──
    from services.rag_service import RAGService
    rag = RAGService()

    # Note: We typically DON'T wipe here because the main docs are already ingested.
    # This script appends notebook content alongside existing documentation.
    if not args.no_wipe:
        print("\n📌 Running in APPEND mode (use --no-wipe to be explicit)")
        print("   Existing documentation chunks will be preserved.")

    all_stats = {}

    # ── Ingest local ALMA notebooks ──
    if os.path.exists(LOCAL_NOTEBOOKS_DIR):
        print("\n" + "─" * 70)
        print(f"📚 Ingesting local ALMA Science Archive Notebooks")
        print(f"   Path: {LOCAL_NOTEBOOKS_DIR}")
        stats = ingest_directory(
            LOCAL_NOTEBOOKS_DIR,
            "alma-science-archive-notebooks",
            {
                "url": "https://almascience.org/alma-data/archive/archive-notebooks/",
                "category": "tutorial",
                "topics": ["ALMA", "archive", "query", "astroquery"],
            },
            rag,
        )
        all_stats["alma-science-archive-notebooks"] = stats
    else:
        print(f"\n⚠ Local notebooks not found at {LOCAL_NOTEBOOKS_DIR}")

    # ── Ingest cloned repos ──
    if not args.local_only:
        for repo in REPOS:
            repo_path = os.path.join(CLONE_DIR, repo["name"])
            if not os.path.exists(repo_path):
                print(f"\n⏭ Skipping {repo['name']} (not cloned, use --clone)")
                continue

            print("\n" + "─" * 70)
            print(f"📚 Ingesting: {repo['name']}")
            print(f"   {repo['description']}")

            stats = ingest_directory(repo_path, repo["name"], repo, rag)
            all_stats[repo["name"]] = stats

    # ── Create indexes ──
    print("\n" + "─" * 70)
    print("Creating payload indexes...")
    rag.create_metadata_indexes()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  INGESTION SUMMARY")
    print("=" * 70)

    total_chunks = 0
    for name, stats in all_stats.items():
        chunks = stats["chunks"]
        total_chunks += chunks
        nb = stats["notebooks"]
        pdf = stats["pdfs"]
        doc = stats["docs"]
        fail = stats["failed"]
        print(f"\n  {name}:")
        print(f"    📓 Notebooks: {nb} | 📄 PDFs: {pdf} | 📝 Docs: {doc} | 📦 Chunks: {chunks}", end="")
        if fail:
            print(f" | ✗ Failed: {fail}")
        else:
            print()

    print(f"\n  Total chunks ingested: {total_chunks}")

    # Post-ingestion stats
    collection_stats = rag.get_collection_stats()
    print(f"  Collection stats: {collection_stats}")

    print("\n✅ Notebook ingestion complete!")


if __name__ == "__main__":
    main()
