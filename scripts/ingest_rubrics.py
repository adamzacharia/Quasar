# -*- coding: utf-8 -*-
"""Ingest ALMA reviewer guidelines and proposing guidance into the rubrics collection."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv
load_dotenv()

from services.rag_service import RAGService

rag = RAGService()

rubric_dir = os.path.join("docs", "rubrics")
files = [f for f in os.listdir(rubric_dir) if f.endswith(".txt")]

print(f"Found {len(files)} rubric files to ingest:")
for f in files:
    print(f"  - {f}")

for f in files:
    path = os.path.join(rubric_dir, f)
    print(f"\nIngesting {f}...")
    result = rag.ingest_rubric(
        path,
        progress_callback=lambda msg, pct: print(f"  [{pct}%] {msg}")
    )
    print(f"  Result: {result}")

# Verify
from services.vector_db import get_collection_info
info = get_collection_info("proposal_rubrics")
print(f"\nRubrics collection now has {info.get('vectors_count', '?')} chunks")

# Quick search test
test_results = rag.search_rubrics("What are the review criteria for ALMA proposals", k=3)
print(f"\nTest query returned {len(test_results)} results:")
for i, d in enumerate(test_results):
    print(f"  [{i+1}] src={d.metadata.get('source_file','?')}  text={d.page_content[:100]}")
