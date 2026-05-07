"""Check metadata on specific files that showed 'unknown' in the UI."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from services.vector_db import search_vectors, scroll_all
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Simulate the actual query
query = "What is the ALMA proprietary period?"
query_vector = embeddings.embed_query(query)

# Search like the agent does (top 3)
from services.vector_db import search_vectors
hits = search_vectors("alma_general", query_vector, limit=5)

print(f"=== Top {len(hits)} results for: '{query}' ===\n")
for i, hit in enumerate(hits):
    p = hit["payload"]
    src = p.get("source", p.get("source_file", "Unknown"))
    page = p.get("page", "???MISSING???")
    doc_year = p.get("doc_year", "???MISSING???")
    doc_month = p.get("doc_month", "???MISSING???")
    score = round(hit["score"], 4)
    
    # Recreate the header logic from agent.py
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    if doc_month and isinstance(doc_month, int) and doc_month in month_names:
        date_label = f"Date: {month_names[doc_month]} {doc_year}"
    else:
        date_label = f"Year: {doc_year}"
    
    header_parts = [f"Source: {src}"]
    if page and page != "?" and str(page) != "?":
        header_parts.append(f"Page {page}")
    header_parts.append(date_label)
    header_parts.append(f"Relevance: {score}")
    
    header = f"[{', '.join(header_parts)}]"
    
    print(f"--- Result {i+1} ---")
    print(f"  Raw metadata keys: {sorted(p.keys())}")
    print(f"  source_file = {p.get('source_file')}")
    print(f"  page        = {page}")
    print(f"  doc_year    = {doc_year}")
    print(f"  doc_month   = {doc_month}")
    print(f"  score       = {score}")
    print(f"  HEADER:  {header}")
    print(f"  text[:150]  = {p.get('text', '')[:150]}")
    print()
