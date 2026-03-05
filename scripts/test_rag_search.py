"""Test RAG search with full metadata (page numbers) for each document"""
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from services.rag_service import RAGService

rag = RAGService()
stats = rag.get_collection_stats()

output = []
output.append(f"Vector Store Stats: {stats}")
output.append("=" * 80)

# Test queries for each document type
test_queries = [
    ("ALMA Technical Handbook", "What frequency does ALMA Band 6 cover?"),
    ("ALMA Proposer's Guide", "How do I submit an ALMA proposal?"),
    ("ALMA OT Quickstart", "How do I install the ALMA Observing Tool?"),
    ("ALMA Pipeline Guide", "What are the steps in the ALMA calibration pipeline?"),
    ("ALMA User Policies", "What is the proprietary period for ALMA data?"),
    ("ALMA Large Program", "What is an ALMA large program?"),
]

for doc_name, query in test_queries:
    output.append(f"\n{'='*80}")
    output.append(f"Testing: {doc_name}")
    output.append(f"Query: {query}")
    output.append("-" * 80)
    
    results = rag.search(query, k=2)
    
    if not results:
        output.append("No results found!")
        continue
    
    for i, r in enumerate(results):
        output.append(f"\n[Result {i+1}]")
        output.append(f"  Source File: {r.metadata.get('source_file', 'unknown')}")
        output.append(f"  Page Number: {r.metadata.get('page', 'N/A')}")
        output.append(f"  All Metadata: {r.metadata}")
        output.append(f"  Content Preview: {r.page_content[:400]}...")

output.append("\n" + "=" * 80)
output.append("METADATA CHECK COMPLETE")

# Write to file
with open('rag_test_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print("Results written to rag_test_results.txt")
print('\n'.join(output[:20]))  # Print first 20 lines to console
