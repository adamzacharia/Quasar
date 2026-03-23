import os
import sys
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
    print("=" * 60)
    print("Ingesting docs/pdfs into Qdrant")
    print("=" * 60)
    
    rag = RAGService()
    
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "pdfs")
    
    if not os.path.exists(docs_dir):
        print(f"Directory not found: {docs_dir}")
        return
        
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs in {docs_dir}")
    
    results = {}
    for pdf in pdf_files:
        file_path = os.path.join(docs_dir, pdf)
        print(f"Ingesting {pdf}...")
        try:
            success = rag.ingest_document(file_path)
            results[pdf] = success
        except Exception as e:
            print(f"Error ingesting {pdf}: {e}")
            results[pdf] = False
            
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    success_count = sum(1 for v in results.values() if v)
    print(f"Successful: {success_count}/{len(results)}")
    
    for doc, status in results.items():
        icon = "OK" if status else "FAIL"
        print(f"[{icon}] {doc}")
        
    stats = rag.get_collection_stats()
    print(f"\nFinal vector store stats: {stats}")

if __name__ == "__main__":
    main()
