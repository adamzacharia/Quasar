import sys
import os
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from services.rag_service import RAGService
from dotenv import load_dotenv

load_dotenv()

def main():
    pdf_path = "c:/Users/Asus/Desktop/archive-primer.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    print("Initializing RAG Service...")
    rag = RAGService()
    
    print(f"Ingesting {pdf_path}...")
    success = rag.ingest_document(pdf_path)
    
    if success:
        print("Ingestion Successful!")
    else:
        print("Ingestion Failed.")

if __name__ == "__main__":
    main()
