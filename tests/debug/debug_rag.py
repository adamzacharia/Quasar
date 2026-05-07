import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent
sys.path.append(str(root_path))

from services.rag_service import RAGService
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Initializing RAG Service...")
    rag = RAGService()
    
    query = "What is the difference between Band 3 and Band 6?"
    print(f"\nQuery: {query}")
    
    # Use similarity_search_with_score to see distances
    results = rag.vector_store.similarity_search_with_score(query, k=5)
    
    with open("rag_debug_results.txt", "w", encoding="utf-8") as f:
        if results:
            f.write(f"Found {len(results)} documents:\n")
            for i, (doc, score) in enumerate(results):
                f.write(f"\n--- Document {i+1} (Score: {score:.4f}) ---\n")
                f.write(doc.page_content + "\n")
        else:
            f.write("No documents found.\n")
            
    print("Debug results written to rag_debug_results.txt")

if __name__ == "__main__":
    main()
