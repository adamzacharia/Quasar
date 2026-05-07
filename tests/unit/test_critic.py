import asyncio
from services.proposal_critic import ProposalCriticService
from services.rag_service import RAGService
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

async def test():
    critic = ProposalCriticService()
    rag = RAGService()
    
    # We will do a generic search first since we didn't populate the database
    # But RAG service should still handle empty searches gracefully
    
    def on_progress(msg, pct):
        print(f"[{pct}%] {msg}")
        
    res = critic.review_proposal("test_proposal.pdf", rag, on_progress)
    print("\n\n--- Critique ---")
    print(res.get("critique", "No critique generated"))
    if "error" in res:
        print("ERROR:", res.get("error"))

if __name__ == "__main__":
    asyncio.run(test())
