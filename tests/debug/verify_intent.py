print("Starting verify_intent.py...", flush=True)
import os
import sys
from dotenv import load_dotenv

print("Loading .env...", flush=True)
load_dotenv()

sys.path.insert(0, os.path.abspath("."))

print("Importing QuasarAgent...", flush=True)
try:
    from core.agent import QuasarAgent
    print("QuasarAgent imported.", flush=True)
except ImportError as e:
    print(f"ImportError: {e}", flush=True)
    sys.exit(1)

def test_intent():
    print("Initializing Agent...", flush=True)
    try:
        agent = QuasarAgent()
        print("Agent initialized.", flush=True)
    except Exception as e:
        print(f"Failed to initialize agent: {e}", flush=True)
        return

    queries = [
        "Find ALMA data for Sz65",
        "How do I visualize FITS files?",
        "Search for Band 6 data",
        "What is the sensitivity of ALMA?"
    ]

    print("\n--- Testing Intent Classification ---", flush=True)
    for q in queries:
        print(f"\nQuery: {q}", flush=True)
        intent = agent.determine_intent(q)
        print(f"Result: {intent}", flush=True)

    print("\n--- Testing Entity Extraction ---", flush=True)
    search_queries = [
        "Find ALMA data for Sz65 in Band 6",
        "Search for high resolution data of M31",
        "Find data for project 2019.1.00123.S"
    ]
    for q in search_queries:
        print(f"\nQuery: {q}", flush=True)
        entities = agent.extract_entities(q)
        print(f"Result: {entities}", flush=True)

if __name__ == "__main__":
    test_intent()
