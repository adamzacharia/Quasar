import sys
sys.stdout = open('internal_log.txt', 'w')
sys.stderr = sys.stdout

print("Starting verify_intent_internal.py...")
import os
from dotenv import load_dotenv

print("Loading .env...")
load_dotenv()

sys.path.insert(0, os.path.abspath("."))

print("Importing QuasarAgent...")
try:
    from core.agent import QuasarAgent
    print("QuasarAgent imported.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_intent():
    print("Initializing Agent...")
    try:
        agent = QuasarAgent()
        print("Agent initialized.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    queries = [
        "Find ALMA data for Sz65",
        "How do I visualize FITS files?",
        "Search for Band 6 data",
        "What is the sensitivity of ALMA?"
    ]

    print("\n--- Testing Intent Classification ---")
    for q in queries:
        print(f"\nQuery: {q}")
        intent = agent.determine_intent(q)
        print(f"Result: {intent}")

    print("\n--- Testing Entity Extraction ---")
    search_queries = [
        "Find ALMA data for Sz65 in Band 6",
        "Search for high resolution data of M31",
        "Find data for project 2019.1.00123.S"
    ]
    for q in search_queries:
        print(f"\nQuery: {q}")
        entities = agent.extract_entities(q)
        print(f"Result: {entities}")

if __name__ == "__main__":
    test_intent()
