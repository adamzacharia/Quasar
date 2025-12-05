import sys
import os
from pathlib import Path

# Add project root
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from core.agent import QuasarAgent

def test_search_command():
    print("Initializing Agent...")
    try:
        agent = QuasarAgent()
        
        query = "@search what is the diffrence between band 6 and band 7"
        print(f"\nTesting Query: '{query}'")
        
        print("Calling stream_general_response...")
        response = agent.stream_general_response(query)
        
        print("\nResponse received:")
        print(response)
        print("\nSUCCESS: No 400 Error.")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search_command()
