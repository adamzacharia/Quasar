
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

from core.agent import QuasarAgent, AgentConfig

def run_live_test():
    print("--- STARTING LIVE ALMA INTEGRATION TEST ---")
    
    # Check for API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        return

    # Initialize Agent with Real Services (No Mocks)
    print("Initializing QuasarAgent...")
    try:
        agent = QuasarAgent()
        print("Agent Initialized Successfully.")
    except Exception as e:
        print(f"Agent Initialization Failed: {e}")
        return

    # Test Query
    query = "Find ALMA observations for HL Tau"
    print(f"\nProcessing Query: '{query}'")
    
    try:
        response = agent.stream_general_response(query)
        print("\n--- AGENT RESPONSE ---")
        print(response)
        
        # Verify if data was found
        if agent.last_search_results is not None and not agent.last_search_results.empty:
            count = len(agent.last_search_results)
            print(f"\n[SUCCESS] Found {count} real records from ALMA Archive!")
            print(f"Columns found: {agent.last_search_results.columns.tolist()}")
            
            # Robust column selection
            cols_to_show = ['target_name', 's_ra', 's_dec', 'frequency']
            if 'Band' in agent.last_search_results.columns:
                cols_to_show.append('Band')
            elif 'band_number' in agent.last_search_results.columns:
                cols_to_show.append('band_number')
                
            print("First 3 results:")
            print(agent.last_search_results.head(3)[cols_to_show].to_string())
            
            # Save to file for proof
            agent.last_search_results.to_csv("live_test_results.csv", index=False)
            print("\nResults saved to live_test_results.csv")
        else:
            print("\n[FAILURE] No records found. ALminer connection might be broken.")
            
    except Exception as e:
        print(f"Query Processing Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_live_test()
