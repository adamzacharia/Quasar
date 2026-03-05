import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock cgi to prevent pyvo crash on Python 3.13 if necessary
import types
try:
    import cgi
except ImportError:
    mock_cgi = types.ModuleType("cgi")
    mock_cgi.parse_header = lambda x: (x, {}) # minimalistic mock
    sys.modules["cgi"] = mock_cgi

from core.agent import QuasarAgent, AgentConfig

def test_ads_tool_fix():
    print("--- Testing ADS Tool Fix in QuasarAgent ---")
    
    ads_key = os.getenv("NASA_ADS_API_KEY")
    if not ads_key:
        print("WARNING: No NASA_ADS_API_KEY found in environment.")
        return
        
    config = AgentConfig()
    agent = QuasarAgent(config)
    
    # Get the registered search_papers tool
    tool = agent.tool_registry.get_tool("search_papers")
    
    if not tool:
        print("ERROR: search_papers tool not found in registry.")
        sys.exit(1)
        
    print(f"Tool found: {tool.name}")
    
    # Simulate LLM executing the tool with a natural-language-like query
    # (The LLM shouldn't do this anymore since the prompt tells it to use ADS fields,
    # but if it does, our builder should catch it).
    print("Executing tool with: 'recent ALMA papers on molecular clouds' ...")
    
    # We pass max_results=3 so it's fast
    result = tool.execute(query="recent ALMA papers on molecular clouds", max_results=3)
    
    print(f"\nResult object from tool execution:")
    print(result)
    
    if result.get("success"):
        print(f"\nSuccess! Found {result.get('count')} papers.")
        print(f"Actual ADS Query built: {result.get('ads_query')}")
        
        # Verify the agent stored the papers correctly for the frontend
        last_result = agent.last_run_result
        if last_result and "papers" in last_result:
            papers = last_result["papers"]
            print(f"\nagent.last_run_result populated correctly. Found {len(papers)} papers in property.")
            if len(papers) > 0:
                print(f"Top paper title: {papers[0].get('title')}")
                print("\nTEST PASSED")
            else:
                print("\nTEST FAILED: Paper list is empty.")
                sys.exit(1)
        else:
            print("\nTEST FAILED: agent.last_run_result does not contain 'papers'.")
            sys.exit(1)
    else:
        print(f"\nTEST FAILED: Tool returned error: {result.get('error')}")
        sys.exit(1)

if __name__ == "__main__":
    test_ads_tool_fix()
