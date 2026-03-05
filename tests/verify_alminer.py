
import sys
import os
import json
from unittest.mock import MagicMock

# Mock rich if missing
try:
    from rich.console import Console
except ImportError:
    print("Mocking rich...")
    class Console:
        def print(self, *args, **kwargs): pass
        def status(self, *args, **kwargs): 
            class Status:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            return Status()
    sys.modules['rich.console'] = MagicMock()
    sys.modules['rich.console'].Console = Console
    sys.modules['rich.table'] = MagicMock()
    sys.modules['rich.panel'] = MagicMock()

# Mock alminer if missing
try:
    import alminer
except ImportError:
    print("Mocking alminer...")
    sys.modules['alminer'] = MagicMock()

# Mock other dependencies if needed
try:
    import pandas as pd
except ImportError:
    print("Mocking pandas...")
    sys.modules['pandas'] = MagicMock()
    pd = sys.modules['pandas']
    pd.DataFrame = MagicMock
    pd.DataFrame.empty = True
    pd.concat = MagicMock(return_value=pd.DataFrame())

# Now import our code
# We need to make sure we can import services.search and core.agent
# They might import other things that are missing.
# We'll try to import and if it fails, we can't verify much.

try:
    from services.search import SearchService
    from integrations.alminer_client import ALminerClient
    from core.agent import QuasarAgent, AgentConfig
except ImportError as e:
    print(f"Failed to import application modules: {e}")
    # If we can't import, we can't test.
    sys.exit(1)

def test_agent_tool_execution():
    print("\nTesting Agent tool execution logic...")
    
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = 'I will search for it. [TOOL: search_by_target {"target_name": "Sz65", "facility": "ALMA"}]'
    mock_client.responses.create.return_value = mock_response
    
    # Initialize agent with mocked client
    config = AgentConfig(api_key="fake-key")
    agent = QuasarAgent(config)
    agent.client = mock_client
    
    # Mock tool registry execute to avoid actual execution
    agent.tool_registry.execute = MagicMock(return_value={"status": "success", "data": "mock data"})
    
    # We want to test process_query but we need to control the loop.
    # The loop calls responses.create.
    # First call returns tool call.
    # Second call (after tool execution) should return final answer.
    
    mock_response_final = MagicMock()
    mock_response_final.output_text = "I found the data."
    
    mock_client.responses.create.side_effect = [mock_response, mock_response_final]
    
    # Run process_query
    result = agent.process_query("Find Sz65")
    
    print(f"Agent result: {result}")
    
    # Verify tool was executed
    if agent.tool_registry.execute.called:
        print("✅ Tool execution verified!")
        return True
    else:
        print("❌ Tool was NOT executed.")
        return False

if __name__ == "__main__":
    if test_agent_tool_execution():
        print("\nVerification passed!")
        sys.exit(0)
    else:
        print("\nVerification failed.")
        sys.exit(1)
