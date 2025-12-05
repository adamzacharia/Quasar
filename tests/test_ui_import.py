import sys
import os
from pathlib import Path

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

try:
    from ui.app import QuasarAgent, AgentConfig
    print("Successfully imported QuasarAgent from ui.app (which imports from core.agent)")
    
    config = AgentConfig(api_key="test", ads_api_key="test")
    agent = QuasarAgent(config=config)
    print("Successfully initialized QuasarAgent")
    
    # Check if it has the new RAG service
    if hasattr(agent, 'rag_service'):
        print("Agent has rag_service")
    else:
        print("ERROR: Agent missing rag_service")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
