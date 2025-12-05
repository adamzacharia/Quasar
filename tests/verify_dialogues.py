
import sys
import os
import unittest
import traceback
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules
sys.modules['streamlit'] = MagicMock()
sys.modules['astropy'] = MagicMock()
sys.modules['astropy.coordinates'] = MagicMock()
sys.modules['astropy.units'] = MagicMock()
sys.modules['astroquery'] = MagicMock()
sys.modules['astroquery.alma'] = MagicMock()
sys.modules['astroquery.simbad'] = MagicMock()
sys.modules['pyvo'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['rich'] = MagicMock()
sys.modules['rich.console'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['services.rag_service'] = MagicMock()
sys.modules['services.memory_service'] = MagicMock()

# Import
try:
    from core.agent import QuasarAgent, AgentConfig
except ImportError:
    with open("verification_results.log", "w") as f:
        f.write("Import Failed:\n")
        traceback.print_exc(file=f)
    sys.exit(1)

LOG_FILE = "verification_results.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")

class TestAgentMemory(unittest.TestCase):
    def setUp(self):
        # Setup mocks
        self.mock_search = MagicMock()
        
        # Mock results
        results_df = MagicMock()
        results_df.empty = False
        results_df.__len__.return_value = 50
        # Mock to_dict to return a list of dicts
        results_df.to_dict.return_value = [{"target_name": "HL Tau", "freq_min": 100}]
        results_df.head.return_value = results_df
        
        self.mock_search.search_by_target.return_value = results_df
        self.mock_search.search_alma_with_keywords.return_value = results_df
        
        # Initialize agent
        config = AgentConfig(verbose=False, api_key="dummy-key")
        self.agent = QuasarAgent(config)
        self.agent.search_service = self.mock_search
        
        self.agent.client = MagicMock()
        
    def test_tool_feedback_loop(self):
        log("Starting test_tool_feedback_loop")
        try:
            tool_call_response = 'I will search for HL Tau. [TOOL: search_by_target {"target_name": "HL Tau"}]'
            
            mock_chunk = MagicMock()
            mock_chunk.choices[0].delta.content = tool_call_response
            self.agent.client.chat.completions.create.return_value = [mock_chunk]
            
            # Execute
            response = self.agent.stream_general_response("Search for HL Tau")
            log(f"Agent Response: {response}")
            
            # Check mocks
            if self.mock_search.search_by_target.called:
                log("search_by_target WAS called")
            else:
                log("search_by_target WAS NOT called")
                
            # Check memory
            history = self.agent.memory.get_history()
            log(f"All Memory Roles: {[m['role'] for m in history]}")
            
            system_msgs = [m for m in history if m["role"] == "system" and "Tool" in m["content"]]
            if system_msgs:
                log(f"Found System Msg: {system_msgs[0]['content']}")
            else:
                log("No System Msgs found")
                
            self.assertTrue(len(system_msgs) > 0, "Tool output missing from memory")
            self.assertIn("Found 1 records", system_msgs[0]["content"])
            log("test_tool_feedback_loop PASSED")
            
        except Exception as e:
            log(f"Exception in test: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
            raise e

if __name__ == '__main__':
    # Clear log
    with open(LOG_FILE, "w") as f:
        f.write("Test Run Started\n")
    unittest.main()
