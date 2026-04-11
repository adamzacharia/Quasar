import logging
logging.basicConfig(level=logging.INFO)

from core.agent import QuasarAgent, AgentConfig
from dotenv import load_dotenv
import os
import json

load_dotenv()
print("Starting agent...")
agent = QuasarAgent(AgentConfig())
print("Calling extraction...")
res = agent._extract_paper_details('1812.04040', 'What is the exact angular resolution achieved for AS 209?')
print("=== RESULT ===")
print(json.dumps(res, indent=2))
