import sys
import logging
from core.agent import QuasarAgent, AgentConfig

logging.basicConfig(level=logging.INFO)

# Run end-to-end test on the agent's new _reproduce_paper_methods
config = AgentConfig()
agent = QuasarAgent(config=config)

# DSHARP paper
bibcode = '2018ApJ...869L..41A'

print(f"Testing lit-to-code on {bibcode}...")
result = agent._reproduce_paper_methods(bibcode)

print("\n--- AGENT RESULT ---")
print(result)
