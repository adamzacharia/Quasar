"""Quick test for _detect_beyond_cutoff and the web search integration."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.platform == "win32" and "pwd" not in sys.modules:
    import types
    _pwd = types.ModuleType("pwd")
    _pwd.getpwuid = lambda uid: type("pw", (), {"pw_name": "user"})()
    sys.modules["pwd"] = _pwd

if "cgi" not in sys.modules:
    import email.message, types
    _cgi = types.ModuleType("cgi")
    def _ph(line):
        m = email.message.EmailMessage()
        m['content-type'] = line
        return m.get_content_type(), m.get_params() or {}
    _cgi.parse_header = _ph
    sys.modules["cgi"] = _cgi

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from core.agent import QuasarAgent, AgentConfig

agent = QuasarAgent(AgentConfig())

# Test detection
test_queries = [
    ("How many ALMA bands are there?", False),
    ("What bands have public data as of March 25, 2026?", True),
    ("How many bands have data in the ALMA archive as of 2025?", True),
    ("What is the latest ALMA cycle?", False),          # freshness word, no explicit date → no web search
    ("Search ALMA for TW Hya observations", False),
    ("What recent data releases are available this year?", False),  # freshness word, no explicit date → no web search
    ("Tell me about ALMA Band 6 specifications", False),
    ("How many proposals were submitted in November 2024?", True),
    ("What happened in October 2024?", False),  # within cutoff
    ("What happened in December 2024?", True),   # after cutoff
]

print("\n=== Detection Tests ===\n")
for query, expected in test_queries:
    result = agent._detect_beyond_cutoff(query)
    detected = result is not None
    status = "[PASS]" if detected == expected else "[FAIL] WRONG"
    print(f"  {status}  detected={detected:5}  expected={expected:5}  \"{query[:65]}\"")

# Run one actual query with web search
print("\n\n=== Live Test: Query with date beyond cutoff ===\n")
query = "How many observing bands are available with ALMA? How many bands have public data available in the ALMA Science Archive as of March 25, 2026?"

tokens = []
statuses = []

def on_token(t):
    tokens.append(t)

def on_status(step, state):
    statuses.append((step, state))
    icon = "[...]" if state == "running" else "[OK]"
    print(f"  {icon} [{state}] {step}")

print(f"  Query: \"{query[:80]}...\"\n")

response = agent.stream_response_api(
    query=query,
    user_id="cutoff-test",
    on_token=on_token,
    on_status=on_status,
)

print(f"\n=== Response ({len(response)} chars) ===\n")
print(response[:2000])
if len(response) > 2000:
    print(f"\n... ({len(response) - 2000} more chars)")

# Check web section present
if "Web Search Results" in response:
    print("\n\n[PASS] WEB SEARCH RESULTS SECTION FOUND IN RESPONSE!")
else:
    print("\n\n[!]  No web search results section found")
