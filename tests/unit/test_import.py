print("Starting import...")
try:
    from core.agent import QuasarAgent
    print("Import successful")
    agent = QuasarAgent()
    print("Agent initialized")
except Exception as e:
    print(f"Error: {e}")
