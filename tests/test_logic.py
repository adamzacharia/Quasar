import json

# Mock classes — updated for Responses API shape
class MockOpenAI:
    def __init__(self):
        self.responses = self

    def create(self, model, input, temperature=0.1, instructions=None,
               text=None, max_output_tokens=None, **kwargs):
        content = input
        if "INTENT_CLASSIFICATION" in content or "classify the user's intent" in content:
            if "Search for ALMA data" in content and "Sz65" not in content:
                return MockResponse('{"intent": "SEARCH", "confidence": 0.9}')
            if "Sz65" in content:
                return MockResponse('{"intent": "SEARCH", "confidence": 0.9}')
        
        if "ENTITY_EXTRACTION" in content or "Extract the following entities" in content:
            if "Search for ALMA data" in content and "Sz65" not in content:
                return MockResponse('{"source_name": null, "band": null}')
            if "Sz65" in content:
                return MockResponse('{"source_name": "Sz65", "band": null}')

        return MockResponse("I don't know")

class MockResponse:
    def __init__(self, content):
        self.output_text = content

class MockMemory:
    def __init__(self):
        self.messages = []
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

class AgentLogicTest:
    def __init__(self):
        self.client = MockOpenAI()
        self.memory = MockMemory()
        self.config = type('Config', (), {'model': 'gpt-4', 'verbose': True})()

    def determine_intent(self, query):
        # Simplified logic from agent.py — Responses API
        response = self.client.responses.create(
            model=self.config.model,
            input="classify the user's intent: " + query,
            temperature=0.1
        )
        return json.loads(response.output_text)

    def extract_entities(self, query):
        # Simplified logic from agent.py — Responses API
        response = self.client.responses.create(
            model=self.config.model,
            input="Extract the following entities: " + query,
            temperature=0.1
        )
        return json.loads(response.output_text)

    def process_query(self, query):
        print(f"Processing: {query}")
        self.memory.add_message("user", query)
        
        intent_data = self.determine_intent(query)
        intent = intent_data.get("intent", "QUESTION")
        print(f"Intent: {intent}")
        
        if intent == "SEARCH":
            entities = self.extract_entities(query)
            print(f"Entities: {entities}")
            
            source_name = entities.get("source_name")
            
            if not source_name and "search" in query.lower():
                response = "I can help you search for ALMA data. Which astronomical source are you interested in?"
                self.memory.add_message("assistant", response)
                return response
            
            return f"Searching for {source_name}..."

# Run Test
test = AgentLogicTest()
print("\n--- Test 1: Ambiguous Query ---")
response = test.process_query("Search for ALMA data")
print(f"Response: {response}")

if response == "I can help you search for ALMA data. Which astronomical source are you interested in?":
    print("SUCCESS: Back-and-forth logic triggered!")
else:
    print("FAILURE: Back-and-forth logic NOT triggered.")
