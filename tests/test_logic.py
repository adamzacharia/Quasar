import json

# Mock classes
class MockOpenAI:
    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature, response_format=None, max_tokens=None):
        content = messages[0]['content']
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
        self.choices = [MockChoice(content)]

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockMessage:
    def __init__(self, content):
        self.content = content

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
        # Simplified logic from agent.py
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": "classify the user's intent: " + query}],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)

    def extract_entities(self, query):
        # Simplified logic from agent.py
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": "Extract the following entities: " + query}],
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)

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
