import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from core.prompts import ENTITY_EXTRACTION_PROMPT

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_extraction(query, description):
    print(f"\n--- Testing: {description} ---")
    print(f"Query: '{query}'")
    
    prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw_content = response.choices[0].message.content
        print(f"Raw Content: {raw_content}")
        result = json.loads(raw_content)
        print(f"Result: {result}")
        
        source = result.get("source_name")
        if source == "Sz65":
            print("✅ SUCCESS: Extracted 'Sz65'")
        else:
            print(f"❌ FAILURE: Expected 'Sz65', got '{source}'")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

# Test Case 1: Complex Query
test_extraction(
    "Make a table of different ALMA datasets for Sz65, sorted by best sensitivity and angular resolution",
    "Complex Query"
)

# Test Case 2: Clarification Response
test_extraction(
    "the source is Sz65",
    "Clarification Response"
)
