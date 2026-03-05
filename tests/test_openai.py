import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key present: {bool(api_key)}")

try:
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Hello",
        max_output_tokens=5
    )
    print("Response:", response.output_text)
except Exception as e:
    print("Error:", e)
