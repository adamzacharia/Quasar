import os
from openai import OpenAI

def main():
    client = OpenAI()

    tools = [{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }]

    print("--- Round 1 (Asking for weather) ---")
    response_stream = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        tools=tools,
        stream=True
    )
    
    last_id = None
    for event in response_stream:
        if event.type == "response.created":
            last_id = event.response.id

    print(f"Round 1 ended. last_id = {last_id}")
    print("Pretending we failed to send the tool output...")

    print("\n--- Round 2 (User asks another question) ---")
    try:
        response_stream = client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": "Actually, what about New York?"}],
            tools=tools,
            previous_response_id=last_id,
            stream=True
        )
        for event in response_stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="")
        print()
    except Exception as e:
        print("API Error:", e)

if __name__ == "__main__":
    main()
