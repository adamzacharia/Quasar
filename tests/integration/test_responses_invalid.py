import os
import json
from openai import OpenAI

def main():
    client = OpenAI()

    # Define a simple tool
    tools = [{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }]

    print("--- Round 1 ---")
    response_stream = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        tools=tools,
        stream=True
    )
    
    last_id = None
    call_id = None
    for event in response_stream:
        if event.type == "response.created":
            last_id = event.response.id
        elif event.type == "response.output_item.added":
            item = event.item
            if getattr(item, 'type', None) == 'function_call':
                call_id = getattr(item, 'call_id', None)

    tool_results = [
        {
            "type": "function_call_output",
            "call_id": "fc_12345_invalid",
            "output": "The weather is sunny and 25C.",
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": "The weather is sunny and 25C.",
        }
    ]

    print("\n--- Round 2 ---")
    try:
        response_stream = client.responses.create(
            model="gpt-4o",
            input=tool_results,
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
