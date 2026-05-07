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

    # Round 1
    print("--- Round 1 ---")
    response_stream = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": "What is the weather in Tokyo?"}],
        tools=tools,
        stream=True
    )
    
    last_id = None
    function_calls = {}
    item_id_to_call_id = {}
    output_text = ""

    for event in response_stream:
        if event.type == "response.created":
            last_id = event.response.id
            print(f"Created response: {last_id}")
        elif event.type == "response.output_item.added":
            item = event.item
            if getattr(item, 'type', None) == 'function_call':
                call_id = getattr(item, 'call_id', None)
                item_id = getattr(item, 'id', None)
                name = getattr(item, 'name', 'unknown')
                cid = call_id or item_id
                if cid:
                    function_calls[cid] = {"name": name, "arguments": "", "call_id": cid, "_item_id": item_id}
                    if item_id and item_id != cid:
                        item_id_to_call_id[item_id] = cid
                    if call_id and call_id != item_id:
                        item_id_to_call_id[call_id] = cid
        elif event.type == "response.output_item.done":
            item = event.item
            if getattr(item, 'type', None) == 'function_call':
                final_call_id = getattr(item, 'call_id', None)
                item_id = getattr(item, 'id', None)
                if final_call_id:
                    old_cid = item_id_to_call_id.get(item_id, item_id)
                    if old_cid in function_calls:
                        function_calls[old_cid]["call_id"] = final_call_id
                    elif item_id in function_calls:
                        function_calls[item_id]["call_id"] = final_call_id
        elif event.type == "response.function_call_arguments.delta":
            raw_id = getattr(event, 'call_id', None) or getattr(event, 'item_id', None)
            cid = item_id_to_call_id.get(raw_id, raw_id)
            if cid and cid in function_calls:
                function_calls[cid]["arguments"] += event.delta
        elif event.type == "response.completed":
            completed_resp = getattr(event, 'response', None)
            if completed_resp and hasattr(completed_resp, 'output'):
                for out_item in completed_resp.output:
                    if getattr(out_item, 'type', None) == 'function_call':
                        final_cid = getattr(out_item, 'call_id', None)
                        item_id = getattr(out_item, 'id', None)
                        fn_name = getattr(out_item, 'name', '')
                        fn_args = getattr(out_item, 'arguments', '')
                        if final_cid:
                            old_key = item_id_to_call_id.get(item_id, item_id)
                            if old_key in function_calls:
                                function_calls[old_key]["call_id"] = final_cid
                            elif item_id in function_calls:
                                function_calls[item_id]["call_id"] = final_cid
                            elif final_cid not in function_calls:
                                function_calls[final_cid] = {
                                    "name": fn_name,
                                    "arguments": fn_args,
                                    "call_id": final_cid,
                                    "_item_id": item_id,
                                }
    
    print("Function calls collected:", json.dumps(function_calls, indent=2))
    
    if not function_calls:
        print("No function calls!")
        return

    # Mock tool results
    tool_results = []
    for fc in function_calls.values():
        tool_results.append({
            "type": "function_call_output",
            "call_id": fc["call_id"],
            "output": "The weather is sunny and 25C.",
        })
    
    print("Sending tool results:", json.dumps(tool_results, indent=2))
    print(f"To previous_response_id: {last_id}")

    # Round 2
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
