
import re
import json

def extract_tool_calls(response: str):
    tool_calls = []
    # Pattern from agent.py
    pattern = r'\[TOOL:\s*(\w+)\s*({.*?})\]'
    matches = re.finditer(pattern, response, re.DOTALL)

    print(f"Scanning response: '{response}'")
    for match in matches:
        tool_name = match.group(1)
        param_str = match.group(2)
        print(f"Match found! Tool: {tool_name}, Params: {param_str}")
        try:
            params = json.loads(param_str)
            tool_calls.append((tool_name, params))
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")

    return tool_calls

test_str = 'I will search for HL Tau. [TOOL: search_by_target {"target_name": "HL Tau"}]'
results = extract_tool_calls(test_str)
print("Extracted:", results)

if not results:
    print("FAILURE: No tools extracted")
else:
    print("SUCCESS")
