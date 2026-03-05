import os
import sys

# Set dummy key to bypass validation during introspection
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"

try:
    import openai
    print(f"OpenAI Version: {openai.__version__}")
    
    print("\n[1] Checking 'openai.beta.assistants'")
    try:
        if hasattr(openai, 'beta') and hasattr(openai.beta, 'assistants'):
            print(" -> FOUND")
        else:
            print(" -> NOT FOUND")
    except Exception as e:
        print(f" -> ERROR: {e}")

    print("\n[2] Checking 'openai.responses'")
    try:
        # Check if 'responses' exists in top level
        if hasattr(openai, 'responses'):
            print(" -> FOUND")
        else:
            print(" -> NOT FOUND")
    except Exception as e:
        print(f" -> ERROR: {e}")

    print("\n[3] Checking 'openai.conversations'")
    try:
        if hasattr(openai, 'conversations'):
            print(" -> FOUND")
        else:
            print(" -> NOT FOUND")
    except Exception as e:
        print(f" -> ERROR: {e}")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
