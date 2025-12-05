import openai
import inspect

print(f"OpenAI Version: {openai.__version__}")

print("\nChecking for 'beta.assistants':")
try:
    print(openai.beta.assistants)
    print(" - Found")
except AttributeError:
    print(" - Not Found")

print("\nChecking for 'responses':")
try:
    print(openai.responses)
    print(" - Found")
except AttributeError:
    print(" - Not Found")

print("\nChecking for 'conversations':")
try:
    print(openai.conversations)
    print(" - Found")
except AttributeError:
    print(" - Not Found")
