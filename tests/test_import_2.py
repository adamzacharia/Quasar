print("Starting import test...")
import os
import sys
from dotenv import load_dotenv
from rich.console import Console

print("Imports done.")
load_dotenv()
print("Env loaded.")

try:
    from core.agent import QuasarAgent
    print("QuasarAgent imported.")
    agent = QuasarAgent()
    print("Agent initialized.")
except Exception as e:
    print(f"Error: {e}")
