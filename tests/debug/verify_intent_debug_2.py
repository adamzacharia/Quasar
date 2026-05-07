print("1. Start")
import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from core.agent import QuasarAgent

load_dotenv()
print("2. Loaded env")

try:
    agent = QuasarAgent()
    print("3. Agent initialized")
except Exception as e:
    print(f"Error: {e}")

print("4. Done")
