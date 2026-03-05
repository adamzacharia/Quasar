import sys
from pathlib import Path

# Simulate the path fix in app.py
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    print(f"Adding {root_path} to sys.path")
    sys.path.append(str(root_path))
else:
    print(f"{root_path} already in sys.path")

try:
    from core.prompts import ENTITY_EXTRACTION_PROMPT
    print("SUCCESS: Imported ENTITY_EXTRACTION_PROMPT")
except ImportError as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
