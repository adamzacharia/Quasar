import os
import sys
import pprint
import warnings

# Mock removed standard libraries for Python 3.13 compatibility
import types
if 'cgi' not in sys.modules:
    sys.modules['cgi'] = types.ModuleType('cgi')
    sys.modules['cgi'].parse_header = lambda x: (x, {})

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

try:
    from services.rag_service import RAGService
    rag = RAGService(user_id="test_user")
    stats = rag.get_collection_stats()
    print("=== Qdrant Collection Stats ===")
    pprint.pprint(stats)
except Exception as e:
    import traceback
    traceback.print_exc()
