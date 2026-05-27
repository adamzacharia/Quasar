import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from core.llm_client import LLMClient
from core.langfuse_integration import get_langfuse

def test_non_streaming():
    print("--- Testing non-streaming DeepSeek ---")
    client = LLMClient(model="deepseek-v4-pro")
    
    # Enable Langfuse tracing
    lf = get_langfuse()
    if not lf:
        print("Langfuse is not enabled!")
        return
        
    trace = lf.trace(name="test_deepseek_cost_non_streaming", user_id="test_user")
    
    import core.llm_client
    core.llm_client.set_langfuse_parent(trace)
    
    try:
        resp = client.responses.create(
            model="deepseek-v4-pro",
            input="Answer in exactly three words: what is astronomy?",
            instructions="You are a helpful science assistant.",
        )
        print("Response:", resp.output_text)
        print("Usage Object in Response:", resp.usage)
    finally:
        lf.flush()
        print("Flushed Langfuse traces.")

if __name__ == "__main__":
    test_non_streaming()
