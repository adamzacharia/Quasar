print("Starting dotenv test...")
try:
    from dotenv import load_dotenv
    print("dotenv imported")
    load_dotenv()
    print("dotenv loaded")
except Exception as e:
    print(f"Error: {e}")
