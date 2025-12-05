print("Starting alminer import...")
try:
    import alminer
    print("alminer imported")
except ImportError:
    print("alminer not found")
except Exception as e:
    print(f"Error: {e}")
