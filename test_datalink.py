from integrations.datalink import DataLinkClient
import sys

def main():
    client = DataLinkClient()
    # Replace this with a valid UID if you know one. This one might work.
    mous_uid = "uid://A001/X135e/X737" 
    print(f"Testing list_files with UID: {mous_uid}")
    
    try:
        results = client.list_files(mous_uid)
        print(f"Success! Found {len(results)} results:")
        for r in results[:3]:
            print(f" - {r.get('filename')}: {r.get('content_length')} bytes")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
