import weaviate

try:
    weaviate_client = weaviate.connect_to_local()
    print("✅ Connected to Weaviate")
except Exception as e:
    print(f"❌ Weaviate connection failed: {e}")
    weaviate_client = None

