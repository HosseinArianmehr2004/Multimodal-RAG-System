import weaviate
import weaviate.classes as wvc


# 1. Connect to Weaviate 
try:
    client = weaviate.connect_to_local()
    print("Successfully connected to Weaviate!")
except Exception as e:
    print(f"Failed to connect to Weaviate: {e}")
    exit()

collection_name = "Multimodal_Collection"

# 2. Delete the previous Collection if it exists
if client.collections.exists(collection_name):
    print(f"Collection '{collection_name}' already exists. Deleting it.")
    client.collections.delete(collection_name)

# 3. Define and create the Collection (new method v4)
print(f"Creating collection '{collection_name}'...")
my_collection = client.collections.create(
    name=collection_name,
    properties=[
        wvc.config.Property(name="modality", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="contentId", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="filePath", data_type=wvc.config.DataType.TEXT),
    ]
)
print(f"Collection '{collection_name}' created successfully.")

client.close()
