import json
import weaviate
from weaviate.classes.config import Property, DataType

# === 1. Connect to local Weaviate ===
client = weaviate.connect_to_local()
print("✅ Connected to local Weaviate!")

# === 2. Define (or recreate) the schema ===
class_name = "Multimodal_Collection"

# Delete old class if exists
if client.collections.exists(class_name):
    client.collections.delete(class_name)

# Create new collection
client.collections.create(
    name=class_name,
    properties=[
        Property(name="modality", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="contentId", data_type=DataType.TEXT),
        Property(name="filePath", data_type=DataType.TEXT),
    ],
    vectorizer_config=None,  # We'll supply our own vectors
)
print("✅ Schema created!")

# === 3. Load JSON file ===
json_path = "../../exported_data_from_weaviate_cloud_10k_texts_5k_images_25k_audio_trasnscriptions.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 4. Get the collection handle ===
collection = client.collections.get(class_name)

# === 5. Batch import ===
print(f"🚀 Importing {len(data)} objects into Weaviate...")

with collection.batch.dynamic() as batch:
    for entry in data:
        batch.add_object(
            properties={
                "modality": entry["modality"],
                "content": entry["content"],
                "contentId": entry["contentId"],
                "filePath": entry["filePath"],
            },
            vector=entry.get("vector"),
        )

print("✅ Import complete!")

client.close()
