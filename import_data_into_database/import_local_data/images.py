import os
import torch
import weaviate
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================
#   Weaviate Connection
# ============================================
try:
    weaviate_client = weaviate.connect_to_local()
    print("✅ Connected to Weaviate")
    collection = weaviate_client.collections.get("Multimodal_Collection")
except Exception as e:
    print(f"❌ Weaviate connection failed: {e}")
    weaviate_client = None


# ============================================
#   CLIP Model Setup
# ============================================
MODEL_NAME = "ViT-B-32"
PRETRAINED_LOCAL_PATH = (
    "./open_clip_weights/ViT-B-32-openai/open_clip_model.safetensors"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED_LOCAL_PATH
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
clip_model.to(DEVICE).eval()


# ============================================
#   Create Embedding Function
# ============================================
def get_embedding(modality: str, image_name: Union[str, None]) -> np.ndarray:
    mod = modality.lower()
    if mod == "image":
        """Return normalized CLIP image embedding."""
        if not isinstance(image_name, str):
            raise ValueError("`image_name` must be a string.")

        img_path = f"./content/image_dataset/{image_name}"
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = clip_model.encode_image(x)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().reshape(-1)
    else:
        raise ValueError("`modality` must be 'image'.")


# ============================================
#   Data Storage Functions
# ============================================
def store_image_item(item_id: str, img_name: str):
    """Store image embedding and metadata in Weaviate."""
    embedding = get_embedding("image", img_name)
    relative_path = f"/content/image_dataset/{img_name}"
    properties = {
        "contentId": item_id,
        "modality": "image",
        "filePath": relative_path,
        "content": "",
    }
    collection.data.insert(properties=properties, vector=embedding.tolist())


# ============================================
#   Batch Processing Functions
# ============================================
def process_images(image_folder: str, max_workers: int = 8):
    """Process and store image embeddings in parallel."""
    images = [
        file_path
        for file_path in os.listdir(image_folder)
        if file_path.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                store_image_item, os.path.splitext(os.path.basename(img_name))[0], img_name
            )
            for img_name in images
        ]
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="📸 Processing images"
        ):
            pass

    print(f"✅ Processed {len(images)} images.")


# ============================================
#   Main Execution
# ============================================
if __name__ == "__main__":
    image_folder = "./content/image_dataset"
    process_images(image_folder)
    weaviate_client.close()
