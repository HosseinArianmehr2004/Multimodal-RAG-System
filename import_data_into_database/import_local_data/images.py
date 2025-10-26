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
    "../../open_clip_weights/ViT-B-32-openai/open_clip_model.safetensors"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED_LOCAL_PATH
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
clip_model.to(DEVICE).eval()


# ============================================
#   Utility Functions
# ============================================
def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to 1D numpy array."""
    return tensor.detach().cpu().numpy().reshape(-1)


def embed_image(image_path: str) -> np.ndarray:
    """Return normalized CLIP image embedding."""
    if not isinstance(image_path, str):
        raise ValueError("`image_path` must be a string.")

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = clip_model.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)
    return _to_numpy(features)


def get_embedding(modality: str, input_data: Union[str, None]) -> np.ndarray:
    """Wrapper for modality-specific embedding."""
    mod = modality.lower()
    if mod == "image":
        return embed_image(input_data)
    else:
        raise ValueError("`modality` must be 'image'.")


# ============================================
#   Data Storage Functions
# ============================================
def store_image_item(item_id: str, image_path: str):
    """Store image embedding and metadata in Weaviate."""
    abs_path = os.path.abspath(image_path).replace("\\", "/")
    relative_path = f"/content/image_dataset/{image_path}"
    embedding = get_embedding("image", abs_path)
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
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                store_image_item, os.path.splitext(os.path.basename(f))[0], f
            )
            for f in image_files
        ]
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="📸 Processing images"
        ):
            pass

    print(f"✅ Processed {len(image_files)} images.")


# ============================================
#   Main Execution
# ============================================
if __name__ == "__main__":
    image_folder = "../../content/image_dataset"
    process_images(image_folder)
    weaviate_client.close()
