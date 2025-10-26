import os
import base64
import numpy as np
import torch
import weaviate.classes.query
from PIL import Image
from .database import weaviate_client
from .ai_models import clip_model, tokenizer, preprocess
from .config import DEVICE, WEAVIATE_COLLECTION_NAME


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().reshape(-1)


def embed_text(text: str) -> np.ndarray:
    tokens = tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return _to_numpy(text_features)


def embed_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return _to_numpy(image_features)


def get_embedding(modality: str, data: str):
    if modality == "text":
        return embed_text(data)
    elif modality == "image":
        return embed_image(data)
    raise ValueError("modality must be 'text' or 'image'")


def search_by_embedding(query_embedding, modality: str, top_k=3):
    try:
        collection = weaviate_client.collections.get(WEAVIATE_COLLECTION_NAME)
        results = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=top_k,
            filters=weaviate.classes.query.Filter.by_property("modality").equal(
                modality
            ),
        )
        found = []
        if results.objects:
            for obj in results.objects:
                found.append({"properties": obj.properties})
        return found
    except Exception as e:
        print(f"Error searching {modality}: {e}")
        return []
