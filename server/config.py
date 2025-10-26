import os
import torch
from dotenv import load_dotenv
from pathlib import Path


PROJECT_ROOT_PATH = Path.cwd().as_posix()
print(f"Project root path: {PROJECT_ROOT_PATH}")

WEAVIATE_COLLECTION_NAME = "Multimodal_Collection"

# === Load environment variables ===
load_dotenv()  # Looks for .env in current working directory or parents

# === Device setup ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CLIP Model Config ===
CLIP_MODEL_NAME = "ViT-B-32"
PRETRAINED_LOCAL_PATH = f"{PROJECT_ROOT_PATH}/open_clip_weights/ViT-B-32-openai/open_clip_model.safetensors"

# === Whisper Model Config ===
WHISPER_MODEL = "small"

# === LLM Config ===
LLM_MODEL_NAME = "openai/gpt-5-image-mini"
LLM_API_BASE = "https://openrouter.ai/api/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY")  # <- Loaded from .env

if not LLM_API_KEY:
    raise ValueError("❌ Missing LLM_API_KEY in .env file!")

print("✅ Environment key loaded successfully!")
