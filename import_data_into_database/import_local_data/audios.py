import os
import json
import torch
import weaviate
import whisper
import open_clip
import numpy as np
import soundfile as sf
from tqdm import tqdm
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed


AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small", device=device)

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


def embed_text(text: str) -> np.ndarray:
    """Return normalized CLIP text embedding."""
    if not isinstance(text, str):
        raise ValueError("`text` must be a string.")

    tokens = tokenizer([text]).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return _to_numpy(features)


def get_embedding(modality: str, input_data: Union[str, None]) -> np.ndarray:
    """Wrapper for modality-specific embedding."""
    mod = modality.lower()
    if mod == "text":
        return embed_text(input_data)
    else:
        raise ValueError("`modality` must be 'text'.")


# ============================================
#   Data Storage Functions
# ============================================
def store_audio_item(item_id: str, transcription_text: str, audio_path: str = ""):
    """Store audio transcription embedding and metadata in Weaviate."""
    relative_path = f"/content/audio_dataset/{audio_path}"
    embedding = get_embedding("text", transcription_text)
    properties = {
        "contentId": item_id,
        "modality": "audio",
        "filePath": relative_path,
        "content": transcription_text,
    }
    print(relative_path)
    collection.data.insert(properties=properties, vector=embedding.tolist())


# ============================================
#   Batch Processing Functions
# ============================================
def process_audio_json(json_path: str, max_workers: int = 8):
    """Process and store audio embeddings using metadata JSON."""
    with open(json_path, "r") as f:
        metadata = json.load(f)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for item in metadata:
            file_path = item.get("filepath", "").strip()
            transcription = item.get("transcription", "").strip()
            if not (file_path and os.path.exists(file_path) and transcription):
                continue
            content_id = str(item.get("id", os.path.basename(file_path)))
            futures.append(
                executor.submit(store_audio_item, content_id, transcription, os.path.basename(file_path))
            )

        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="🎧 Processing audios"
        ):
            pass

    print(f"✅ Processed {len(metadata)} audio files.")


# ============================================
#   Audio Transcription
# ============================================
def transcribe_to_english(path):
    res = whisper_model.transcribe(path, task="transcribe")
    lang, text = res.get("language", ""), res.get("text", "").strip()
    return text if lang.startswith("en") else whisper_model.transcribe(path, task="translate", language="en").get("text", "").strip()

def transcribe_audios(folder, out_json="results.json"):
    if not os.path.isdir(folder): raise ValueError(f"No folder: {folder}")
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXTS)])

    results = [
        {
            "id": i + 1,
            "filename": os.path.basename(p),
            "filepath": os.path.abspath(p).replace("\\", "/"),
            "transcription": transcribe_to_english(p)
        }
        for i, p in enumerate(tqdm(files, desc="Processing"))
    ]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(results)} items to {out_json}")
    return results


# ============================================
#   Main Execution
# ============================================
if __name__ == "__main__":
    audio_folder = "../../content/audio_dataset"
    audio_json = f"{audio_folder}/audio_metadata.json"

    transcribe_audios(audio_folder, audio_json)
    process_audio_json(audio_json)
    weaviate_client.close()

