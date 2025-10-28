import os
import tempfile
import requests
import numpy as np
from PIL import Image
from typing import List, Tuple
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Form
from deep_translator import GoogleTranslator

from .utils import embed_text, embed_image, search_by_embedding
from .ai_models import whisper_model
from .config import PROJECT_ROOT_PATH
from .llm import feed_data_into_llm


router = APIRouter(prefix="/search", tags=["Search"])

# --- Helper Functions ---
def normalize_text_to_english(text: str) -> str:
    """
    Convert any input text into English if it is not already English.
    Works offline for language detection + uses Google Translate API.
    """
    if not text or text.strip() == "":
        return text

    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def normalize_results(results_list):
    """Converts Weaviate result list to a simple list of properties."""
    out = []
    for r in results_list or []:
        props = r.get("properties") if isinstance(r, dict) else None
        if props:
            out.append(props)
    return out


async def save_temp_file(file: UploadFile) -> str:
    """Saves UploadFile to a temp path and returns the path."""
    # Ensure a file extension, default to .tmp if none
    suffix = os.path.splitext(file.filename)[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(await file.read())
        return tf.name


def process_audio(temp_path: str) -> Tuple[List[float], dict]:
    """
    Transcribes audio, normalizes text, embeds it,
    and returns (embedding, metadata_dict).
    """
    # 1. Whisper -> Text + Language Detection
    whisper_res = whisper_model.transcribe(temp_path)
    detected_text = whisper_res.get("text", "").strip()
    detected_lang = whisper_res.get("language", "")

    # 2. If Persian -> Translate to English
    if detected_lang.startswith("fa"):
        processed_text = normalize_text_to_english(detected_text)
    else:
        processed_text = detected_text

    # 3. Embed transcribed text
    q_emb = embed_text(processed_text)

    metadata = {
        "file_name": os.path.basename(temp_path),
        "detected_text": detected_text,
        "processed_text": processed_text,
        "detected_language": detected_lang,
    }
    return q_emb, metadata


# --- Main Multimodal Endpoint ---
@router.post("/multimodal")
async def search_multimodal(
    query: str = Form(default=""), files: List[UploadFile] = File(default=[])
):

    # # --- 🔹 Print all incoming data immediately ---
    # print(f"\n=== Incoming Frontend Data ===")
    # print(f"Query: {query}")
    # print(f"Files ({len(files)}): {[file.filename for file in files]}")


    temp_paths = []
    embeddings = []
    response_data = {}

    try:
        has_text = bool(query and query.strip())

        # --- 1. Process Inputs ---
        if has_text:
            english_query = normalize_text_to_english(query)
            text_emb = embed_text(english_query)
            embeddings.append(text_emb)
            response_data["original_query"] = query
            response_data["processed_query"] = english_query

        audio_transcriptions = []
        image_files = []
        for file in files:
            temp_path = await save_temp_file(file)
            temp_paths.append(temp_path)

            if file.content_type.startswith("image/"):
                img_emb = embed_image(temp_path)
                embeddings.append(img_emb)
                try:
                    img_obj = Image.open(temp_path).copy()
                except Exception as e:
                    print(f"⚠️ Failed to open image {temp_path}: {e}")
                    img_obj = None
                image_files.append(img_obj)

            elif file.content_type.startswith("audio/"):
                audio_emb, audio_meta = process_audio(temp_path)
                embeddings.append(audio_emb)
                audio_transcriptions.append(audio_meta)

        if audio_transcriptions:
            if len(audio_transcriptions) == 1 and not has_text and len(files) == 1:
                response_data.update(audio_transcriptions[0])
            else:
                response_data["transcriptions"] = audio_transcriptions

        # --- 2. Determine Final Search Vector ---
        if not embeddings:
            return JSONResponse(
                {"error": "No valid text or media found to process."}, status_code=400
            )

        if len(embeddings) == 1:
            q_emb = embeddings[0]
            response_data["search_mode"] = "single"
        else:
            q_emb_array = np.array(embeddings)
            mean_emb = np.mean(q_emb_array, axis=0)
            q_emb = mean_emb
            response_data["search_mode"] = f"multimodal_mean ({len(embeddings)} inputs)"

        # --- 3. Perform Search ---
        text_res = search_by_embedding(q_emb, "text", top_k=5)
        image_res = search_by_embedding(q_emb, "image", top_k=5)
        audio_res = search_by_embedding(q_emb, "audio", top_k=5)

        normalized_texts = normalize_results(text_res)
        normalized_images = normalize_results(image_res)
        normalized_audios = normalize_results(audio_res)

        response_data.update(
            {
                "text_results": normalized_texts,
                "image_results": normalized_images,
                "audio_results": normalized_audios,
            }
        )

        llm_data = {
            "user_text_query": query if has_text else None,
            "user_image_queries": image_files if image_files else None,
            "user_audio_queries": (
                [a.get("processed_text") for a in audio_transcriptions]
                if audio_transcriptions
                else None
            ),
            "retrieved_texts": (
                [t.get("content") for t in normalized_texts[:1]]
                if normalized_texts
                else None
            ),
            "retrieved_images": (
                [
                    (
                        Image.open(PROJECT_ROOT_PATH + i.get("filePath"))
                        if i.get("filePath")
                        else (
                            Image.open(requests.get(i.get("url"), stream=True).raw)
                            if i.get("url")
                            else None
                        )
                    )
                    for i in normalized_images[:1]
                ]
                if normalized_images
                else None
            ),
            "retrieved_audios": (
                [
                    a.get("content") or a.get("processed_text")
                    for a in normalized_audios[:1]
                ]
                if normalized_audios
                else None
            ),
        }

        llm_data_modified_for_front = {
            "retrieved_texts": (
                [t.get("content") for t in normalized_texts[:1]]
                if normalized_texts
                else None
            ),
            "retrieved_images": (
                [i.get("filePath") or i.get("url") for i in normalized_images[:1]]
                if normalized_images
                else None
            ),
            "retrieved_audios": (
                [a.get("filePath") for a in normalized_audios[:1]]
                if normalized_audios
                else None
            ),
        }

        response_data["llm_response"] = feed_data_into_llm(llm_data)
        response_data["llm_data"] = llm_data_modified_for_front
        return JSONResponse(response_data)

    except Exception as e:
        print(f"An error occurred in /multimodal endpoint: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
