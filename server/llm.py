import io
import os
import base64
import requests
from PIL import Image
from openai import OpenAI
from .config import LLM_MODEL_NAME, LLM_API_BASE, LLM_API_KEY, PROJECT_ROOT_PATH


# === Initialize OpenAI Client ===
client = OpenAI(
    base_url=LLM_API_BASE,
    api_key=LLM_API_KEY
)


def to_pil_image(img_candidate):
    """Convert input to a PIL.Image (RGB)."""
    if img_candidate is None:
        return None

    if isinstance(img_candidate, Image.Image):
        return img_candidate.convert("RGB")

    if isinstance(img_candidate, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img_candidate)).convert("RGB")
        except Exception:
            return None

    if isinstance(img_candidate, str):
        if os.path.exists(img_candidate):
            try:
                return Image.open(img_candidate).convert("RGB")
            except Exception:
                return None
        if img_candidate.startswith("http://") or img_candidate.startswith("https://"):
            try:
                resp = requests.get(img_candidate, timeout=5)
                if resp.status_code == 200:
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                return None
        return None

    if isinstance(img_candidate, dict):
        b = img_candidate.get("bytes") or img_candidate.get("content") or img_candidate.get("data")
        if isinstance(b, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                return None
        path = img_candidate.get("path") or img_candidate.get("file")
        if isinstance(path, str) and os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                return None
        return None

    return None


def image_to_base64_data_url(pil_img):
    """Convert PIL.Image to base64 data URL (JPEG)."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def feed_data_into_llm(llm_data: dict) -> str:
    """Send multimodal data (text + image + audio) to OpenRouter model."""
    def section(title: str, content: str) -> str:
        return f"\n\n### {title}\n{content.strip()}"

    sections = []

    # --- User Text ---
    if llm_data.get("user_text_query"):
        sections.append(section("🧍 User Text Query", llm_data["user_text_query"]))

    # --- User Images ---
    image_base64s = []
    if llm_data.get("user_image_queries"):
        for img in llm_data["user_image_queries"]:
            pil_img = to_pil_image(img)
            if pil_img:
                pil_img = pil_img.resize((128, 128), Image.LANCZOS)
                image_base64s.append(image_to_base64_data_url(pil_img))
        sections.append(section("User Image Queries", f"{len(image_base64s)} image(s) attached." if image_base64s else "<no usable images>"))

    # --- User Audio ---
    if llm_data.get("user_audio_queries"):
        aud_list = "\n\n".join(
            [f"🎧 Audio {i+1} (Transcription):\n{t}" for i, t in enumerate(llm_data["user_audio_queries"])]
        )
        sections.append(section("User Audio Queries", aud_list))

    # --- Retrieved Texts ---
    if llm_data.get("retrieved_texts"):
        txt_list = "\n\n".join(
            [f"📄 Retrieved Text {i+1}:\n{t}" for i, t in enumerate(llm_data["retrieved_texts"])]
        )
        sections.append(section("Retrieved Texts", txt_list))

    # --- Retrieved Images ---
    retrieved_image_base64s = []
    if llm_data.get("retrieved_images"):
        for img in llm_data["retrieved_images"]:
            pil_img = to_pil_image(img)
            if pil_img:
                pil_img = pil_img.resize((128, 128), Image.LANCZOS)
                retrieved_image_base64s.append(image_to_base64_data_url(pil_img))

        sections.append(section("Retrieved Images", f"{len(retrieved_image_base64s)} image(s) attached." if retrieved_image_base64s else "<no usable retrieved images>"))

    # --- Retrieved Audios ---
    if llm_data.get("retrieved_audios"):
        raud_list = "\n\n".join(
            [f"🔊 Retrieved Audio {i+1} (Transcription):\n{t}" for i, t in enumerate(llm_data["retrieved_audios"])]
        )
        sections.append(section("Retrieved Audio Transcriptions", raud_list))

    # --- Merge all context ---
    full_context = "\n\n".join(sections).strip() or "No multimodal content provided."

    # === Construct message content ===
    content_list = [{"type": "text", "text": full_context}]

    # Add images as base64 URLs
    for img_url in image_base64s + retrieved_image_base64s:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": img_url}
        })

    # === Call the LLM ===
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,  # e.g. "openai/gpt-5-image-mini"
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Multimodal RAG Chatbot"
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a multimodal reasoning assistant.\n"
                        "You receive user inputs and retrieved multimodal data (text, image, audio).\n"
                        "If the user asked a question, answer it using relevant data.\n"
                        "If not, describe the multimodal inputs clearly in Markdown."
                    )
                },
                {
                    "role": "user",
                    "content": content_list
                }
            ]
        )

        response_text = completion.choices[0].message.content
    except Exception as e:
        response_text = f"⚠️ LLM call failed: {e}"

    # === Logging ===
    log_dir = os.path.join(PROJECT_ROOT_PATH, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "llm_full_context.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("📌 === FULL CONTEXT SENT TO LLM ===\n\n")
            f.write(full_context + "\n\n")
            f.write("🔹" * 50 + "\n\n")
            f.write("📌 === LLM RESPONSE ===\n\n")
            f.write(response_text + "\n")
        print(f"✅ Full context written to {log_path}")
    except Exception as e:
        print(f"⚠️ Failed to write LLM log: {e}")

    return response_text
