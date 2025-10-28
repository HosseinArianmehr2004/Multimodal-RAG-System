import os
import re
import time
import torch
import weaviate
import open_clip
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


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
#   Environment setup
# ============================================
load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")
if not API_KEY:
    raise ValueError(
        "Please set your API key in the environment variable OPENROUTER_API_KEY."
    )


# ============================================
#   Initialize LLM model
# ============================================
LLM_model = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    temperature=0.1,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate the following Persian (Farsi) text to clear, natural English.\n"
            "Return only the translation (no extra explanation).\n\n"
            "Persian text:\n{text}",
        )
    ]
)

parser = StrOutputParser()
vision_chain = prompt | LLM_model | parser


# ============================================
#   Create embedding and store in DB
# ============================================
def get_embedding(modality: str, input_data: Union[str, None]) -> np.ndarray:
    """Wrapper for modality-specific embedding."""
    mod = modality.lower()
    if mod == "text":
        """Return normalized CLIP text embedding."""
        if not isinstance(input_data, str):
            raise ValueError("`input_data` must be a string.")

        tokens = tokenizer([input_data]).to(DEVICE)
        with torch.no_grad():
            features = clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().reshape(-1)
    else:
        raise ValueError("`modality` must be 'text'.")


def store_text_item(item_id: str, text_data: str):
    """Store text embedding and metadata in Weaviate."""
    embedding = get_embedding("text", text_data)
    properties = {
        "contentId": item_id,
        "modality": "text",
        "filePath": "",
        "content": text_data,
    }
    collection.data.insert(properties=properties, vector=embedding.tolist())


def process_texts(input_csv: str, max_workers: int = 8):
    """Process and store text embeddings in parallel."""
    df = pd.read_csv(input_csv)
    texts = df["text"].astype(str).tolist()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(store_text_item, f"text_{i+1}", text)
            for i, text in enumerate(texts)
        ]
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="📝 Processing texts"
        ):
            pass

    print(f"✅ Processed {len(texts)} text items.")


# ============================================
#   Translate
# ============================================
def is_persian(text: str) -> bool:
    """Return True if the text contains at least one Persian character."""
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r"[\u0600-\u06FF]", text))


def translate_text(text: str, retries=2, pause=1.0) -> str:
    # ---------- Translate a single text ----------
    if not isinstance(text, str) or not text.strip():
        return ""

    prompt_input = {"text": text}

    attempt = 0
    while attempt <= retries:
        try:
            translated = vision_chain.invoke(prompt_input)
            return translated.strip()
        except Exception as e:
            attempt += 1
            print(f"⚠️ Translation error (attempt {attempt}): {e}")
            if attempt > retries:
                return ""
            time.sleep(pause * attempt)


def translate_csv(input_csv: str, text_column="text", workers=4):
    """
    Takes a CSV file, translates the specified text column,
    and saves the output in the same folder as the input file with
    the name <original_name>_translated.csv.
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"File '{input_csv}' does not exist.")

    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' does not exist in the file. Columns: {df.columns.tolist()}"
        )

    texts = df[text_column].fillna("").astype(str).tolist()
    results = [""] * len(texts)

    # Use ThreadPoolExecutor to speed up translation (optional)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(translate_text, texts[i]): i for i in range(len(texts))
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Translating"
            ):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    print(f"⚠️ Error at index {idx}: {e}")
                    results[idx] = ""
    else:
        for i, t in enumerate(tqdm(texts, desc="Translating")):
            results[i] = translate_text(t)

    df["translated"] = results

    # Save output file in the same folder as the input file
    folder, filename = os.path.split(input_csv)
    name, ext = os.path.splitext(filename)
    output_csv = os.path.join(folder, f"{name}_translated{ext}")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Translation complete. Saved to {output_csv}")
    return output_csv


def prepare_final_csv(
    input_csv: str, text_column="text", translated_column="translated", workers=2
):
    """
    Process a CSV file where each row can be Persian or English.
    - English rows are added directly to final.csv
    - Persian rows are translated and also added to the same 'text' column
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"File '{input_csv}' does not exist.")

    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' does not exist in the file. Columns: {df.columns.tolist()}"
        )

    # Split English and Persian rows
    english_rows = df[~df[text_column].apply(is_persian)]
    persian_rows = df[df[text_column].apply(is_persian)]

    # Paths
    folder, filename = os.path.split(input_csv)
    name, ext = os.path.splitext(filename)
    final_csv = os.path.join(folder, f"{name}_final.csv")
    temp_csv = os.path.join(folder, f"{name}_temp.csv")

    # Save English rows directly to final.csv
    english_rows[[text_column]].to_csv(final_csv, index=False, encoding="utf-8-sig")

    if persian_rows.empty:
        print("No Persian text found. Final file saved.")
        return final_csv

    # Save Persian rows to temp.csv for translation
    persian_rows[[text_column]].to_csv(temp_csv, index=False, encoding="utf-8-sig")

    # Translate Persian rows
    translated_csv = translate_csv(temp_csv, text_column=text_column, workers=workers)

    # Load translated text
    translated_df = pd.read_csv(translated_csv)

    # We only need the translated text under the same column name 'text'
    translated_text = translated_df[[translated_column]].rename(
        columns={translated_column: text_column}
    )

    # Combine English + translated rows (all under one column 'text')
    final_df = pd.concat(
        [english_rows[[text_column]], translated_text], ignore_index=True
    )

    # Save final CSV (only one column: 'text')
    final_df.to_csv(final_csv, index=False, encoding="utf-8-sig")

    # Clean up
    os.remove(temp_csv)
    os.remove(translated_csv)

    print(f"✅ Final CSV prepared (single 'text' column): {final_csv}")
    return final_csv


# ============================================
#   Main Execution
# ============================================
if __name__ == "__main__":
    # Step 1: Input file (contains Persian + English)
    input_csv = "./content/text_dataset/text_data.csv"

    # Step 2: Prepare final English-only CSV
    final_csv = prepare_final_csv(input_csv, text_column="text", workers=2)

    # Step 3: Generate and store embeddings in Weaviate
    process_texts(final_csv)
    weaviate_client.close()
