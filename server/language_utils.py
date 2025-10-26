# server/language_utils.py
from deep_translator import GoogleTranslator

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
