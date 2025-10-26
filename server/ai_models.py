import torch
import whisper
import open_clip
from .config import WHISPER_MODEL, CLIP_MODEL_NAME, PRETRAINED_LOCAL_PATH, DEVICE


""" Whisper """
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)


""" Open_CLIP """
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL_NAME, pretrained=PRETRAINED_LOCAL_PATH
)
tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
clip_model.to(DEVICE)
clip_model.eval()
