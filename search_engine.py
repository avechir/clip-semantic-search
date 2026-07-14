import json
import os
import torch
from transformers import AutoProcessor, CLIPModel

class ImageSearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32", vectors_path="img_vectors.json"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        with open(vectors_path, "r") as file:
            self.image_vectors = json.load(file)

    # def get_text_embedding(self, text: str) -> torch.Tensor:
        