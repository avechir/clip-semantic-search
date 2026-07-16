import json
import os
import logging
import torch
from transformers import AutoProcessor, CLIPModel

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class ImageSearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32", vectors_path="img_vectors.json"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Using device: {self.device.upper()}")

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.info("CLIP model loaded successfully.")

        if not os.path.exists(vectors_path):
            logger.error(f"Database file '{vectors_path}' not found!")
            raise FileNotFoundError(f"Database file '{vectors_path}' not found.")
        
        with open(vectors_path, "r") as file:
            self.image_vectors = json.load(file)
        logger.info(f"Successfully loaded {len(self.image_vectors)} image vectors.")

    def get_text_embedding(self, text: str) -> torch.Tensor:
        logger.info(f"Vectorizing text query: '{text}'")
        # Tokenize text and move tensors to the correct device
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        # Get features
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features.pooler_output
            # L2 normalization
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Extract the 1D vector (shape [512] instead of [1, 512])
        return text_features[0]
    
    def search(self, text_query: str, top_k: int = 3) -> list:
        logger.info(f"Starting search for query: '{text_query}' (top_k={top_k})")
        query_vector = self.get_text_embedding(text_query)
        results = []
        for img_name, img_vec in self.image_vectors.items():
            img_tensor = torch.tensor(img_vec).to(self.device)
            if img_tensor.ndim > 1: img_tensor = img_tensor[0]
            img_tensor = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
            similarity = torch.dot(query_vector, img_tensor).item()
            results.append((img_name, similarity))
    
        results.sort(reverse=True, key=lambda x: x[1])
        logger.info(f"Search completed. Found {len(results)} total matches.")
        return results[:top_k]
        

if __name__ == "__main__":
    logger.info("=== Running local Search Engine test ===")
    try:
        # Initialize engine
        engine = ImageSearchEngine()
        
        # Test with a sample text query
        test_query = "something cozy" 
        
        # Execute search
        matches = engine.search(test_query, top_k=3)
        
        print("\n --- TOP MATCHES ---")
        for rank, (img_name, score) in enumerate(matches, 1):
            print(f"Rank {rank}: {img_name} | Similarity Score: {score:.4f}")
        print("----------------------\n")
        
    except Exception as e:
        logger.exception("An error occurred during the local test:")