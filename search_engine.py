import logging
import torch
from transformers import AutoProcessor, CLIPModel
import chromadb

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class ImageSearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32", db_path: str = "chroma_db", collection_name: str="image_embeddings"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Using device: {self.device.upper()}")

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        logger.info("CLIP model loaded successfully.")

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        logger.info(f"Successfully initialized collection '{collection_name}' at '{db_path}'. "
                    f"Current size: {self.collection.count()} embeddings.")

    def get_text_embedding(self, text: str) -> list[float]:
        logger.info(f"Vectorizing text query: '{text}'")
        # Tokenize text and move tensors to the correct device
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        # Get features
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features.pooler_output
            # L2 normalization
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Extract the 1D list
        return text_features[0].cpu().tolist()
    
    def search(self, text_query: str, top_k: int = 3) -> list[tuple[str, float]]:
        logger.info(f"Starting search for query: '{text_query}' (top_k={top_k})")
        query_vector = self.get_text_embedding(text_query)
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        image_ids = results['ids'][0]
        distances = results['distances'][0]
        formatted_results = []
        for img_id, dist in zip(image_ids, distances):
            similarity = 1 - dist
            formatted_results.append((img_id, similarity))
        return formatted_results
        

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