import os
import json
import logging
import chromadb
import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class VectorDBInitializer:
    def __init__(self, db_path: str = "chroma_db", collection_name: str = "image_embeddings"):
        """
        Initialize ChromaDB persistent client and collection.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        logger.info(f"Successfully initialized collection '{collection_name}' at '{db_path}'. "
                    f"Current size: {self.collection.count()} embeddings.")

    def load_json_vectors(self, json_path: str) -> dict:
        """
        Read and parse image vectors from a JSON file.
        """
        if not os.path.exists(json_path):
            logger.error(f"File '{json_path}' not found!")
            raise FileNotFoundError(f"File '{json_path}' not found.")

        try:
            with open(json_path, "r") as file:
                image_vectors = json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from '{json_path}': {e}")
            raise

        if not isinstance(image_vectors, dict):
            logger.error(f"JSON data in '{json_path}' must be a dictionary, got {type(image_vectors).__name__}")
            raise ValueError(f"Expected dict, got {type(image_vectors).__name__}")
        logger.info(f"Successfully loaded {len(image_vectors)} image vectors.")
        return image_vectors

    def populate_db(self, json_path: str = "img_vectors.json") -> None:
        """
        Migrate vectors from JSON file into ChromaDB collection.
        """
        image_vectors = self.load_json_vectors(json_path)
        ids = list(image_vectors.keys())
        embeddings = np.array(list(image_vectors.values()))
        self.collection.add(ids=ids, embeddings=embeddings)
        logger.info(f"Successfully added vectors to the DB. Total embeddings in collection: {self.collection.count()}")

if __name__ == "__main__":
    initializer = VectorDBInitializer()
    initializer.populate_db("img_vectors.json")