import logging
from query_processor import QueryProcessor
from search_engine import ImageSearchEngine

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class MultimodalSearchPipeline:
    """
    Orchestrates LLM query preprocessing and CLIP vector image retrieval
    into a unified end-to-end search pipeline.
    """
    def __init__(self, dotenv_path: str = ".env"):
        logger.info("Initializing Multimodal Search Pipeline...")
        self.query_processor = QueryProcessor(dotenv_path=dotenv_path)
        self.search_engine = ImageSearchEngine()
        logger.info("Pipeline initialized successfully.")

    def run(self, raw_query: str, top_k: int = 3) -> dict:
        """
        Executes full end-to-end workflow:
        1. Preprocesses raw user input using LLM.
        2. Queries ChromaDB vector database using CLIP embeddings.
        """
        logger.info(f"--- Pipeline Execution Started for: '{raw_query}' ---")

        # Step 1: LLM Preprocessing 
        processed_prompt = self.query_processor.process_query(raw_query)

        # Step 2: Vector Search via CLIP & ChromaDB
        search_results = self.search_engine.search(processed_prompt, top_k=top_k)

        logger.info("--- Pipeline Execution Completed ---")

        return {
            "raw_query": raw_query,
            "processed_prompt": processed_prompt,
            "results": search_results,
        }


if __name__ == "__main__":
    logger.info("=== Running Multimodal Pipeline Integration Test ===")

    # Initialize full pipeline
    pipeline = MultimodalSearchPipeline()

    # Test query with negation and Ukrainian language
    test_query = "покажи зображення, де є дикі тварини, але не коні"

    # Run search
    response = pipeline.run(test_query, top_k=3)

    # Output formatted results
    print("\n" + "=" * 55)
    print(f"RAW USER QUERY     : {response['raw_query']}")
    print(f"PROCESSED LLM PROMPT: {response['processed_prompt']}")
    print("-" * 55)
    print("TOP MATCHED IMAGES:")
    for rank, (img_name, score) in enumerate(response["results"], 1):
        print(f"  [{rank}] {img_name:<35} | Similarity: {score:.4f}")
    print("=" * 55 + "\n")