# Clip Semantic Search 
A simple, local Python backend that allows searching for images using natural language queries (semantic search) instead of exact keywords or filenames.

The project is split into five modules:

1. **`img-vectors_clip.ipynb`** (Image Processing):
   * Scans the local images folder.
   * Uses the CLIP model to convert each image into a vector embedding.
   * Saves all embeddings into a local file (`image_vectors.json`) to avoid re-calculating them every time.

2. **`init_db.py`** (Database Initialization):
   * Reads embeddings from `image_vectors.json`.
   * Loads embeddings into a local ChromaDB vector collection.

3. **`query_processor.py`** (Query Preprocessing):
   * Uses LangChain and Google Gemini to preprocess user queries.
   * Translates queries to English, handles negations, and expands visual keywords.

4. **`search_engine.py`** (Text Search):
   * Takes the text query.
   * Converts the text into a vector using the same CLIP model.
   * Queries ChromaDB to find and return the best matching images (cosine similarity).

5. **`search_pipeline.py`** (Search Pipeline):
   * Connects `query_processor.py` and `search_engine.py` into a single search workflow.

## Tech Stack

* **Language**: Python
* **Libraries**: PyTorch, Hugging Face Transformers (CLIP), Pillow (PIL), LangChain (Google Gemini)
* **Vector Database**: ChromaDB
