# Clip Semantic Search 
A simple, local Python backend that allows searching for images using natural language queries (semantic search) instead of exact keywords or filenames.

The project is split into two simple scripts:

1. **`encode_images.py`** (Image Processing):
   * Scans the local images folder.
   * Uses the CLIP model to convert each image into a vector embedding.
   * Saves all embeddings into a local file (`image_vectors.json`) to avoid re-calculating them every time.

2. **`search.py`** (Text Search):
   * Takes the text query.
   * Converts the text into a vector using the same CLIP model.
   * Calculates the cosine similarity between the text vector and the pre-computed image vectors to find the best match.

## Tech Stack

* **Language**: Python
* **Libraries**: PyTorch, Hugging Face Transformers (CLIP), Pillow (PIL)

## Future Plans

* Replace the local JSON storage with a dedicated vector database for better scalability.
