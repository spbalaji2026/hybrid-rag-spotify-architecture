
import os
import json
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "spotify_chunks.json"
vector_dir = BASE_DIR / "vector_store"

vector_dir.mkdir(parents=True, exist_ok=True)

# Step 2: Load environment variables
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing in .env")

client = OpenAI(api_key=api_key)

model = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# Step 3: Load text chunks
with open(input_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("No text chunks found!")

texts = [chunk["text"] for chunk in chunks]

print("Total chunks loaded:", len(texts))

# Step 4: Generate embeddings
print("Generating embeddings using OpenAI...")

response = client.embeddings.create(
    model=model,
    input=texts
)

# Preserve embedding order using API indices
sorted_data = sorted(response.data, key=lambda item: item.index)

embeddings = [item.embedding for item in sorted_data]

# Step 5: Convert embeddings to NumPy array
vectors = np.array(embeddings, dtype=np.float32)

# Step 6: Normalize vectors for cosine similarity
faiss.normalize_L2(vectors)

dimension = vectors.shape[1]

print("Embedding dimensions:", dimension)

# Step 7: Create FAISS index
index = faiss.IndexFlatIP(dimension)

# Step 8: Add vectors
index.add(vectors)

print("Vectors stored in FAISS:", index.ntotal)

# Step 9: Save FAISS index
faiss.write_index(
    index,
    str(vector_dir / "spotify.index")
)

# Step 10: Save chunk metadata
with open(
    vector_dir / "metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "embedding_model": model,
            "dimension": dimension,
            "chunks": chunks
        },
        file,
        ensure_ascii=False,
        indent=2
    )

print("\nFAISS Vector Database created successfully!")
print("Index saved:", vector_dir / "spotify.index")
print("Metadata saved:", vector_dir / "metadata.json")
