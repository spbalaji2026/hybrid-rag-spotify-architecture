
import os
import json
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_store"

# Step 2: Load API key
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing in .env")

client = OpenAI(api_key=api_key)

# Step 3: Load existing FAISS index
index = faiss.read_index(
    str(VECTOR_DIR / "spotify.index")
)

# Step 4: Load chunk metadata
with open(
    VECTOR_DIR / "metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

chunks = metadata["chunks"]
model = metadata["embedding_model"]

if index.ntotal != len(chunks):
    raise ValueError("FAISS index and metadata count do not match")

print("FAISS database loaded successfully!")
print("Total vectors:", index.ntotal)

# Step 5: Ask a question
question = input("\nEnter your question: ").strip()

if not question:
    raise ValueError("Question cannot be empty")

# Step 6: Convert question into embedding
response = client.embeddings.create(
    model=model,
    input=question
)

query_vector = np.array(
    [response.data[0].embedding],
    dtype=np.float32
)

# Step 7: Normalize question vector
faiss.normalize_L2(query_vector)

# Step 8: Search top 3 matching chunks
top_k = min(3, index.ntotal)

scores, indices = index.search(
    query_vector,
    top_k
)

# Step 9: Display search results
print("\n--- TOP SEARCH RESULTS ---")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):
    if idx == -1:
        continue

    chunk = chunks[int(idx)]

    print(f"\nResult {rank}")
    print("Chunk ID:", chunk["chunk_id"])
    print("PDF Page:", chunk["page"])
    print("Cosine Similarity:", round(float(score), 4))
    print("Text:")
    print(chunk["text"])
    print("-" * 50)
