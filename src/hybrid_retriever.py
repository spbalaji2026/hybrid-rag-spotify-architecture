
import os
import re
import json
from pathlib import Path

import faiss
import numpy as np
import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vector_store"
GRAPH_DIR = BASE_DIR / "knowledge_graph"

# Step 2: Load OpenAI API key
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing")

client = OpenAI(api_key=api_key)

# Step 3: Load FAISS
index = faiss.read_index(
    str(VECTOR_DIR / "spotify.index")
)

with open(
    VECTOR_DIR / "metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

chunks = metadata["chunks"]
embedding_model = metadata["embedding_model"]

if index.ntotal != len(chunks):
    raise ValueError("FAISS index and metadata mismatch")

# Step 4: Load Knowledge Graph
with open(
    GRAPH_DIR / "spotify_graph.json",
    "r",
    encoding="utf-8"
) as file:
    graph_data = json.load(file)

graph = nx.node_link_graph(
    graph_data,
    edges="links"
)

# Step 5: Define text matching helper
def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

stopwords = {
    "what", "which", "who", "where", "when",
    "why", "how", "does", "do", "is", "are",
    "the", "a", "an", "in", "of", "for",
    "to", "and", "used", "use", "uses",
    "technologies", "technology"
}

# Step 6: FAISS retrieval
def search_faiss(question, top_k=3):
    response = client.embeddings.create(
        model=embedding_model,
        input=question
    )

    query_vector = np.array(
        [response.data[0].embedding],
        dtype=np.float32
    )

    faiss.normalize_L2(query_vector)

    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "score": float(score),
            "text": chunk["text"]
        })

    return results

# Step 7: Knowledge Graph retrieval
def search_graph(question):
    question_words = tokenize(question) - stopwords

    matched_nodes = []

    for node in graph.nodes():
        node_words = tokenize(str(node))

        if node_words and node_words.issubset(question_words):
            matched_nodes.append(node)

    results = []
    seen = set()

    for node in matched_nodes:
        edges = list(graph.out_edges(node, data=True))
        edges += list(graph.in_edges(node, data=True))

        for source, target, data in edges:
            key = (
                source,
                target,
                data.get("relation"),
                data.get("chunk_id")
            )

            if key in seen:
                continue

            seen.add(key)

            results.append({
                "source": source,
                "relation": data.get("relation"),
                "target": target,
                "page": data.get("page"),
                "chunk_id": data.get("chunk_id"),
                "evidence": data.get("evidence")
            })

    return results

# Step 8: Ask question
question = input("\nEnter your question: ").strip()

if not question:
    raise ValueError("Question cannot be empty")

print("\nRunning Hybrid Retrieval...")

vector_results = search_faiss(question)
graph_results = search_graph(question)

# Step 9: Display FAISS results
print("\n========== FAISS RESULTS ==========")

for number, item in enumerate(vector_results, start=1):
    print(f"\nResult {number}")
    print("PDF Page:", item["page"])
    print("Similarity:", round(item["score"], 4))
    print("Text:", item["text"])

# Step 10: Display graph results
print("\n======= KNOWLEDGE GRAPH RESULTS =======")

if not graph_results:
    print("No graph relationships matched.")

for number, item in enumerate(graph_results, start=1):
    print(f"\nRelationship {number}")
    print(
        f"{item['source']} "
        f"--[{item['relation']}]--> "
        f"{item['target']}"
    )
    print("PDF Page:", item["page"])
    print("Evidence:", item["evidence"])

# Step 11: Summary
print("\n========== HYBRID SUMMARY ==========")
print("FAISS chunks retrieved:", len(vector_results))
print("Graph relationships retrieved:", len(graph_results))
print("Hybrid retrieval completed successfully!")
