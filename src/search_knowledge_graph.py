
import json
import re
from pathlib import Path

import networkx as nx

# Step 1: Locate project folder
BASE_DIR = Path(__file__).resolve().parent.parent

graph_path = (
    BASE_DIR
    / "knowledge_graph"
    / "spotify_graph.json"
)

# Step 2: Load Knowledge Graph
with open(graph_path, "r", encoding="utf-8") as file:
    graph_data = json.load(file)

graph = nx.node_link_graph(
    graph_data,
    edges="links"
)

print("Knowledge Graph loaded successfully!")
print("Total nodes:", graph.number_of_nodes())
print("Total relationships:", graph.number_of_edges())

# Step 3: Ask a question
question = input("\nEnter your question: ").strip()

if not question:
    raise ValueError("Question cannot be empty")

# Step 4: Match graph entity names in the question
def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

question_words = tokenize(question)

# Ignore common question words
stopwords = {
    "what", "which", "who", "where", "when",
    "why", "how", "does", "do", "is", "are",
    "the", "a", "an", "in", "of", "for",
    "to", "and", "used", "use", "uses",
    "technologies", "technology"
}

question_words -= stopwords

matched_nodes = []

for node in graph.nodes():
    node_words = tokenize(str(node))

    if node_words and node_words.issubset(question_words):
        matched_nodes.append(node)

# Step 5: Display matching relationships
print("\n--- MATCHED ENTITIES ---")

if not matched_nodes:
    print("No matching graph entities found.")
else:
    for node in matched_nodes:
        print("-", node)

print("\n--- GRAPH SEARCH RESULTS ---")

results = []
seen = set()

for node in matched_nodes:
    # Find relationships where the entity is the source
    for source, target, data in graph.out_edges(
        node,
        data=True
    ):
        key = (
            source,
            target,
            data.get("relation"),
            data.get("chunk_id")
        )

        if key not in seen:
            seen.add(key)
            results.append((source, target, data))

    # Find relationships where the entity is the target
    for source, target, data in graph.in_edges(
        node,
        data=True
    ):
        key = (
            source,
            target,
            data.get("relation"),
            data.get("chunk_id")
        )

        if key not in seen:
            seen.add(key)
            results.append((source, target, data))

if not results:
    print("No relationships found for this question.")

for number, (source, target, data) in enumerate(
    results,
    start=1
):
    print(f"\nResult {number}")
    print(
        f"{source} --[{data.get('relation')}]--> {target}"
    )
    print("PDF Page:", data.get("page"))
    print("Evidence:", data.get("evidence"))
    print("-" * 50)

print("\nTotal relationships retrieved:", len(results))
