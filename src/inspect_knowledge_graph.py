
import json
from pathlib import Path

import networkx as nx

# Step 1: Locate the saved Knowledge Graph
BASE_DIR = Path(__file__).resolve().parent.parent

graph_path = (
    BASE_DIR
    / "knowledge_graph"
    / "spotify_graph.json"
)

# Step 2: Load JSON data
with open(graph_path, "r", encoding="utf-8") as file:
    graph_data = json.load(file)

# Step 3: Reconstruct NetworkX graph
graph = nx.node_link_graph(
    graph_data,
    edges="links"
)

print("Knowledge Graph loaded successfully!")
print("Total nodes:", graph.number_of_nodes())
print("Total relationships:", graph.number_of_edges())

# Step 4: Inspect first 10 relationships
print("\n--- FIRST 10 RELATIONSHIPS ---")

for number, (source, target, data) in enumerate(
    graph.edges(data=True),
    start=1
):
    if number > 10:
        break

    print(f"\nRelationship {number}")
    print("Source:", source)
    print("Relation:", data.get("relation"))
    print("Target:", target)
    print("PDF Page:", data.get("page"))
    print("Chunk ID:", data.get("chunk_id"))
    print("Evidence:", data.get("evidence"))
    print("-" * 50)

# Step 5: Check whether every relationship has provenance
missing_evidence = 0

for source, target, data in graph.edges(data=True):
    if not all([
        data.get("relation"),
        data.get("page"),
        data.get("chunk_id"),
        data.get("evidence")
    ]):
        missing_evidence += 1

print("\n--- VALIDATION SUMMARY ---")
print("Relationships missing provenance:", missing_evidence)
print("Inspection completed!")
