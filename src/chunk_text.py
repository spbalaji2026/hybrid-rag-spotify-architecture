
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "spotify_pages.json"
output_path = BASE_DIR / "data" / "processed" / "spotify_chunks.json"

# Step 2: Load extracted PDF pages
with open(input_path, "r", encoding="utf-8") as file:
    pages = json.load(file)

print("Total PDF pages loaded:", len(pages))

# Step 3: Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Step 4: Split each page into chunks
all_chunks = []

for page in pages:
    page_number = page["page"]
    page_text = page["text"]

    chunks = text_splitter.split_text(page_text)

    for chunk_number, chunk in enumerate(chunks, start=1):
        all_chunks.append({
            "chunk_id": f"page_{page_number}_chunk_{chunk_number}",
            "page": page_number,
            "text": chunk
        })

    print(f"Page {page_number}: {len(chunks)} chunks created")

# Step 5: Save chunks as JSON
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, ensure_ascii=False, indent=2)

print("\nText chunking completed successfully!")
print("Total chunks created:", len(all_chunks))
print("Output saved at:", output_path)

# Step 6: Display a sample chunk
if all_chunks:
    print("\n--- SAMPLE CHUNK ---")
    print("Chunk ID:", all_chunks[0]["chunk_id"])
    print("Page:", all_chunks[0]["page"])
    print("Text:", all_chunks[0]["text"][:300])
