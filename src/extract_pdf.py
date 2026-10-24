
from pathlib import Path
from pypdf import PdfReader
import json

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Input PDF
pdf_path = BASE_DIR / "data" / "pdfs" / "spotify_web_app_architecture.pdf"

# Output JSON
output_path = BASE_DIR / "data" / "processed" / "spotify_pages.json"

# Check PDF
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

# Create output directory
output_path.parent.mkdir(parents=True, exist_ok=True)

# Read PDF
reader = PdfReader(str(pdf_path))

pages_data = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    pages_data.append({
        "page": page_number,
        "text": text.strip()
    })

    print(f"Page {page_number}: {len(text)} characters extracted")

# Save JSON with UTF-8 encoding
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(pages_data, file, ensure_ascii=False, indent=2)

print("\nPDF extraction completed!")
print("Total pages:", len(pages_data))
print("JSON saved at:", output_path)
