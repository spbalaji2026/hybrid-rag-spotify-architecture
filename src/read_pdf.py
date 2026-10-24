
from pathlib import Path
from pypdf import PdfReader

# Find the project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of the teacher's PDF
pdf_path = BASE_DIR / "data" / "pdfs" / "spotify_web_app_architecture.pdf"

# Check whether the PDF exists
if not pdf_path.exists():
    print("ERROR: PDF file not found!")
else:
    # Open and read the PDF
    reader = PdfReader(str(pdf_path))

    # Count pages
    print("PDF loaded successfully!")
    print("Total pages:", len(reader.pages))

    # Extract text from the first page
    first_page = reader.pages[0]
    text = first_page.extract_text()

    print("\n--- FIRST PAGE CONTENT ---\n")
    print(text if text else "No extractable text found.")
