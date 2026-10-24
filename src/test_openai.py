
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Locate the project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Step 2: Load the API key from .env
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

# Step 3: Verify API key is configured
if not api_key or api_key == "your_actual_openai_api_key_here":
    raise ValueError("OPENAI_API_KEY is missing in .env")

print("API key loaded successfully!")

# Step 4: Create OpenAI client
client = OpenAI(api_key=api_key, timeout=20.0, max_retries=0)

# Step 5: Generate a test embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Spotify uses microservices for its web application."
)

# Step 6: Display result
embedding = response.data[0].embedding

print("OpenAI connection successful!")
print("Embedding model: text-embedding-3-small")
print("Vector dimensions:", len(embedding))
print("Embedding generated successfully!")
