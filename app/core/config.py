import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing in .env")

TABLE_NAME = "pdf_embeddings"

MAX_FILE_SIZE = 20 * 1024 * 1024
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150