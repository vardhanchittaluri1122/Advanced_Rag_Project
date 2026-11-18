
import asyncio
import sys

# Fix Psycopg async compatibility on Windows.
# This must execute before PGEngine creates its background event loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore

from app.core.config import DATABASE_URL, TABLE_NAME


def create_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        output_dimensionality=768
    )


@lru_cache(maxsize=1)
def create_engine():
    return PGEngine.from_connection_string(
        url=DATABASE_URL
    )


@lru_cache(maxsize=1)
def get_vector_store():
    embeddings = create_embeddings()
    engine = create_engine()

    return PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=embeddings
    )
