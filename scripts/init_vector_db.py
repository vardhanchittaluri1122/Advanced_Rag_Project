import asyncio
import sys

# Must be configured before PGEngine creates its background event loop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

from app.database.vector_store import (
    create_embeddings,
    create_engine
)

from app.core.config import TABLE_NAME


def initialize_vector_database():

    print("Connecting to PostgreSQL...")

    embeddings = create_embeddings()

    # Create PGEngine only after setting Windows event-loop policy
    engine = create_engine()

    print("Checking Gemini embedding dimensions...")

    vector_size = len(
        embeddings.embed_query("Test embedding dimension")
    )

    print("Embedding dimensions:", vector_size)

    print("Creating vector table...")

    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=vector_size
    )

    print("Vector table initialized successfully!")


if __name__ == "__main__":
    initialize_vector_database()