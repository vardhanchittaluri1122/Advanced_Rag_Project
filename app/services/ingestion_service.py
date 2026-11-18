import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.pdf_loader import load_pdf
from app.database.vector_store import get_vector_store


def ingest_pdf(file_path, filename):

    # 1. Load PDF
    documents = load_pdf(file_path)

    if not documents:
        raise ValueError("No content found in PDF")

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks generated")

    # 3. Add metadata
    document_id = str(uuid.uuid4())

    for chunk in chunks:

        chunk.metadata["filename"] = filename

        chunk.metadata["document_id"] = document_id

        chunk.metadata["source"] = filename

    # 4. Get vector database
    vector_store = get_vector_store()

    # 5. Generate embeddings and store chunks
    vector_store.add_documents(chunks)

    return {
        "document_id": document_id,
        "filename": filename,
        "total_pages": len(documents),
        "total_chunks": len(chunks),
        "message": "PDF uploaded and indexed successfully"
    }