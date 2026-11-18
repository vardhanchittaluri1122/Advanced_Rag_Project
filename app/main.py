from fastapi import FastAPI

from app.api.routes.upload import router as upload_router
from app.api.routes.query import router as query_router

app = FastAPI(
    title="Advanced RAG System",
    description="Advanced RAG document ingestion and retrieval API",
    version="1.0.0"
)

app.include_router(
    upload_router,
    prefix="/api/v1",
    tags=["Document Ingestion"]
)

app.include_router(
    query_router,
    prefix="/api/v1",
    tags=["Document Retrieval"]
)


@app.get("/")
def home():
    return {"message": "Advanced RAG System is running"}