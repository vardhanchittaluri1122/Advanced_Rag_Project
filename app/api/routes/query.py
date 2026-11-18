import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.query_service import (
    hybrid_retrieve_documents,
    retrieve_documents,
    search_documents,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query")
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/retrieve/hybrid")
async def retrieve_hybrid_documents(request: QueryRequest):
    """Combine semantic vector and PostgreSQL full-text retrieval with RRF."""
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = await run_in_threadpool(
            hybrid_retrieve_documents,
            user_query,
            request.top_k,
        )
        return {
            "query": user_query,
            "top_k": request.top_k,
            "method": "hybrid_rrf",
            "count": len(results),
            "results": results,
        }
    except Exception:
        logger.exception("Hybrid document retrieval failed")
        raise HTTPException(
            status_code=500,
            detail="Hybrid document retrieval failed. Check server logs.",
        )


@router.post("/retrieve")
async def retrieve_matching_documents(request: QueryRequest):
    """Search the vector database and return matching chunks without an LLM call."""
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = await run_in_threadpool(
            retrieve_documents,
            user_query,
            request.top_k,
        )
        return {
            "query": user_query,
            "top_k": request.top_k,
            "count": len(results),
            "results": results,
        }
    except Exception:
        logger.exception("Document retrieval failed")
        raise HTTPException(
            status_code=500,
            detail="Document retrieval failed. Check server logs.",
        )


@router.post("/query")
async def query_documents(request: QueryRequest):

    user_query = request.query.strip()

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:
        result = await run_in_threadpool(
            search_documents,
            user_query,
            request.top_k
        )

        return {
            "query": user_query,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception:
        logger.exception("Document search failed")
        raise HTTPException(
            status_code=500,
            detail="Document search failed. Check server logs."
        )
