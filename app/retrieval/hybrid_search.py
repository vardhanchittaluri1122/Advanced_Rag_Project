
from app.retrieval.semantic_search import semantic_search_documents
from app.retrieval.keyword_search import keyword_search
from app.retrieval.rrf import reciprocal_rank_fusion


HYBRID_CANDIDATE_MULTIPLIER = 4


def hybrid_retrieve_documents(
    query: str,
    top_k: int = 5
):
    """
    Combine semantic and keyword search using RRF.
    """

    candidate_limit = top_k * HYBRID_CANDIDATE_MULTIPLIER

    # 1. Semantic retrieval
    dense_documents = semantic_search_documents(
        query=query,
        top_k=candidate_limit
    )

    # Convert LangChain documents into a common format
    dense_results = [
        {
            "content": document.page_content,
            "metadata": document.metadata or {}
        }
        for document in dense_documents
    ]

    # 2. Keyword retrieval
    lexical_results = keyword_search(
        query=query,
        top_k=candidate_limit
    )

    # 3. Combine using RRF
    fused_results = reciprocal_rank_fusion(
        dense_documents=dense_results,
        lexical_documents=lexical_results,
        top_k=top_k
    )

    # 4. Format final results
    results = []

    for rank, item in enumerate(fused_results, start=1):

        metadata = item["metadata"]
        page = metadata.get("page")

        results.append({
            "rank": rank,
            "filename": metadata.get(
                "filename",
                "Unknown file"
            ),
            "page": page + 1 if isinstance(page, int) else None,
            "content": item["content"],
            "rrf_score": round(item["rrf_score"], 6),
            "dense_rank": item["dense_rank"],
            "lexical_rank": item["lexical_rank"]
        })

    return results
