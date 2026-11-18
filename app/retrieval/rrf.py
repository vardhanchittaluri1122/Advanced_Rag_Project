
RRF_K = 60


def reciprocal_rank_fusion(
    dense_documents,
    lexical_documents,
    top_k: int = 5
):
    """
    Combine semantic and keyword results using RRF.
    """

    fused = {}

    retrieval_channels = [
        ("dense", dense_documents),
        ("lexical", lexical_documents)
    ]

    for channel, documents in retrieval_channels:

        for rank, document in enumerate(documents, start=1):

            content = document["content"]
            metadata = document.get("metadata") or {}

            item = fused.setdefault(
                content,
                {
                    "content": content,
                    "metadata": dict(metadata),
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "lexical_rank": None
                }
            )

            item["rrf_score"] += 1 / (RRF_K + rank)

            item[f"{channel}_rank"] = rank

            for key, value in metadata.items():
                item["metadata"].setdefault(key, value)

    ordered = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )[:top_k]

    return ordered
