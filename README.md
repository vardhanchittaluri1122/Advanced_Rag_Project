# Advanced RAG System

## Retrieve matching chunks

After uploading and indexing a PDF, use the retrieval endpoint to inspect the top matching chunks from the vector database. This step performs semantic search only; it does not call the language model.

```http
POST /api/v1/retrieve
Content-Type: application/json

{
  "query": "What does the document say about ...?",
  "top_k": 3
}
```

The response includes the normalized query, number of returned chunks, and each match's rank, filename, page, and content. `top_k` defaults to 5 and accepts values from 1 to 20.

The existing `POST /api/v1/query` endpoint remains the full RAG flow: it retrieves chunks, sends them as context to Gemini, and returns an answer with source references.

## Step 1: Hybrid retrieval with RRF

Use the hybrid endpoint to compare dense semantic search with PostgreSQL full-text search. It returns retrieved chunks only and does not call Gemini's chat model.

```http
POST /api/v1/retrieve/hybrid
Content-Type: application/json

{
  "query": "What does the document say about ...?",
  "top_k": 5
}
```

Dense search finds conceptually similar wording through embeddings. Lexical search matches query terms using PostgreSQL `websearch_to_tsquery`. Reciprocal Rank Fusion (RRF) merges the two ordered lists with `1 / (60 + rank)` per list. It uses ranks instead of comparing incompatible raw scores. The response exposes each match's `dense_rank`, `lexical_rank`, and `rrf_score` so the fusion can be inspected. The endpoint retrieves up to four times `top_k` candidates from each method before returning the fused top results.

Example PowerShell call:

```powershell
$body = @{ query = "your question"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/retrieve/hybrid" `
  -Method Post -ContentType "application/json" -Body $body
```
