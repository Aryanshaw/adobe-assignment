import os
import re
import cohere
from openai import AsyncOpenAI
from langchain_core.tools import tool
from app.logger import logger
from app.db.qdrant import qdrant_db
from app.db.bm25 import bm25_db
from app.agent.utils import build_metadata_filter, rrf_fuse, format_results_with_citations

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cohere_client = cohere.AsyncClientV2(api_key=os.getenv("CO_API_KEY"))


@tool
async def retrieve_unstructured_context(query: str) -> str:
    """
    Searches company documents for relevant content.
    Automatically runs hybrid semantic + keyword search with reranking.

    Use for: strategy, risks, operational updates, narrative content,
    qualitative context, anything not requiring numeric computation.

    Args:
        query: precise description of what content is needed,
               including time period if relevant

    Returns:
        Top 5 relevant document excerpts with [source, page] citations.
    """
    # Step 1 — Embed query
    logger.info(f"Embedding query for unstructured retrieval: '{query[:50]}...'")
    emb_resp = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_vector = emb_resp.data[0].embedding

    # Step 2 — Dense ANN search (Qdrant)
    collection_name = os.getenv("UNSTRUCTURED_QDRANT_COLLECTION_NAME", "leadership_docs")
    payload_filter = build_metadata_filter(query)

    dense_results = await qdrant_db.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=payload_filter,
        limit=20
    )
    logger.info(f"Qdrant dense search returned {len(dense_results)} candidates")

    # Step 3 — BM25 sparse search (in-process)
    sparse_results = []
    if bm25_db.index and bm25_db.corpus:
        tokens = re.findall(r"\w+", query.lower())
        bm25_scores = await bm25_db.get_scores(tokens)
        top_bm25_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:20]
        sparse_results = [
            bm25_db.corpus[i]
            for i in top_bm25_idx
            if bm25_scores[i] > 0
        ]
        logger.info(f"BM25 sparse search returned {len(sparse_results)} candidates")
    else:
        logger.warning("BM25 index not loaded — falling back to dense-only retrieval.")

    # Step 4 — RRF fusion
    fused = rrf_fuse(dense_results, sparse_results)
    logger.info(f"RRF fusion reduced candidates to {len(fused)}")

    if not fused:
        return "No relevant documents found."

    try:
        logger.info(f"Reranking {len(fused)} documents with Cohere v3.5...")
        reranked = await cohere_client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=[r["text"] for r in fused],
            top_n=5
        )
        final = [fused[r.index] for r in reranked.results]
        logger.info("Cohere reranking complete")
    except Exception as e:
        logger.warning(f"Cohere rerank failed: {e}. Using RRF top-5.")
        final = fused[:5]

    # Step 6 — Format with citations
    return format_results_with_citations(final)
