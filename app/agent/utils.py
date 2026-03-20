import re
import json
from qdrant_client.models import Filter, FieldCondition, MatchValue


def build_metadata_filter(query: str) -> Filter | None:
    """Build Qdrant filter from query content hints."""
    conditions = []
    if conditions:
        return Filter(must=conditions)
    return None


def rrf_fuse(dense: list, sparse: list, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion of dense (Qdrant) and sparse (BM25) results.
    Discards raw scores, uses only rank positions.
    Returns top 20 deduplicated chunks sorted by RRF score as list of dicts.
    """
    scores = {}
    chunk_map = {}

    for rank, result in enumerate(dense):
        cid = result.id
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunk_map[cid] = {
            "text":   result.payload.get("text", ""),
            "source": result.payload.get("source_file", ""),
            "page":   result.payload.get("page", ""),
            "id":     cid,
        }

    for rank, result in enumerate(sparse):
        if isinstance(result, dict):
            cid = result.get("id", f"bm25_{rank}")
            normalized = {
                "text": result.get("text", ""),
                "source": result.get("source", result.get("source_file", "unknown")),
                "page": result.get("page", ""),
                "id": cid,
            }
        else:
            cid = f"bm25_{rank}"
            normalized = {
                "text": str(result),
                "source": "unknown",
                "page": "",
                "id": cid,
            }
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = normalized

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids[:20]]


def format_results_with_citations(results: list[dict]) -> str:
    """Format top results as SOURCE N [filename, page] citations."""
    if not results:
        return "No relevant documents found."

    parts = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "unknown")
        page   = r.get("page", "")
        text   = r.get("text", "")
        citation = f"[{source}, page {page}]" if page else f"[{source}]"
        parts.append(f"SOURCE {i} {citation}\n{text}")

    return "\n\n".join(parts)
