import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("rag-service.retrievals.rrf")


def reciprocal_rank_fusion_multi(
    result_lists: Optional[List[Optional[List[Dict[str, Any]]]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists into a single ranking using Reciprocal Rank Fusion (RRF).

    Formula:
        score(d) = sum( 1 / (k + rank_i(d)) )

    Where:
        d = document chunk
        rank_i(d) = 1-based ranking position of document d in result list i
        k = configurable smoothing constant

    Args:
        result_lists: A list of result lists, where each result list contains dict items.
                      Each item dict must contain at least a string "chunk_id" key.
                      Original search scores are ignored.
        k: Configurable constant (default 60). Must be greater than 0.

    Returns:
        A list of dicts with keys "chunk_id" and "rrf_score", sorted descending by "rrf_score".
        Empty or None inputs are handled gracefully without crashing.
        Malformed entries are ignored.
    """
    if k <= 0:
        raise ValueError(f"Constant k must be greater than 0, got {k}")

    if result_lists is None:
        return []

    scores: Dict[str, float] = {}

    for index, result_list in enumerate(result_lists):
        if result_list is None or not isinstance(result_list, list):
            continue

        seen_in_current_list = set()
        rank = 1

        for item in result_list:
            if not isinstance(item, dict):
                # Ignore non-dictionary items
                continue

            chunk_id = item.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str) or not chunk_id.strip():
                # Ignore malformed entries without chunk_id or non-string chunk_id
                continue

            # If the same chunk_id exists multiple times in the SAME list,
            # we only consider the highest ranking occurrence (the first one).
            if chunk_id in seen_in_current_list:
                continue
            seen_in_current_list.add(chunk_id)

            # Calculate and accumulate RRF score
            reciprocal_rank_score = 1.0 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + reciprocal_rank_score
            
            # Increment rank for the next unique valid item in this list
            rank += 1

    # Sort items descending by score
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Format return list
    return [{"chunk_id": chunk_id, "rrf_score": score} for chunk_id, score in sorted_items]


def reciprocal_rank_fusion(
    vector_results: Optional[List[Dict[str, Any]]],
    keyword_results: Optional[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) to combine vector search results and keyword search results.

    Args:
        vector_results: Ranked list of vector search results.
        keyword_results: Ranked list of keyword search results.
        k: Configurable constant (default 60).

    Returns:
        Combined list of results sorted descending by rrf_score.
    """
    return reciprocal_rank_fusion_multi([vector_results, keyword_results], k)
