# src/tools/hybrid_retriever.py
"""
Hybrid retriever: combines keyword, semantic, and metadata search
using Reciprocal Rank Fusion (RRF).
"""

import logging

from langdetect import detect, LangDetectException

from src.config import (
    RERANK_CANDIDATE_K,
    RERANK_ENABLED,
    RERANK_SCORE_THRESHOLD,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
    USE_BM25,
)
from src.tools.keyword_search import keyword_search, load_chunks
from src.tools.metadata_filter import apply_metadata_boost, detect_language
from src.tools.reranker import rerank_results
from src.tools.semantic_search import semantic_search

log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------

def rrf_fusion(
    ranked_lists: list[list[dict]],
    weights: list[float] = None,
    k: int = 60,
) -> list[dict]:
    """
    Weighted Reciprocal Rank Fusion.
    weights: list of floats, one per ranked_list.
             Default is equal weights (1.0 each).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores:  dict[int, float] = {}
    chunk_store: dict[int, dict]  = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.get("chunk_id", -1)
            if cid == -1:
                continue
            rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + weight * (1.0 / (k + rank))
            chunk_store[cid] = result

    merged = []
    for cid, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = chunk_store[cid].copy()
        result["rrf_score"] = round(rrf_score, 6)
        result["score"]     = round(rrf_score, 6)
        merged.append(result)

    return merged


# ---------------------------------------------------------------
# Main hybrid search
# ---------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    use_bm25: bool = USE_BM25,
    use_semantic: bool = True,
    use_metadata_boost: bool = True,
    use_rerank: bool = RERANK_ENABLED,
    candidate_k: int = RERANK_CANDIDATE_K,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline.

    Steps:
    1. Semantic search top 10    → ranked list A
    2. Optional BM25 search top 10 → ranked list B
    3. RRF fusion + deduplication → merged ranking
    4. Metadata boosting         → final score adjustment
    5. Rerank merged top 10      → final top_k results

    Args:
        query:               User question
        top_k:               Final number of results (default 5)
        use_bm25:            Include BM25 keyword search
        use_semantic:        Include semantic vector search
        use_metadata_boost:  Apply metadata score adjustments
        use_rerank:          Apply local cross-encoder reranking
        candidate_k:         Candidates per method before fusion
    """
    ranked_lists = []
    weights = []

    # Step 1 - Semantic
    if use_semantic:
        sem_results = semantic_search(
            query, top_k=candidate_k,
            score_threshold=RETRIEVAL_SCORE_THRESHOLD,
            candidate_k=candidate_k,
        )
        ranked_lists.append(sem_results)
        weights.append(0.8)
        log.info(f"Semantic: {len(sem_results)} candidates")

    # Step 2 - BM25
    if use_bm25:
        bm25_results = keyword_search(query, top_k=candidate_k, candidate_k=candidate_k)
        ranked_lists.append(bm25_results)
        weights.append(0.2)
        log.info(f"BM25: {len(bm25_results)} candidates")

    if not ranked_lists:
        return []

    # Step 3 - Weighted RRF fusion
    fused      = rrf_fusion(ranked_lists, weights=weights, k=60)
    candidates = fused[:candidate_k]

    # Step 4 - Enrich candidates with metadata (required for boost factors)
    if use_metadata_boost:
        chunks       = load_chunks()
        chunk_by_id  = {c.get("chunk_id", i): c for i, c in enumerate(chunks)}
        for result in candidates:
            cid              = result.get("chunk_id", -1)
            chunk            = chunk_by_id.get(cid, {})
            result["metadata"] = chunk.get("metadata", {})

    # Step 5 - Rerank (sigmoid-normalised scores, min_score filter)
    if use_rerank:
        reranked = rerank_results(
            query, candidates, top_k=top_k, min_score=RERANK_SCORE_THRESHOLD
        )
    else:
        reranked = candidates[:top_k]

    # Step 6 - Metadata boost applied AFTER reranking so it is not
    #           overridden by the reranker score replacement
    if use_metadata_boost:
        final = apply_metadata_boost(reranked, query)
    else:
        final = reranked

    log.info(f"Hybrid search: '{query[:50]}' → {len(final)} results")
    return final


# ---------------------------------------------------------------
# Convenience wrappers for evaluation
# ---------------------------------------------------------------

def search_bm25_only(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """BM25 only — for MAP evaluation comparison."""
    return hybrid_search(
        query, top_k=top_k,
        use_bm25=True, use_semantic=False, use_metadata_boost=False,
        use_rerank=False
    )


def search_semantic_only(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Semantic only — for MAP evaluation comparison."""
    return hybrid_search(
        query, top_k=top_k,
        use_bm25=False, use_semantic=True, use_metadata_boost=False,
        use_rerank=False
    )


def search_hybrid_no_boost(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """BM25 + semantic without metadata boost — for ablation study."""
    return hybrid_search(
        query, top_k=top_k,
        use_bm25=True, use_semantic=True, use_metadata_boost=False,
        use_rerank=False
    )


def search_full_hybrid(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Full hybrid with metadata boosting and optional reranking."""
    return hybrid_search(
        query, top_k=top_k,
        use_bm25=USE_BM25, use_semantic=True, use_metadata_boost=True,
        use_rerank=RERANK_ENABLED
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = search_full_hybrid("how does drought affect olive production", top_k=3)
    for r in results:
        print(f"Score={r['score']:.6f} | RRF={r['rrf_score']:.6f} | "
              f"Boost={r.get('boost', 1.0):.3f}")
        print(f"Source: {r['source'][:40]} p{r['page']}")
        print(r["text"][:200])
        print()
