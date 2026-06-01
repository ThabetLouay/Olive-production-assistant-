# src/tools/reranker.py
"""
Local cross-encoder reranker for second-stage RAG retrieval.
"""

import logging
import math

from sentence_transformers import CrossEncoder

from src.config import RERANK_DEVICE, RERANK_MODEL

log = logging.getLogger(__name__)

_model = None


def _sigmoid(x: float) -> float:
    """Maps raw cross-encoder logit → probability in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def get_reranker() -> CrossEncoder:
    """Load and cache the local reranker model."""
    global _model
    if _model is None:
        import torch
        log.info(f"Loading reranker model: {RERANK_MODEL} on {RERANK_DEVICE}")
        kwargs = {"torch_dtype": torch.float16} if RERANK_DEVICE == "cuda" else {}
        _model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE, max_length=384, model_kwargs=kwargs)
    return _model


def rerank_results(
    query: str,
    results: list[dict],
    top_k: int,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Rerank retrieved chunks by direct query/chunk relevance.

    Raw cross-encoder logits are passed through sigmoid so the final
    score is a probability in [0, 1] — interpretable and comparable
    across queries.  Chunks scoring below min_score are dropped; if
    every chunk falls below the threshold the top-1 is kept so the
    agent always receives at least one result.
    """
    if top_k <= 0 or not results:
        return []

    model      = get_reranker()
    pairs      = [(query, r.get("text", "")) for r in results]
    raw_scores = model.predict(pairs)

    reranked = []
    for result, raw in zip(results, raw_scores):
        item                        = result.copy()
        score                       = round(_sigmoid(float(raw)), 6)
        item["score_before_rerank"] = item.get("score", 0.0)
        item["rerank_score"]        = score
        item["score"]               = score
        item["method"]              = f"{item.get('method', 'retrieval')}+rerank"
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    if min_score > 0.0:
        above = [r for r in reranked if r["rerank_score"] >= min_score]
        reranked = above if above else reranked[:1]

    top = reranked[:top_k]
    log.info(
        f"Reranked {len(results)} candidates → {len(top)} kept "
        f"(scores: max={top[0]['rerank_score']:.3f} "
        f"min={top[-1]['rerank_score']:.3f})"
    )
    return top
