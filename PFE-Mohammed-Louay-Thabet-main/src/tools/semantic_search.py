# src/tools/semantic_search.py
"""
Semantic search using bge-m3 embeddings + Qdrant vector store.
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from src.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBED_MODEL, EMBED_DEVICE,
    RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------
_model  = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import torch
        log.info(f"Loading embedding model: {EMBED_MODEL} on {EMBED_DEVICE}")
        kwargs = {"torch_dtype": torch.float16} if EMBED_DEVICE == "cuda" else {}
        _model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE, model_kwargs=kwargs)
    return _model


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def embed_query(query: str) -> list[float]:
    """
    Embed a query string using bge-m3.
    Uses asymmetric retrieval prefix for better accuracy.
    """
    model = get_model()
    prefixed = f"Represent this agricultural question for retrieving documents: {query}"
    return model.encode(prefixed, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------

def semantic_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    domain_filter: str = None,
    content_type_filter: str = None,
    candidate_k: int = 20,
) -> list[dict]:
    """
    Semantic vector search via Qdrant.

    Args:
        query:                Natural language question
        top_k:                Number of results to return
        score_threshold:      Minimum cosine similarity score
        domain_filter:        Filter by 'olive_agronomy' or 'related_domain'
        content_type_filter:  Filter by 'text' or 'table'
        candidate_k:          Internal candidate pool size

    Returns:
        List of result dicts sorted by cosine similarity descending
    """
    client    = get_client()
    query_vec = embed_query(query)

    # Build optional Qdrant filter
    must_conditions = []
    if domain_filter:
        must_conditions.append(
            FieldCondition(key="domain", match=MatchValue(value=domain_filter))
        )
    if content_type_filter:
        must_conditions.append(
            FieldCondition(key="content_type", match=MatchValue(value=content_type_filter))
        )

    search_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec,
        limit=candidate_k,
        score_threshold=score_threshold,
        query_filter=search_filter,
        with_payload=True,
    ).points

    output = []
    for r in results:
        output.append({
            "text":     r.payload.get("text", ""),
            "score":    round(float(r.score), 6),
            "source":   r.payload.get("source_pdf", "unknown"),
            "page":     r.payload.get("page", "?"),
            "section":  r.payload.get("section", "unknown"),
            "domain":   r.payload.get("domain", "unknown"),
            "language": r.payload.get("language", "unknown"),
            "type":     r.payload.get("content_type", "text"),
            "chunk_id": r.payload.get("chunk_id", r.id),
            "method":   "semantic",
        })

    final = output[:top_k]
    log.info(f"Semantic search: '{query[:50]}' → {len(final)} results")
    return final


def semantic_search_two_tier(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Two-tier retrieval:
    1. Search olive_agronomy domain first (priority 1)
    2. Fall back to related_domain if fewer than 2 results
    """
    results = semantic_search(
        query=query, top_k=top_k,
        domain_filter="olive_agronomy"
    )
    if len(results) < 2:
        log.info("Tier 1 insufficient — falling back to related_domain")
        fallback = semantic_search(
            query=query, top_k=top_k,
            domain_filter="related_domain"
        )
        results.extend(fallback)
    return results[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = semantic_search("drought resistance olive tree Tunisia", top_k=3)
    for r in results:
        print(f"Score={r['score']:.4f} | {r['source'][:40]} p{r['page']}")
        print(r["text"][:200])
        print()