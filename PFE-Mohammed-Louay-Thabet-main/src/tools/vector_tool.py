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

_model  = None
_client = None

def _get_model():
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    return _model

def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client

def search_documents(query, top_k=RETRIEVAL_TOP_K, domain_filter=None, content_type=None, exclude_ocr_noise=False):
    model  = _get_model()
    client = _get_client()

    query_text = f"Represent this agricultural question for retrieving documents: {query}"
    query_vec  = model.encode(query_text, normalize_embeddings=True).tolist()

    must_conditions = []
    if domain_filter:
        must_conditions.append(FieldCondition(key="domain", match=MatchValue(value=domain_filter)))
    if content_type:
        must_conditions.append(FieldCondition(key="content_type", match=MatchValue(value=content_type)))
    if exclude_ocr_noise:
        must_conditions.append(FieldCondition(key="ocr_noise_flag", match=MatchValue(value=False)))

    search_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec,
        limit=top_k,
        score_threshold=RETRIEVAL_SCORE_THRESHOLD,
        query_filter=search_filter,
        with_payload=True,
    ).points

    output = []
    for r in results:
        output.append({
            "text":     r.payload.get("text", ""),
            "score":    round(r.score, 4),
            "source":   r.payload.get("source_pdf", "unknown"),
            "page":     r.payload.get("page", "?"),
            "section":  r.payload.get("section", "unknown"),
            "domain":   r.payload.get("domain", "unknown"),
            "language": r.payload.get("language", "unknown"),
            "type":     r.payload.get("content_type", "text"),
        })

    log.info(f"Vector search: {len(output)} results")
    return output

def search_with_fallback(query, top_k=RETRIEVAL_TOP_K):
    results = search_documents(query=query, top_k=top_k, domain_filter="olive_agronomy")
    if len(results) < 2:
        log.info("Tier 1 insufficient - falling back to related_domain")
        fallback = search_documents(query=query, top_k=top_k, domain_filter="related_domain")
        results.extend(fallback)
    return results[:top_k]
