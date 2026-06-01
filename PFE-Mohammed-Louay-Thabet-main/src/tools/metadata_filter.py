# src/tools/metadata_filter.py
"""
Metadata-based scoring, boosting and filtering for RAG retrieval.
Applied after BM25 and semantic search, before final ranking.
"""

import logging
from langdetect import detect, LangDetectException

log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Boost configuration
# ---------------------------------------------------------------

DOMAIN_BOOST = {
    "olive_agronomy": 1.30,   # primary domain — 30% boost
    "related_domain": 0.85,   # secondary domain — 15% penalty
}

CONTENT_TYPE_BOOST = {
    "table": 1.20,            # tables carry dense structured info
    "text":  1.00,
}

OCR_NOISE_PENALTY = {
    True:  0.85,              # noisy OCR — 15% penalty
    False: 1.00,
}

PRIORITY_BOOST = {
    1: 1.00,                  # primary documents — no change
    2: 0.90,                  # secondary documents — 10% penalty
}

LANGUAGE_BOOST = {
    "match":   1.15,          # query and chunk same language
    "mismatch": 0.95,         # different languages
    "unknown": 1.00,          # can't determine
}

# ---------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect language of a text string. Returns ISO code or 'unknown'."""
    try:
        return detect(text) if len(text.strip()) > 30 else "unknown"
    except LangDetectException:
        return "unknown"


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------

def compute_metadata_boost(chunk: dict, query_lang: str = "unknown") -> float:
    """
    Compute a metadata-based score multiplier for a chunk.

    Factors:
    - Domain relevance (olive_agronomy vs related_domain)
    - Content type (table vs text)
    - OCR quality (noise flag)
    - Document retrieval priority
    - Language match between query and chunk

    Args:
        chunk:      Chunk dict with 'metadata' key
        query_lang: Detected language of the query ('fr', 'en', etc.)

    Returns:
        Float multiplier (>1.0 = boost, <1.0 = penalty)
    """
    meta  = chunk.get("metadata", {})
    score = 1.0

    # Domain
    domain = meta.get("domain", "")
    score *= DOMAIN_BOOST.get(domain, 1.0)

    # Content type
    ctype = meta.get("content_type", "text")
    score *= CONTENT_TYPE_BOOST.get(ctype, 1.0)

    # OCR noise
    noisy = meta.get("ocr_noise_flag", False)
    score *= OCR_NOISE_PENALTY.get(bool(noisy), 1.0)

    # Retrieval priority
    priority = meta.get("retrieval_priority", 1)
    score *= PRIORITY_BOOST.get(int(priority), 1.0)

    # Language match
    chunk_lang = meta.get("language", "unknown")
    if query_lang != "unknown" and chunk_lang != "unknown":
        if query_lang == chunk_lang:
            score *= LANGUAGE_BOOST["match"]
        else:
            score *= LANGUAGE_BOOST["mismatch"]
    else:
        score *= LANGUAGE_BOOST["unknown"]

    return round(score, 4)


def apply_metadata_boost(
    results: list[dict],
    query: str,
) -> list[dict]:
    """
    Apply metadata boosting to a list of search results.
    Modifies the 'score' field and adds a 'boost' field.

    Args:
        results: List of result dicts from keyword or semantic search
        query:   Original query string (used for language detection)

    Returns:
        Results with updated scores, sorted by boosted score descending
    """
    query_lang = detect_language(query)

    for result in results:
        boost = compute_metadata_boost(result, query_lang)
        original_score = result.get("score", 0.0)
        result["boost"]          = boost
        result["score_original"] = round(original_score, 6)
        result["score"]          = round(original_score * boost, 6)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def filter_by_domain(
    results: list[dict],
    domain: str,
) -> list[dict]:
    """Filter results to only include chunks from a specific domain."""
    return [r for r in results if r.get("domain") == domain]


def filter_by_language(
    results: list[dict],
    language: str,
) -> list[dict]:
    """Filter results to only include chunks in a specific language."""
    return [r for r in results if r.get("language") == language]


def filter_by_content_type(
    results: list[dict],
    content_type: str,
) -> list[dict]:
    """Filter results to only include chunks of a specific content type."""
    return [r for r in results if r.get("type") == content_type]


def explain_boost(chunk: dict, query_lang: str = "unknown") -> dict:
    """
    Returns a breakdown of each boost factor for a chunk.
    Useful for debugging and evaluation.
    """
    meta = chunk.get("metadata", {})
    return {
        "domain_boost":    DOMAIN_BOOST.get(meta.get("domain", ""), 1.0),
        "content_boost":   CONTENT_TYPE_BOOST.get(meta.get("content_type", "text"), 1.0),
        "ocr_penalty":     OCR_NOISE_PENALTY.get(bool(meta.get("ocr_noise_flag", False)), 1.0),
        "priority_boost":  PRIORITY_BOOST.get(int(meta.get("retrieval_priority", 1)), 1.0),
        "language_boost":  (
            LANGUAGE_BOOST["match"]
            if query_lang == meta.get("language", "unknown") != "unknown"
            else LANGUAGE_BOOST["unknown"]
        ),
        "total":           compute_metadata_boost(chunk, query_lang),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with a dummy chunk
    test_chunk = {
        "text": "Test chunk about olive drought resistance",
        "metadata": {
            "domain": "olive_agronomy",
            "content_type": "text",
            "ocr_noise_flag": False,
            "retrieval_priority": 1,
            "language": "fr",
        }
    }
    boost = compute_metadata_boost(test_chunk, query_lang="fr")
    breakdown = explain_boost(test_chunk, query_lang="fr")
    print(f"Total boost: {boost}")
    print(f"Breakdown: {breakdown}")