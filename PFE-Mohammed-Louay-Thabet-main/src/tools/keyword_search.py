# src/tools/keyword_search.py
"""
BM25 keyword search over the olive RAG knowledge base.
Handles French + English text with domain-specific term preservation.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from src.config import RETRIEVAL_TOP_K

log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "olive_chunks.jsonl"

# ---------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------
try:
    STOPWORDS_FR = set(stopwords.words("french"))
    STOPWORDS_EN = set(stopwords.words("english"))
except Exception:
    STOPWORDS_FR = set()
    STOPWORDS_EN = set()

STOPWORDS = STOPWORDS_FR | STOPWORDS_EN

# Domain terms never removed as stopwords
DOMAIN_TERMS = {
    "olive", "olivier", "olives", "oleiculture", "oléiculture",
    "sfax", "médenine", "medenine", "tunisie", "tunisia",
    "sécheresse", "secheresse", "drought", "irrigation",
    "bactrocera", "oleae", "production", "récolte", "harvest",
    "huile", "oil", "rendement", "yield", "précipitation",
    "rainfall", "température", "temperature", "chilling",
    "floraison", "flowering", "variété", "variety",
    "pesticide", "fertilisation", "fertilization", "pruning",
    "taille", "sol", "soil", "enracinement", "root",
}

# ---------------------------------------------------------------
# Singleton cache
# ---------------------------------------------------------------
_bm25   = None
_chunks = None


def load_chunks() -> list[dict]:
    """Load all chunks from JSONL file. Cached after first call."""
    global _chunks
    if _chunks is None:
        _chunks = []
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    _chunks.append(json.loads(line))
        log.info(f"Loaded {len(_chunks)} chunks")
    return _chunks


def build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Build BM25 index from all chunks. Cached after first call."""
    global _bm25
    chunks = load_chunks()
    if _bm25 is None:
        log.info("Building BM25 index...")
        tokenized = [tokenize(c["text"]) for c in chunks]
        _bm25 = BM25Okapi(tokenized)
        log.info(f"BM25 index ready — {len(chunks)} documents indexed")
    return _bm25, chunks


# ---------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    - Lowercase + remove punctuation
    - Remove French + English stopwords
    - Preserve olive domain-specific terms
    - Handle French accented characters
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\-àâäéèêëïîôùûüçœæ]", " ", text)

    try:
        tokens = word_tokenize(text, language="french")
    except Exception:
        tokens = text.split()

    cleaned = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t in DOMAIN_TERMS:
            cleaned.append(t)
        elif t not in STOPWORDS:
            cleaned.append(t)
    return cleaned


# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------

def keyword_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    candidate_k: int = 20,
) -> list[dict]:
    """
    BM25 keyword search.

    Args:
        query:       Natural language question
        top_k:       Number of results to return
        candidate_k: Internal candidate pool size

    Returns:
        List of result dicts sorted by BM25 score descending
    """
    bm25, chunks = build_bm25_index()
    tokens = tokenize(query)

    if not tokens:
        log.warning("BM25: query produced no tokens after preprocessing")
        return []

    scores      = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:candidate_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        chunk = chunks[int(idx)]
        results.append({
            "text":     chunk["text"],
            "score":    round(score, 6),
            "source":   chunk["metadata"].get("source_pdf", "unknown"),
            "page":     chunk["metadata"].get("page", "?"),
            "section":  chunk["metadata"].get("section", "unknown"),
            "domain":   chunk["metadata"].get("domain", "unknown"),
            "language": chunk["metadata"].get("language", "unknown"),
            "type":     chunk["metadata"].get("content_type", "text"),
            "chunk_id": chunk.get("chunk_id", int(idx)),
            "method":   "bm25",
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    final = results[:top_k]
    log.info(f"BM25 search: '{query[:50]}' → {len(final)} results")
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = keyword_search("drought resistance olive tree Tunisia", top_k=3)
    for r in results:
        print(f"Score={r['score']:.4f} | {r['source'][:40]} p{r['page']}")
        print(r["text"][:200])
        print()