# src/ingestion/web_scraper.py
#
# Web scraper module for the Olive RAG system.
# Scrapes agricultural URLs (FAO, ONAGRI, IOC, agridata.tn, GIFruits, custom)
# and ingests the clean text into Qdrant as searchable knowledge chunks.
#
# Usage:
#   from src.ingestion.web_scraper import run_scraper_pipeline
#   result = run_scraper_pipeline(extra_urls=["https://..."])

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBED_MODEL, EMBED_DEVICE, EMBED_BATCH_SIZE,
    SCRAPER_URLS_FILE, SCRAPER_CHUNK_SIZE, SCRAPER_CHUNK_OVERLAP,
)

log = logging.getLogger(__name__)

VECTOR_SIZE = 1024   # bge-m3 fixed output dimension

# ---------------------------------------------------------------
# Default seed URLs — one per known source
# Users can add more via data/scraper_urls.txt or the Streamlit UI
# ---------------------------------------------------------------
DEFAULT_URLS = [
    # FAO — olive/olive oil statistics
    "https://www.fao.org/faostat/en/#data/QCL",
    # International Olive Council
    "https://www.internationaloliveoil.org/what-we-do/economic-affairs-promotion-unit/world-olive-oil-figures/",
    # ONAGRI — Tunisian agricultural statistics
    "http://www.onagri.nat.tn",
    # GIFruits — Tunisian fruit interprofessional group
    "http://www.gifruit.nat.tn",
    # agridata.tn — open data portal Tunisia
    "https://agridata.tn",
]

# HTTP request headers to mimic a browser (reduces blocks on some sites)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; OliveRAG/1.0; "
        "+https://github.com/olive-rag)"
    ),
    "Accept-Language": "fr-TN,fr;q=0.9,ar;q=0.8,en;q=0.7",
}


# ---------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------

def _fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    """Downloads HTML from a URL. Returns None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return None


def _extract_text(html: str) -> str:
    """
    Extracts clean readable text from HTML using BeautifulSoup.
    Removes scripts, styles, nav bars, and footers.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Prefer main content containers if they exist
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id="content") or
        soup.find(class_="content") or
        soup.body
    )

    text = (main or soup).get_text(separator="\n")

    # Collapse whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _detect_language(text: str) -> str:
    """
    Heuristic language detection based on character frequency.
    Returns 'ar', 'fr', or 'en'.
    """
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars  = max(len(text), 1)

    if arabic_chars / total_chars > 0.2:
        return "ar"

    french_markers = re.findall(
        r'\b(le|la|les|de|du|des|et|en|un|une|pour|dans|avec|sur|par|est)\b',
        text.lower()
    )
    if len(french_markers) > 10:
        return "fr"

    return "en"


# ---------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------

def _chunk_text(
    text: str,
    chunk_size: int = SCRAPER_CHUNK_SIZE,
    overlap: int = SCRAPER_CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits text into overlapping chunks by sentence boundaries.
    Tries to respect sentence endings ('. ', '! ', '? ') before hard-splitting.
    """
    # Split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            # Start new chunk with overlap from end of previous
            if overlap > 0 and current:
                words = current.split()
                overlap_words = words[-max(1, overlap // 6):]
                current = " ".join(overlap_words) + " " + sentence
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    # Filter out very short chunks (likely navigation remnants)
    return [c for c in chunks if len(c) >= 80]


def _make_chunk_id(url: str, chunk_index: int) -> int:
    """Generates a stable integer ID for a chunk from its URL + index."""
    raw = f"{url}::chunk::{chunk_index}"
    return int(hashlib.md5(raw.encode()).hexdigest(), 16) % (2**63)


# ---------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------

def _ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info(f"Created Qdrant collection '{QDRANT_COLLECTION}'")


def _upsert_points(
    client: QdrantClient,
    model: SentenceTransformer,
    points_data: list[dict],
):
    """Embeds and upserts a list of chunk dicts into Qdrant."""
    texts = [
        f"Represent this agricultural document about olive farming: {d['text']}"
        for d in points_data
    ]

    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        embeddings = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())

    points = []
    for d, vector in zip(points_data, all_embeddings):
        points.append(PointStruct(
            id=d["id"],
            vector=vector,
            payload={
                "text":               d["text"],
                "source_pdf":         d["url"],
                "domain":             "olive_agronomy",
                "content_type":       "web",
                "language":           d["language"],
                "topic":              "olive_farming",
                "doc_type":           "web_article",
                "retrieval_priority": "secondary",
                "ocr_noise_flag":     False,
                "page":               d["chunk_index"],
                "section":            d.get("title", ""),
                "char_count":         len(d["text"]),
            },
        ))

    for i in range(0, len(points), 100):
        client.upsert(collection_name=QDRANT_COLLECTION, points=points[i: i + 100])

    log.info(f"Upserted {len(points)} web chunks into Qdrant")


# ---------------------------------------------------------------
# Load URL list from file
# ---------------------------------------------------------------

def load_urls_from_file(path: Path = SCRAPER_URLS_FILE) -> list[str]:
    """
    Reads URLs from data/scraper_urls.txt (one per line).
    Lines starting with '#' are treated as comments and ignored.
    """
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


# ---------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------

def run_scraper_pipeline(
    extra_urls: Optional[list[str]] = None,
    include_defaults: bool = True,
    delay_seconds: float = 1.5,
) -> dict:
    """
    Main pipeline:
      1. Collects URLs (defaults + file + extra_urls passed in)
      2. Scrapes and extracts clean text from each
      3. Chunks text
      4. Embeds and upserts into Qdrant

    Args:
        extra_urls:        List of URLs added on-demand from Streamlit UI
        include_defaults:  Whether to scrape DEFAULT_URLS
        delay_seconds:     Polite delay between requests

    Returns:
        dict with 'scraped', 'failed', 'chunks_added'
    """
    # Build final URL list (deduplicated)
    all_urls: list[str] = []
    if include_defaults:
        all_urls.extend(DEFAULT_URLS)
    all_urls.extend(load_urls_from_file())
    if extra_urls:
        all_urls.extend(extra_urls)

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    log.info(f"Web scraper: {len(unique_urls)} URLs to process")

    model  = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _ensure_collection(client)

    scraped     = 0
    failed      = 0
    chunks_added = 0
    all_points  = []

    for url in unique_urls:
        log.info(f"Scraping: {url}")
        html = _fetch_html(url)

        if not html:
            failed += 1
            continue

        text = _extract_text(html)
        if len(text) < 200:
            log.warning(f"Too little text extracted from {url} ({len(text)} chars), skipping")
            failed += 1
            continue

        lang   = _detect_language(text)
        chunks = _chunk_text(text)
        domain = urlparse(url).netloc

        for idx, chunk in enumerate(chunks):
            all_points.append({
                "id":          _make_chunk_id(url, idx),
                "text":        chunk,
                "url":         url,
                "language":    lang,
                "chunk_index": idx,
                "title":       domain,
            })

        scraped      += 1
        chunks_added += len(chunks)
        log.info(f"  -> {len(chunks)} chunks from {domain}")

        time.sleep(delay_seconds)  # be polite to servers

    # Upsert all collected points in one batch call
    if all_points:
        _upsert_points(client, model, all_points)

    result = {
        "scraped":      scraped,
        "failed":       failed,
        "chunks_added": chunks_added,
        "total_urls":   len(unique_urls),
    }
    log.info(f"Web scraper complete: {result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_scraper_pipeline()
    print(f"\nResult: {result}")
