# src/ingestion/load_documents.py

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "olive_chunks.jsonl"

# ---------------------------------------------------------------
# Config — override via .env
# ---------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "olive_knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")   # "cuda" on VM with GPU
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 32))
VECTOR_SIZE = 1024   # bge-m3 output dimension — fixed


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def build_embedding_text(chunk: dict) -> str:
    """
    Prefix tells bge-m3 this is a document to index (not a query).
    This is required for asymmetric retrieval to work correctly.
    Query prefix is added at retrieval time in vector_tool.py.
    """
    meta = chunk["metadata"]
    domain = meta.get("domain", "")
    topic = meta.get("topic", "")
    lang = meta.get("language", "")

    prefix = (
        f"Represent this {domain} agricultural document "
        f"about {topic} in {lang}: "
    )
    return prefix + chunk["text"]


def create_collection_if_missing(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        log.info(
            f"Collection '{COLLECTION_NAME}' already exists — skipping creation")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    log.info(f"Created collection '{COLLECTION_NAME}'")


def upsert_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[dict],
):
    log.info(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")

    texts = [build_embedding_text(c) for c in chunks]

    # Embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i: i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,   # required for cosine similarity
            show_progress_bar=False,
        )
        all_embeddings.extend(embeddings.tolist())

    # Build Qdrant points
    points = []
    for chunk, vector in zip(chunks, all_embeddings):
        meta = chunk["metadata"]
        points.append(PointStruct(
            id=chunk["chunk_id"],
            vector=vector,
            payload={
                # Retrieval filters — used in vector_tool.py
                "domain":             meta.get("domain"),
                "retrieval_priority": meta.get("retrieval_priority"),
                "language":           meta.get("language"),
                "content_type":       meta.get("content_type"),
                "ocr_noise_flag":     meta.get("ocr_noise_flag", False),

                # Provenance — shown in LLM citations
                "source_pdf":         meta.get("source_pdf"),
                "page":               meta.get("page"),
                "section":            meta.get("section"),
                "topic":              meta.get("topic"),
                "doc_type":           meta.get("doc_type"),
                "char_count":         meta.get("char_count"),

                # Full text — returned with search results
                "text": chunk["text"],
            },
        ))

    # Upsert in batches of 100
    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting"):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i: i + batch_size],
        )

    log.info(f"✅ Upserted {len(points)} points into '{COLLECTION_NAME}'")


def verify_collection(client: QdrantClient):
    info = client.get_collection(COLLECTION_NAME)
    log.info(f"Collection info: {info.points_count} points indexed")

    # Test retrieval — sanity check (query_points replaces search in qdrant-client ≥1.7)
    test_vec = [0.0] * VECTOR_SIZE
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_vec,
        limit=1,
    )
    results = response.points
    if results:
        log.info(f"Test search OK — top result: "
                 f"{results[0].payload.get('source_pdf')} "
                 f"p{results[0].payload.get('page')}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def run_pipeline():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}\n"
            f"Run the Colab notebook first and place "
            f"olive_chunks.jsonl in data/processed/"
        )

    log.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    chunks = load_chunks(CHUNKS_PATH)
    create_collection_if_missing(client)
    upsert_chunks(client, model, chunks)
    verify_collection(client)

    log.info("✅ Document ingestion pipeline complete")


if __name__ == "__main__":
    run_pipeline()
