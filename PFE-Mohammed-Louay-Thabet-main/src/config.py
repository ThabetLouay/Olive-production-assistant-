# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

BASE_DIR      = Path(__file__).resolve().parents[1]
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Database
DB_HOST     = os.getenv("POSTGRES_HOST",     "127.0.0.1")
DB_PORT     = os.getenv("POSTGRES_PORT",     "5433")
DB_NAME     = os.getenv("POSTGRES_DB",       "olive_db")
DB_USER     = os.getenv("POSTGRES_USER",     "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "louay")
DB_URL      = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Qdrant
QDRANT_HOST       = os.getenv("QDRANT_HOST",       "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT",   "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "olive_knowledge")

# Embedding
EMBED_MODEL      = os.getenv("EMBED_MODEL",       "BAAI/bge-m3")
import torch
EMBED_DEVICE     = os.getenv("EMBED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Ollama
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL   = os.getenv("DEEPSEEK_MODEL",   "deepseek-chat")

# RAG
RETRIEVAL_TOP_K           = int(os.getenv("RETRIEVAL_TOP_K",           "5"))  # keep consistent with MAP evaluation (K=5)
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.25"))
USE_BM25                  = _env_bool("USE_BM25", True)
RERANK_ENABLED            = _env_bool("RERANK_ENABLED", True)
RERANK_MODEL              = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE             = os.getenv("RERANK_DEVICE", EMBED_DEVICE)
RERANK_CANDIDATE_K        = int(os.getenv("RERANK_CANDIDATE_K", "15"))
RERANK_SCORE_THRESHOLD    = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.3"))
CHUNK_SIZE                = int(os.getenv("CHUNK_SIZE",                "1200"))

# PDF processor
PDF_CHUNK_SIZE    = int(os.getenv("PDF_CHUNK_SIZE",    "1200"))
PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "150"))
PDF_MIN_CHUNK     = int(os.getenv("PDF_MIN_CHUNK",     "120"))

# Web scraper
SCRAPER_URLS_FILE     = Path(os.getenv("SCRAPER_URLS_FILE", str(BASE_DIR / "data" / "scraper_urls.txt")))
SCRAPER_CHUNK_SIZE    = int(os.getenv("SCRAPER_CHUNK_SIZE",    "800"))
SCRAPER_CHUNK_OVERLAP = int(os.getenv("SCRAPER_CHUNK_OVERLAP", "100"))
