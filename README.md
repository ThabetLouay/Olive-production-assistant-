# 🫒 Olive RAG System
### Framework for a Common Agricultural Data Space — Tunisian Olive Sector

> End-of-study project — Military Academy Fondouk Jedid | Génie Informatique | 2025–2026
> **Author:** Mohamed Louay Thabet
> **Supervisors:** LT Col Aymen Yahyaoui · Col Faicel Yakoubi
> **Domain expert:** Dr. Olfa Elloumi, Olive Tree Institute of Sfax

---

## Overview

The Olive RAG System is an open-source framework for a **common agricultural data space** tailored to the Tunisian olive sector. It combines a **polyglot persistence layer**, a **six-stage hybrid retrieval pipeline**, and a **multi-backend LLM architecture** to deliver grounded agronomic decision support in **Arabic, French, and English** — including Tunisian dialect.

The system is designed around three non-negotiable principles:

- **Data sovereignty** — ODRL-based usage policies enforced at every access point
- **Openness** — exclusively open-source stack, deployable on a university server or modest cloud instance
- **Accessibility** — Arabic-first support including Tunisian Darija, with cross-lingual retrieval

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 5 — Interfaces                  │
│   Olive Production Assistant (chatbot)  ·  AdvisoryAI   │
│              Streamlit · streamlit-authenticator         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Layer 4 — Generative AI Engine              │
│  src/agent.py · src/tools/hybrid_retriever.py           │
│  Multi-backend LLM (Ollama + DeepSeek API)              │
│  6-stage hybrid retrieval · bge-m3 · bge-reranker-v2-m3 │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│           Layer 3 — Common Data Space Core               │
│  TimescaleDB · Qdrant · PostgreSQL · MinIO · Redis       │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│          Layer 2 — Ingestion & Validation Gateway        │
│  FastAPI · Validation Agent · ODRL policies · EDC        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                Layer 1 — Data Sources                    │
│   PDF research documents · CSV/Excel · Simulated IoT    │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Tool | Role |
|---|---|---|
| API framework | FastAPI | REST endpoints, JWT auth, OpenAPI docs |
| Time-series DB | TimescaleDB (port 5433) | Climate and production data (428 rows, 1990–2025) |
| Vector store | Qdrant (port 6333) | chunks, collection: `olive_knowledge` |
| Relational DB | PostgreSQL + PostGIS | Structured farm data, geospatial queries |
| Object storage | MinIO | PDFs, satellite imagery, lab files |
| Cache | Redis | Session storage, API caching |
| LLM inference | Ollama (port 11434) | Local model serving |
| Embedding model | BAAI/bge-m3 | 1024-dim multilingual embeddings |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder reranking |
| Interfaces | Streamlit (port 8501) | Farmer chatbot + agronomist dashboard |
| Containerization | Docker + Docker Compose | Full stack orchestration |
| Cloud | Azure VM — Ubuntu 22.04, Tesla T4 16GB | Deployment environment |
| Conda env | `pfe-louay` | Python environment |

---

## LLM Backends

| Backend key | Label | Type |
|---|---|---|
| `llama3.1:8b` | LLaMA 3.1 8B (default) | Local — Ollama |
| `mistral:7b` | Mistral 7B | Local — Ollama |
| `deepseek-r1:7b` | DeepSeek R1 7B | Local — Ollama (reasoner) |
| `qwen2.5:7b` | Qwen 2.5 7B | Local — Ollama |
| `deepseek-chat` | DeepSeek V3 | Cloud — DeepSeek API |
| `deepseek-reasoner` | DeepSeek R1 | Cloud — DeepSeek API (reasoner) |

Default: `llama3.1:8b`

---

## Hybrid Retrieval Pipeline

Every query runs through six stages:

```
Query
  │
  ├─► Stage 1: bge-m3 semantic search (Qdrant HNSW)   weight = 0.8
  ├─► Stage 2: BM25 keyword search (rank_bm25)          weight = 0.2
  │
  ▼
Stage 3: Weighted RRF fusion
         Score(d) = 0.8·1/(60+r_sem) + 0.2·1/(60+r_bm25)
  │
  ▼
Stage 4: Metadata enrichment from chunk payloads
  │
  ▼
Stage 5: Cross-encoder reranking — bge-reranker-v2-m3
         top-15 candidates → min_score=0.3 → top-5
  │
  ▼
Stage 6: Metadata boosting
         olive_agronomy ×1.30 · lang match ×1.15
         table ×1.20 · ocr_noise ×0.85
  │
  ▼
Top-5 ranked chunks with source and page
```

---



## Quick Start

### Prerequisites

- Docker + Docker Compose
- Conda
- NVIDIA GPU with CUDA (Tesla T4 recommended) or CPU fallback
- Ollama installed and running

### 1. Clone and configure

```bash
git clone https://github.com/ThabetLouay/olive-rag-system.git
cd olive-rag-system
cp .env.example .env
# Edit .env with your credentials
```

### 2. Start services

```bash
docker-compose up -d
```

This starts: Qdrant (:6333), TimescaleDB (:5433), MinIO, Redis, Streamlit (:8501)

### 3. Set up the Conda environment

```bash
conda activate pfe-louay
pip install -r requirements.txt
```

### 4. Pull LLM models via Ollama

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull deepseek-r1:7b
ollama pull qwen2.5:7b
```

### 5. Run the agent (CLI)

```bash
python -m src.agent --backend llama3.1:8b
```

```bash
# Disable RAG (baseline mode)
python -m src.agent --backend deepseek-chat --no-db
```

### 6. Launch the Streamlit interface

```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## Environment Variables

```env
# Database
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433
POSTGRES_DB=olive_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=olive_knowledge

# Embedding
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=cuda

# Ollama
OLLAMA_HOST=http://localhost:11434

# DeepSeek API (optional — cloud backends only)
DEEPSEEK_API_KEY=your_api_key

# RAG
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.25
USE_BM25=true
RERANK_ENABLED=true
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATE_K=15
RERANK_SCORE_THRESHOLD=0.3
CHUNK_SIZE=1200
PDF_CHUNK_OVERLAP=150
```

---








## Standards Compliance

| Standard | Compliance |
|---|---|
| W3C DCAT v2 | Full |
| W3C ODRL 2.2 | Full |
| OAuth 2.0 / JWT | Full |
| IDSA RAM v4 | Partial (lightweight EDC adaptation) |
| ISO 19115 | Partial (core elements) |
| AGROVOC (FAO) | Extended (olive-specific terms) |
| GAIA-X Trust Framework | Conceptual alignment |

---

## Deployment

The full system runs on a single Azure VM:

- **OS:** Ubuntu 22.04 LTS
- **CPU:** 4 vCPUs
- **RAM:** 27 GB
- **GPU:** NVIDIA Tesla T4 — 16 GB VRAM


---

## Expert Evaluation

The system was evaluated in cooperation with **Dr. Olfa Elloumi**, Researcher at the **Olive Tree Institute of Sfax**, who reviewed and annotated the system's agronomic responses against domain ground truth.

---

## License

This project is open-source and was developed as an end-of-study project at the Military Academy Fondouk Jedid. All components use open-source licenses. See individual package licenses for details.

---

## Citation

```
Thabet Mohamed Louay. (2026). Framework for a Common Agricultural Data Space
Using Generative AI — Application to the Tunisian Olive Sector.
End of study project, Military Academy Fondouk Jedid, Génie Informatique.
Supervisors: LT Col A. Yahyaoui, Col F. Yakoubi,Dr. Olfa Elloumi.
```
