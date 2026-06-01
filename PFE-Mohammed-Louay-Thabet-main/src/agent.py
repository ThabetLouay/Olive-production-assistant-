# src/agent.py

import logging
import re
import time

import ollama
from openai import OpenAI

from src.config import (
    DEEPSEEK_API_KEY,
    RETRIEVAL_TOP_K,
)
from src.tools.hybrid_retriever import search_full_hybrid
from src.tools.sql_tool import (
    get_summary_stats,
    get_annual_production,
    get_drought_years,
)

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# LLM Backend Registry
# All supported backends — add new ones here
# ---------------------------------------------------------------

LLM_BACKENDS = {
    # Local models via Ollama
    "llama3.1:8b": {
        "type":        "ollama",
        "model":       "llama3.1:8b",
        "label":       "LLaMA 3.1 8B (local GPU)",
        "is_reasoner": False,
    },
    "mistral:7b": {
        "type":        "ollama",
        "model":       "mistral:7b",
        "label":       "Mistral 7B (local GPU)",
        "is_reasoner": False,
    },
    "deepseek-r1:7b": {
        "type":        "ollama",
        "model":       "deepseek-r1:7b",
        "label":       "DeepSeek R1 7B (local GPU)",
        "is_reasoner": True,   # strips <think> tags
    },
    "qwen2.5:7b": {
        "type":        "ollama",
        "model":       "qwen2.5:7b",
        "label":       "Qwen 2.5 7B (local GPU)",
        "is_reasoner": False,
    },

    # Cloud models via DeepSeek API
    "deepseek-chat": {
        "type":        "deepseek_api",
        "model":       "deepseek-chat",
        "label":       "DeepSeek V3 (cloud API)",
        "is_reasoner": False,
    },
    "deepseek-reasoner": {
        "type":        "deepseek_api",
        "model":       "deepseek-reasoner",
        "label":       "DeepSeek R1 (cloud API)",
        "is_reasoner": True,
    },
}

# Default backend — change this to switch globally
DEFAULT_BACKEND = "llama3.1:8b"

# ---------------------------------------------------------------
# DeepSeek API client (lazy init)
# ---------------------------------------------------------------
_deepseek_client = None

def get_deepseek_client() -> OpenAI:
    global _deepseek_client
    if _deepseek_client is None:
        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY not set in .env — cannot use cloud backend"
            )
        _deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
    return _deepseek_client


# ---------------------------------------------------------------
# DeepSeek-R1 <think> tag parser
# ---------------------------------------------------------------

def parse_reasoner_response(raw: str) -> dict:
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer   = raw[think_match.end():].strip()
    else:
        thinking = ""
        answer   = raw.strip()
    answer = re.sub(r"<[^>]+>", "", answer).strip()
    return {"thinking": thinking, "answer": answer}


# ---------------------------------------------------------------
# Context assembly — SQL + Vector (always both tracks)
# ---------------------------------------------------------------

def build_context(question: str, use_db: bool = True) -> dict:
    """
    use_db=True  → fetch from TimescaleDB + Qdrant (Config A and B)
    use_db=False → skip DB entirely, return empty context (Config C)
    """
    sql_context    = ""
    vector_context = ""

    if not use_db:
        log.info("DB access disabled — returning empty context")
        return {"sql_context": "", "vector_context": ""}

    # ── SQL track ──────────────────────────────────────────────
    try:
        stats = get_summary_stats()
        sql_context += f"Dataset overview: {stats}\n\n"

        years = re.findall(r"\b(19|20)\d{2}\b", question)
        if len(years) >= 2:
            df = get_annual_production(
                start_year=int(years[0]),
                end_year=int(years[-1]),
            )
        else:
            df = get_annual_production()

        if not df.empty:
            sql_context += "Annual production + climate data (Médenine):\n"
            sql_context += df.to_string(index=False) + "\n\n"

        droughts = get_drought_years(spi_threshold=-0.1)
        if not droughts.empty:
            sql_context += "Years with drought stress (SPI < -0.1):\n"
            sql_context += droughts.to_string(index=False) + "\n\n"

    except Exception as e:
        log.error(f"SQL error: {e}")
        sql_context = "Statistical data temporarily unavailable.\n"

    # ── Hybrid vector track ────────────────────────────────────
    try:
        results = search_full_hybrid(question, top_k=RETRIEVAL_TOP_K)
        if results:
            vector_context = "Relevant scientific knowledge:\n\n"
            for i, r in enumerate(results):
                filename = r["source"].split("/")[-1]   # strip path, keep filename
                vector_context += (
                    f"[Doc {i+1}: {filename}, page {r['page']}, "
                    f"relevance {r['score']:.4f}]\n"
                    f"{r['text']}\n\n"
                )
        else:
            vector_context = "No relevant documents found.\n"
    except Exception as e:
        log.error(f"Hybrid search error: {e}")
        vector_context = "Document search temporarily unavailable.\n"

    return {
        "sql_context":    sql_context,
        "vector_context": vector_context,
    }


# ---------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------

def build_prompt(question: str, context: dict, use_db: bool = True) -> str:
    sql_ctx    = context.get("sql_context", "")
    vector_ctx = context.get("vector_context", "")

    if use_db:
        data_section = f"""## Statistical Data — Médenine Olive Production (TimescaleDB)
{sql_ctx}

## Scientific Knowledge — Research Documents (Hybrid Search)
{vector_ctx}"""
    else:
        data_section = "## No external data provided — answer from your own knowledge."

    return f"""You are an expert agricultural assistant specialising in olive \
farming in the Médenine region of Tunisia.
Your audience includes both farmers and agricultural researchers.

INSTRUCTIONS:
- Give a detailed, structured answer (minimum 150 words).
- Explain the WHY behind every fact, not just the fact itself.
- If numerical data is available, always include and interpret it agronomically.
- Structure your answer with clear sections when appropriate.
- Be practical and actionable for farmers.
- Be quantitatively precise for researchers.
- Answer in the SAME LANGUAGE as the question (French / English / Arabic).
- Never invent numbers or facts not present in the data below.
- If data is insufficient, say so clearly.
- Do NOT add any inline citations or source labels inside the answer text.
- End your answer with a "## Sources" section listing the documents and data you used:
  format each line as: [Doc N] filename — page X  (for research documents)
  or: [Data] Médenine production & climate dataset (for statistical figures)

{data_section}

## Question
{question}

## Answer
"""


# ---------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------

def _call_ollama(model: str, prompt: str, is_reasoner: bool) -> dict:
    log.info(f"Calling Ollama ({model})...")
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_ctx":     4096,
            "num_predict": 1024,
        },
    )
    raw = response["message"]["content"]
    if is_reasoner:
        return parse_reasoner_response(raw)
    return {"thinking": "", "answer": raw.strip()}


def _call_deepseek_api(model: str, prompt: str, is_reasoner: bool) -> dict:
    log.info(f"Calling DeepSeek API ({model})...")
    client   = get_deepseek_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    raw       = response.choices[0].message.content or ""
    reasoning = getattr(
        response.choices[0].message, "reasoning_content", ""
    ) or ""
    if is_reasoner and not reasoning:
        parsed    = parse_reasoner_response(raw)
        reasoning = parsed["thinking"]
        raw       = parsed["answer"]
    return {"thinking": reasoning, "answer": raw.strip()}


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def ask(
    question: str,
    backend:  str  = DEFAULT_BACKEND,
    use_db:   bool = True,
) -> dict:
    """
    Main entry point. Switchable backend and DB access.

    Parameters
    ----------
    question : str
        The user query.
    backend : str
        Key from LLM_BACKENDS dict. Examples:
            "llama3.1:8b"       → local GPU, best quality
            "mistral:7b"        → local GPU, fast
            "deepseek-r1:7b"    → local GPU, reasoning
            "deepseek-chat"     → cloud API, best quality
            "deepseek-reasoner" → cloud API, reasoning
    use_db : bool
        True  → fetch TimescaleDB + Qdrant context (RAG mode)
        False → no DB access, pure LLM knowledge (baseline mode)

    Returns
    -------
    dict with keys:
        answer        : str   — final answer text
        thinking      : str   — chain of thought (reasoner models only)
        backend       : str   — which backend was used
        use_db        : bool  — whether DB was queried
        response_time : float — seconds taken
    """
    if backend not in LLM_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Choose from: {list(LLM_BACKENDS.keys())}"
        )

    cfg = LLM_BACKENDS[backend]
    log.info(f"Backend: {cfg['label']} | DB: {use_db} | Q: {question}")

    start   = time.time()
    context = build_context(question, use_db=use_db)
    prompt  = build_prompt(question, context, use_db=use_db)

    if cfg["type"] == "ollama":
        result = _call_ollama(cfg["model"], prompt, cfg["is_reasoner"])
    elif cfg["type"] == "deepseek_api":
        result = _call_deepseek_api(cfg["model"], prompt, cfg["is_reasoner"])
    else:
        raise ValueError(f"Unknown backend type: {cfg['type']}")

    result["backend"]       = backend
    result["use_db"]        = use_db
    result["response_time"] = round(time.time() - start, 2)
    result["context"]       = context  # for RAGAS evaluation

    log.info(f"Done in {result['response_time']}s")
    return result


# ---------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Olive RAG Agent")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=list(LLM_BACKENDS.keys()),
        help="LLM backend to use",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable DB access (pure LLM baseline)",
    )
    args = parser.parse_args()

    cfg = LLM_BACKENDS[args.backend]
    print("=== Olive RAG Agent ===")
    print(f"LLM    : {cfg['label']}")
    print(f"DB     : {'Disabled (baseline mode)' if args.no_db else 'TimescaleDB + Qdrant'}")
    print(f"Search : BM25 + Semantic + Metadata")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = ask(question, backend=args.backend, use_db=not args.no_db)

        if result.get("thinking"):
            print("\n[Chain of thought]")
            thinking = result["thinking"]
            print(thinking[:500] + ("..." if len(thinking) > 500 else ""))

        print(f"\nAnswer: [{result['backend']} | {result['response_time']}s]")
        print(result["answer"])
        print("\n" + "=" * 50 + "\n")
