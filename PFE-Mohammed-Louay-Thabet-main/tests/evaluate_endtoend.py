# tests/evaluate_endtoend.py
"""
End-to-end RAG evaluation — compares three configurations on the same 20-query test set.

  Config A  Full Local RAG   — Ollama LLM  + Qdrant + TimescaleDB
  Config B  DeepSeek + RAG   — DeepSeek API + Qdrant + TimescaleDB
  Config C  DeepSeek Only    — DeepSeek API, NO local DB (pure LLM baseline)

Why three configs?
  A vs B  → same RAG pipeline, different LLM quality (local vs cloud)
  B vs C  → same LLM (DeepSeek), with vs without retrieval — measures RAG contribution
  A vs C  → full local RAG vs pure cloud LLM — total comparison

─────────────────────────────────────────
SCORING  (0–100 per config)
─────────────────────────────────────────
Two independent scores are computed:

  Automated Score (always available):
    Retrieval   0–50 pts  context_hit_rate × 50   (0 for no-DB configs by design)
    Richness    0–30 pts  min(avg_words / 250, 1) × 30
    Reliability 0–20 pts  (successful_queries / 20) × 20

  LLM Judge Score (optional, needs DeepSeek API, add --judge flag):
    DeepSeek rates each answer 1–10 on Relevance + Accuracy + Detail
    Score = average_total_points / 30 × 100

─────────────────────────────────────────
Metrics stored per query × config:
  response_time_s   wall-clock seconds
  answer_words      word count of the answer
  context_hit       did Qdrant retrieve a relevant source? (RAG configs only)
  judge_score       1–10 per criterion if --judge used, else null
  answer            full answer text verbatim

Results written to:
  tests/results/endtoend_<YYYYMMDD_HHMM>_full.json    all detail, every query
  tests/results/endtoend_<YYYYMMDD_HHMM>_summary.csv  one row per query × config
  tests/results/endtoend_<YYYYMMDD_HHMM>_scores.json  final scores per config

─────────────────────────────────────────
Usage:
  # Ollama only — Config A, no API key needed
  python tests/evaluate_endtoend.py --configs local

  # DeepSeek configs — B + C, needs DEEPSEEK_API_KEY in .env
  python tests/evaluate_endtoend.py --configs deepseek

  # All three configs
  python tests/evaluate_endtoend.py --configs all

  # All three + LLM judge scoring (needs DeepSeek API)
  python tests/evaluate_endtoend.py --configs all --judge

  # Quick smoke test — first 3 queries only
  python tests/evaluate_endtoend.py --configs all --queries 3

  # Override Ollama model
  python tests/evaluate_endtoend.py --configs local --local-backend mistral:7b
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Test set  (same 20 queries as evaluate_rag.py)
# ──────────────────────────────────────────────────────────────────────────────

TEST_SET = [
    # Drought & climate stress
    {
        "query":         "What are the physiological effects of drought stress on olive trees?",
        "relevant_docs": ["Drought stress effects", "Technologie innovante"],
        "category":      "drought",
    },
    {
        "query":         "How does water deficit affect olive oil quality?",
        "relevant_docs": ["Drought stress effects", "water used by the olive tree"],
        "category":      "drought",
    },
    {
        "query":         "What drought resistance mechanisms do olive trees use?",
        "relevant_docs": ["Drought stress effects", "Technologie innovante"],
        "category":      "drought",
    },
    {
        "query":         "Comment les oliviers s'adaptent à la sécheresse?",
        "relevant_docs": ["Technologie innovante", "Drought stress effects"],
        "category":      "drought",
    },
    {
        "query":         "What is the impact of soil water content on olive stomatal conductance?",
        "relevant_docs": ["water used by the olive tree", "Drought stress effects"],
        "category":      "drought",
    },
    # Irrigation & water management
    {
        "query":         "What are the water requirements of olive trees in arid regions?",
        "relevant_docs": ["water used by the olive tree", "Drought stress effects"],
        "category":      "irrigation",
    },
    {
        "query":         "How should I irrigate my olive farm during summer drought?",
        "relevant_docs": ["Technologie innovante", "water used by the olive tree"],
        "category":      "irrigation",
    },
    {
        "query":         "What irrigation techniques reduce water stress in olive orchards?",
        "relevant_docs": ["water used by the olive tree", "Technologie innovante"],
        "category":      "irrigation",
    },
    # Production & yield
    {
        "query":         "What climate variables affect olive yield the most?",
        "relevant_docs": ["emperature-related", "1.Elloumi", "Drought stress effects"],
        "category":      "production",
    },
    {
        "query":         "How does warm winter affect olive flowering and fruit set?",
        "relevant_docs": ["1.Elloumi", "Drought stress effects"],
        "category":      "production",
    },
    {
        "query":         "What is the alternate bearing phenomenon in olive trees?",
        "relevant_docs": ["emperature-related", "1.Elloumi"],
        "category":      "production",
    },
    {
        "query":         "How do chilling hours affect olive production?",
        "relevant_docs": ["emperature-related", "1.Elloumi"],
        "category":      "production",
    },
    # Olive oil quality
    {
        "query":         "What factors affect the polyphenol content of olive oil?",
        "relevant_docs": ["Olive tree  leaf", "Drought stress effects"],
        "category":      "quality",
    },
    {
        "query":         "What are the health benefits of olive leaf extracts?",
        "relevant_docs": ["Olive tree  leaf"],
        "category":      "quality",
    },
    {
        "query":         "How does olive leaf composition vary by cultivar?",
        "relevant_docs": ["Olive tree  leaf"],
        "category":      "quality",
    },
    # Pests & diseases
    {
        "query":         "What are the main pests affecting olive trees during drought?",
        "relevant_docs": ["Technologie innovante", "Drought stress effects"],
        "category":      "pests",
    },
    {
        "query":         "How does Bactrocera oleae affect olive production?",
        "relevant_docs": ["Technologie innovante", "Drought stress effects"],
        "category":      "pests",
    },
    # Farming systems & sustainability
    {
        "query":         "What sustainable farming practices improve olive production?",
        "relevant_docs": ["evolution and sustainability", "Technologie innovante"],
        "category":      "sustainability",
    },
    {
        "query":         "How does pruning affect olive tree productivity?",
        "relevant_docs": ["evolution and sustainability", "water used by the olive tree"],
        "category":      "sustainability",
    },
    # Tunisia specific
    {
        "query":         "What are the challenges of olive growing in arid Tunisia?",
        "relevant_docs": ["1.Elloumi", "Technologie innovante", "Drought stress effects"],
        "category":      "tunisia",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation configurations
# ──────────────────────────────────────────────────────────────────────────────

def build_configs(local_backend: str) -> dict:
    return {
        "local_rag": {
            "label":       f"Config A — Full Local RAG ({local_backend})",
            "backend":     local_backend,
            "use_db":      True,
            "llm":         local_backend,
            "description": f"Ollama {local_backend} + Qdrant hybrid search + TimescaleDB",
        },
        "deepseek_rag": {
            "label":       "Config B — DeepSeek API + RAG",
            "backend":     "deepseek-chat",
            "use_db":      True,
            "llm":         "deepseek-chat",
            "description": "DeepSeek V3 (cloud) + Qdrant hybrid search + TimescaleDB",
        },
        "deepseek_only": {
            "label":       "Config C — DeepSeek API Only (no RAG baseline)",
            "backend":     "deepseek-chat",
            "use_db":      False,
            "llm":         "deepseek-chat",
            "description": "DeepSeek V3 (cloud), no local DB — pure LLM knowledge",
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Automated metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def context_hit(context: dict, relevant_docs: list[str]) -> bool:
    """True if the retrieved vector context contains at least one relevant source."""
    vector_ctx = context.get("vector_context", "").lower()
    if not vector_ctx or "no relevant documents" in vector_ctx:
        return False
    return any(rd.lower() in vector_ctx for rd in relevant_docs)


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


# ──────────────────────────────────────────────────────────────────────────────
# LLM Judge — DeepSeek rates each answer (optional)
# ──────────────────────────────────────────────────────────────────────────────

_judge_client = None

def _get_judge_client():
    """Lazy-load DeepSeek client for judging. Reads .env automatically."""
    global _judge_client
    if _judge_client is None:
        from dotenv import load_dotenv
        import os
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in .env — cannot use --judge")
        from openai import OpenAI
        _judge_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _judge_client


JUDGE_PROMPT = """\
You are an expert evaluator for an agricultural question-answering system about olive farming.

Rate the following ANSWER to the given QUESTION on three criteria.
Respond ONLY with a JSON object, no explanation.

QUESTION: {question}

ANSWER: {answer}

Rate each criterion from 1 (very poor) to 10 (excellent):
  relevance   — Does the answer directly address the question asked?
  accuracy    — Are the agricultural/scientific facts correct and plausible?
  completeness — Is the answer detailed and thorough enough to be useful?

Respond with exactly this JSON (integers only):
{{"relevance": <1-10>, "accuracy": <1-10>, "completeness": <1-10>}}
"""


def judge_answer(question: str, answer: str) -> dict:
    """
    Ask DeepSeek to rate one answer. Returns {"relevance":x,"accuracy":y,"completeness":z}.
    Returns None on any error so one failure does not abort the run.
    """
    if not answer or answer.startswith("ERROR:"):
        return {"relevance": 0, "accuracy": 0, "completeness": 0}
    try:
        client   = _get_judge_client()
        prompt   = JUDGE_PROMPT.format(question=question, answer=answer[:2000])
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
        scores = json.loads(raw)
        return {
            "relevance":    int(scores.get("relevance",    0)),
            "accuracy":     int(scores.get("accuracy",     0)),
            "completeness": int(scores.get("completeness", 0)),
        }
    except Exception as exc:
        log.warning(f"Judge failed for query '{question[:40]}': {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Composite score calculator
# ──────────────────────────────────────────────────────────────────────────────

def compute_automated_score(records: list[dict]) -> dict:
    """
    Automated score (0–100), no API needed.

    Components:
      Retrieval   0–50 pts  context_hit_rate × 50
                            (automatically 0 for no-DB configs — that's intentional)
      Richness    0–30 pts  min(avg_words / 250, 1.0) × 30
      Reliability 0–20 pts  (ok_queries / total) × 20
    """
    total = len(records)
    ok    = sum(1 for r in records if not r.get("error"))

    hits  = [r["context_hit"] for r in records if r["context_hit"] is not None]
    hit_rate  = sum(hits) / len(hits) if hits else 0.0  # 0 if no-DB config

    avg_words = sum(r["answer_words"] for r in records) / total if total else 0

    retrieval_pts   = hit_rate * 50
    richness_pts    = min(avg_words / 250, 1.0) * 30
    reliability_pts = (ok / total) * 20 if total else 0

    total_score = retrieval_pts + richness_pts + reliability_pts

    avg_time = sum(r["response_time_s"] for r in records) / total if total else 0

    return {
        "automated_score":    round(total_score, 1),
        "retrieval_pts":      round(retrieval_pts, 1),
        "richness_pts":       round(richness_pts, 1),
        "reliability_pts":    round(reliability_pts, 1),
        "context_hit_rate":   round(hit_rate, 3),
        "avg_words":          round(avg_words, 1),
        "avg_time_s":         round(avg_time, 1),
        "ok_queries":         ok,
        "total_queries":      total,
    }


def compute_judge_score(records: list[dict]) -> dict:
    """
    Aggregate the per-query judge scores into a 0–100 score.
    judge_score = avg(relevance + accuracy + completeness) / 30 × 100
    """
    rated = [r for r in records if r.get("judge_scores") is not None]
    if not rated:
        return {"judge_score": None, "judge_relevance": None,
                "judge_accuracy": None, "judge_completeness": None,
                "judge_n": 0}

    avg_rel  = sum(r["judge_scores"]["relevance"]    for r in rated) / len(rated)
    avg_acc  = sum(r["judge_scores"]["accuracy"]     for r in rated) / len(rated)
    avg_comp = sum(r["judge_scores"]["completeness"] for r in rated) / len(rated)
    total    = (avg_rel + avg_acc + avg_comp) / 30 * 100

    return {
        "judge_score":        round(total, 1),
        "judge_relevance":    round(avg_rel,  2),
        "judge_accuracy":     round(avg_acc,  2),
        "judge_completeness": round(avg_comp, 2),
        "judge_n":            len(rated),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-query runner
# ──────────────────────────────────────────────────────────────────────────────

def run_query(ask_fn, query: str, backend: str, use_db: bool) -> dict:
    try:
        return ask_fn(question=query, backend=backend, use_db=use_db)
    except Exception as exc:
        log.error(f"Query failed [{backend}|db={use_db}]: {exc}")
        return {
            "answer":        f"ERROR: {exc}",
            "thinking":      "",
            "backend":       backend,
            "use_db":        use_db,
            "response_time": 0.0,
            "context":       {},
            "error":         str(exc),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Single-config evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_config(
    config_key:  str,
    config:      dict,
    ask_fn,
    test_set:    list[dict],
    use_judge:   bool = False,
) -> list[dict]:
    label   = config["label"]
    backend = config["backend"]
    use_db  = config["use_db"]

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  {config['description']}")
    print(f"{'=' * 72}")
    if use_judge:
        print("  LLM judge: ON  (DeepSeek will rate each answer)")

    records = []

    for i, test in enumerate(test_set, start=1):
        query    = test["query"]
        rel_docs = test["relevant_docs"]

        print(f"  Q{i:>2}/20  {query[:58]}", end="", flush=True)

        raw   = run_query(ask_fn, query, backend, use_db)
        answer      = raw.get("answer", "")
        ctx         = raw.get("context", {})
        resp_time   = raw.get("response_time", 0.0)
        error       = raw.get("error", None)

        hit   = context_hit(ctx, rel_docs) if use_db else None
        words = count_words(answer)

        # Optional LLM judge
        judge_scores = None
        if use_judge and not error:
            judge_scores = judge_answer(query, answer)

        hit_str   = f" ctx={'✓' if hit else '✗'}" if use_db else " ctx=—"
        judge_str = ""
        if judge_scores:
            total_j = judge_scores["relevance"] + judge_scores["accuracy"] + judge_scores["completeness"]
            judge_str = f" judge={total_j}/30"

        status = "✅" if not error else "❌"
        print(f"  {status}  {resp_time:.1f}s  {words}w{hit_str}{judge_str}")

        records.append({
            # identification
            "config_key":       config_key,
            "config_label":     label,
            "llm":              config["llm"],
            "use_db":           use_db,
            "description":      config["description"],
            # query
            "query_num":        i,
            "query":            query,
            "category":         test["category"],
            "relevant_docs":    rel_docs,
            # metrics
            "response_time_s":  round(resp_time, 2),
            "answer_words":     words,
            "context_hit":      hit,
            "judge_scores":     judge_scores,
            # content — stored verbatim, nothing truncated
            "answer":           answer,
            "thinking_excerpt": raw.get("thinking", "")[:300],
            # diagnostics
            "error":            error,
        })

    # Per-config quick summary
    n_ok    = sum(1 for r in records if not r["error"])
    avg_t   = sum(r["response_time_s"] for r in records) / len(records)
    avg_w   = sum(r["answer_words"]    for r in records) / len(records)
    db_note = ""
    if use_db:
        hit_rate = sum(1 for r in records if r["context_hit"]) / len(records)
        db_note  = f" | context hit {hit_rate:.0%}"
    print(f"\n  {n_ok}/20 ok | avg {avg_t:.1f}s/q | avg {avg_w:.0f} words{db_note}")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Results storage
# ──────────────────────────────────────────────────────────────────────────────

def save_results(
    all_records:  list[dict],
    all_scores:   dict,
    run_ts:       str,
    results_dir:  Path,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON — every field, every query, every answer verbatim
    json_path = results_dir / f"endtoend_{run_ts}_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": run_ts,
                "total_records": len(all_records),
                "configs_run":   list({r["config_key"] for r in all_records}),
                "scores":        all_scores,
                "results":       all_records,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n  Full results  → {json_path}")

    # CSV summary — one row per query × config, key metrics only
    csv_rows = []
    for r in all_records:
        j = r.get("judge_scores") or {}
        csv_rows.append({
            "config":          r["config_label"],
            "llm":             r["llm"],
            "use_db":          r["use_db"],
            "query_num":       r["query_num"],
            "category":        r["category"],
            "query":           r["query"][:80],
            "response_time_s": r["response_time_s"],
            "answer_words":    r["answer_words"],
            "context_hit":     r["context_hit"],
            "judge_relevance":    j.get("relevance"),
            "judge_accuracy":     j.get("accuracy"),
            "judge_completeness": j.get("completeness"),
            "error":           r["error"] is not None,
        })
    csv_path = results_dir / f"endtoend_{run_ts}_summary.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  CSV summary   → {csv_path}")

    # Scores JSON — leaderboard snapshot
    scores_path = results_dir / f"endtoend_{run_ts}_scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({"run_timestamp": run_ts, "scores": all_scores}, f, indent=2)
    print(f"  Scores        → {scores_path}")

    return json_path, csv_path, scores_path


# ──────────────────────────────────────────────────────────────────────────────
# Final comparison + leaderboard
# ──────────────────────────────────────────────────────────────────────────────

def print_leaderboard(all_scores: dict, configs: dict):
    print(f"\n{'=' * 72}")
    print("  FINAL SCORES  (0–100 per config)")
    print(f"{'=' * 72}")
    print(f"  {'Config':<42} {'Auto':>6}  {'Judge':>6}  {'Ret':>6}  {'Rich':>6}  {'Rely':>6}")
    print(f"  {'-'*42} {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}  {'------':>6}")

    rows = []
    for key, sc in all_scores.items():
        label = configs[key]["label"]
        auto  = sc["automated"]["automated_score"]
        judge = sc["judge"].get("judge_score")
        ret   = sc["automated"]["retrieval_pts"]
        rich  = sc["automated"]["richness_pts"]
        rely  = sc["automated"]["reliability_pts"]
        judge_str = f"{judge:6.1f}" if judge is not None else f"{'N/A':>6}"
        print(f"  {label:<42} {auto:6.1f}  {judge_str}  {ret:6.1f}  {rich:6.1f}  {rely:6.1f}")
        rows.append((key, auto, judge))

    # Score breakdown legend
    print(f"\n  Score components:")
    print(f"    Auto  = Retrieval (0–50) + Richness (0–30) + Reliability (0–20)")
    print(f"    Judge = DeepSeek rates Relevance+Accuracy+Completeness (0–100)  [--judge]")
    print(f"    Ret   = context_hit_rate × 50  (0 by design for no-DB configs)")
    print(f"    Rich  = min(avg_words/250, 1) × 30")
    print(f"    Rely  = (ok_queries/20) × 20")

    # Key comparisons (only when both configs ran)
    keys_run = list(all_scores.keys())
    print(f"\n{'─' * 72}")
    print("  KEY COMPARISONS")
    print(f"{'─' * 72}")

    if "deepseek_rag" in keys_run and "deepseek_only" in keys_run:
        delta_auto  = (all_scores["deepseek_rag"]["automated"]["automated_score"]
                       - all_scores["deepseek_only"]["automated"]["automated_score"])
        delta_judge = None
        if (all_scores["deepseek_rag"]["judge"]["judge_score"] is not None and
                all_scores["deepseek_only"]["judge"]["judge_score"] is not None):
            delta_judge = (all_scores["deepseek_rag"]["judge"]["judge_score"]
                           - all_scores["deepseek_only"]["judge"]["judge_score"])
        judge_note = f" | judge Δ={delta_judge:+.1f}" if delta_judge is not None else ""
        print(f"  B vs C  RAG contribution to DeepSeek :  auto Δ={delta_auto:+.1f}{judge_note}")
        if delta_auto > 0:
            print(f"          → RAG adds +{delta_auto:.1f} pts to DeepSeek's score")
        else:
            print(f"          → RAG not helping DeepSeek on this test set ({delta_auto:+.1f} pts)")

    if "local_rag" in keys_run and "deepseek_rag" in keys_run:
        delta_auto  = (all_scores["deepseek_rag"]["automated"]["automated_score"]
                       - all_scores["local_rag"]["automated"]["automated_score"])
        delta_judge = None
        if (all_scores["deepseek_rag"]["judge"]["judge_score"] is not None and
                all_scores["local_rag"]["judge"]["judge_score"] is not None):
            delta_judge = (all_scores["deepseek_rag"]["judge"]["judge_score"]
                           - all_scores["local_rag"]["judge"]["judge_score"])
        judge_note = f" | judge Δ={delta_judge:+.1f}" if delta_judge is not None else ""
        print(f"  B vs A  Cloud LLM quality over local :  auto Δ={delta_auto:+.1f}{judge_note}")

    if "local_rag" in keys_run and "deepseek_only" in keys_run:
        delta_auto  = (all_scores["local_rag"]["automated"]["automated_score"]
                       - all_scores["deepseek_only"]["automated"]["automated_score"])
        delta_judge = None
        if (all_scores["local_rag"]["judge"]["judge_score"] is not None and
                all_scores["deepseek_only"]["judge"]["judge_score"] is not None):
            delta_judge = (all_scores["local_rag"]["judge"]["judge_score"]
                           - all_scores["deepseek_only"]["judge"]["judge_score"])
        judge_note = f" | judge Δ={delta_judge:+.1f}" if delta_judge is not None else ""
        print(f"  A vs C  Full local RAG vs cloud only :  auto Δ={delta_auto:+.1f}{judge_note}")

    # Winner
    if rows:
        print(f"\n{'─' * 72}")
        best_auto  = max(rows, key=lambda x: x[1])
        print(f"  🏆 Best automated score : {configs[best_auto[0]]['label']}  ({best_auto[1]:.1f}/100)")
        judged = [(k, j) for k, _, j in rows if j is not None]
        if judged:
            best_judge = max(judged, key=lambda x: x[1])
            print(f"  🏆 Best judge score     : {configs[best_judge[0]]['label']}  ({best_judge[1]:.1f}/100)")


def print_per_category(all_records: list[dict]):
    """Breakdown of context_hit by category and config."""
    rag_records = [r for r in all_records if r["use_db"]]
    if not rag_records:
        return

    df = pd.DataFrame(rag_records)
    pivot = df.pivot_table(
        index="category",
        columns="config_label",
        values="context_hit",
        aggfunc="mean",
    ).round(2)

    print(f"\n  Context hit rate by category (RAG configs only):")
    print(pivot.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# CLI + main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end RAG evaluation: Config A (local) vs B (DeepSeek+RAG) vs C (DeepSeek only)"
    )
    p.add_argument(
        "--configs",
        choices=["local", "deepseek", "all"],
        default="all",
        help="local=Config A | deepseek=B+C | all=A+B+C",
    )
    p.add_argument(
        "--local-backend",
        default="llama3.1:8b",
        dest="local_backend",
        help="Ollama model for Config A (default: llama3.1:8b)",
    )
    p.add_argument(
        "--judge",
        action="store_true",
        help="Rate each answer with DeepSeek (needs DEEPSEEK_API_KEY). Adds ~1 API call per query.",
    )
    p.add_argument(
        "--queries",
        type=int,
        default=None,
        metavar="N",
        help="Run only first N queries (quick smoke test)",
    )
    return p.parse_args()


def main():
    args        = parse_args()
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = Path(__file__).resolve().parent / "results"
    configs     = build_configs(args.local_backend)

    if args.configs == "local":
        selected = ["local_rag"]
    elif args.configs == "deepseek":
        selected = ["deepseek_rag", "deepseek_only"]
    else:
        selected = ["local_rag", "deepseek_rag", "deepseek_only"]

    test_set = TEST_SET if args.queries is None else TEST_SET[: args.queries]

    print("=" * 72)
    print("  Olive RAG — End-to-End Evaluation")
    print(f"  Run timestamp : {run_ts}")
    print(f"  Queries       : {len(test_set)} / {len(TEST_SET)}")
    print(f"  Configs       : {', '.join(selected)}")
    print(f"  LLM judge     : {'ON (DeepSeek)' if args.judge else 'OFF (use --judge to enable)'}")
    print("=" * 72)

    from src.agent import ask

    all_records: list[dict] = []
    run_start = time.time()

    for key in selected:
        records = evaluate_config(key, configs[key], ask, test_set, use_judge=args.judge)
        all_records.extend(records)

    total_elapsed = time.time() - run_start
    print(f"\n  Total wall time: {total_elapsed:.0f}s  ({len(all_records)} query-runs)")

    # Compute scores per config
    all_scores: dict[str, dict] = {}
    for key in selected:
        recs = [r for r in all_records if r["config_key"] == key]
        all_scores[key] = {
            "config_label": configs[key]["label"],
            "llm":          configs[key]["llm"],
            "use_db":       configs[key]["use_db"],
            "automated":    compute_automated_score(recs),
            "judge":        compute_judge_score(recs),
        }

    # Save everything
    save_results(all_records, all_scores, run_ts, results_dir)

    # Print leaderboard and comparisons
    print_leaderboard(all_scores, configs)
    print_per_category(all_records)

    print(f"\n  Done. All results in tests/results/")


if __name__ == "__main__":
    main()
