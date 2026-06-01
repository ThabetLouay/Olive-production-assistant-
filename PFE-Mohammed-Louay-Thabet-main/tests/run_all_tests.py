# tests/run_all_tests.py
"""
Master test runner — runs both evaluation suites and prints one unified report.

  Part 1  Retrieval quality  (evaluate_rag.py logic)
          Measures MAP, P@1, P@3, P@5, Recall@5 for BM25 / Semantic / Hybrid

  Part 2  End-to-end quality  (evaluate_endtoend.py logic)
          Measures automated score + optional LLM judge score for:
            Config A  Full Local RAG   (Ollama + Qdrant + TimescaleDB)
            Config B  DeepSeek + RAG   (DeepSeek API + Qdrant + TimescaleDB)
            Config C  DeepSeek Only    (DeepSeek API, no local DB)

Usage:
  # Retrieval + Config A only (no API key needed)
  python tests/run_all_tests.py --configs local

  # Retrieval + Configs B + C (needs DEEPSEEK_API_KEY)
  python tests/run_all_tests.py --configs deepseek

  # Everything — all 3 configs + LLM judge scoring
  python tests/run_all_tests.py --configs all --judge

  # Quick smoke test (3 queries only, all configs)
  python tests/run_all_tests.py --configs all --judge --queries 3

  # Change Ollama model
  python tests/run_all_tests.py --configs all --local-backend mistral:7b

Results saved to:
  tests/results/full_report_<YYYYMMDD_HHMM>.json   everything in one file
  tests/results/full_report_<YYYYMMDD_HHMM>.csv    all rows (retrieval + e2e)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── imports from our two eval modules ────────────────────────────────────────
from tests.evaluate_rag import (
    TEST_SET       as RETRIEVAL_TEST_SET,
    evaluate_method,
    search_bm25_only,
    search_semantic_only,
    search_full_hybrid,
)
from tests.evaluate_endtoend import (
    TEST_SET       as E2E_TEST_SET,
    build_configs,
    evaluate_config,
    compute_automated_score,
    compute_judge_score,
    save_results,
)


# ──────────────────────────────────────────────────────────────────────────────
# Part 1 — Retrieval evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_retrieval_eval() -> list[dict]:
    """Run BM25 / Semantic / Hybrid retrieval evaluation. Returns list of metric dicts."""
    print(f"\n{'=' * 72}")
    print("  PART 1 — RETRIEVAL QUALITY  (MAP evaluation, K=5)")
    print(f"{'=' * 72}")

    methods = [
        ("BM25 only",                    search_bm25_only),
        ("Semantic only",                search_semantic_only),
        ("Hybrid (BM25+Semantic+Boost)", search_full_hybrid),
    ]

    results = []
    for name, fn in methods:
        r = evaluate_method(name, fn, RETRIEVAL_TEST_SET, k=5)
        results.append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Part 2 — End-to-end evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_e2e_eval(
    selected:      list[str],
    local_backend: str,
    use_judge:     bool,
    test_set:      list[dict],
) -> tuple[list[dict], dict]:
    """Run end-to-end configs. Returns (all_records, all_scores)."""
    print(f"\n{'=' * 72}")
    print("  PART 2 — END-TO-END QUALITY  (agent.ask(), 20 queries)")
    print(f"{'=' * 72}")

    from src.agent import ask

    configs     = build_configs(local_backend)
    all_records = []

    for key in selected:
        records = evaluate_config(key, configs[key], ask, test_set, use_judge=use_judge)
        all_records.extend(records)

    all_scores = {}
    for key in selected:
        recs = [r for r in all_records if r["config_key"] == key]
        all_scores[key] = {
            "config_label": configs[key]["label"],
            "llm":          configs[key]["llm"],
            "use_db":       configs[key]["use_db"],
            "automated":    compute_automated_score(recs),
            "judge":        compute_judge_score(recs),
        }

    return all_records, all_scores, configs


# ──────────────────────────────────────────────────────────────────────────────
# Unified report printer
# ──────────────────────────────────────────────────────────────────────────────

def print_unified_report(
    retrieval_results: list[dict],
    all_scores:        dict,
    configs:           dict,
    total_time_s:      float,
    run_ts:            str,
):
    W = 72
    print(f"\n{'=' * W}")
    print("  OLIVE RAG — COMPLETE EVALUATION REPORT")
    print(f"  Run : {run_ts}   |   Total time : {total_time_s:.0f}s")
    print(f"{'=' * W}")

    # ── Retrieval table ──────────────────────────────────────────────────────
    if retrieval_results:
        print(f"\n  RETRIEVAL QUALITY  (BM25 / Semantic / Hybrid, K=5, 20 queries)")
        print(f"  {'─' * 68}")
        header = f"  {'Method':<34}  {'MAP':>6}  {'P@1':>6}  {'P@3':>6}  {'P@5':>6}  {'R@5':>6}"
        print(header)
        print(f"  {'─' * 68}")
        for r in retrieval_results:
            flag = " ◀ best" if r["MAP"] == max(x["MAP"] for x in retrieval_results) else ""
            print(
                f"  {r['method']:<34}  "
                f"{r['MAP']:>6.4f}  "
                f"{r['P@1']:>6.4f}  "
                f"{r['P@3']:>6.4f}  "
                f"{r['P@5']:>6.4f}  "
                f"{r['Recall@5']:>6.4f}"
                f"{flag}"
            )

    # ── End-to-end table ─────────────────────────────────────────────────────
    if all_scores:
        print(f"\n  END-TO-END QUALITY  (full agent pipeline, 20 queries)")
        print(f"  {'─' * 68}")
        print(
            f"  {'Config':<36}  {'DB':>4}  "
            f"{'Auto/100':>8}  {'Judge/100':>9}  "
            f"{'AvgTime':>7}  {'AvgWords':>8}  {'CtxHit':>6}"
        )

        print(f"  {'─' * 68}")

        best_auto  = max(all_scores.values(), key=lambda x: x["automated"]["automated_score"])
        judge_vals = [x["judge"]["judge_score"] for x in all_scores.values() if x["judge"]["judge_score"] is not None]
        best_judge_score = max(judge_vals) if judge_vals else None

        for key, sc in all_scores.items():
            auto   = sc["automated"]["automated_score"]
            judge  = sc["judge"]["judge_score"]
            db_str = "Yes" if sc["use_db"] else "No"
            label  = configs[key]["label"]

            # Pull timing and word counts from automated breakdown
            avg_time  = sc["automated"].get("avg_time_s", 0)
            avg_words = sc["automated"].get("avg_words",  0)
            hit_rate  = sc["automated"]["context_hit_rate"]

            judge_str  = f"{judge:9.1f}" if judge is not None else f"{'N/A':>9}"
            hit_str    = f"{hit_rate:.0%}" if sc["use_db"] else "  —"
            auto_flag  = " ◀" if auto  == best_auto["automated"]["automated_score"] else ""
            judge_flag = " ◀" if (judge is not None and judge == best_judge_score) else ""

            print(
                f"  {label:<36}  {db_str:>4}  "
                f"{auto:>8.1f}  {judge_str}  "
                f"{avg_time:>6.1f}s  {avg_words:>8.0f}  {hit_str:>6}"
                f"{auto_flag}{judge_flag}"
            )

        # Score breakdown reminder
        print(f"\n  Score components:")
        print(f"    Auto  = Retrieval(0-50) + Richness(0-30) + Reliability(0-20)")
        print(f"    Judge = DeepSeek rates Relevance+Accuracy+Completeness (needs --judge)")
        print(f"    CtxHit = % queries where Qdrant retrieved a relevant source")

    # ── Key comparisons ──────────────────────────────────────────────────────
    if all_scores and len(all_scores) > 1:
        print(f"\n  KEY COMPARISONS")
        print(f"  {'─' * 68}")

        def delta_str(key_a, key_b, label):
            if key_a not in all_scores or key_b not in all_scores:
                return
            da = all_scores[key_a]["automated"]["automated_score"]
            db_ = all_scores[key_b]["automated"]["automated_score"]
            d_auto = da - db_
            ja = all_scores[key_a]["judge"]["judge_score"]
            jb = all_scores[key_b]["judge"]["judge_score"]
            d_judge_str = f"  judge Δ={ja - jb:+.1f}" if (ja and jb) else ""
            direction = "▲ better" if d_auto > 0 else "▼ worse"
            print(f"  {label:<40}  auto Δ={d_auto:+.1f}  {direction}{d_judge_str}")

        delta_str("deepseek_rag",  "deepseek_only", "B vs C  RAG value for DeepSeek")
        delta_str("deepseek_rag",  "local_rag",     "B vs A  Cloud vs local LLM")
        delta_str("local_rag",     "deepseek_only", "A vs C  Local RAG vs cloud only")

    # ── Winner ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    if retrieval_results:
        best_ret = max(retrieval_results, key=lambda x: x["MAP"])
        print(f"  Best retrieval  : {best_ret['method']}  (MAP={best_ret['MAP']:.4f})")
    if all_scores:
        best_e2e_key = max(all_scores, key=lambda k: all_scores[k]["automated"]["automated_score"])
        best_e2e     = all_scores[best_e2e_key]
        j_str = f"  judge={best_e2e['judge']['judge_score']:.1f}" if best_e2e["judge"]["judge_score"] else ""
        print(f"  Best end-to-end : {best_e2e['config_label']}  (auto={best_e2e['automated']['automated_score']:.1f}{j_str})")
    print(f"{'=' * W}")


# ──────────────────────────────────────────────────────────────────────────────
# Save combined results
# ──────────────────────────────────────────────────────────────────────────────

def save_combined(
    retrieval_results: list[dict],
    e2e_records:       list[dict],
    all_scores:        dict,
    run_ts:            str,
    results_dir:       Path,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Combined JSON ────────────────────────────────────────────────────────
    json_path = results_dir / f"full_report_{run_ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp":   run_ts,
                "retrieval":       retrieval_results,
                "endtoend_scores": all_scores,
                "endtoend_detail": e2e_records,
            },
            f, ensure_ascii=False, indent=2,
        )

    # ── Combined CSV — two sections in one file ──────────────────────────────
    rows = []

    for r in retrieval_results:
        rows.append({
            "section":    "retrieval",
            "name":       r["method"],
            "llm":        "—",
            "use_db":     "—",
            "MAP":        r["MAP"],
            "P@1":        r["P@1"],
            "P@3":        r["P@3"],
            "P@5":        r["P@5"],
            "Recall@5":   r["Recall@5"],
            "auto_score": "—",
            "judge_score":"—",
            "ctx_hit_pct":"—",
            "avg_words":  "—",
        })

    for key, sc in all_scores.items():
        rows.append({
            "section":    "endtoend",
            "name":       sc["config_label"],
            "llm":        sc["llm"],
            "use_db":     sc["use_db"],
            "MAP":        "—",
            "P@1":        "—",
            "P@3":        "—",
            "P@5":        "—",
            "Recall@5":   "—",
            "auto_score": sc["automated"]["automated_score"],
            "judge_score":sc["judge"]["judge_score"] if sc["judge"]["judge_score"] else "N/A",
            "ctx_hit_pct":f"{sc['automated']['context_hit_rate']:.0%}" if sc["use_db"] else "—",
            "avg_words":  sc["automated"]["avg_words"],
        })

    csv_path = results_dir / f"full_report_{run_ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n  Saved → {json_path}")
    print(f"  Saved → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run all Olive RAG tests and print one unified report"
    )
    p.add_argument(
        "--configs",
        choices=["local", "deepseek", "all"],
        default="all",
        help="Which end-to-end configs to run  (local=A | deepseek=B+C | all=A+B+C)",
    )
    p.add_argument(
        "--local-backend",
        default="llama3.1:8b",
        dest="local_backend",
        help="Ollama model for Config A  (default: llama3.1:8b)",
    )
    p.add_argument(
        "--judge",
        action="store_true",
        help="Rate each answer with DeepSeek judge (needs DEEPSEEK_API_KEY)",
    )
    p.add_argument(
        "--queries",
        type=int,
        default=None,
        metavar="N",
        help="Run only first N queries — quick smoke test",
    )
    p.add_argument(
        "--skip-retrieval",
        action="store_true",
        dest="skip_retrieval",
        help="Skip Part 1 (retrieval MAP eval) — faster if you only want e2e results",
    )
    return p.parse_args()


def main():
    args        = parse_args()
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = Path(__file__).resolve().parent / "results"

    if args.configs == "local":
        selected = ["local_rag"]
    elif args.configs == "deepseek":
        selected = ["deepseek_rag", "deepseek_only"]
    else:
        selected = ["local_rag", "deepseek_rag", "deepseek_only"]

    test_set = E2E_TEST_SET if args.queries is None else E2E_TEST_SET[: args.queries]

    print("=" * 72)
    print("  Olive RAG — Full Test Suite")
    print(f"  Run        : {run_ts}")
    print(f"  Queries    : {len(test_set)} / 20")
    print(f"  E2E configs: {', '.join(selected)}")
    print(f"  LLM judge  : {'ON' if args.judge else 'OFF (add --judge to enable)'}")
    print("=" * 72)

    wall_start = time.time()

    # Part 1 — Retrieval
    retrieval_results = []
    if not args.skip_retrieval:
        retrieval_results = run_retrieval_eval()

    # Part 2 — End-to-end
    e2e_records, all_scores, configs = run_e2e_eval(
        selected, args.local_backend, args.judge, test_set
    )

    total_time = time.time() - wall_start

    # Save combined files
    save_combined(retrieval_results, e2e_records, all_scores, run_ts, results_dir)

    # Print unified table
    print_unified_report(retrieval_results, all_scores, configs, total_time, run_ts)


if __name__ == "__main__":
    main()
