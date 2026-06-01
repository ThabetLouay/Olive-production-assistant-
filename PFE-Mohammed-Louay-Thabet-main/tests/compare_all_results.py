# tests/compare_all_results.py
"""
Reads every result file in tests/results/ and merges them into one
final comparison table.

Handles:
  full_report_*.json   — from run_all_tests.py
  endtoend_*_full.json — from evaluate_endtoend.py (older format)

Usage:
  python tests/compare_all_results.py
  python tests/compare_all_results.py --save          # also saves a merged CSV
  python tests/compare_all_results.py --best-only     # only show best run per config
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_full_report(path: Path) -> dict:
    """Load a full_report_*.json file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_endtoend_report(path: Path) -> dict:
    """Load an endtoend_*_full.json file and normalise to full_report shape."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Re-compute scores per config from stored records
    records = data.get("results", [])
    scores  = {}
    for key in set(r["config_key"] for r in records):
        recs = [r for r in records if r["config_key"] == key]
        hits     = [r["context_hit"] for r in recs if r["context_hit"] is not None]
        hit_rate = sum(hits) / len(hits) if hits else 0.0
        avg_words = sum(r["answer_words"] for r in recs) / len(recs)
        avg_time  = sum(r["response_time_s"] for r in recs) / len(recs)
        ok        = sum(1 for r in recs if not r.get("error"))

        retrieval_pts   = hit_rate * 50
        richness_pts    = min(avg_words / 250, 1.0) * 30
        reliability_pts = (ok / len(recs)) * 20
        auto_score      = retrieval_pts + richness_pts + reliability_pts

        rated = [r for r in recs if r.get("judge_scores") is not None]
        if rated:
            avg_rel  = sum(r["judge_scores"]["relevance"]    for r in rated) / len(rated)
            avg_acc  = sum(r["judge_scores"]["accuracy"]     for r in rated) / len(rated)
            avg_comp = sum(r["judge_scores"]["completeness"] for r in rated) / len(rated)
            judge_score = (avg_rel + avg_acc + avg_comp) / 30 * 100
        else:
            judge_score = None

        label = recs[0]["config_label"]
        scores[key] = {
            "config_label": label,
            "llm":          recs[0]["llm"],
            "use_db":       recs[0]["use_db"],
            "automated": {
                "automated_score":  round(auto_score, 1),
                "context_hit_rate": round(hit_rate, 3),
                "avg_words":        round(avg_words, 1),
                "avg_time_s":       round(avg_time, 1),
                "ok_queries":       ok,
                "total_queries":    len(recs),
                "retrieval_pts":    round(retrieval_pts, 1),
                "richness_pts":     round(richness_pts, 1),
                "reliability_pts":  round(reliability_pts, 1),
            },
            "judge": {"judge_score": round(judge_score, 1) if judge_score else None},
        }

    return {
        "run_timestamp": data.get("run_timestamp", path.stem),
        "retrieval":     [],
        "endtoend_scores": scores,
        "endtoend_detail": records,
    }


def collect_all_reports() -> list[dict]:
    """Read and return all result files, sorted by timestamp ascending."""
    reports = []

    for p in sorted(RESULTS_DIR.glob("full_report_*.json")):
        try:
            r = load_full_report(p)
            r["_source_file"] = p.name
            reports.append(r)
        except Exception as e:
            print(f"  Warning: could not read {p.name}: {e}")

    for p in sorted(RESULTS_DIR.glob("endtoend_*_full.json")):
        try:
            r = load_endtoend_report(p)
            r["_source_file"] = p.name
            reports.append(r)
        except Exception as e:
            print(f"  Warning: could not read {p.name}: {e}")

    reports.sort(key=lambda x: x.get("run_timestamp", ""))
    return reports


# ──────────────────────────────────────────────────────────────────────────────
# Build comparison tables
# ──────────────────────────────────────────────────────────────────────────────

def build_retrieval_table(reports: list[dict]) -> pd.DataFrame:
    """
    Retrieval metrics are deterministic — take the first run that has them.
    Returns a DataFrame with columns: method, MAP, P@1, P@3, P@5, Recall@5
    """
    for r in reports:
        rows = r.get("retrieval", [])
        if rows:
            return pd.DataFrame(rows)[["method", "MAP", "P@1", "P@3", "P@5", "Recall@5"]]
    return pd.DataFrame()


def build_e2e_table(reports: list[dict], best_only: bool) -> pd.DataFrame:
    """
    For each config_key, collect results from every run.
    If best_only=True: keep the row with the highest auto_score per config.
    """
    rows = []
    for report in reports:
        ts    = report.get("run_timestamp", "?")
        src   = report.get("_source_file", "?")
        n_q   = len(set(
            r["query_num"]
            for r in report.get("endtoend_detail", [])
        ))

        for key, sc in report.get("endtoend_scores", {}).items():
            auto  = sc["automated"]
            judge = sc["judge"]
            rows.append({
                "run":          ts,
                "source_file":  src,
                "n_queries":    n_q or "?",
                "config":       sc["config_label"],
                "config_key":   key,
                "llm":          sc["llm"],
                "use_db":       sc["use_db"],
                "auto_score":   auto["automated_score"],
                "ret_pts":      auto["retrieval_pts"],
                "rich_pts":     auto["richness_pts"],
                "rely_pts":     auto["reliability_pts"],
                "ctx_hit_%":    round(auto["context_hit_rate"] * 100, 0) if sc["use_db"] else None,
                "avg_words":    auto["avg_words"],
                "avg_time_s":   auto.get("avg_time_s", None),
                "ok_queries":   auto["ok_queries"],
                "judge_score":  judge["judge_score"],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if best_only:
        df = (
            df.sort_values("auto_score", ascending=False)
              .drop_duplicates(subset="config_key", keep="first")
              .reset_index(drop=True)
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Print tables
# ──────────────────────────────────────────────────────────────────────────────

W = 78

def print_retrieval(df: pd.DataFrame):
    if df.empty:
        print("  No retrieval results found.")
        return
    print(f"\n  RETRIEVAL QUALITY  (MAP evaluation, K=5)")
    print(f"  {'─' * (W-2)}")
    print(f"  {'Method':<36}  {'MAP':>7}  {'P@1':>6}  {'P@3':>6}  {'P@5':>6}  {'R@5':>6}")
    print(f"  {'─' * (W-2)}")
    best_map = df["MAP"].max()
    for _, row in df.iterrows():
        flag = "  ◀ best" if row["MAP"] == best_map else ""
        print(
            f"  {row['method']:<36}  "
            f"{row['MAP']:>7.4f}  "
            f"{row['P@1']:>6.4f}  "
            f"{row['P@3']:>6.4f}  "
            f"{row['P@5']:>6.4f}  "
            f"{row['Recall@5']:>6.4f}"
            f"{flag}"
        )


def print_e2e_all_runs(df: pd.DataFrame):
    """Show every run for every config side by side."""
    if df.empty:
        print("  No end-to-end results found.")
        return

    print(f"\n  END-TO-END — ALL RUNS")
    print(f"  {'─' * (W-2)}")
    print(
        f"  {'Timestamp':<16}  {'Config':<36}  {'DB':>3}  "
        f"{'#Q':>3}  {'Auto':>6}  {'Judge':>6}  "
        f"{'CtxHit':>6}  {'AvgW':>5}  {'AvgT':>5}"
    )
    print(f"  {'─' * (W-2)}")

    # Group by config_key for readability
    for config_key in ["local_rag", "deepseek_rag", "deepseek_only"]:
        subset = df[df["config_key"] == config_key]
        if subset.empty:
            continue
        for _, row in subset.iterrows():
            db_s     = "Yes" if row["use_db"] else "No"
            judge_s  = f"{row['judge_score']:6.1f}" if pd.notna(row["judge_score"]) else "   N/A"
            ctx_s    = f"{row['ctx_hit_%']:5.0f}%" if pd.notna(row["ctx_hit_%"]) else "    —"
            avgw_s   = f"{row['avg_words']:5.0f}" if pd.notna(row["avg_words"]) else "    —"
            avgt_s   = f"{row['avg_time_s']:4.1f}s" if pd.notna(row["avg_time_s"]) else "   —"
            label    = row["config"][:36]
            print(
                f"  {row['run']:<16}  {label:<36}  {db_s:>3}  "
                f"{int(row['n_queries']) if str(row['n_queries']).isdigit() else '?':>3}  "
                f"{row['auto_score']:>6.1f}  {judge_s:>6}  "
                f"{ctx_s:>6}  {avgw_s:>5}  {avgt_s:>5}"
            )
        print()


def print_e2e_best(df: pd.DataFrame):
    """Best row per config — the definitive comparison."""
    if df.empty:
        return

    print(f"\n  END-TO-END — BEST RESULT PER CONFIG")
    print(f"  {'─' * (W-2)}")
    print(
        f"  {'Config':<40}  {'DB':>3}  "
        f"{'Auto/100':>8}  {'Judge/100':>9}  "
        f"{'CtxHit':>6}  {'AvgWords':>8}  {'AvgTime':>7}"
    )
    print(f"  {'─' * (W-2)}")

    best_auto  = df["auto_score"].max()
    judge_vals = df["judge_score"].dropna()
    best_judge = judge_vals.max() if not judge_vals.empty else None

    for _, row in df.iterrows():
        db_s     = "Yes" if row["use_db"] else "No"
        judge_s  = f"{row['judge_score']:9.1f}" if pd.notna(row["judge_score"]) else f"{'N/A':>9}"
        ctx_s    = f"{row['ctx_hit_%']:.0f}%" if pd.notna(row["ctx_hit_%"]) else "—"
        avgw_s   = f"{row['avg_words']:.0f}" if pd.notna(row["avg_words"]) else "—"
        avgt_s   = f"{row['avg_time_s']:.1f}s" if pd.notna(row["avg_time_s"]) else "—"
        a_flag   = " ◀" if row["auto_score"] == best_auto else ""
        j_flag   = " ◀" if (best_judge and pd.notna(row["judge_score"]) and row["judge_score"] == best_judge) else ""
        print(
            f"  {row['config'][:40]:<40}  {db_s:>3}  "
            f"{row['auto_score']:>8.1f}  {judge_s}  "
            f"{ctx_s:>6}  {avgw_s:>8}  {avgt_s:>7}"
            f"{a_flag}{j_flag}"
        )

    print(f"\n  Score: Auto = Retrieval(0-50) + Richness(0-30) + Reliability(0-20)")
    print(f"         Judge = DeepSeek rates Relevance+Accuracy+Completeness /100")
    print(f"         CtxHit = % queries where Qdrant retrieved a relevant source")

    # Key comparisons
    keys_present = set(df["config_key"])
    print(f"\n  {'─' * (W-2)}")
    print(f"  KEY COMPARISONS (using best run per config)")

    def compare(k_a, k_b, label):
        if k_a not in keys_present or k_b not in keys_present:
            return
        a_auto  = df[df["config_key"] == k_a]["auto_score"].max()
        b_auto  = df[df["config_key"] == k_b]["auto_score"].max()
        d_auto  = a_auto - b_auto
        a_judge = df[df["config_key"] == k_a]["judge_score"].dropna()
        b_judge = df[df["config_key"] == k_b]["judge_score"].dropna()
        d_j_str = ""
        if not a_judge.empty and not b_judge.empty:
            d_j_str = f"   judge Δ={a_judge.max() - b_judge.max():+.1f}"
        arrow = "▲ better" if d_auto > 0 else "▼ worse "
        print(f"  {label:<42}  auto Δ={d_auto:+.1f}  {arrow}{d_j_str}")

    compare("deepseek_rag",  "deepseek_only", "B vs C  RAG value for DeepSeek")
    compare("deepseek_rag",  "local_rag",     "B vs A  Cloud vs local LLM")
    compare("local_rag",     "deepseek_only", "A vs C  Local RAG vs pure cloud")


# ──────────────────────────────────────────────────────────────────────────────
# Save merged CSV
# ──────────────────────────────────────────────────────────────────────────────

def save_merged(
    ret_df:  pd.DataFrame,
    e2e_df:  pd.DataFrame,
    best_df: pd.DataFrame,
):
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = RESULTS_DIR / f"merged_comparison_{ts}.csv"

    rows = []
    if not ret_df.empty:
        for _, r in ret_df.iterrows():
            rows.append({
                "section": "retrieval", "run": "—", "config": r["method"],
                "llm": "—", "use_db": "—", "n_queries": 20,
                "MAP": r["MAP"], "P@1": r["P@1"], "P@3": r["P@3"],
                "P@5": r["P@5"], "Recall@5": r["Recall@5"],
                "auto_score": "—", "judge_score": "—",
                "ctx_hit_%": "—", "avg_words": "—", "avg_time_s": "—",
            })

    for _, r in best_df.iterrows():
        rows.append({
            "section": "endtoend_best", "run": r["run"], "config": r["config"],
            "llm": r["llm"], "use_db": r["use_db"], "n_queries": r["n_queries"],
            "MAP": "—", "P@1": "—", "P@3": "—", "P@5": "—", "Recall@5": "—",
            "auto_score": r["auto_score"], "judge_score": r["judge_score"],
            "ctx_hit_%": r["ctx_hit_%"], "avg_words": r["avg_words"],
            "avg_time_s": r["avg_time_s"],
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n  Saved merged CSV → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge and compare all saved evaluation results"
    )
    p.add_argument("--save",      action="store_true", help="Save merged CSV to results/")
    p.add_argument("--best-only", action="store_true", dest="best_only",
                   help="Only show best run per config, skip per-run detail")
    return p.parse_args()


def main():
    args    = parse_args()
    reports = collect_all_reports()

    if not reports:
        print(f"No result files found in {RESULTS_DIR}")
        return

    print(f"{'=' * W}")
    print(f"  OLIVE RAG — MERGED COMPARISON  ({len(reports)} result files)")
    print(f"  Source: {RESULTS_DIR}")
    print(f"{'=' * W}")

    print(f"\n  Files found:")
    for r in reports:
        n_configs = len(r.get("endtoend_scores", {}))
        has_ret   = "retrieval ✓" if r.get("retrieval") else "retrieval —"
        has_judge = "judge ✓" if any(
            sc["judge"]["judge_score"] is not None
            for sc in r.get("endtoend_scores", {}).values()
        ) else "judge —"
        print(f"    {r['run_timestamp']}  {r['_source_file']:<45}  "
              f"{n_configs} configs  {has_ret}  {has_judge}")

    # Build tables
    ret_df  = build_retrieval_table(reports)
    e2e_df  = build_e2e_table(reports, best_only=False)
    best_df = build_e2e_table(reports, best_only=True)

    # Print retrieval
    print_retrieval(ret_df)

    # Print per-run detail (unless --best-only)
    if not args.best_only:
        print_e2e_all_runs(e2e_df)

    # Print best per config
    print_e2e_best(best_df)

    print(f"\n{'=' * W}")

    if args.save:
        save_merged(ret_df, e2e_df, best_df)


if __name__ == "__main__":
    main()
