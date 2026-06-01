# tests/evaluate_rag.py
"""
MAP (Mean Average Precision) evaluation for the Olive RAG hybrid retriever.

Test set: 20 questions manually annotated with relevant source documents.
We evaluate at K=1, K=3, K=5.

Metrics computed:
  - Precision@K     per query
  - Average Precision (AP) per query
  - Mean Average Precision (MAP) across all queries
  - Recall@K        per query

Comparison:
  - BM25 only
  - Semantic only
  - Hybrid (BM25 + Semantic + Metadata boost)
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING)

from src.tools.hybrid_retriever import (
    search_bm25_only,
    search_semantic_only,
    search_full_hybrid,
)

# ---------------------------------------------------------------
# Test set
# Each entry:
#   query          — the question
#   relevant_docs  — list of source_pdf substrings that are relevant
#                    (partial match is fine — we check if substring in result source)
#   notes          — why this question tests what it tests
# ---------------------------------------------------------------

TEST_SET = [
    # ── Drought & climate stress ─────────────────────────────
    {
        "query": "What are the physiological effects of drought stress on olive trees?",
        "relevant_docs": ["Drought stress effects", "Technologie innovante"],
        "notes": "Core drought physiology — should retrieve drought paper + French guide",
    },
    {
        "query": "How does water deficit affect olive oil quality?",
        "relevant_docs": ["Drought stress effects", "water used by the olive tree"],
        "notes": "Water stress impact on oil — drought paper + water use paper",
    },
    {
        "query": "What drought resistance mechanisms do olive trees use?",
        "relevant_docs": ["Drought stress effects", "Technologie innovante"],
        "notes": "Resistance mechanisms — both drought documents",
    },
    {
        "query": "Comment les oliviers s'adaptent à la sécheresse?",
        "relevant_docs": ["Technologie innovante", "Drought stress effects"],
        "notes": "French query — should prioritise French document",
    },
    {
        "query": "What is the impact of soil water content on olive stomatal conductance?",
        "relevant_docs": ["water used by the olive tree", "Drought stress effects"],
        "notes": "Stomatal response — water use paper primary",
    },

    # ── Irrigation & water management ───────────────────────
    {
        "query": "What are the water requirements of olive trees in arid regions?",
        "relevant_docs": ["water used by the olive tree", "Drought stress effects"],
        "notes": "Water requirements — water use paper",
    },
    {
        "query": "How should I irrigate my olive farm during summer drought?",
        "relevant_docs": ["Technologie innovante", "water used by the olive tree"],
        "notes": "Practical irrigation advice — French guide + water paper",
    },
    {
        "query": "What irrigation techniques reduce water stress in olive orchards?",
        "relevant_docs": ["water used by the olive tree", "Technologie innovante"],
        "notes": "Irrigation techniques — water use paper + French guide",
    },

    # ── Production & yield ──────────────────────────────────
    {
        "query": "What climate variables affect olive yield the most?",
        "relevant_docs": ["emperature-related", "1.Elloumi", "Drought stress effects"],
        "notes": "Climate-yield relationship — pistachio paper + Elloumi",
    },
    {
        "query": "How does warm winter affect olive flowering and fruit set?",
        "relevant_docs": ["1.Elloumi", "Drought stress effects"],
        "notes": "Winter temperature effect — Elloumi paper primary",
    },
    {
        "query": "What is the alternate bearing phenomenon in olive trees?",
        "relevant_docs": ["emperature-related", "1.Elloumi"],
        "notes": "Alternate bearing — pistachio paper + Elloumi",
    },
    {
        "query": "How do chilling hours affect olive production?",
        "relevant_docs": ["emperature-related", "1.Elloumi"],
        "notes": "Chilling requirements — pistachio paper primary",
    },

    # ── Olive oil quality ────────────────────────────────────
    {
        "query": "What factors affect the polyphenol content of olive oil?",
        "relevant_docs": ["Olive tree  leaf", "Drought stress effects"],
        "notes": "Polyphenols — leaf paper + drought paper",
    },
    {
        "query": "What are the health benefits of olive leaf extracts?",
        "relevant_docs": ["Olive tree  leaf"],
        "notes": "Leaf bioactive compounds — leaf paper exclusively",
    },
    {
        "query": "How does olive leaf composition vary by cultivar?",
        "relevant_docs": ["Olive tree  leaf"],
        "notes": "Cultivar differences in leaves — leaf paper",
    },

    # ── Pests & diseases ─────────────────────────────────────
    {
        "query": "What are the main pests affecting olive trees during drought?",
        "relevant_docs": ["Technologie innovante", "Drought stress effects"],
        "notes": "Drought-related pests — French guide + drought paper",
    },
    {
        "query": "How does Bactrocera oleae affect olive production?",
        "relevant_docs": ["Technologie innovante", "Drought stress effects"],
        "notes": "Olive fly — French guide primary",
    },

    # ── Farming systems & sustainability ─────────────────────
    {
        "query": "What sustainable farming practices improve olive production?",
        "relevant_docs": ["evolution and sustainability", "Technologie innovante"],
        "notes": "Sustainable systems — evolution paper + French guide",
    },
    {
        "query": "How does pruning affect olive tree productivity?",
        "relevant_docs": ["evolution and sustainability", "water used by the olive tree"],
        "notes": "Pruning effects — evolution paper",
    },

    # ── Tunisia specific ─────────────────────────────────────
    {
        "query": "What are the challenges of olive growing in arid Tunisia?",
        "relevant_docs": ["1.Elloumi", "Technologie innovante", "Drought stress effects"],
        "notes": "Tunisia-specific — Elloumi paper primary",
    },
]


# ---------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------

def is_relevant(result: dict, relevant_docs: list[str]) -> bool:
    """Check if a result is relevant based on source document name."""
    source = result.get("source", "").lower()
    return any(rd.lower() in source for rd in relevant_docs)


def precision_at_k(results: list[dict], relevant_docs: list[str], k: int) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    if not results:
        return 0.0
    top_k = results[:k]
    relevant_count = sum(1 for r in top_k if is_relevant(r, relevant_docs))
    return relevant_count / k


def average_precision(results: list[dict], relevant_docs: list[str], k: int = 5) -> float:
    """
    Average Precision (AP): average of precision values at each relevant result.
    AP = (1/R) * Σ P@k * rel(k)
    where R = total relevant documents in top-K, rel(k) = 1 if result k is relevant.
    """
    if not results:
        return 0.0

    score     = 0.0
    n_relevant = 0

    for i, result in enumerate(results[:k], start=1):
        if is_relevant(result, relevant_docs):
            n_relevant += 1
            score += n_relevant / i

    if n_relevant == 0:
        return 0.0
    return score / n_relevant


def recall_at_k(results: list[dict], relevant_docs: list[str], k: int) -> float:
    """
    Recall@K: fraction of relevant documents found in top-K.
    Since we don't know the exact chunk IDs, we approximate by
    counting unique relevant sources found vs total relevant sources.
    """
    if not results or not relevant_docs:
        return 0.0
    found_sources = set()
    for r in results[:k]:
        for rd in relevant_docs:
            if rd.lower() in r.get("source", "").lower():
                found_sources.add(rd)
    return len(found_sources) / len(relevant_docs)


def evaluate_method(
    method_name: str,
    search_fn,
    test_set: list[dict],
    k: int = 5,
) -> dict:
    """Run evaluation for one search method across all test queries."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name} @ K={k}")
    print(f"{'='*60}")

    ap_scores  = []
    p_at_1     = []
    p_at_3     = []
    p_at_5     = []
    recall_scores = []

    for i, test in enumerate(test_set, start=1):
        query        = test["query"]
        relevant     = test["relevant_docs"]
        results      = search_fn(query, top_k=k)

        ap  = average_precision(results, relevant, k=k)
        p1  = precision_at_k(results, relevant, k=1)
        p3  = precision_at_k(results, relevant, k=min(3, k))
        p5  = precision_at_k(results, relevant, k=k)
        rec = recall_at_k(results, relevant, k=k)

        ap_scores.append(ap)
        p_at_1.append(p1)
        p_at_3.append(p3)
        p_at_5.append(p5)
        recall_scores.append(rec)

        status = "✅" if ap > 0 else "❌"
        print(f"  Q{i:>2} {status} AP={ap:.3f} P@1={p1:.2f} P@3={p3:.2f} "
              f"P@5={p5:.2f} R@5={rec:.2f} | {query[:50]}")

    map_score = sum(ap_scores) / len(ap_scores)
    results_summary = {
        "method":    method_name,
        "MAP":       round(map_score, 4),
        "P@1":       round(sum(p_at_1) / len(p_at_1), 4),
        "P@3":       round(sum(p_at_3) / len(p_at_3), 4),
        "P@5":       round(sum(p_at_5) / len(p_at_5), 4),
        "Recall@5":  round(sum(recall_scores) / len(recall_scores), 4),
    }

    print(f"\n  MAP={map_score:.4f} | "
          f"P@1={results_summary['P@1']:.4f} | "
          f"P@3={results_summary['P@3']:.4f} | "
          f"P@5={results_summary['P@5']:.4f} | "
          f"R@5={results_summary['Recall@5']:.4f}")

    return results_summary


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("🫒 Olive RAG — Retrieval Evaluation (MAP)")
    print(f"Test set: {len(TEST_SET)} queries")
    print("Methods: BM25 | Semantic | Hybrid (BM25+Semantic+Metadata)")

    results = []

    # BM25 only
    results.append(evaluate_method(
        "BM25 only",
        search_bm25_only,
        TEST_SET,
        k=5,
    ))

    # Semantic only
    results.append(evaluate_method(
        "Semantic only",
        search_semantic_only,
        TEST_SET,
        k=5,
    ))

    # Full hybrid
    results.append(evaluate_method(
        "Hybrid (BM25+Semantic+Metadata)",
        search_full_hybrid,
        TEST_SET,
        k=5,
    ))

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")

    df = pd.DataFrame(results).set_index("method")
    print(df.to_string())

    # Best method
    best = df["MAP"].idxmax()
    print(f"\n🏆 Best method by MAP: {best} (MAP={df.loc[best,'MAP']:.4f})")

    # Improvement
    bm25_map    = df.loc["BM25 only", "MAP"]
    sem_map     = df.loc["Semantic only", "MAP"]
    hybrid_map  = df.loc["Hybrid (BM25+Semantic+Metadata)", "MAP"]

    if bm25_map > 0:
        print(f"📈 Hybrid vs BM25:     +{((hybrid_map - bm25_map)/bm25_map*100):.1f}%")
    if sem_map > 0:
        print(f"📈 Hybrid vs Semantic: +{((hybrid_map - sem_map)/sem_map*100):.1f}%")

    # Save results
    output_path = Path("tests/evaluation_results.csv")
    df.to_csv(output_path)
    print(f"\n💾 Results saved to {output_path}")