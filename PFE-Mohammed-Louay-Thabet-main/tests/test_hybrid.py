# test_hybrid.py
import logging
logging.basicConfig(level=logging.WARNING)

from src.tools.keyword_search import keyword_search
from src.tools.semantic_search import semantic_search
from src.tools.metadata_filter import apply_metadata_boost
from src.tools.hybrid_retriever import (
    search_bm25_only,
    search_semantic_only,
    search_full_hybrid,
)

query = "how does drought affect olive production in Tunisia"

print("=" * 60)
print("BM25 ONLY")
print("=" * 60)
for r in search_bm25_only(query, top_k=3):
    print(f"Score={r['score']:.4f} | {r['source'][:35]} p{r['page']}")
    print(r["text"][:150])
    print()

print("=" * 60)
print("SEMANTIC ONLY")
print("=" * 60)
for r in search_semantic_only(query, top_k=3):
    print(f"Score={r['score']:.4f} | {r['source'][:35]} p{r['page']}")
    print(r["text"][:150])
    print()

print("=" * 60)
print("FULL HYBRID (BM25 + Semantic + Metadata)")
print("=" * 60)
for r in search_full_hybrid(query, top_k=3):
    print(f"Score={r['score']:.6f} | Boost={r.get('boost',1):.3f} | {r['source'][:35]} p{r['page']}")
    print(r["text"][:150])
    print()