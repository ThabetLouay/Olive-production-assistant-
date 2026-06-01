# test_tools.py
from src.tools.vector_tool import search_with_fallback
from src.tools.sql_tool import get_summary_stats, get_drought_years

# Test vector search
print("=== VECTOR SEARCH ===")
results = search_with_fallback("olive drought resistance Tunisia", top_k=3)
for r in results:
    print(f"Score={r['score']} | {r['source'][:40]} p{r['page']}")
    print(r['text'][:150])
    print()

# Test SQL
print("=== SQL STATS ===")
stats = get_summary_stats()
print(stats)

print("=== DROUGHT YEARS ===")
droughts = get_drought_years()
print(f"Found {len(droughts)} drought years")
print(droughts.head(3))