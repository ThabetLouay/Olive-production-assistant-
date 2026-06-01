import json
from pathlib import Path

chunks_path = Path("data/processed/olive_chunks.jsonl")
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f if l.strip()]

print(f"Total chunks: {len(chunks)}\n")

# Show all unique sources with counts
sources = {}
for c in chunks:
    src = c["metadata"]["source_pdf"]
    lang = c["metadata"].get("language", "?")
    key = f"{src} [{lang}]"
    sources[key] = sources.get(key, 0) + 1

print("=== ALL SOURCES ===")
for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
    print(f"  {count:>4} chunks | {src}")