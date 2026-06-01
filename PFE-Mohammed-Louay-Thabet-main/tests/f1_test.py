def retrieval_f1(retrieved: list[str], relevant: list[str]) -> dict:
    """
    Compute retrieval Precision, Recall, and F1.

    Args:
        retrieved: list of retrieved chunk IDs (or texts)
        relevant:  list of ground truth relevant chunk IDs (or texts)

    Returns:
        dict with precision, recall, f1
    """
    retrieved_set = set(retrieved)
    relevant_set  = set(relevant)

    tp = len(retrieved_set & relevant_set)  # true positives

    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall    = tp / len(relevant_set)  if relevant_set  else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


# --- Evaluate over a dataset ---
def evaluate_retrieval(dataset: list[dict]) -> dict:
    """
    dataset: list of {"query": ..., "retrieved": [...], "relevant": [...]}
    Returns macro-averaged metrics.
    """
    scores = [retrieval_f1(d["retrieved"], d["relevant"]) for d in dataset]

    avg = lambda key: sum(s[key] for s in scores) / len(scores)
    return {
        "precision": avg("precision"),
        "recall":    avg("recall"),
        "f1":        avg("f1"),
    }


# --- Example usage ---
dataset = [
    {
        "query": "What is the return policy?",
        "retrieved": ["chunk_3", "chunk_7", "chunk_12"],   # your RAG retrieved these
        "relevant":  ["chunk_3", "chunk_7"],               # ground truth
    },
    {
        "query": "How do I reset my password?",
        "retrieved": ["chunk_1", "chunk_9"],
        "relevant":  ["chunk_1", "chunk_5", "chunk_9"],
    },
]

results = evaluate_retrieval(dataset)
print(results)
