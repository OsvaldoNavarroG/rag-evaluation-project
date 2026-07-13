from functools import lru_cache
from sentence_transformers import CrossEncoder
from typing import List


@lru_cache
def get_reranker() -> CrossEncoder:
    """
    Lazily load the cross-encoder reranker.

    Loading is deferred until the first rerank call (and cached thereafter)
    so importing this module dows not download or load a model. This keeps
    the module importable in environments where reranking is never used
    (e.g. unit tests of the retrieval or attribution layers)
    """
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, retrieved_results):
    reranker: CrossEncoder = get_reranker()
    pairs: List[tuple] = [(query, r["chunk"]) for r in retrieved_results]
    scores = reranker.predict(pairs)

    for i, r in enumerate(retrieved_results):
        r["rerank_score"] = float(scores[i])

    return sorted(retrieved_results, key=lambda x: x["rerank_score"], reverse=True)
