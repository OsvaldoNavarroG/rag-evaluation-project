from typing import Any


def hybrid_retrieve(
    query: str, dense_retrieve_fn, bm25_retriever, k_dense: int = 5, k_bm25: int = 5
) -> list[dict[str, Any]]:
    """
    Combines dense and BM25 retrieval.
    Deduplicates chunks and keeps metadata.
    """

    dense_results: list[dict[str, Any]] = dense_retrieve_fn(query, k=k_dense)
    bm25_results: list[dict[str, Any]] = bm25_retriever.retrieve(query, k=k_bm25)

    combined: dict[str, dict[str, Any]] = {}

    # Add dense results
    for r in dense_results:
        combined[r["chunk"]] = r

    # Add BM25 results (if new)
    for r in bm25_results:
        if r["chunk"] not in combined:
            combined[r["chunk"]] = r

    return list(combined.values())
