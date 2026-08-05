from typing import Any


class MultiQueryRetriever:
    """
    Expands a query and aggregates retrieval results
    """

    def __init__(self, retriever_fn):
        """
        retriever_fn: function(query)-> list of results
        """
        self.retriever_fn = retriever_fn

    def retrieve(self, expanded_queries: list[str]) -> list[dict[str, Any]]:
        all_results: dict[str, dict[str, Any]] = {}

        for q in expanded_queries:
            results: list[dict[str, Any]] = self.retriever_fn(q)

            for r in results:
                chunk = r["chunk"]

                # Deduplicate by chunk text while preserving first occurrence
                if chunk not in all_results:
                    all_results[chunk] = r

        return list(all_results.values())
