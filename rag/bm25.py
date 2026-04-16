import numpy as np
from rank_bm25 import BM25Okapi
from typing import Any, Dict, List, Tuple


class BM25Retriever:
    def __init__(self, chunks: List[str]):
        self.chunks: List[str] = chunks
        self.tokenized_chunks = [self._tokenize(text=c) for c in chunks]
        self.bm25 = BM25Okapi(corpus=self.tokenized_chunks)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        tokenized_query: List[str] = self._tokenize(text=query)
        scores: np.ndarray = self.bm25.get_scores(tokenized_query)

        ranked: List[Tuple[str, float]] = sorted(
            zip(self.chunks, scores), key=lambda x: x[1], reverse=True
        )

        return [
            {"chunk": c, "score": float(s), "source": "bm25"} for c, s in ranked[:k]
        ]
