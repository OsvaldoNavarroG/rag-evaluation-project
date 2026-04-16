from typing import Any, Dict, List
import numpy as np




def dense_retrieve(query, index, chunks, model, k=5) -> List[Dict[str, Any]]:
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results: List[Dict[str, Any]] = []
    for i, idx in enumerate(indices[0]):
        results.append(
            {"chunk": chunks[idx], "score": float(distances[0][i]), "source": "dense"}
        )
    return results
