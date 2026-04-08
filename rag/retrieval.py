from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({"chunk": chunks[idx], "score": float(distances[0][i])})
    return results
