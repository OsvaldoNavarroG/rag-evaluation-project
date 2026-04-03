import numpy as np
from typing import List
from rag.ingestion import load_documents, chunk_text, embed_chunks, build_index
from rag.retrieval import retrieve
from rag.generation import generate_answer
from rag.evaluation import evaluate_answer

# 1. Load and prepare data
text: str = load_documents(path="data/docs.txt")
chunks = chunk_text(text=text)
embeddings: np.ndarray = embed_chunks(chunks=chunks)
index = build_index(embeddings=embeddings)

# 2. Test queries (you will expand this later)
test_data: List[dict] = [
    {
        "question": "What is the main topic of the document?",
        "expected": "machine learning",
    }
]

# 3. Run pipeline
for item in test_data:
    query: str = item["question"]
    expected: str = item["expected"]

    retrieved = retrieve(query=query, index=index, chunks=chunks)
    answer = generate_answer(query=query, context_chunks=retrieved)

    print("\n---")
    print("Q:", query)
    print("A:", answer)

    score = evaluate_answer(answer, expected)
    print("Correct:", score)