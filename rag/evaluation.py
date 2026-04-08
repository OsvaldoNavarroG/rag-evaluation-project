from rag.ingestion import chunk_text_sentences, embed_chunks, build_index
from rag.retrieval import retrieve
from rag.reranking import rerank
from rag.generation import generate_answer


def evaluate_answer(predicted, expected):
    predicted = predicted.lower()
    expected = expected.lower()

    return expected in predicted


def run_pipeline(chunking_fn, text, test_data, label):
    print(f"\n===== {label} =====")

    chunks = chunking_fn(text)
    embeddings = embed_chunks(chunks=chunks)
    index = build_index(embeddings=embeddings)

    results = []

    for item in test_data:
        query = item["question"]
        expected = item["expected"]

        # baseline retrieval
        # retrieved: list = retrieve(query=query, index=index, chunks=chunks, k=3)
        
        retrieved: list = retrieve(query=query, index=index, chunks=chunks, k=10)
        reranked: list = rerank(query=query, retrieved_results=retrieved)
        retrieved: list = reranked[:3]

        retrieved_texts = [r if isinstance(r, str) else r["chunk"] for r in retrieved]
        answer = generate_answer(query=query, context_chunks=retrieved_texts)
        # Metrics
        is_correct = evaluate_answer(predicted=answer, expected=expected)
        hit = any(expected.lower() in c.lower() for c in retrieved_texts)
        results.append({"question": query, "correct": is_correct, "retrieval_hit": hit})

        print("\n---")
        print("Q:", query)
        print("Hit:", hit)
        print("Correct:", is_correct)

    return results


def compare_chunking_approaches(text, test_data):
    from rag.ingestion import chunk_text as naive_chunk_text

    naive_results = run_pipeline(
        chunking_fn=naive_chunk_text,
        text=text,
        test_data=test_data,
        label="Naive Chunking",
    )
    sentence_results = run_pipeline(
        chunking_fn=chunk_text_sentences,
        text=text,
        test_data=test_data,
        label="Sentence Chunking",
    )
    print("\n===== SUMMARY =====")
    print("Naive:", summarize(results=naive_results))
    print("Sentence:", summarize(results=sentence_results))


def summarize(results):
    total: int = len(results)
    correct = sum(r["correct"] for r in results)
    hits = sum(r["retrieval_hit"] for r in results)

    return {"accuracy": correct / total, "retrieval_hit_rate": hits / total}
