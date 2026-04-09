from rag.ingestion import chunk_text_sentences, embed_chunks, build_index
from rag.retrieval import retrieve
from rag.reranking import rerank
from rag.generation import generate_answer


def normalize(text: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - remove spaces
    """
    return text.lower().replace("", "")


def is_grounded(answer: str, context_chunks: list) -> bool:
    """
    Checks wether the answer is supported by the retrieved context.

    Current heuristic:
    - Returns True if the answer (or part of it) appears in any retrieved chunk
    """
    answer_norm: str = normalize(text=answer)

    for chunk in context_chunks:
        if answer_norm in normalize(text=chunk):
            return True

    return False


def is_grounded_top1(answer: str, context_chunks: list) -> bool:
    """
    Checks if the answer is supported by the TOP retrieved chunk only.

    This measures ranking quality (precision).
    """
    if not context_chunks:
        return False

    return normalize(text=answer) in normalize(context_chunks[0])


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
        #retrieved: list = retrieve(query=query, index=index, chunks=chunks, k=3)

        # reranked retrieval
        retrieved: list = retrieve(query=query, index=index, chunks=chunks, k=10)
        reranked: list = rerank(query=query, retrieved_results=retrieved)
        retrieved: list = reranked[:3]

        retrieved_texts = [r if isinstance(r, str) else r["chunk"] for r in retrieved]
        answer = generate_answer(query=query, context_chunks=retrieved_texts)
        # Metrics
        is_correct = evaluate_answer(predicted=answer, expected=expected)
        hit = any(expected.lower() in c.lower() for c in retrieved_texts)
        grounded: bool = is_grounded(answer=answer, context_chunks=retrieved_texts)
        grounded_top1: bool = is_grounded_top1(
            answer=answer, context_chunks=retrieved_texts
        )

        # debug snippets
        # if not grounded:
        #     print("\n[UNGROUNDED ANSWER]")
        #     print("Q:", query)
        #     print("A:", answer)
        #     print("\nTop chunks:")
        #     for c in retrieved_texts:
        #         print("-", c[:120])

        if grounded and not grounded_top1:
            print("\n[RANKING ISSUE]")
            print("Q:", query)
            print("A:", answer)

            print("\nTop chunk:")
            print("-", retrieved_texts[0][:120])
            print("\nCorrect chunk exists but not ranked first:")
            for c in retrieved_texts[1:]:
                print("-", c[:120])

        results.append(
            {
                "question": query,
                "correct": is_correct,
                "retrieval_hit": hit,
                "grounded": grounded,
                "grounded_top1": grounded_top1,
            }
        )

        # print("\n---")
        # print("Q:", query)
        # print("Hit:", hit)
        # print("Correct:", is_correct)

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


def summarize(results) -> dict:
    total: int = len(results)
    correct: int = sum(r["correct"] for r in results)
    hits: int = sum(r["retrieval_hit"] for r in results)
    grounded: int = sum(r["grounded"] for r in results)

    return {
        "accuracy": correct / total,
        "retrieval_hit_rate": hits / total,
        "groundedness": grounded / total,
        "grounded_top1": sum(r["grounded_top1"] for r in results) / total,
    }
