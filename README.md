# RAG Retrieval Optimization Project

⏱️ Time to read: ~1 minute

---

## 🚀 Key Results

Improved retrieval performance in a RAG system:

* **Retrieval Hit Rate:** 9% → 86% (+77pp)
* **Answer Accuracy:** 86% → 91%

Identified **chunking as the primary bottleneck** and improved answer quality using **cross-encoder re-ranking**.

---

## 📌 Overview

This project implements and evaluates a **Retrieval-Augmented Generation (RAG)** pipeline, focusing on:

* Diagnosing retrieval vs generation failures
* Improving retrieval quality through better chunking
* Increasing answer accuracy via re-ranking

---

## 🧠 Pipeline

```text
Documents → Chunking → Embeddings → FAISS Index
Query → Retrieval (Top-K) → Re-ranking → LLM → Answer
```

---

## ⚙️ Tech Stack

* FAISS — vector search
* sentence-transformers — embeddings + cross-encoder reranker
* OpenAI API — answer generation

---

## 🧪 Evaluation Setup

* Dataset: ~1500+ words (ML concepts)
* Chunking: 50-word chunks (naive vs sentence-based)
* Test set: 25 queries (easy / medium / hard)

### Metrics

* **Accuracy** — correctness of generated answer
* **Retrieval Hit Rate** — whether correct context was retrieved

---

## 📊 Results

| Method            | Accuracy | Hit Rate |
| ----------------- | -------- | -------- |
| Naive Chunking    | 0.86     | 0.09     |
| Sentence Chunking | 0.86     | 0.86     |
| + Re-ranking      | **0.91** | 0.86     |

---

## 🔍 Example Improvement

**Query:**
“What is machine learning used for in image tasks?”

**Before re-ranking:**

* Retrieved: generic ML usage, image processing
* Answer: partially correct

**After re-ranking:**

* Retrieved: image recognition (correct chunk)
* Answer: correct

→ Re-ranking improves **context precision**, leading to better answers

---

## 🧠 Key Insights

* **Chunking dominates retrieval performance**
  → Sentence-based chunking improved hit rate dramatically

* **LLM knowledge can mask retrieval failures**
  → High accuracy does not guarantee correct retrieval

* **Re-ranking improves precision, not recall**
  → Better ordering of retrieved chunks improves answer quality

---

## 🚀 How to Run

```bash
pip install sentence-transformers faiss-cpu openai python-dotenv
python main.py
```

---

## 🎯 What This Demonstrates

* End-to-end RAG system design
* Retrieval vs generation error analysis
* Practical improvements using IR techniques
* Measurable, data-driven performance gains

---

## 🔧 Next Steps

* Hybrid search (BM25 + embeddings)
* Query expansion
* Groundedness evaluation
