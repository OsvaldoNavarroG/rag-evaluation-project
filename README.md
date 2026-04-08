# RAG System with Retrieval Optimization

## 📌 Summary

Built an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline and improved its performance through **chunking strategy and re-ranking**.

Focus: **measure, diagnose, and fix retrieval quality**

---

## ⚙️ Stack

* FAISS (vector search)
* sentence-transformers (embeddings + reranker)
* OpenAI API (generation)

---

## 🧠 Pipeline

```text
Documents → Chunking → Embeddings → FAISS
Query → Retrieval (Top-K) → Re-ranking → LLM → Answer
```

---

## 🧪 Evaluation Setup

* ~1500+ word dataset
* 25 queries (easy / medium / hard)
* Metrics:

  * **Accuracy**
  * **Retrieval Hit Rate**

---

## 📊 Results

| Method            | Accuracy | Hit Rate |
| ----------------- | -------- | -------- |
| Naive Chunking    | 0.86     | 0.09     |
| Sentence Chunking | 0.86     | 0.86     |
| + Re-ranking      | **0.91** | 0.86     |

---

## 🔍 Key Insights

* **Chunking dominates performance**
  → Hit rate improved **0.09 → 0.86**

* **LLM knowledge can hide retrieval failures**
  → High accuracy does not guarantee correct retrieval

* **Re-ranking improves precision, not recall**
  → Accuracy improved without changing hit rate

---

## 🚀 Run

```bash
pip install sentence-transformers faiss-cpu openai python-dotenv
python main.py
```

---

## 🎯 What This Shows

* Built a **modular RAG pipeline from scratch**
* Identified bottlenecks via **metrics-driven analysis**
* Improved system using **IR techniques (chunking + reranking)**
* Demonstrated **measurable gains**

---

## 🔧 Next Steps

* Hybrid search (BM25 + embeddings)
* Query expansion
* Groundedness evaluation
