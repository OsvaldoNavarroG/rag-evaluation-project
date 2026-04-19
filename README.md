# RAG Retrieval Optimization & Evaluation

⏱️ Time to read: ~1 minute

---

## 🚀 Key Results

Evaluated multiple RAG configurations using **heuristic and LLM-based metrics**:

* **Retrieval Hit Rate:** ~86% (consistently high across methods)
* **LLM Accuracy:** up to **91%**
* **LLM Groundedness:** up to **91%**
* **Groundedness (heuristic):** ~41–50%

---

## 🔥 Key Findings

### 1. Evaluation Gap (Critical)

* Heuristic groundedness: ~0.41
* LLM groundedness: ~0.91

👉 Traditional metrics significantly **underestimate semantic correctness**

---

### 2. Hybrid Retrieval Improves Ranking (Not Recall)

* Retrieval hit rate unchanged (~0.86)
* Improved:

  * answer accuracy
  * groundedness
  * top-ranked chunk quality

---

### 3. Multi-Query Retrieval Did NOT Improve Performance

| Metric             | Hybrid + Rerank | Multi-Query + Hybrid + Rerank |
| ------------------ | --------------- | ----------------------------- |
| Retrieval Hit Rate | 0.86            | 0.86                          |
| Accuracy (Naive)   | **0.77**        | 0.73                          |
| LLM Groundedness   | **0.91**        | 0.86                          |

👉 Multi-query introduced **more noise without improving recall**

---

## 🧠 Core Insight

> **Multi-query retrieval is only beneficial when the system is recall-limited.**

In this project:

* Recall already high (~0.86)
* Bottleneck = **ranking precision**, not retrieval

👉 Adding more queries increased candidate noise and degraded performance

---

## 📌 Overview

This project builds and evaluates a RAG pipeline focusing on:

* Retrieval vs generation error analysis
* Hallucination detection (groundedness)
* Heuristic vs semantic evaluation
* Retrieval strategy comparison
* Query expansion analysis

---

## 🧠 Pipeline

```text
Documents → Chunking → Embeddings → FAISS
Query → Retrieval (Dense / Hybrid / Multi-Query) → Reranking → LLM → Answer
```

---

## ⚙️ Tech Stack

* FAISS — dense vector retrieval
* BM25 (rank-bm25) — lexical retrieval
* sentence-transformers — embeddings + cross-encoder reranker
* OpenAI API — generation + LLM-based evaluation

---

## 🧪 Retrieval Strategies Compared

### Dense Retrieval

* Semantic similarity
* Strong baseline recall

### Hybrid Retrieval (BM25 + Dense)

* Combines semantic + keyword signals
* Improves candidate diversity

### Hybrid + Reranking (**Best Performing**)

* Cross-encoder selects best chunks
* Improves answer quality and grounding

### Multi-Query Retrieval (LLM-based Expansion)

* Generates multiple query variants
* Intended to improve recall

❗ Result: No improvement due to already high recall

---

## 📊 Best Configuration

| Metric               | Value    |
| -------------------- | -------- |
| Retrieval Hit Rate   | 0.86     |
| Accuracy             | 0.77     |
| Groundedness         | 0.50     |
| Top-1 Groundedness   | 0.45     |
| **LLM Accuracy**     | **0.91** |
| **LLM Groundedness** | **0.91** |

---

## 🔍 Key System Insight

```text
If recall is already high:
→ Improving retrieval inputs (multi-query) adds noise
→ Improving ranking (reranking, scoring) adds value
```

---

## 🧠 Evaluation Framework

### Heuristic Metrics

* Accuracy
* Retrieval Hit Rate
* Groundedness
* Top-1 Groundedness

### LLM-based Metrics

* LLM Accuracy
* LLM Groundedness

---

## 🎯 What This Demonstrates

* End-to-end RAG system design
* Retrieval vs ranking decomposition
* Hybrid retrieval (BM25 + dense)
* Cross-encoder reranking
* Hallucination detection
* LLM-as-judge evaluation
* Query expansion analysis
* **Understanding of when advanced techniques fail**

---

## 🚀 How to Run

```bash
pip install sentence-transformers faiss-cpu rank-bm25 openai python-dotenv
python main.py
```

---

## 🔧 Next Steps

* Score-weighted hybrid retrieval (BM25 vs dense)
* Larger / noisier datasets to stress recall
* Query difficulty benchmarking
