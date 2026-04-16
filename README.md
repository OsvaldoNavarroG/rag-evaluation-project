# RAG Retrieval Optimization & Evaluation

⏱️ Time to read: ~1 minute

---

## 🚀 Key Results

Evaluated multiple RAG configurations with **heuristic and LLM-based metrics**:

* **Retrieval Hit Rate:** ~86% (stable across methods)
* **LLM Accuracy:** up to **91%**
* **LLM Groundedness:** up to **91%**
* **Groundedness (heuristic):** ~41–50%

### 🔥 Key Findings

1. **Evaluation gap:**
   Heuristic groundedness (~0.41) vs LLM groundedness (~0.91)
   → **+50pp difference due to semantic evaluation**

2. **Hybrid retrieval improves ranking, not recall:**

   * No change in hit rate (~0.86)
   * Significant improvement in answer quality and grounding

3. **Best configuration:**
   → **Hybrid retrieval + reranking**

---

## 📌 Overview

This project builds and evaluates a RAG pipeline focusing on:

* Retrieval vs generation error analysis
* Hallucination detection via groundedness
* Heuristic vs semantic evaluation
* Retrieval strategy comparison (dense vs hybrid)

---

## 🧠 Pipeline

```text id="xg3m6z"
Documents → Chunking → Embeddings → FAISS
Query → Retrieval → (Hybrid) → Reranking → LLM → Answer
```

---

## ⚙️ Tech Stack

* FAISS — dense vector retrieval
* BM25 (rank-bm25) — lexical retrieval
* sentence-transformers — embeddings + reranker
* OpenAI API — generation + LLM-based evaluation

---

## 🧪 Retrieval Strategies Compared

### 1. Dense Retrieval

* Semantic similarity (embeddings)
* Strong recall
* Weaker keyword precision

---

### 2. Hybrid Retrieval (BM25 + Dense)

* Combines:

  * semantic matching (dense)
  * keyword matching (BM25)
* Adds candidate diversity

---

### 3. Hybrid + Reranking (**Best**)

* Cross-encoder selects best chunks
* Improves ranking quality and grounding

---

## 📊 Results (Best Configuration)

| Metric               | Value    |
| -------------------- | -------- |
| Retrieval Hit Rate   | 0.86     |
| Accuracy             | 0.82     |
| Groundedness         | 0.45     |
| Top-1 Groundedness   | 0.45     |
| **LLM Accuracy**     | **0.91** |
| **LLM Groundedness** | **0.91** |

---

## 🔍 Key Insight: Ranking vs Recall

```text id="o0o9e4"
Dense only:
  High recall, moderate ranking quality

Hybrid:
  Same recall, more diverse candidates

Hybrid + rerank:
  Best ranking → best answers
```

### Conclusion

Hybrid retrieval did **not** improve recall but improved:

* answer accuracy
* groundedness
* top-ranked chunk quality

---

## 🔍 Why Hybrid Helps

Dense retrieval:

* captures meaning
* misses exact terms

BM25:

* captures keywords
* misses semantics

👉 Hybrid combines both, giving the reranker **better candidates to choose from**

---

## 🔍 Chunking Interaction

* Sentence chunking → strong baseline
* Naive chunking → benefits more from hybrid

### Insight

> Hybrid retrieval is most useful when baseline retrieval quality is suboptimal.

---

## 🧠 Evaluation Framework

### Heuristic (lexical)

* Accuracy
* Retrieval Hit Rate
* Groundedness
* Top-1 Groundedness

### LLM-based (semantic)

* LLM Accuracy
* LLM Groundedness

---

## 🔍 Evaluation Gap

```text id="6u6s5h"
Heuristic groundedness: ~0.41
LLM groundedness:       ~0.91
```

→ Traditional metrics underestimate performance due to paraphrasing

---

## 🎯 What This Demonstrates

* End-to-end RAG system design
* Retrieval vs ranking decomposition
* Hybrid retrieval (BM25 + dense)
* Cross-encoder reranking
* Hallucination detection
* **Semantic evaluation with LLM-as-judge**
* Real-world limitations of heuristic metrics

---

## 🚀 How to Run

```bash id="5i2m3g"
pip install sentence-transformers faiss-cpu rank-bm25 openai python-dotenv
python main.py
```

---

## 🔧 Next Steps

* Query expansion / multi-query retrieval
* Larger / more ambiguous datasets
* Hybrid score weighting (BM25 vs dense)
