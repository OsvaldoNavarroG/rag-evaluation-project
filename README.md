# RAG Retrieval Optimization & Evaluation

⏱️ Time to read: ~1 minute

---

## 🚀 Key Results

Built and evaluated a Retrieval-Augmented Generation (RAG) system with **multi-layer evaluation**:

* **Retrieval Hit Rate:** up to **86%**
* **Heuristic Groundedness:** ~41%
* **LLM Groundedness:** **~91% (+50pp)**
* **Answer Accuracy (LLM-evaluated):** **~91%**

### 🔥 Key Finding

Heuristic (string-based) evaluation **severely underestimates performance**.
LLM-based evaluation reveals the system is **far more accurate and grounded than surface metrics suggest**.

---

## 📌 Overview

This project implements a RAG pipeline and focuses on:

* Separating **retrieval vs generation errors**
* Detecting hallucinations via **groundedness metrics**
* Comparing **heuristic vs semantic evaluation**
* Understanding when **reranking helps (or doesn’t)**

---

## 🧠 Pipeline

```text id="y0y5sk"
Documents → Chunking → Embeddings → FAISS Index
Query → Retrieval (Top-K) → Re-ranking → LLM → Answer
```

---

## ⚙️ Tech Stack

* FAISS — vector search
* sentence-transformers — embeddings + cross-encoder reranker
* OpenAI API — generation + LLM-based evaluation

---

## 🧪 Evaluation Framework

### Metrics

**Heuristic (lexical):**

* Accuracy
* Retrieval Hit Rate (recall)
* Groundedness (answer appears in any chunk)
* Top-1 Groundedness (ranking precision)

**LLM-based (semantic):**

* LLM Accuracy
* LLM Groundedness

---

## 📊 Results (Sentence Chunking)

| Metric                   | Value    |
| ------------------------ | -------- |
| Accuracy (heuristic)     | 0.73     |
| Retrieval Hit Rate       | 0.86     |
| Groundedness (heuristic) | 0.41     |
| Top-1 Groundedness       | 0.41     |
| **LLM Accuracy**         | **0.91** |
| **LLM Groundedness**     | **0.91** |

---

## 🔍 Evaluation Gap (Critical Insight)

```text id="g3y8tb"
Heuristic groundedness: 0.41
LLM groundedness:       0.91
```

→ **+50 percentage point difference**

### Why?

Heuristic evaluation fails on:

* paraphrases
* longer answers
* semantic equivalence

LLM evaluation captures:

* meaning
* context alignment
* true grounding

---

## 🔍 Example: Semantic Grounding

**Question:**
“What does reinforcement learning rely on?”

**Answer:**
“Reinforcement learning relies on an agent interacting with an environment and receiving rewards or penalties.”

**Context:**
“...agents learn through rewards and penalties...”

→ Heuristic: ❌ not grounded
→ LLM judge: ✅ grounded

---

## 🧠 Key Insights

### 1. Chunking Dominates Retrieval Performance

* Sentence-based chunking significantly improves hit rate
* Poor chunking breaks retrieval entirely

---

### 2. LLMs Can Mask Retrieval Failures

* High accuracy without grounding indicates reliance on prior knowledge
* Prompting is required to enforce context usage

---

### 3. Heuristic Metrics Are Misleading

* String matching underestimates performance
* Cannot capture paraphrasing or semantic equivalence

---

### 4. LLM-Based Evaluation Is Essential

* Captures true correctness and grounding
* Aligns with real-world GenAI evaluation practices

---

### 5. Reranking Depends on Data Difficulty

* Limited impact in low-ambiguity datasets
* Improves performance mainly when retrieval candidates are noisy

---

## 🚀 How to Run

```bash id="4qrb6q"
pip install sentence-transformers faiss-cpu openai python-dotenv
python main.py
```

---

## 🎯 What This Demonstrates

* End-to-end RAG system design
* Retrieval vs generation error decomposition
* Hallucination detection via groundedness
* Precision vs recall evaluation in retrieval
* **Semantic evaluation with LLM-as-judge**
* Real-world limitations of heuristic metrics

---

## 🔧 Next Steps

* Larger / more ambiguous datasets
* Hybrid retrieval (BM25 + embeddings)
* Multi-query retrieval
