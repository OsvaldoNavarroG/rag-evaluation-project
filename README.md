# RAG Retrieval Optimization & Evaluation

⏱️ Time to read: ~1 minute

---

## 🚀 Key Results

Built and analyzed a Retrieval-Augmented Generation (RAG) system with **multi-level evaluation**:

* **Retrieval Hit Rate:** up to **86%**
* **Groundedness (recall):** ~50%
* **Top-1 Groundedness (precision):** ~45%
* **Answer Accuracy:** up to **73%**

Identified:

* **Chunking as the primary driver of retrieval performance**
* **Prompting as critical for enforcing grounding**
* **Reranking impact depends on dataset difficulty**

---

## 📌 Overview

This project implements a RAG pipeline and focuses on:

* Separating **retrieval vs generation errors**
* Measuring **groundedness (hallucination detection)**
* Evaluating **ranking quality vs retrieval recall**
* Understanding when **reranking actually helps**

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

## 🧪 Evaluation Framework

### Metrics

* **Accuracy** → Is the answer correct?
* **Retrieval Hit Rate** → Was the correct chunk retrieved? (recall)
* **Groundedness** → Is the answer supported by *any* retrieved chunk?
* **Top-1 Groundedness** → Is the answer supported by the *top-ranked* chunk? (precision)

---

## 📊 Results

### Without Re-ranking

| Method            | Accuracy | Hit Rate | Grounded | Top-1 Grounded |
| ----------------- | -------- | -------- | -------- | -------------- |
| Naive Chunking    | 0.73     | 0.82     | 0.41     | 0.41           |
| Sentence Chunking | 0.68     | 0.86     | 0.50     | 0.45           |

---

### With Re-ranking

| Method            | Accuracy | Hit Rate | Grounded | Top-1 Grounded |
| ----------------- | -------- | -------- | -------- | -------------- |
| Naive Chunking    | 0.64     | 0.77     | 0.36     | 0.36           |
| Sentence Chunking | **0.73** | 0.86     | 0.45     | 0.45           |

---

## 🔍 Example: Ranking Failure

**Query:**
“What helps reduce overfitting in models?”

**Top retrieved chunk (incorrect ranking):**

```
Data preprocessing includes cleaning, normalizing...
Overfitting occurs when...
```

**Correct chunk (ranked lower):**

```
Regularization techniques help prevent overfitting...
```

→ Retrieval succeeded, but **ranking failed**
→ Demonstrates need for ranking-aware evaluation

---

## 🧠 Key Insights

### 1. Chunking Dominates Retrieval Performance

* Sentence-based chunking significantly improves hit rate
* Poor chunking can completely break retrieval

---

### 2. LLMs Can Mask Retrieval Failures

* High accuracy without grounding indicates reliance on prior knowledge
* Prompting is required to enforce context usage

---

### 3. Groundedness ≠ Accuracy

* Correct answers are not always supported by retrieved context
* Groundedness is essential to detect hallucinations

---

### 4. Reranking Improves Context Quality, Not Always Ranking Metrics

* Improved answer accuracy in some cases
* Did **not significantly improve top-1 grounding** in this dataset
* Indicates:

  > Reranking effectiveness depends on retrieval difficulty and data ambiguity

---

### 5. Precision vs Recall in Retrieval

* Hit rate measures **recall**
* Top-1 groundedness measures **ranking precision**
* Both are required for proper evaluation

---

## 🚀 How to Run

```bash
pip install sentence-transformers faiss-cpu openai python-dotenv
python main.py
```

---

## 🎯 What This Demonstrates

* End-to-end RAG system design
* Retrieval vs generation error decomposition
* Hallucination detection via groundedness
* Precision vs recall evaluation in retrieval
* Real-world limitations of reranking

---

## 🔧 Next Steps

* Increase dataset difficulty to better expose reranking gains
* LLM-as-judge evaluation for semantic groundedness
* Hybrid retrieval (BM25 + embeddings)
