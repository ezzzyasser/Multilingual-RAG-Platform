

# Multilingual RAG Platform

An end-to-end **Retrieval-Augmented Generation (RAG)** system designed for high-performance, multilingual question answering. This project implements a scalable architecture capable of processing diverse knowledge domains using semantic search and Large Language Models (LLMs).

## 🚀 Project Overview

The platform leverages the **Natural Questions** dataset to provide context-aware answers. By combining **Sentence Transformers** for multilingual embeddings and **FAISS** for vector indexing, the system ensures efficient and accurate information retrieval across multiple languages.

## 🛠️ Technical Stack

* **Backend:** FastAPI (Async support) 


* **Vector Database:** FAISS 


* **Embeddings:** Sentence Transformers (Multilingual) 


* **LLM Integration:** Groq-based API 


* **Database:** SQLAlchemy (MySQL/SQLite) 


* **Data Science:** Pandas, NumPy, Scikit-learn 



---

## 📂 Implementation Phases

### Phase 1: Dataset & Foundation

* **Data Pre-processing:** Extraction of question-answer pairs from the Natural Questions dataset.


* **Chunking & Metadata:** Implementation of optimized text chunking with metadata enrichment (domain, difficulty, and question type).


* **Vectorization:** Multilingual embedding pipeline with a FAISS index for similarity search.



### Phase 2: Advanced RAG Features

* **Query Enhancement:** Normalization, query expansion, and multi-turn conversation context management.


* **Reranking:** Relevance scoring and ranking algorithms to refine top-K results.


* **Generation Logic:** Context-aware prompt engineering with fallback mechanisms for low-confidence scenarios.


* **Optimization:** Intelligent caching and batch processing to reduce latency.



### Phase 3: Production API & Evaluation

* **REST Endpoints:** * `POST /ask-question`: Primary Q&A interface.


* `POST /evaluate`: System performance assessment.


* `GET /health`: System status monitoring.




* **Analytics:** SQLAlchemy models to track conversation history and query performance.


* **Benchmarking:** Automated evaluation using Precision@K, Recall@K, BLEU, and ROUGE scores.



---

## 📊 Evaluation Results

The system is benchmarked on:

* **Retrieval Accuracy:** Precision and Recall at various K-levels.


* **Generation Quality:** Linguistic evaluation via ROUGE and BLEU metrics.


* **Performance:** Latency and throughput benchmarking for production readiness.



## 🔧 Installation & Setup
1. **Create Virtual Environment:**
```bash
python -m venv venv

```
```bash
venv\Scripts\Activate

```
2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure Environment:**
Add your `GROQ_API_KEY` and database credentials to a `.env` file.
4. **Run the API:**
```bash
uvicorn main:app --reload

```



