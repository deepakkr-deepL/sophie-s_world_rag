# sophie-s_world_rag

# 📘 Sophie's World – RAG (Retrieval Augmented Generation)

This project is a **Retrieval Augmented Generation (RAG)** system built on top of the novel **"Sophie's World" by Jostein Gaarder**.

The system allows users to ask questions about the book and get answers **strictly grounded in the novel text**, using vector search and an LLM.

---

## ✨ Features

* 📚 Uses **Sophie's World** as the single source of truth
* 🔎 Semantic search using **ChromaDB** (vector database)
* 🧠 Multiple retrieval strategies:

  * Similarity Retriever
  * MMR Retriever
  * Multi-Query Retriever
* 🧩 Custom chunking & preprocessing pipeline
* 🧪 Hallucination-controlled prompt design
* 🖥️ Interactive **Streamlit UI**
* 🧱 Clean `src/`-based project structure

---

## 🏗️ Project Architecture

```
sophie's_world_rag/
│
├── app.py                  # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── chains/serve_chain.py              # Chain factory (serve_rag_chain)
│   ├── chians/chains.py            # RAG chain definitions
│   ├── config/config.py            # Central configuration
│   ├── prompts/prompts.py           # Prompt templates
│   └── utils/utils.py             # Helper functions (format_docs etc.)
│
├── data/                    # Raw & preprocessed novel data
├── chroma_db/               # Chroma vector store
├── requirements.txt
└── README.md
```

---

## 🔄 RAG Pipeline Overview

1. **Data Preprocessing**

   * Raw novel text is cleaned
   * Split into overlapping chunks
   * Chunk size & overlap configurable

2. **Embedding Generation**

   * Each chunk converted to embeddings
   * Stored in **ChromaDB**

3. **Retrieval**

   * User query → embedding
   * Top-k relevant chunks retrieved
   * Optional MMR / Multi-query expansion

4. **Generation**

   * Retrieved chunks injected into prompt
   * LLM generates answer using ONLY provided context

---

## 🧠 Prompt Philosophy

The system prompt is designed to:

* Treat retrieved chunks as the **primary source of truth**
* Prevent use of external or general knowledge
* Explicitly refuse answers when context is insufficient

This helps minimize hallucinations and keeps answers grounded in the novel.

---

## ▶️ How to Run

### 1️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧪 Example Questions

* Who is Sophie?
* Why does Sophie receive mysterious letters?
* What does philosophy mean in the early chapters?

If the answer is not present in the retrieved excerpts, the system will clearly say so.

---

## ⚙️ Configuration

All major settings are controlled via `config.py`:

* Embedding model
* LLM model
* Chunk size & overlap
* Retriever strategy

This allows easy experimentation without touching core logic.

---

## 🚀 Future Improvements

* Source citations (chapter / chunk id)
* Chat-style multi-turn memory
* Hybrid retriever (BM25 + vector)
* RAG evaluation (RAGAS)
* Dockerized deployment

---

## 📌 Notes

* `.venv` is excluded via `.gitignore`
* Project follows production-style Python packaging
* Designed for learning, experimentation, and extension

---

## 🙌 Author

Built as a hands-on RAG learning project using **Sophie's World** as a knowledge base.

---

Happy exploring philosophy with RAG 🚀
