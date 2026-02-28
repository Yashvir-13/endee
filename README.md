# 📚 Philosopher RAG

> **Grounded Question Answering over Schopenhauer's Texts using Endee Vector DB**

A production-quality RAG system that answers philosophical questions about Arthur Schopenhauer's *The World as Will and Representation* by retrieving relevant passages from a vector database and generating grounded responses via an LLM.

Built with **10 advanced RAG techniques** and powered by **Endee** — a high-performance open-source vector database with native hybrid search support.

---

## ✨ Features

| # | Feature | Description |
|:-:|---------|-------------|
| 1 | **Hybrid Search** | Dense (e5-large-v2, 1024-dim) + Sparse (TF-IDF) retrieval via Endee's native hybrid index |
| 2 | **Re-ranking** | LLM-based passage scoring for improved precision |
| 3 | **Evaluation Pipeline** | 10 curated questions with automated concept-hit-rate metrics |
| 4 | **Metadata Filtering** | Filter search results by volume using Endee's filter API |
| 5 | **Hierarchical Chunking** | Parent (3K chars) → Child (800 chars) for context-rich retrieval |
| 6 | **HyDE** | Hypothetical Document Embeddings for better recall on abstract queries |
| 7 | **Query Expansion** | Multi-query reformulation and result merging |
| 8 | **Agentic RAG** | Multi-step retrieval with automated quality assessment and query refinement |
| 9 | **Streaming Output** | Token-by-token LLM response for responsive UX |
| 10 | **Conversation Mode** | Multi-turn interactive chat with memory |

---

## 🏗️ Architecture

```
                        ┌──────────────────────────────┐
                        │    Documents/ (3 volumes)     │
                        └──────────────┬───────────────┘
                                       │
                              ┌────────▼────────┐
                              │  Hierarchical    │
                              │  Chunking        │
                              │  Parent → Child  │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
           ┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼──────┐
           │  Dense Embed   │  │ TF-IDF Sparse │  │  Metadata    │
           │ (e5-large-v2)  │  │   Encoding    │  │  Filters     │
           │   1024-dim     │  │   ~24K-dim    │  │  (volume)    │
           └───────┬────────┘  └───────┬───────┘  └───────┬──────┘
                   └───────────┬───────┘──────────────────┘
                               │
                      ┌────────▼────────┐
                      │   Endee DB       │
                      │  Hybrid HNSW     │
                      │  (cosine sim)    │
                      └────────┬─────────┘
                               │
     User Question ──► [HyDE] ──► [Expand] ──► Hybrid Search ──► [Re-rank]
                                                                     │
                                                              ┌──────▼──────┐
                                                              │  LLM (Llama3│
                                                              │  via Ollama)│
                                                              │  + Sources  │
                                                              └─────────────┘
```

### Why Endee?

This project demonstrates deep usage of Endee's capabilities:
- **Hybrid Index** — Dense + sparse vectors in a single index for combined semantic and lexical search
- **Metadata Filtering** — Targeted retrieval by volume using Endee's filter API
- **Batch Upsert** — Efficient batch vector operations via the Python SDK
- **HNSW Search** — Fast approximate nearest neighbor search with cosine similarity

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Purpose |
|-------------|---------|
| Python 3.10+ | Runtime |
| [Endee](https://github.com/endee-ai/endee) | Vector database |
| [Ollama](https://ollama.ai) | LLM inference (Llama3) |

### 1. Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Pull the LLM Model

```bash
ollama pull llama3
```

### 3. Start the Endee Server

```bash
cd endee_fork/endee
./run.sh
```

### 4. Configure (Optional)

```bash
cp .env.example .env
# Edit .env to customize:
#   ENDEE_URL        — Endee server URL (default: http://localhost:8080/api/v1)
#   OLLAMA_MODEL     — LLM model name (default: llama3)
#   EMBEDDING_MODEL  — HuggingFace model ID or local path
#                      (default: intfloat/e5-large-v2, auto-downloaded on first run)
```

### 5. Index the Documents

```bash
python run.py ingest
```

This runs the full ingestion pipeline:
- Loads 3 volumes of Schopenhauer's text
- Creates hierarchical chunks (parent → child)
- Generates dense embeddings (e5-large-v2, 1024-dim)
- Builds TF-IDF sparse vectors
- Stores everything in Endee's hybrid index with metadata filters

### 6. Ask Questions

```bash
python run.py query "Why does Schopenhauer believe suffering is fundamental to life?"
```

---

## 📖 Usage Guide

### Basic Query

```bash
python run.py query "What is the role of art in Schopenhauer's philosophy?"
```

### Advanced Query Flags

| Flag | Feature | What it Does |
|------|---------|--------------|
| `--hyde` | HyDE | Generates a hypothetical answer, then embeds *that* for better recall |
| `--expand` | Query Expansion | Creates 3 alternative phrasings, searches all, merges results |
| `--rerank` | Re-ranking | LLM scores each passage 1-10 and re-sorts by relevance |
| `--stream` | Streaming | Prints the LLM response token-by-token |
| `--filter vol1` | Metadata Filter | Restricts search to a specific volume (vol1, vol2, vol3) |
| `--all` | Everything | Enables HyDE + expand + rerank + stream together |
| `--top_k N` | Result Count | Number of passages to retrieve (default: 5) |

```bash
# Use all advanced features at once
python run.py query "What is the will?" --all

# Filter to Volume 1 only
python run.py query "What does Schopenhauer say about music?" --filter vol1

# Just re-rank and stream
python run.py query "What is the denial of the will to live?" --rerank --stream
```

### Conversation Mode

Interactive multi-turn chat with context memory. The assistant remembers your previous questions.

```bash
python run.py chat
```

```
  You: What does Schopenhauer think about art?
  Assistant: According to Schopenhauer, art serves as a temporary escape from...

  You: How does music differ from other art forms?
  Assistant: Building on what we discussed, Schopenhauer gives music a special status...

  You: quit
```

### Agentic Mode

Multi-step reasoning with automatic quality assessment. The agent evaluates whether retrieved passages are sufficient, and if not, reformulates the query and retrieves again (up to 3 iterations).

```bash
python run.py agent "What is the relationship between will, suffering, and salvation?"
```

### Evaluation

Runs 10 curated philosophical questions and measures concept-hit-rate in retrieved passages.

```bash
python run.py evaluate
```

```
  EVALUATION REPORT
  Questions:              10
  Passing (≥60% hits):    8/10
  Overall concept hit:    42/50 (84%)
  Avg retrieval time:     0.15s
```

---

## 🧠 How It Works

### Ingestion Pipeline

```
Documents/*.txt → Hierarchical Chunking → Dense Embedding → TF-IDF Sparse → Endee Hybrid Index
```

1. **Load** — Reads plain-text files from `Documents/`
2. **Hierarchical Chunk** — Splits into large parent chunks (~3000 chars) then small child chunks (~800 chars). Children are stored as vectors; parents are preserved in metadata for LLM context
3. **Dense Embed** — `intfloat/e5-large-v2` with `"passage: "` prefix (1024-dim)
4. **Sparse Encode** — TF-IDF vectorizer fit on the full corpus (vocabulary saved to `data/tfidf_vectorizer.pkl`)
5. **Store** — Upserted into Endee with dense vector, sparse vector, metadata, and volume filter

### Query Pipeline

```
Question → [HyDE] → [Expand] → Embed → Hybrid Search (dense+sparse) → [Filter] → [Re-rank] → LLM → Answer
```

1. **HyDE** *(optional)* — LLM generates a hypothetical passage; that passage is embedded instead of the raw question
2. **Query Expansion** *(optional)* — LLM generates 3 alternative phrasings; all are searched and results merged
3. **Embed** — `"query: "` prefix for asymmetric E5 retrieval
4. **Hybrid Search** — Endee combines dense cosine similarity with sparse TF-IDF matching
5. **Filter** *(optional)* — Endee filters results by volume metadata
6. **Re-rank** *(optional)* — LLM scores each passage's relevance 1-10
7. **Generate** — Grounded answer using retrieved parent context, with source citations

### Agentic Pipeline

```
Question → Retrieve → Re-rank → Assess sufficiency → [Refine query → Retrieve again] → Answer
```

The agent loops up to 3 times, accumulating unique passages across iterations.

---

## 📁 Project Structure

```
.
├── Documents/                # Schopenhauer text files (3 volumes)
├── src/
│   ├── __init__.py
│   ├── ingest.py             # Ingestion: hierarchical chunk → embed → sparse → store
│   ├── query.py              # Query: HyDE, expand, hybrid search, rerank, stream, chat
│   ├── agent.py              # Agentic multi-step RAG
│   ├── evaluate.py           # Evaluation pipeline with concept-hit metrics
│   ├── embeddings.py         # Embedding abstraction (e5-large-v2)
│   ├── endee_client.py       # Endee SDK wrapper (hybrid search + filtering)
│   ├── sparse.py             # TF-IDF sparse encoder for hybrid search
│   ├── reranker.py           # LLM-based re-ranking via Ollama
│   └── utils.py              # Document loading + hierarchical chunking
├── data/
│   └── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer (generated during ingest)
├── eval_questions.json       # 10 curated evaluation questions
├── run.py                    # CLI entry point
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README.md
```

---

## 🔧 Configuration

All settings are configurable via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENDEE_URL` | `http://localhost:8080/api/v1` | Endee server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model for generation |
| `EMBEDDING_MODEL` | `intfloat/e5-large-v2` | HuggingFace model ID or local path to embedding model |

### Embedding Model

The default embedding model (`intfloat/e5-large-v2`) is automatically downloaded from HuggingFace on first run (~1.3 GB). If you prefer a different model or have a local copy:

```bash
# Use a different HuggingFace model
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Use a local model directory
export EMBEDDING_MODEL=/path/to/your/local/model
```

> **Note:** If using a non-E5 model, the `"query: "` / `"passage: "` prefixes may not be optimal. E5 models are specifically trained for this asymmetric retrieval pattern.

---

## 📊 Evaluation Details

The evaluation suite (`eval_questions.json`) contains 10 questions spanning key Schopenhauer topics:

- Will and representation
- Suffering and pessimism
- Art, music, and aesthetics
- Ethics and compassion
- Free will and determinism
- Platonic Ideas
- Time, space, and principium individuationis

Each question includes expected concepts. The pipeline checks whether retrieved passages contain these concepts, measuring **concept hit rate** as a proxy for retrieval quality.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Database | [Endee](https://github.com/endee-ai/endee) (hybrid HNSW) |
| Dense Embeddings | intfloat/e5-large-v2 via sentence-transformers |
| Sparse Encoding | TF-IDF via scikit-learn |
| LLM | Llama3 via Ollama |
| Language | Python 3.10+ |
