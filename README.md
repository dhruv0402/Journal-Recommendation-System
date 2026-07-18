# Journal Recommendation System

An AI-powered system that recommends academic journals for a research paper based on its title and abstract, using a multi-stage semantic search pipeline.

## How It Works

```
Title Input
    │
    ▼
Phase 0 — Abstract Duplicate Check (FAISS)
    │  If abstract already exists in dataset → reject
    ▼
Phase 1 — Title Duplicate Check (embedding + exact/fuzzy match)
    │  If title is a strong match → warn and reject
    ▼
Phase 2 — Abstract Semantic Matching
    │  FAISS search → scope reranker → learning reranker → final decision
    ▼
RAG Layer — Groq LLM generates a natural language explanation
    ▼
Recommendation: Primary journal + alternates + confidence score
```

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector search | FAISS (IVF flat index) |
| Reranker | Custom learned logistic reranker |
| LLM | Groq API (LLaMA-3.1-8b-instant) |
| API | FastAPI |
| Frontend | Streamlit |

## Setup

```bash
pip install -r requirements.txt

# Copy and fill in your Groq API key
cp .env.example .env
# Edit .env → set GROQ_API_KEY
```

## Run

### Backend (FastAPI)
```bash
uvicorn src.api.main:app --reload
```

### Frontend (Streamlit)
```bash
streamlit run streamlit_app.py
```

### CLI (interactive)
```bash
python -m src.main
```

## Run Tests
```bash
pytest tests/
```

## Rebuild FAISS Index (after dataset update)
```bash
python build_domain_map.py
python build_reranker_training_data.py
python train_reranker.py
```

## Dataset

- `data/master_journals_expanded.csv` — main dataset (~50MB, not committed to git)
- `data/faiss_index.bin` + `data/faiss_meta.pkl` — prebuilt FAISS index (auto-generated on first run)
