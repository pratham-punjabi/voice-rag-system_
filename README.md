# 🎙️ VoiceRAG v2.0 — Groq API + ChromaDB

A production-ready voice-enabled RAG system with **live speech-to-text display**, **Groq LLM**, and **ChromaDB vector storage**.

---

## ✨ What's New in v2.0

| Feature | Before | After |
|---|---|---|
| Vector DB | FAISS (no persistence) | **ChromaDB (persistent)** |
| LLM backend | OpenAI-compat | **Groq API (primary)** |
| Data source | Dataset only | **API + Dataset + Hybrid** |
| Voice display | No transcript shown | **Live transcript appears as you speak** |
| STT fallback | None | **Browser Web Speech API** |
| Text input | None | **Textarea + Send button** |

---

## 🚀 Quick Start

### 1. Set up environment
```bash
cp .env .env.local    # or edit .env directly
# Add your Groq API key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### 2. Install & start backend
```bash
pip install -r requirements.txt
python -m backend.app.main
# → http://localhost:8000/docs
```

### 3. Install & start frontend
```bash
cd voice-rag-frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## ⚙️ Data Source Modes

Set `DATA_SOURCE_MODE` in `.env`:

| Mode | Description |
|---|---|
| `api` | Groq LLM with full knowledge — no dataset needed |
| `dataset` | ChromaDB vector store strictly |
| `hybrid` | ChromaDB first, Groq fallback (**recommended**) |

---

## 🎤 Voice Features

- **Browser Web Speech API** (free, no key needed) — live transcript displayed as you speak
- **Sarvam STT** (optional, set `SARVAM_API_KEY`) — high-accuracy server-side transcription
- Speaking → words appear live in the transcript box → auto-submitted when you click "Send"

---

## 📦 API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/query/text` | Text query |
| POST | `/api/voice/query` | Voice query (audio file) |
| POST | `/api/voice/transcript` | Browser STT transcript |
| POST | `/api/ingest/documents` | Ingest custom documents |
| POST | `/api/ingest/dataset` | Run HuggingFace ingestion |
| GET  | `/api/ingest/status` | ChromaDB index status |
| DELETE | `/api/ingest/collection` | Clear ChromaDB |
| GET  | `/api/health` | System health |

---

## 🔑 Environment Variables

```env
# Required
GROQ_API_KEY=your_groq_api_key_here
DATA_SOURCE_MODE=hybrid

# Optional — for server-side STT
SARVAM_API_KEY=your_sarvam_key_here

# Optional — for dataset mode
DATASET_NAME=ai4bharat/MSMARCO-XL
DATASET_MAX_DOCS=50000
```

---

## 🗄️ Adding Your Own Documents

### Via API
```bash
curl -X POST http://localhost:8000/api/ingest/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc1", "text": "Your document text here...", "title": "My Doc"},
      {"id": "doc2", "text": "Another document...", "title": "Doc 2"}
    ]
  }'
```

### Via HuggingFace Dataset
```bash
# Set in .env:
DATASET_NAME=your/dataset
DATA_SOURCE_MODE=dataset

# Then trigger ingestion:
curl -X POST http://localhost:8000/api/ingest/dataset
```

---

## 🏗️ Architecture

```
Voice/Text Input
     │
     ▼
[Browser Web Speech API] ──► Live Transcript (shown in UI)
     │                              │
     ▼                              ▼
[Sarvam STT (optional)]     [Auto-submit on stop]
     │                              │
     └──────────────────────────────┘
                    │
                    ▼
          FastAPI Orchestrator
                    │
          ┌─────────┴─────────┐
          │                   │
    [ChromaDB]           [Groq LLM]
    vector search         API RAG
          │                   │
          └─────────┬─────────┘
                    │
              [Reranker]
                    │
              [Generator]
                    │
              [Response]
```

---

## 🐳 Docker

```bash
docker-compose up --build
```

Exposes:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
