# Local PDF RAG Assistant

A simple Flask + vanilla JavaScript RAG project for asking questions from uploaded PDFs without using OpenAI or any external AI API. It extracts PDF text locally, builds a lightweight vector index locally, retrieves the most relevant chunks, and answers using a local Ollama model.

## 🎥 Demo Video
https://github.com/sanjeetthakur/askpdf/blob/main/Demo.mp4

## Features

- Upload and analyze PDF files from the browser
- Extract text with `pypdf`
- Chunk and index document text with a local TF-IDF vectorizer
- Store searchable indexes on disk
- Ask questions grounded in retrieved PDF passages
- Generate answers with a local Ollama model
- Retrieval-only fallback if Ollama is not running
- High-tech responsive UI built with plain HTML, CSS, and JavaScript

## Tech Stack

- Backend: Flask
- PDF parsing: pypdf
- Retrieval: local TF-IDF vector search
- Local LLM: Ollama (`llama3.2:3b` by default)
- Frontend: Vanilla HTML, CSS, JavaScript

## Project Structure

```text
.
├── app.py
├── rag/
│   ├── chunking.py
│   ├── document_store.py
│   ├── ollama_client.py
│   ├── pdf_reader.py
│   ├── rag_engine.py
│   └── vectorizer.py
├── static/
│   ├── css/styles.css
│   └── js/app.js
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

## Setup for Evaluators

Follow these steps to get the project running quickly.

### Prerequisites
- Python 3.8+
- Ollama installed: [https://ollama.com](https://ollama.com)
- ~2 GB disk space for the local model

### Quick Start (6 Steps)

**Step 1: Navigate to project directory**

Open terminal and cd to the project.

**Step 2: Create virtual environment**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Mac/Linux (Bash):
```bash
python -m venv .venv
source .venv/bin/activate
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**No API keys required** — everything runs locally.

**Step 4: Pull Ollama model**

```bash
ollama pull llama3.2:3b
```

This downloads the 3B parameter model (~2 GB).

**Step 5: Start Ollama server** (in a separate terminal)

```bash
ollama serve
```

*Windows users:* If you see a port binding error, Ollama is already running — skip this step.

**Step 6: Run Flask app**

In your project terminal (with `.venv` activated):

```bash
python app.py
```

Then open: **http://127.0.0.1:5000**

## Answer Latency Explanation

⏱️ **Why does answering take 5-15 seconds?**

Response time is dominated by **local LLM inference**, not network delays.

### Why It's Slow

1. **Local Model Processing** — The Ollama model runs on your CPU/GPU, generating text token-by-token. This is much slower than cloud APIs using specialized hardware.

2. **Retrieval is fast** (~100-300ms) — TF-IDF vectorization and chunk scoring are quick.

3. **LLM generation is slow** (~8-15 seconds) — The bottleneck. Generating 50-200 tokens at ~5-10 tokens/second is inherent to local inference.

4. **Model Size Trade-off** — `llama3.2:3b` (3 billion parameters) balances speed and quality. Smaller models are faster but less capable.

### Expected Timings

- **PDF upload + indexing:** 2-10 seconds
- **First question:** 8-15 seconds
- **Subsequent questions:** 8-15 seconds

### This is Normal

Local LLM inference without GPU is inherently slower than cloud APIs. For production, you could:
- Use a smaller quantized model (faster)
- Enable GPU acceleration (if available)
- Cache frequent questions
- Use a dedicated GPU server

**This design prioritizes privacy and local control over speed.**

## Configuration

Customize behavior with environment variables:

### Setting Environment Variables

**Windows (PowerShell):**
```powershell
$env:OLLAMA_MODEL="mistral:7b"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:RETRIEVAL_MODEL="local-tfidf"
$env:MAX_UPLOAD_MB="25"
$env:OLLAMA_TIMEOUT="120"
python app.py
```

**Mac/Linux (Bash):**
```bash
export OLLAMA_MODEL="mistral:7b"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export RETRIEVAL_MODEL="local-tfidf"
export MAX_UPLOAD_MB="25"
export OLLAMA_TIMEOUT="120"
python app.py
```

### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2:3b` | Model for answer generation (must be pulled in Ollama) |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server endpoint |
| `RETRIEVAL_MODEL` | `local-tfidf` | Retrieval strategy (currently only `local-tfidf` supported) |
| `MAX_UPLOAD_MB` | `25` | Maximum PDF file size in MB |
| `OLLAMA_TIMEOUT` | `120` | Timeout for LLM response in seconds |

### Model Selection Tips

- **Fast & Light:** `neural-chat:7b-q5` (7B quantized, ~4 GB, faster)
- **Balanced:** `llama3.2:3b` (3B, ~2 GB, good balance) — **Recommended for demo**
- **More Capable:** `llama2:13b` (13B, ~7 GB, slower but better answers)
- **Experimental:** `mistral:7b` (7B, ~4 GB, good for instructions)

Available models: [ollama.ai/library](https://ollama.ai/library)

## How It Works: Architecture & Technology Stack

### End-to-End Flow with Technology Details

#### 1. PDF Upload
- User selects a PDF file via browser (max 25 MB by default)
- **Technology:** HTML5 file input + multipart form POST via Flask

#### 2. Text Extraction (`rag/pdf_reader.py`)
- Uses `pypdf` library to extract selectable text from each PDF page
- **Technology:** `pypdf>=5.1` (pure Python PDF parser)
- **Output:** Raw text with page markers (e.g., `[Page 1]\ntext content`)

#### 3. Text Normalization & Chunking (`rag/chunking.py`)
- **Normalize:** Remove null bytes, collapse whitespace, limit line breaks
- **Chunk:** Split into overlapping segments (default: 900 char chunk, 180 char overlap)
- **Boundary-aware:** Prefers splits at sentence/paragraph breaks for semantic coherence
- **Technology:** Pure Python regex + string operations

#### 4. Vectorization (`rag/vectorizer.py` - Local TF-IDF)
- Each chunk is tokenized: lowercase, stopword removal, minimal preprocessing
- Computes **TF-IDF** (Term Frequency–Inverse Document Frequency) vectors
- Vectors normalized via L2 norm for cosine similarity
- **Technology:** `numpy>=2.1` for matrix operations; custom TF-IDF implementation (no external ML library)
- **Why TF-IDF?** Lightweight, deterministic, keyword-based retrieval without embedding models

#### 5. Index Storage (`rag/document_store.py`)
- Chunks, vectors, and vectorizer metadata persisted to disk
- In-memory cache for fast repeated queries on the same document
- **Technology:** Python `pickle` for serialization; caching in a dict
- **Storage:** `storage/indexes/` (ignored by Git)

#### 6. Question Answering
- Query vectorized using the same TF-IDF vectorizer
- Cosine similarity computed: `query_vector · chunk_vectors^T`
- Top-K (default 5) most similar chunks retrieved
- **Technology:** `numpy` dot product for fast similarity scoring

#### 7. Context Assembly & LLM Generation (`rag/ollama_client.py`)
- Retrieved chunks concatenated into a context prompt
- Prompt sent to Ollama via HTTP API: `POST /api/generate`
- **Local Ollama Model:** `llama3.2:3b` by default (3 billion parameters)
- Model generates answer using only PDF context
- **Technology:** `requests` library for HTTP; Ollama's REST API
- **No cloud calls:** Everything runs locally

#### 8. Response to Browser
- Answer, mode (local_llm or fallback), sources returned as JSON
- Frontend displays answer with ranked source chunks
- **Technology:** Vanilla JavaScript (no framework); dynamic DOM updates

### Architecture Diagram

```
User Browser
    |
    | upload PDF
    v
[Flask App] --> PDF extraction (pypdf)
    |
    v
[Text Chunking] --> Overlapping chunks
    |
    v
[TF-IDF Vectorizer] --> Dense vectors (numpy arrays)
    |
    v
[Document Store] --> Persist (pickle) + In-memory cache
    |
    |
    | ask question
    |
    v
[Vectorize Query] --> Same TF-IDF vectorizer
    |
    v
[Cosine Similarity] --> Score all chunks (numpy)
    |
    v
[Retrieve Top-5] --> Most similar chunks
    |
    v
[Ollama LLM] <-- HTTP POST with context
    |           (llama3.2:3b running locally)
    |
    v
[Generate Answer] --> LLM completion
    |
    v
Return JSON --> Browser displays answer + sources
```

### Why Local-First?

- **Privacy:** No data sent to external servers; all processing on your machine
- **No API Keys:** No dependency on cloud services or subscription costs
- **Offline:** Works without internet (after models are downloaded)
- **Transparency:** Full control over model behavior and data handling

### Fallback Mode

If Ollama is unavailable:
- App gracefully returns retrieval-only mode
- Shows most relevant chunks without LLM synthesis
- User sees raw evidence instead of synthesized answer

### Technology Stack Summary

| Component | Technology | Purpose |
|-----------|-----------|----------|
| Web Server | Flask 3.0+ | HTTP API & static file serving |
| PDF Parsing | pypdf 5.1+ | Extract text from PDFs |
| Vectorization | numpy 2.1+ | TF-IDF computation & similarity |
| LLM Backend | Ollama | Local model inference |
| HTTP Client | requests 2.32+ | Call Ollama API |
| Frontend | Vanilla JS/HTML/CSS | Browser UI (no framework) |
| Persistence | Python pickle | Serialize indexes to disk |

### Performance Characteristics

- **Indexing:** O(n*m) where n=chunks, m=vocabulary size (linear in document length)
- **Retrieval:** O(n) dot products (linear in chunk count)
- **Storage:** Dense numpy arrays + metadata (~50-500 KB per typical PDF)
- **Bottleneck:** LLM generation (most time spent here)

## Notes

- **API-Free:** This app does not call OpenAI, Anthropic, Gemini, or any hosted LLM API. Everything runs locally.
- **Text-Based PDFs Only:** Scanned image PDFs need OCR preprocessing first. This project focuses on PDFs with extractable text (use `pypdf`).
- **Performance Tips:** Use concise PDFs (<50 pages) and `llama3.2:3b` for best demo performance. Larger documents or models will be slower.
- **Storage:** Uploaded PDFs and generated indexes are stored in `storage/`, which is ignored by Git and can be safely deleted.
- **Disk I/O:** Indexes are pickled and cached in memory. First question on a new document loads from disk; subsequent questions use the cache (fast).
- **Privacy:** No telemetry, no logging of questions or answers to external services. All data stays on your machine.
- **GPU Acceleration:** If you have a CUDA-compatible GPU, Ollama will automatically use it for faster inference. Check Ollama documentation.
- **Threading:** Flask runs with `threaded=True` for concurrent request handling; long LLM calls won't block other operations.
