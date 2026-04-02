# Planning With Files API — Usage Guide

A FastAPI service that answers natural-language questions about a folder of
descriptively-named text files, using Claude as the reasoning engine.

---

## Overview

The system has two distinct stages:

1. **Ingestion** — A PDF is split into per-page `.txt` files whose names encode
   the entities and topics found on each page (`extract_pages.py`).
2. **Query** — A REST API accepts a question and a folder path, then delegates
   to an agentic Claude loop that autonomously searches and reads relevant files
   using tool-use, and returns the answer with source citations (`api/`).

---

## How It Works (end-to-end)

```
User query
    │
    ▼
POST /query  { "query": "...", "folder_path": "..." }
    │
    ├─ 1. Validate folder exists
    │
    ├─ 2. ask_claude()  — agentic tool-use loop
    │       │
    │       ├─ Claude receives query + 3 tools
    │       │
    │       ├─ Claude calls list_files(folder_path)
    │       │   → sees descriptive filenames (metadata)
    │       │
    │       ├─ Claude calls list_files(folder_path, "*overtime*")
    │       │   → narrows by filename pattern (glob)
    │       │
    │       ├─ Claude calls search_files(folder_path, "ariana")
    │       │   → finds files containing term (grep)
    │       │
    │       ├─ Claude calls read_file(path)
    │       │   → reads full content of relevant files
    │       │
    │       ├─ (repeats tool calls as needed, max 10 rounds)
    │       │
    │       └─ Claude returns final answer with citations
    │
    └─ 3. Return { "answer": "...", "sources": ["page_042_...", ...] }
```

---

## Key Components

### `extract_pages.py` — PDF ingestion (one-time setup)

Converts a PDF into a folder of individually-named `.txt` files.

**What it does:**
- Opens the PDF with **pdfplumber** for native text extraction.
- Falls back to **Tesseract OCR** (via pdf2image) for scanned/image-based pages.
- Runs **spaCy NER** (`en_core_web_sm`) on each page's text to find:
  - Named entities: `PERSON`, `ORG`, `GPE`, `LAW` (up to 4)
  - Top frequent nouns as topic keywords (up to 3)
- Writes each page to `matheson_deposition_pages/page_NNN_<entities>_<topics>.txt`

**Output filename convention:**
```
page_077_ariana_arellano_molly_chuen_overtime_employee.txt
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^
         NER entities (people, orgs, etc) Topic nouns
```

This naming scheme is what makes the agentic approach work well — Claude can
scan filenames as metadata (via `list_files`) to identify relevant pages
before reading any file contents.

**Run it:**
```bash
uv run --with pdfplumber --with spacy --with pytesseract --with pdf2image --with Pillow \
    python extract_pages.py
```

---

### `api/tools.py` — Tool definitions for the agentic loop (NEW)

Defines three tools that Claude can call during its agentic loop, implementing
the **Manus progressive disclosure** principle: glob (filenames as metadata)
→ grep (content search) → read (full content).

| Tool | Purpose |
|---|---|
| `list_files` | List `.txt` files in a directory, optionally filtered by a glob pattern (e.g., `*overtime*`). Filenames encode NLP-extracted entities and topics, so scanning names alone reveals which pages mention specific people, places, or subjects. |
| `search_files` | Search file contents for a term (case-insensitive, grep-style). Returns matching filenames with brief context snippets. |
| `read_file` | Read the full content of a specific file, truncated at 5 000 characters. |

Claude decides which tools to call and in what order. The system prompt guides
it toward the progressive disclosure strategy (list → search → read), but
Claude is free to adapt based on the query.

---

### `api/claude_client.py` — Agentic Claude loop

| Constant | Default |
|---|---|
| `DEFAULT_MODEL` | `claude-sonnet-4-20250514` |
| `DEFAULT_MAX_TOKENS` | `4096` |
| `MAX_TOOL_ROUNDS` | `10` |

The core function `ask_claude()` runs a **multi-turn tool-use conversation**:

1. The user query is sent to Claude along with the three tool definitions.
2. If Claude responds with `tool_use`, the tool is executed locally and the
   result is appended to the conversation.
3. The loop repeats until Claude returns a final text answer or the safety
   limit of **10 tool call rounds** is reached.
4. **Sources** are tracked automatically — every file read via `read_file`
   is added to the sources list.
5. A **system prompt** guides Claude to follow the progressive disclosure
   strategy: list files first, search contents next, read only what is needed.

**Error handling:**
- `APIKeyMissingError` — `ANTHROPIC_API_KEY` not set → HTTP 500
- `ClaudeAPIError` (auth, rate limit, timeout, general) → HTTP 502

---

### `api/file_utils.py` — File discovery (legacy)

This module provided deterministic keyword-based pre-filtering in the
original implementation. It is **no longer used by the `/query` endpoint** —
the agentic approach replaced the deterministic pre-filtering with Claude's
own tool calls. The module remains available for other uses.

| Function | Purpose |
|---|---|
| `list_text_files(folder)` | Returns sorted list of all `.txt` paths in a directory |
| `filter_files_by_query(files, query, max_files=10)` | Scores files by keyword overlap and returns top matches |

---

### `api/main.py` — FastAPI application

The `/query` endpoint validates the folder path and then delegates entirely
to the agentic `ask_claude()` loop. There is no pre-filtering step — Claude
discovers and reads files autonomously using its tools.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/query` | Main query endpoint — delegates to agentic loop |

**`POST /query` request body:**
```json
{
  "query": "What overtime did Ariana work?",
  "folder_path": "/data/matheson_deposition_pages"
}
```

**`POST /query` response:**
```json
{
  "answer": "According to page_077_ariana_arellano_..., Ariana worked...",
  "sources": [
    "page_077_ariana_arellano_molly_chuen_overtime_employee.txt",
    "page_082_ariana_arellano_schedule_hour.txt"
  ]
}
```

**How Claude decides what to search:**
Instead of a deterministic scoring formula, Claude dynamically decides what
to search for based on the query. It may list all files, filter by glob
pattern, grep for specific terms, or combine strategies — adapting its
approach to each question.

**Error responses:**

| Status | Condition |
|---|---|
| `400` | `folder_path` does not exist or is not a directory; empty `query` |
| `500` | `ANTHROPIC_API_KEY` missing; unexpected internal error |
| `502` | Claude API returned an error (auth, rate limit, timeout) |

---

## Setup

### Prerequisites
- Python 3.11+
- An Anthropic API key

### Install

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running

### Option 1 — Console script (local)

```bash
serve
# Server starts at http://localhost:8000
```

### Option 2 — Uvicorn directly (local, with auto-reload)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3 — Docker Compose

```bash
# Build and start (reads ANTHROPIC_API_KEY from .env automatically)
docker compose up --build

# The matheson_deposition_pages/ folder is volume-mounted into the container
# at /data/matheson_deposition_pages
```

The container exposes port `8000` and restarts automatically unless stopped.

---

## Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What overtime did Ariana work?",
    "folder_path": "/absolute/path/to/matheson_deposition_pages"
  }'
```

When running via Docker Compose, use the volume-mounted path:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What overtime did Ariana work?",
    "folder_path": "/data/matheson_deposition_pages"
  }'
```

---

## Interactive API Docs

FastAPI generates Swagger UI automatically:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Dependency Summary

| Package | Role |
|---|---|
| `fastapi` | HTTP framework |
| `uvicorn[standard]` | ASGI server |
| `anthropic` | Anthropic Python SDK (Claude API) |
| `python-dotenv` | Load `.env` for local development |
| `pdfplumber` | PDF text extraction (ingestion only) |
| `spacy` | NER and NLP for filename generation (ingestion only) |
| `pytesseract` / `pdf2image` | OCR fallback for scanned PDFs (ingestion only) |
