# Planning With Files API

A FastAPI service that answers natural-language questions about a folder of descriptively-named text files, using Claude as an agentic reasoning engine with tool-use.

---

## How It Works

The system operates in two stages:

1. **Ingestion** (one-time) — A PDF is split into per-page `.txt` files with NLP-generated descriptive filenames (`extract_pages.py`).
2. **Query** (runtime) — A REST API accepts a question and folder path, then runs an agentic Claude loop that autonomously discovers, scores, and reads relevant files using tool-use, returning the answer with source citations.

### Architecture

```
┌──────────┐         ┌──────────────┐         ┌─────────────────┐         ┌────────────┐
│  Client   │ ──────► │  FastAPI      │ ──────► │  Claude API     │ ──────► │ Filesystem │
│ (curl/app)│ ◄────── │  main.py      │ ◄────── │  (tool-use)     │ ◄────── │ (.txt files)│
└──────────┘  JSON    └──────────────┘  HTTP    └─────────────────┘  tools  └────────────┘
```

### Request Flow

```
POST /query { "query": "...", "folder_path": "..." }
  │
  ├─ 1. Pydantic validates request (non-empty query and folder_path)
  ├─ 2. Checks folder exists and is a directory (400 if not)
  │
  ├─ 3. ask_claude(query, folder_path) — enters agentic loop
  │       │
  │       ├─ Sends query + 3 tool definitions + system prompt to Claude
  │       │
  │       ├─ AGENTIC LOOP (max 5 rounds):
  │       │   ├─ Claude calls find_files(folder, search_terms)
  │       │   │   → scores files by filename metadata + content matching
  │       │   │
  │       │   ├─ Claude calls search_files(folder, term)
  │       │   │   → grep-style content search with context snippets
  │       │   │
  │       │   ├─ Claude calls read_files([path1, path2, ...])
  │       │   │   → batch reads up to 10 files (5000 chars each)
  │       │   │
  │       │   └─ Repeat until Claude has enough info or limit reached
  │       │
  │       ├─ Claude returns final answer with source citations
  │       └─ Sources tracked from every read_files/read_file call
  │
  └─ 4. Return { "answer": "...", "sources": [...], "stats": {...} }
```

---

## Components

### `api/main.py` — FastAPI Application

The HTTP entry point. Defines request/response models with Pydantic validation and delegates all query logic to the agentic `ask_claude()` function.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/query` | Main query endpoint — runs agentic tool-use loop |

**`POST /query` request:**

```json
{
  "query": "What overtime did Ariana work?",
  "folder_path": "/data/matheson_deposition_pages"
}
```

**`POST /query` response:**

```json
{
  "answer": "According to page 77, Ariana worked...",
  "sources": [
    "page_077_ariana_arellano_molly_chuen_overtime_employee.txt"
  ],
  "stats": {
    "api_calls": 3,
    "tool_rounds": 2,
    "tool_calls": {"find_files": 1, "read_files": 1},
    "input_tokens": 4200,
    "output_tokens": 850,
    "total_tokens": 5050
  }
}
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| `400` | `folder_path` does not exist or is not a directory |
| `422` | Pydantic validation error (empty `query` or `folder_path`) |
| `500` | `ANTHROPIC_API_KEY` missing; unexpected internal error |
| `502` | Claude API error (authentication, rate limit, timeout) |

---

### `api/claude_client.py` — Agentic Tool-Use Loop

The core engine. Runs a multi-turn conversation with Claude where Claude autonomously calls tools to discover and read files.

**Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Claude model used for queries |
| `DEFAULT_MAX_TOKENS` | `4096` | Max response tokens per API call |
| `MAX_TOOL_ROUNDS` | `5` | Safety limit on tool-use iterations |

**How the loop works:**

1. User query is formatted as `"Query: {query}\n\nDocument folder: {folder_path}"` and sent to Claude with the three tool definitions and a system prompt.
2. If Claude responds with `stop_reason: "tool_use"`, each tool call in the response is executed locally via `execute_tool()`. Results are appended back as `tool_result` messages.
3. The loop repeats (step 1-2) until Claude returns `stop_reason: "end_turn"` (final answer) or the 5-round safety limit is reached.
4. If the limit is hit, a final prompt forces Claude to answer with whatever information it has gathered, with tools disabled.
5. **Source tracking**: Every file path passed to `read_files` or `read_file` is recorded in a `sources` list (deduplicated by filename).
6. **Stats tracking**: A `QueryStats` dataclass tracks API calls, tool rounds, per-tool call counts, and input/output token usage.

**System prompt strategy:**

The system prompt instructs Claude to:
- Act as a research assistant answering from documents only
- Use `find_files` first with broad search terms (names, topics, synonyms)
- Use `read_files` to batch-read top-scoring results
- Answer based only on document content, citing filenames
- Fall back to `search_files` for exact phrase matching if `find_files` misses

**Error handling:**

| Exception | Maps to | Trigger |
|-----------|---------|---------|
| `APIKeyMissingError` | HTTP 500 | `ANTHROPIC_API_KEY` env var empty or missing |
| `ClaudeAPIError` | HTTP 502 | Authentication failure, rate limit, timeout, or other API error |

---

### `api/tools.py` — Tool Definitions and Handlers

Defines three tools that Claude can call during its agentic loop. Implements the **progressive disclosure** principle: score filenames (cheap) before reading content (expensive).

**Tools:**

| Tool | Purpose | Cost |
|------|---------|------|
| `find_files` | Score and rank files against search terms using weighted filename + content matching | Medium (reads first 5000 chars of each file for grep scoring) |
| `search_files` | Grep-style content search — find files containing a specific term with context snippets | Medium (scans all .txt files) |
| `read_files` | Batch-read full content of up to 10 files (5000 chars per file) | High (full content loaded) |

**`find_files` — smart file discovery:**

Claude provides search terms; the server scores files deterministically:

```
For each .txt file in folder:
    filename_tokens = tokenize filename (strip page_NNN prefix, split on _)
    glob_score = count of search terms matching filename tokens
    grep_score = count of search terms found in file content (first 5000 chars)
    combined = glob_score * 2 + grep_score
                          ^^^
                  Filename matches weighted 2x because NLP-curated
                  names are high-confidence metadata signals

Sort by combined score descending, return top N results
Fallback: if no file scores > 0, return first N files alphabetically
```

**`search_files` — grep fallback:**

Searches all `.txt` files for a term (case-insensitive). Returns matching filenames with the first matching line as a context snippet (200 chars max). Used when `find_files` doesn't surface the right documents.

**`read_files` — batch content retrieval:**

Reads up to 10 files at once, each truncated at 5000 characters. Returns content sections separated by `---` dividers with filename headers.

**Tool dispatch:**

All tools are registered in a `TOOL_MAP` dictionary and dispatched via `execute_tool(name, input)`. Unknown tool names return an error string (never raises).

---

### `api/file_utils.py` — File Scoring Engine

The deterministic scoring backend used by `find_files`. Implements the Manus context-engineering principle of using glob (filename pattern matching) and grep (content search) for progressive disclosure.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `list_text_files(folder)` | Returns sorted list of all `.txt` paths in a directory |
| `filter_files_by_query(files, query, max_files=10)` | Scores and ranks files by keyword overlap |

**Scoring algorithm detail:**

1. **Query tokenization** (`_tokenize_query`): Split on whitespace/punctuation, lowercase, drop numeric-only tokens.
2. **Filename tokenization** (`_tokenize_filename`): Strip `.txt` extension, remove `page_NNN_` prefix, split on `_`, lowercase, drop numeric tokens.
3. **Glob score**: `len(query_terms & filename_tokens)` — set intersection.
4. **Grep score**: Count of query terms found (case-insensitive substring) in first 5000 chars of file content.
5. **Combined**: `glob_score * 2 + grep_score`.
6. **Sort**: By combined score descending, filename ascending for ties.
7. **Fallback**: If all scores are 0, return first `max_files` in alphabetical order.

**Why filename matches are weighted 2x:**

Filenames follow a convention where NLP-extracted entities and topics are encoded directly:
```
page_077_ariana_arellano_molly_chuen_overtime_employee.txt
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         NER entities (PERSON, ORG, GPE, LAW) + topic nouns
```

A filename match is high-confidence because these tokens were curated by spaCy NER, not arbitrary text.

---

### `extract_pages.py` — PDF Ingestion Pipeline

A one-time preprocessing script that converts a PDF into individually-named `.txt` files with NLP-based descriptive filenames.

**Process:**

1. Opens PDF with **pdfplumber** for native text extraction.
2. If fewer than 50% of pages yield text, falls back to **Tesseract OCR** via `pdf2image` (converts pages to 300 DPI images).
3. Runs **spaCy NER** (`en_core_web_sm`) on each page's text to extract:
   - Up to 4 named entities: `PERSON`, `ORG`, `GPE`, `LAW`
   - Up to 3 topic keywords: most frequent non-stop-word nouns
4. Writes each page to `page_NNN_<entities>_<topics>.txt`.

**Filename convention:**

```
page_077_ariana_arellano_molly_chuen_overtime_employee.txt
│        │                           │
│        └─ NER entities (people,    └─ Topic nouns (frequent
│           orgs, places, laws)         non-stop-word nouns)
└─ Zero-padded page number
```

Max filename length: 120 characters. Tokens are sanitized (alphanumeric + underscore only, lowercased).

**Run it:**

```bash
uv run --with pdfplumber --with spacy --with pytesseract --with pdf2image --with Pillow \
    python extract_pages.py
```

---

### `api/serve.py` — Console Script Entry Point

A minimal wrapper that runs `uvicorn api.main:app` on `0.0.0.0:8000`. Registered as the `serve` console script in `pyproject.toml`.

---

## Setup

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

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

### Dependencies

| Package | Role |
|---------|------|
| `fastapi` | HTTP framework |
| `uvicorn[standard]` | ASGI server |
| `anthropic` | Anthropic Python SDK (Claude API) |
| `python-dotenv` | Load `.env` for local development |

Ingestion-only (not needed at runtime):

| Package | Role |
|---------|------|
| `pdfplumber` | PDF text extraction |
| `spacy` + `en_core_web_sm` | NER and NLP for filename generation |
| `pytesseract` / `pdf2image` / `Pillow` | OCR fallback for scanned PDFs |

---

## Running

### Option 1 — Console script

```bash
serve
# Server starts at http://localhost:8000
```

### Option 2 — Uvicorn with auto-reload

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3 — Docker Compose

```bash
# Build and start (reads ANTHROPIC_API_KEY from .env)
docker compose up --build
```

The Docker setup:
- Uses `python:3.12-slim` base image
- Installs dependencies via `uv` for speed
- Exposes port 8000
- Volume-mounts `matheson_deposition_pages/` to `/data/matheson_deposition_pages`
- Restarts automatically unless stopped

---

## Example Requests

### Local

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What overtime did Ariana work?",
    "folder_path": "/absolute/path/to/matheson_deposition_pages"
  }'
```

### Docker (using volume-mounted path)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What overtime did Ariana work?",
    "folder_path": "/data/matheson_deposition_pages"
  }'
```

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Interactive docs

FastAPI generates Swagger UI automatically:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Typical Tool-Use Sequence

A real query might look like this inside the agentic loop:

```
Round 1:  Claude calls find_files(folder, ["ariana", "overtime", "hours", "work"])
          → Returns ranked files: [score=5] page_077_ariana_..._overtime.txt, ...

Round 2:  Claude calls read_files([page_077_..., page_082_..., page_019_...])
          → Returns content of 3 files (5000 chars each max)

Round 3:  Claude returns final answer citing page_077 and page_082
          → stop_reason = "end_turn"
```

In more complex cases, Claude may call `search_files` as a fallback or run multiple rounds of `find_files` with different terms.

---

## Safety Mechanisms

| Mechanism | Value | Purpose |
|-----------|-------|---------|
| `MAX_TOOL_ROUNDS` | 5 | Prevents runaway loops and unbounded API costs |
| File char limit | 5000 | Bounds per-file context size in both tools and scoring |
| Batch read limit | 10 files | Caps the number of files read in a single `read_files` call |
| Graceful fallback | Final prompt | If max rounds hit, Claude answers with what it has (tools disabled) |
| Tool error handling | String returns | Tool errors never crash; Claude sees the error and adapts |
| Score fallback | First N files | If no files score > 0, returns first N alphabetically for baseline context |

---

## Project Structure

```
planning-with-files/
├── api/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI app, endpoints, Pydantic models
│   ├── claude_client.py     # Agentic tool-use loop, Claude API client
│   ├── tools.py             # Tool definitions + handlers (find/search/read)
│   ├── file_utils.py        # Glob+grep scoring engine
│   └── serve.py             # Console script entry point
├── tests/
│   └── test_api.py          # Tests for tools, agentic loop, and endpoints
├── extract_pages.py         # PDF → NLP-named .txt files (one-time ingestion)
├── pyproject.toml           # Package config and dependencies
├── Dockerfile               # Container image definition
├── docker-compose.yml       # Docker Compose service config
├── .env.example             # Template for environment variables
└── .dockerignore            # Files excluded from Docker build context
```
