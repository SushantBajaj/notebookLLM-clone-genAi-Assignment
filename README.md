# NoteBook LLM Clone

A small NotebookLM-style assignment project: upload documents, let the backend parse and index them, then ask questions grounded in those sources.

The app has a plain HTML/CSS/JavaScript frontend and a FastAPI backend. The backend extracts text from PDF, DOC, DOCX, or CSV files, chunks the text, stores a local FAISS index, runs a basic CRAG retrieval loop, and asks Gemini to answer using only the final selected context.

## What It Does

- Uploads multiple source documents.
- Lets users choose the current chat scope with checkboxes in the sidebar.
- Lets users inspect document chunks from the sidebar.
- Shows the exact retrieved source chunks behind each answer.
- Runs a basic CRAG with query rewriting, retrieval grading, and one corrective retrieval branch.
- Supports PDF, DOC, DOCX, and CSV files.
- Extracts readable text from the uploaded file.
- Splits the text into semantic chunks for retrieval.
- Builds a local FAISS vector index using local HuggingFace sentence embeddings.
- Sends only the most relevant chunks across uploaded documents to Gemini during chat.
- Renders assistant Markdown in the frontend, including bold text, lists, links, and code blocks.
- Adds storage checks for Railway-style limited volume storage.

## Frontend Quality-of-Life Features

- The sidebar keeps long filenames, metadata, document actions, and chunk controls contained in compact source cards.
- Each uploaded source has an `Inspect` control for browsing chunk numbers and previewing one chunk at a time.
- Assistant answers can show their retrieved sources without dumping all source text at once.
- Source cards inside answers show the final selected context only, capped at 12 chunks, with retrieval path metadata for how each chunk was found.
- User messages include a quiet `Info` control that can reveal the retrieval grade, corrective retry status, final source count, grader rationale, and rewritten query.
- In the chat composer, `Shift + Up` recalls previously sent prompts and `Shift + Down` moves forward through that history back toward the current draft.

## Project Structure

```text
.
├── app.js                     # Frontend behavior and API calls
├── index.html                 # Main UI
├── styles.css                 # App styling
├── railway.json               # Railway deployment config
├── requirements.txt           # Root dependency file for Railway
└── backend/
    ├── main.py                # FastAPI routes: health, upload, chat
    ├── document_parser.py     # PDF/DOC/DOCX/CSV text extraction
    ├── rag_pipeline.py        # Chunking, embeddings, FAISS indexing/retrieval
    ├── llm_chat.py            # CRAG loop, prompts, final answer response flow
    ├── gemini_gateway.py      # Gemini API wrapper, key rotation, timeouts
    ├── requirements.txt       # Backend dependency list
    └── data/                  # Uploaded files, metadata, extracted text, indexes
```

## Local Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from the example:

```bash
cp .env.example .env
```

Then add your Gemini key:

```env
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-flash-latest
```

Start the backend:

```bash
uvicorn backend.main:app --reload
```

Open `index.html` in the browser, or serve the static files with any simple static server.

## Deployment Notes

The backend is configured for Railway through `railway.json`:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Railway needs the app to listen on `$PORT`, so keep that start command.

The frontend currently calls the deployed Railway backend from `app.js`:

```js
const API_BASE_URL = "https://notebookllm-clone-genai-assignment-production-aaf3.up.railway.app";
```

If you deploy your own Railway service, replace that URL with your generated backend domain. For local-only development, use:

```js
const API_BASE_URL = "http://127.0.0.1:8000";
```

## Environment Variables

```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional comma-separated pool. If set, the backend rotates to the next key on 429/503.
GEMINI_API_KEYS=your_first_key_here,your_second_key_here
GEMINI_MODEL=gemini-flash-latest
CRAG_MODEL=gemini-2.5-flash-lite
CRAG_TIMEOUT_SECONDS=20
GEMINI_TIMEOUT_SECONDS=90

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
CHUNKING_STRATEGY=semantic
SEMANTIC_BREAKPOINT_THRESHOLD_TYPE=percentile
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT=90
RETRIEVAL_STRATEGY=similarity

# Railway free volume limit: 0.5 GB = 536870912 bytes.
STORAGE_LIMIT_BYTES=536870912
STORAGE_SAFETY_MARGIN_BYTES=33554432
VECTOR_METADATA_BYTES_PER_CHUNK=2048
```

The storage settings are intentionally visible because generated files can be much larger than the uploaded file. A compressed PDF may upload as a small file but expand into much more extracted text and index data.

`GEMINI_API_KEYS` is optional. When multiple keys are configured, the backend keeps using the current key until Gemini returns a retryable `429` or `503`, then switches to the next key in a circular pool. Logs include the current step, model, and key index, but never print the key itself.

`GEMINI_MODEL` controls the final answer model and keeps `gemini-flash-latest` as the preferred default. `CRAG_MODEL` is separate and is used only for query rewriting and retrieval grading. `CRAG_TIMEOUT_SECONDS` prevents rewrite/grading calls from hanging the chat; if grading times out or fails, the backend falls back to a weak grade when chunks exist so the corrective branch still runs. `GEMINI_TIMEOUT_SECONDS` is the default timeout for other Gemini generation calls.

## RAG Strategy

The RAG pipeline uses LangChain's `SemanticChunker` in [backend/rag_pipeline.py](backend/rag_pipeline.py). The current settings are:

```python
CHUNKING_STRATEGY = "semantic"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
SEMANTIC_BREAKPOINT_THRESHOLD_TYPE = "percentile"
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT = 90
RETRIEVAL_K = 12
```

Semantic chunking embeds sentence groups and starts a new chunk when adjacent text becomes meaningfully different. The percentile threshold keeps splits focused on stronger topic shifts instead of fixed character counts.

At chat time, the backend runs a basic CRAG loop before final answer generation:

1. Rewrite the user's query for semantic retrieval.
2. Retrieve the initial top 12 chunks with the rewritten query.
3. Grade whether that context is good, weak, or bad for the original query.
4. If the grade is weak or bad, retry retrieval with both the original and rewritten query.
5. Deduplicate by document id and chunk index, then send only the final top 12 chunks to Gemini.

This is a single corrective pass, not an unbounded retry loop. If the first retrieval grade is `good`, the initial rewritten-query results are used. If the grade is `weak` or `bad`, the backend retrieves once with both the original and rewritten query, merges those candidates, deduplicates them, and sends only the final selected top 12 chunks to Gemini.

The final answer model still uses `GEMINI_MODEL` and its existing fallback list. `CRAG_MODEL` is only for query rewriting and retrieval grading.

The `/chat` response includes CRAG metadata for the frontend `Info` control:

- `rewritten_query`
- `retrieval_grade`
- `retrieval_rationale`
- `corrective_retry`
- `final_source_count`

Each returned source can also include `retrieval_paths`, such as `initial_rewritten_query`, `retry_original_query`, or `retry_rewritten_query`.

## Storage Behavior

Uploaded and generated files are stored under:

```text
backend/data/documents/
```

For every upload, the backend stores:

- the original uploaded file
- `metadata.json`
- `extracted_text.txt`
- `chunks.json`
- a FAISS index folder

Before building the vector index, the backend estimates storage from the actual parsed text and chunk count. If the file would exceed the configured limit, the upload is removed and the user gets a clear error instead of a vague failure.

Check local disk usage with:

```bash
du -sh backend/data
```

## API Endpoints

```text
GET  /health
POST /upload
DELETE /documents/{document_id}
GET  /documents/{document_id}/chunks
POST /chat
```

`/upload` expects the file body directly and reads the original filename from the `X-Filename` header.

`/documents/{document_id}` removes the uploaded file, extracted text, metadata, and FAISS index for that document.

`/documents/{document_id}/chunks` returns the chunk numbers and text for a processed document, which powers the sidebar chunk inspector.

`/chat` accepts either a single `document_id` or multiple `document_ids`:

```json
{
  "document_ids": ["first_uploaded_document_id", "second_uploaded_document_id"],
  "message": "What do these documents say about the topic?"
}
```

The chat response includes `answer`, `sources`, and `crag`. Each source contains the document name, chunk number, exact chunk text used as retrieved context, similarity score, and retrieval path metadata. The frontend renders only those final selected sources, not all intermediate retry candidates.

## Limitations

- This is assignment-grade storage, not production document management.
- Railway volume storage can still fill up if many large files are uploaded, but removing a source from the UI also deletes its backend files.
- Legacy `.doc` parsing is best-effort because old Word files are messy without heavier conversion tooling.
- The local sentence embedding model improves retrieval quality but increases install size and first-run model loading time.
