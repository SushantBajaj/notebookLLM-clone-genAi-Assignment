# NoteBook LLM Clone

A small NotebookLM-style assignment project: upload documents, let the backend parse and index them, then ask questions grounded in those sources.

The app has a plain HTML/CSS/JavaScript frontend and a FastAPI backend. The backend extracts text from PDF, DOC, DOCX, or CSV files, chunks the text, stores a local FAISS index, runs a CRAG retrieval loop, and asks Gemini to answer using only the final selected context.

## What It Does

- Uploads multiple source documents.
- Lets users choose the current chat scope with checkboxes in the sidebar.
- Lets users inspect document chunks from the sidebar.
- Shows the exact retrieved source chunks behind each answer.
- Runs CRAG with query rewriting, retrieval grading, and a corrective branch with HyDE plus query variants.
- Supports PDF, DOC, DOCX, and CSV files.
- Extracts readable text from the uploaded file.
- Splits the text into lightweight recursive chunks for retrieval.
- Builds a local FAISS vector index using deterministic hashing embeddings.
- Sends only the most relevant chunks across uploaded documents to Gemini during chat.
- Renders assistant Markdown in the frontend, including bold text, lists, links, and code blocks.
- Adds storage checks for Railway-style limited volume storage.

## Frontend Quality-of-Life Features

- The sidebar keeps long filenames, metadata, document actions, and chunk controls contained in compact source cards.
- Each uploaded source has an `Inspect` control for browsing chunk numbers and previewing one chunk at a time.
- Assistant answers can show their retrieved sources without dumping all source text at once.
- Source cards inside answers show the final selected context only, capped at 12 chunks, with retrieval pass metadata.
- User messages include a quiet `Info` control that can reveal the retrieval grade, corrective retry status, final source count, grader rationale, rewritten query, generated HyDE passage, and query variants.
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
# Optional comma-separated pool. Leave blank unless you have extra keys.
# Example: GEMINI_API_KEYS=AIza...first,AIza...second
GEMINI_API_KEYS=
GEMINI_MODEL=gemini-flash-latest
CRAG_MODEL=gemini-2.5-flash-lite
CRAG_TIMEOUT_SECONDS=20
GEMINI_TIMEOUT_SECONDS=90
QUERY_VARIANT_COUNT=3
QUERY_VARIANT_RETRIEVAL_K=5
FINAL_CONTEXT_K=10
RERANK_CANDIDATE_K=24

EMBEDDING_MODEL=hashing
EMBEDDING_DIMENSIONS=2048
CHUNKING_STRATEGY=recursive
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
RETRIEVAL_STRATEGY=similarity

# Railway free volume limit: 0.5 GB = 536870912 bytes.
STORAGE_LIMIT_BYTES=536870912
STORAGE_SAFETY_MARGIN_BYTES=33554432
VECTOR_METADATA_BYTES_PER_CHUNK=2048
```

The storage settings are intentionally visible because generated files can be much larger than the uploaded file. A compressed PDF may upload as a small file but expand into much more extracted text and index data.

`GEMINI_API_KEYS` is optional. Put extra keys in one comma-separated line, without spaces required, such as `GEMINI_API_KEYS=AIza...first,AIza...second`. The backend combines `GEMINI_API_KEY` with any keys from `GEMINI_API_KEYS`, ignores placeholder values, removes duplicates, and keeps using the current key until Gemini returns a retryable `429` or `503`. It then switches to the next key in a circular pool. Logs include the current step, model, and key index, but never print the key itself.

`GEMINI_MODEL` controls the final answer model and keeps `gemini-flash-latest` as the preferred default. `CRAG_MODEL` is separate and is used only for query rewriting, retrieval grading, HyDE passage generation, query variant generation, reranking, and answerability checks. `CRAG_TIMEOUT_SECONDS` prevents CRAG helper calls from hanging the chat; if grading times out or fails, the backend falls back to a weak grade when chunks exist so the corrective branch still runs. `GEMINI_TIMEOUT_SECONDS` is the default timeout for other Gemini generation calls. `QUERY_VARIANT_COUNT` and `QUERY_VARIANT_RETRIEVAL_K` control how many focused variants are generated and how many chunks each variant retrieves. `RERANK_CANDIDATE_K` controls how many deduplicated candidates are sent to the reranker, and `FINAL_CONTEXT_K` controls how many final chunks reach the answer model.

## RAG Strategy

The RAG pipeline uses lightweight recursive chunking, deterministic hashing embeddings, and a local FAISS index in [backend/rag_pipeline.py](backend/rag_pipeline.py). The current settings are:

```python
CHUNKING_STRATEGY = "recursive"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "hashing"
EMBEDDING_DIMENSIONS = 2048
RETRIEVAL_K = 12
```

Hashing embeddings avoid a local transformer model, PyTorch, and HuggingFace downloads, which keeps Railway deployments much smaller and faster to start. The tradeoff is that retrieval is more lexical than semantic: exact terms, names, and technical phrases work well, while synonyms and heavy paraphrasing are weaker than transformer embeddings.

At chat time, the backend runs a CRAG loop before final answer generation:

1. Rewrite the user's query for semantic retrieval.
2. Retrieve the initial top 12 chunks with the rewritten query.
3. Grade whether that context is good, weak, or bad for the original query.
4. If the grade is weak or bad, run the corrective branch:
   - retrieve with the original query
   - retrieve with the rewritten query
   - generate a HyDE passage and retrieve with it
   - generate focused query variants and retrieve with each variant
5. Merge the corrective context pool and deduplicate by document id and chunk index.
6. Rerank candidate chunks against the original query.
7. Run an answerability check on the final selected chunks.
8. Send only the filtered final chunks to Gemini.

This is a single corrective pass, not an unbounded retry loop. If the first retrieval grade is `good`, the initial rewritten-query results are used. If the grade is `weak` or `bad`, the backend builds one larger corrective context pool from original-query retrieval, rewritten-query retrieval, HyDE retrieval, and query-variant retrieval. The final answer still receives only the reranked and filtered final chunks.

The final answer model still uses `GEMINI_MODEL` and its existing fallback list. `CRAG_MODEL` is only for CRAG helper steps.

The `/chat` response includes CRAG metadata for the frontend `Info` control:

- `rewritten_query`
- `retrieval_grade`
- `retrieval_rationale`
- `corrective_retry`
- `hyde_used`
- `hyde_passage`
- `query_variants`
- `context_pool_count`
- `corrective_retrieval_counts`
- `rerank`
- `answerability`
- `final_source_count`

Each returned source can also include retrieval pass labels such as `initial_rewritten_query`, `retry_original_query`, `retry_rewritten_query`, `hyde_passage`, or `query_variant_1`.

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
- Hashing embeddings are lightweight and deployment-friendly, but they are more lexical than transformer embeddings and may miss matches that rely on synonyms or paraphrasing.
