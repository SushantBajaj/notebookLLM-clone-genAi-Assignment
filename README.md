# NoteBook LLM Clone

A small NotebookLM-style assignment project: upload one document, let the backend parse and index it, then ask questions grounded in that document.

The app has a plain HTML/CSS/JavaScript frontend and a FastAPI backend. The backend extracts text from PDF, DOC, DOCX, or CSV files, chunks the text, stores a local FAISS index, retrieves relevant chunks, and asks Gemini to answer using that context.

## What It Does

- Uploads one source document at a time.
- Supports PDF, DOC, DOCX, and CSV files.
- Extracts readable text from the uploaded file.
- Splits the text into overlapping chunks for retrieval.
- Builds a local FAISS vector index using deterministic local hash embeddings.
- Sends only the most relevant document chunks to Gemini during chat.
- Renders assistant Markdown in the frontend, including bold text, lists, links, and code blocks.
- Adds storage checks for Railway-style limited volume storage.

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
    ├── llm_chat.py            # Gemini prompt + response flow
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
const API_BASE_URL = "https://notebookllm-clone-genai-assignment-production.up.railway.app";
```

If you deploy your own Railway service, replace that URL with your generated backend domain. For local-only development, use:

```js
const API_BASE_URL = "http://127.0.0.1:8000";
```

## Environment Variables

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest

# Railway free volume limit: 0.5 GB = 536870912 bytes.
STORAGE_LIMIT_BYTES=536870912
STORAGE_SAFETY_MARGIN_BYTES=33554432
VECTOR_METADATA_BYTES_PER_CHUNK=2048
```

The storage settings are intentionally visible because generated files can be much larger than the uploaded file. A compressed PDF may upload as a small file but expand into much more extracted text and index data.

## Chunking Strategy

The RAG pipeline uses LangChain's `RecursiveCharacterTextSplitter` in [backend/rag_pipeline.py](backend/rag_pipeline.py). The current settings are:

```python
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160
RETRIEVAL_K = 4
```

The splitter tries to preserve readable boundaries in this order:

```python
["\n\n", "\n", ". ", " ", ""]
```

That means it first tries to split around paragraphs, then lines, then sentences, then words. The empty string fallback is there so very long unbroken text can still be split instead of producing one oversized chunk.

The chunk size is deliberately moderate. Around 900 characters keeps each retrieved passage small enough for concise prompts while still giving the model enough surrounding context to answer naturally. The 160-character overlap carries a little context from one chunk into the next, which helps when an answer depends on text near a boundary.

At chat time, the backend retrieves the top 4 matching chunks from FAISS and sends only those chunks to Gemini. This keeps prompts smaller and keeps answers grounded in the uploaded file instead of the whole document being sent every time.

## Storage Behavior

Uploaded and generated files are stored under:

```text
backend/data/documents/
```

For every upload, the backend stores:

- the original uploaded file
- `metadata.json`
- `extracted_text.txt`
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
POST /chat
```

`/upload` expects the file body directly and reads the original filename from the `X-Filename` header.

`/chat` expects:

```json
{
  "document_id": "uploaded_document_id",
  "message": "What is this document about?"
}
```

## Limitations

- This is assignment-grade storage, not production document management.
- Local Railway volume storage can fill up, so old uploads may need to be deleted manually.
- Legacy `.doc` parsing is best-effort because old Word files are messy without heavier conversion tooling.
- The local hash embedding is deterministic and free, but not as semantically strong as a real embedding model.

## Security Notes

Do not commit `.env`. The `.gitignore` already excludes it.

If an API key was ever pasted into chat, logs, screenshots, or a public repo, rotate it. Treat it as exposed.
