from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4
import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .llm_chat import chat_with_llm
from .rag_pipeline import ingest_document


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "csv"}

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="NoteBook LLM Clone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().removeprefix(".")


def document_paths(document_id: str) -> dict[str, Path]:
    document_dir = DOCUMENTS_DIR / document_id
    return {
        "dir": document_dir,
        "metadata": document_dir / "metadata.json",
    }


def load_metadata(document_id: str) -> dict:
    paths = document_paths(document_id)
    if not paths["metadata"].exists():
        raise HTTPException(status_code=404, detail="Document not found")

    with paths["metadata"].open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(request: Request) -> dict:
    filename = unquote(request.headers.get("x-filename", "")).strip()
    safe_filename = Path(filename).name
    extension = get_file_extension(safe_filename)

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOC, DOCX, and CSV files are supported.",
        )

    file_bytes = await request.body()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_id = uuid4().hex
    paths = document_paths(document_id)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    stored_file = paths["dir"] / f"original.{extension}"
    with stored_file.open("wb") as output_file:
        output_file.write(file_bytes)

    metadata = {
        "document_id": document_id,
        "filename": safe_filename,
        "content_type": request.headers.get("content-type"),
        "extension": extension,
        "path": str(stored_file),
        "size_bytes": stored_file.stat().st_size,
    }

    with paths["metadata"].open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    try:
        metadata["ingestion"] = ingest_document(metadata)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with paths["metadata"].open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return {
        "document_id": document_id,
        "filename": metadata["filename"],
        "extension": extension,
        "size_bytes": metadata["size_bytes"],
        "character_count": metadata["ingestion"]["character_count"],
        "chunk_count": metadata["ingestion"]["chunk_count"],
        "message": "Document uploaded successfully.",
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    metadata = load_metadata(request.document_id)
    try:
        answer = await chat_with_llm(
            message=request.message,
            document=metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "document_id": request.document_id,
        "answer": answer,
    }
