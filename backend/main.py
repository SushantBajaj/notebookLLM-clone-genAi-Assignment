from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4
import json
import logging
import os
import shutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .llm_chat import chat_with_llm
from .rag_pipeline import EMBEDDING_DIMENSIONS, chunk_text, ingest_document


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "csv"}
DEFAULT_STORAGE_LIMIT_BYTES = 512 * 1024 * 1024
STORAGE_LIMIT_BYTES = int(os.getenv("STORAGE_LIMIT_BYTES", DEFAULT_STORAGE_LIMIT_BYTES))
STORAGE_SAFETY_MARGIN_BYTES = int(os.getenv("STORAGE_SAFETY_MARGIN_BYTES", 32 * 1024 * 1024))
VECTOR_METADATA_BYTES_PER_CHUNK = int(os.getenv("VECTOR_METADATA_BYTES_PER_CHUNK", 2048))

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
    document_id: str | None = Field(default=None, min_length=1)
    document_ids: list[str] = Field(default_factory=list)
    message: str = Field(..., min_length=1)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().removeprefix(".")


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.1f} GB"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def estimate_generated_size(text: str, chunks: list[str]) -> int:
    # Base the guard on parsed output, not upload size; PDFs and DOCX files can be compressed.
    text_size = len(text.encode("utf-8"))
    chunk_text_size = sum(len(chunk.encode("utf-8")) for chunk in chunks)
    vector_size = len(chunks) * EMBEDDING_DIMENSIONS * 4
    metadata_size = len(chunks) * VECTOR_METADATA_BYTES_PER_CHUNK
    return text_size + chunk_text_size + vector_size + metadata_size + STORAGE_SAFETY_MARGIN_BYTES


def storage_limit_message(
    current_size: int,
    required_size: int,
    filename: str,
    character_count: int,
    chunk_count: int,
) -> str:
    available_size = max(STORAGE_LIMIT_BYTES - current_size, 0)
    return (
        f"Storage limit reached while processing {filename}. "
        f"This deployment allows {format_bytes(STORAGE_LIMIT_BYTES)} total storage. "
        f"Already used: {format_bytes(current_size)}. "
        f"Available: {format_bytes(available_size)}. "
        f"The parsed document has {character_count:,} characters and {chunk_count:,} chunks. "
        f"Estimated space still needed for extracted text, chunk text, embeddings, and index metadata: "
        f"{format_bytes(required_size)}. The upload was removed. "
        "Please remove older uploads or use a smaller file."
    )


def ensure_generated_storage_budget(
    text: str,
    chunks: list[str],
    filename: str,
) -> None:
    current_size = directory_size(DATA_DIR)
    required_size = estimate_generated_size(text, chunks)

    if current_size + required_size > STORAGE_LIMIT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=storage_limit_message(
                current_size=current_size,
                required_size=required_size,
                filename=filename,
                character_count=len(text),
                chunk_count=len(chunks),
            ),
        )


def ensure_actual_storage_under_limit(document_dir: Path, filename: str) -> None:
    # Keep this final check because FAISS can write extra index/docstore overhead.
    current_size = directory_size(DATA_DIR)
    if current_size <= STORAGE_LIMIT_BYTES:
        return

    shutil.rmtree(document_dir, ignore_errors=True)
    raise HTTPException(
        status_code=507,
        detail=(
            f"Storage limit reached while processing {filename}. "
            f"After extracting text and building the vector index, usage grew to "
            f"{format_bytes(current_size)} against a {format_bytes(STORAGE_LIMIT_BYTES)} limit. "
            "The upload was removed. Please try a smaller file or clear older uploads."
        ),
    )


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


def get_chat_document_ids(request: ChatRequest) -> list[str]:
    document_ids = [document_id for document_id in request.document_ids if document_id]
    if request.document_id:
        document_ids.append(request.document_id)

    unique_ids = list(dict.fromkeys(document_ids))
    if not unique_ids:
        raise HTTPException(status_code=400, detail="At least one document_id is required.")

    return unique_ids


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
        metadata["ingestion"] = ingest_document(
            metadata,
            storage_guard=lambda text, chunks: ensure_generated_storage_budget(
                text,
                chunks,
                safe_filename,
            ),
        )
    except HTTPException:
        shutil.rmtree(paths["dir"], ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(paths["dir"], ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ensure_actual_storage_under_limit(paths["dir"], safe_filename)

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


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> dict[str, str]:
    paths = document_paths(document_id)
    if not paths["dir"].exists():
        raise HTTPException(status_code=404, detail="Document not found")

    shutil.rmtree(paths["dir"], ignore_errors=True)
    return {"document_id": document_id, "message": "Document removed."}


@app.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str) -> dict:
    metadata = load_metadata(document_id)
    ingestion = metadata.get("ingestion", {})
    chunks_path_value = ingestion.get("chunks_path")
    if chunks_path_value:
        chunks_path = Path(chunks_path_value)
        if chunks_path.exists():
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            return {
                "document_id": document_id,
                "filename": metadata["filename"],
                "chunks": [
                    {
                        "chunk_index": index,
                        "text": chunk,
                    }
                    for index, chunk in enumerate(chunks)
                ],
            }

    text_path_value = ingestion.get("text_path")
    if not text_path_value:
        raise HTTPException(status_code=404, detail="Document chunks not found")

    text_path = Path(text_path_value)
    if not text_path.exists():
        raise HTTPException(status_code=404, detail="Document chunks not found")

    chunks = chunk_text(text_path.read_text(encoding="utf-8"))
    return {
        "document_id": document_id,
        "filename": metadata["filename"],
        "chunks": [
            {
                "chunk_index": index,
                "text": chunk,
            }
            for index, chunk in enumerate(chunks)
        ],
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> dict:
    document_ids = get_chat_document_ids(request)
    documents = [load_metadata(document_id) for document_id in document_ids]
    try:
        result = await chat_with_llm(
            message=request.message,
            documents=documents,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "document_ids": document_ids,
        "answer": result["answer"],
        "sources": result["sources"],
    }
