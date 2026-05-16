from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from threading import Lock
from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from .document_parser import parse_document


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "hashing")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "similarity")
RETRIEVAL_K = 12
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

_embeddings: Embeddings | None = None
_embeddings_lock = Lock()


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = CHUNKING_STRATEGY
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: str = RETRIEVAL_STRATEGY
    k: int = RETRIEVAL_K


class QueryTransformStrategy(Protocol):
    def transform(self, question: str, document: dict) -> str:
        ...


class IdentityQueryTransform:
    def transform(self, question: str, document: dict) -> str:
        return question


class HashingEmbeddings(Embeddings):
    def __init__(self, dimensions: int):
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero.")

        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        features = [*tokens, *bigrams(tokens)]

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector

        return [value / magnitude for value in vector]


class SimilarityRetrievalStrategy:
    def __init__(
        self,
        embeddings: Embeddings,
        query_transform: QueryTransformStrategy | None = None,
    ):
        self.embeddings = embeddings
        self.query_transform = query_transform or IdentityQueryTransform()

    def retrieve(self, document: dict, question: str, limit: int) -> list[dict]:
        vector_path = document.get("ingestion", {}).get("vector_path")
        if not vector_path:
            raise ValueError("Document has not been ingested yet.")

        vector_store = FAISS.load_local(
            vector_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )
        retrieval_query = self.query_transform.transform(question, document)
        results = vector_store.similarity_search_with_score(retrieval_query, k=limit)

        return [
            {
                "text": result.page_content,
                "metadata": result.metadata,
                "score": float(score),
            }
            for result, score in results
        ]


def get_embeddings() -> Embeddings:
    global _embeddings

    if EMBEDDING_MODEL != "hashing":
        raise ValueError("Only EMBEDDING_MODEL=hashing is supported in this lightweight build.")

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = HashingEmbeddings(EMBEDDING_DIMENSIONS)

    return _embeddings


def get_retrieval_strategy(config: RetrievalConfig | None = None) -> SimilarityRetrievalStrategy:
    config = config or RetrievalConfig()
    if config.strategy != "similarity":
        raise ValueError(f"Unsupported retrieval strategy: {config.strategy}")

    return SimilarityRetrievalStrategy(get_embeddings())


def ingest_document(
    document: dict,
    storage_guard: Callable[[str, list[str]], None] | None = None,
) -> dict:
    text = parse_document(document["path"], document["extension"])
    chunks = chunk_text(text)

    # Let the API reject oversized parsed documents before writing generated index files.
    if storage_guard:
        storage_guard(text, chunks)

    document_dir = Path(document["path"]).parent
    text_path = document_dir / "extracted_text.txt"
    chunks_path = document_dir / "chunks.json"
    vector_path = document_dir / "faiss_index"
    text_path.write_text(text, encoding="utf-8")
    chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    vector_store = FAISS.from_texts(
        chunks,
        embedding=get_embeddings(),
        metadatas=[
            {
                "document_id": document["document_id"],
                "filename": document["filename"],
                "chunk_index": index,
            }
            for index, _chunk in enumerate(chunks)
        ],
    )
    vector_store.save_local(str(vector_path))

    return {
        "text_path": str(text_path),
        "chunks_path": str(chunks_path),
        "vector_path": str(vector_path),
        "character_count": len(text),
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "chunking_strategy": CHUNKING_STRATEGY,
    }


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[str]:
    config = config or ChunkingConfig()
    if config.strategy not in {"recursive", "fixed", "semantic"}:
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

    return split_text_recursively(text, config.chunk_size, config.chunk_overlap)


def retrieve_context(
    document: dict,
    question: str,
    limit: int = RETRIEVAL_K,
    retrieval_path: str = "hashing",
) -> list[dict]:
    matches = get_retrieval_strategy().retrieve(document, question, limit)
    for match in matches:
        match["retrieval_path"] = retrieval_path

    return matches


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]


def split_text_recursively(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap must not be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    paragraphs = [normalize_whitespace(paragraph) for paragraph in re.split(r"\n\s*\n", text)]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_text(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        current = paragraph

    if current:
        chunks.append(current)

    return chunks or split_long_text(normalize_whitespace(text), chunk_size, chunk_overlap)


def split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start + max(chunk_size // 2, 1), end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, 0)

    return chunks


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()
