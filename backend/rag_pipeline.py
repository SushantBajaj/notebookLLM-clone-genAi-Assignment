from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from .document_parser import parse_document


load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic")
SEMANTIC_BREAKPOINT_THRESHOLD_TYPE = os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile")
SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT = float(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT", "90"))
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "similarity")
RETRIEVAL_K = 12


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = CHUNKING_STRATEGY
    breakpoint_threshold_type: str = SEMANTIC_BREAKPOINT_THRESHOLD_TYPE
    breakpoint_threshold_amount: float = SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT


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


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


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
    if config.strategy != "semantic":
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

    splitter = SemanticChunker(
        get_embeddings(),
        breakpoint_threshold_type=config.breakpoint_threshold_type,
        breakpoint_threshold_amount=config.breakpoint_threshold_amount,
    )
    return splitter.split_text(text)


def retrieve_context(
    document: dict,
    question: str,
    limit: int = RETRIEVAL_K,
    retrieval_path: str = "semantic",
) -> list[dict]:
    matches = get_retrieval_strategy().retrieve(document, question, limit)
    for match in matches:
        match["retrieval_path"] = retrieval_path

    return matches
