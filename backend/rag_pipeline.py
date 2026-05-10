from pathlib import Path
import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_parser import parse_document


EMBEDDING_MODEL = "local-hashing-embedding"
EMBEDDING_DIMENSIONS = 384
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160
RETRIEVAL_K = 4


class LocalHashEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        length = math.sqrt(sum(value * value for value in vector))
        if length == 0:
            return vector

        return [value / length for value in vector]


def get_embeddings() -> LocalHashEmbeddings:
    return LocalHashEmbeddings()


def ingest_document(document: dict) -> dict:
    text = parse_document(document["path"], document["extension"])
    document_dir = Path(document["path"]).parent
    text_path = document_dir / "extracted_text.txt"
    vector_path = document_dir / "faiss_index"
    text_path.write_text(text, encoding="utf-8")

    chunks = chunk_text(text)
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
        "vector_path": str(vector_path),
        "character_count": len(text),
        "chunk_count": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
    }


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def retrieve_context(document: dict, question: str, limit: int = RETRIEVAL_K) -> list[dict]:
    vector_path = document.get("ingestion", {}).get("vector_path")
    if not vector_path:
        raise ValueError("Document has not been ingested yet.")

    vector_store = FAISS.load_local(
        vector_path,
        embeddings=get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    results = vector_store.similarity_search_with_score(question, k=limit)

    return [
        {
            "text": result.page_content,
            "metadata": result.metadata,
            "score": float(score),
        }
        for result, score in results
    ]
