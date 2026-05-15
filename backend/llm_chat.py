import logging
import os

from dotenv import load_dotenv
from google.genai import types

from .gemini_gateway import get_gemini_gateway
from .rag_pipeline import RETRIEVAL_K, retrieve_context


load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI Assistant who helps resolve the user query using only the retrieved context from the uploaded document sources.

Rule:
- Only answer based on the available document context.
- If the answer is not present in the context, say that the document context does not contain enough information.
- Be concise and cite source filenames and chunk numbers when helpful.
"""

GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


async def chat_with_llm(message: str, documents: list[dict]) -> dict:
    matches = retrieve_document_contexts(documents, message)
    context = format_context(matches)
    source_names = ", ".join(document.get("filename", "uploaded document") for document in documents)
    prompt = build_user_prompt(message, context, source_names)
    gateway = get_gemini_gateway()

    logger.info(
        "rag_context document_ids=%s filenames=%s retrieved_chunks=%s context_chars=%s",
        [document.get("document_id") for document in documents],
        source_names,
        len(matches),
        len(context),
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
    )
    first_model = unique_models(GEMINI_MODELS)[0]
    token_count = await gateway.count_tokens(
        step="final_answer",
        model=first_model,
        contents=prompt,
        config=config,
    )
    logger.info(
        "gemini_request step=final_answer preferred_model=%s prompt_tokens=%s prompt_chars=%s",
        first_model,
        token_count,
        len(prompt),
    )

    result = await gateway.generate_content(
        step="final_answer",
        models=GEMINI_MODELS,
        contents=prompt,
        config=config,
    )
    response = result.response
    usage = response.usage_metadata
    logger.info(
        "gemini_response step=final_answer model=%s key_index=%s prompt_tokens=%s output_tokens=%s total_tokens=%s",
        result.model,
        result.key_index,
        getattr(usage, "prompt_token_count", None),
        getattr(usage, "candidates_token_count", None),
        getattr(usage, "total_token_count", None),
    )

    return {
        "answer": response.text or "The model returned an empty response.",
        "sources": format_sources(matches),
    }


def retrieve_document_contexts(documents: list[dict], message: str) -> list[dict]:
    matches = []

    for document in documents:
        filename = document.get("filename", "uploaded document")
        for match in retrieve_context(document, message):
            match["filename"] = filename
            matches.append(match)

    return sorted(matches, key=lambda match: match["score"])[:RETRIEVAL_K]


def build_user_prompt(message: str, context: str, source_names: str) -> str:
    return f"""Sources: {source_names}

Available context:
{context}

User query:
{message}
"""


def format_context(matches: list[dict]) -> str:
    blocks = []
    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        chunk_index = metadata.get("chunk_index", index - 1)
        filename = match.get("filename") or metadata.get("filename", "uploaded document")
        blocks.append(
            f"[Chunk {chunk_index} | Source: {filename}]\n{match['text']}"
        )

    return "\n\n".join(blocks)


def format_sources(matches: list[dict]) -> list[dict]:
    sources = []
    for match in matches:
        metadata = match.get("metadata", {})
        sources.append(
            {
                "document_id": metadata.get("document_id"),
                "filename": match.get("filename") or metadata.get("filename", "uploaded document"),
                "chunk_index": metadata.get("chunk_index", 0),
                "text": match["text"],
                "score": match["score"],
            }
        )

    return sources


def unique_models(models: list[str]) -> list[str]:
    seen = set()
    unique = []

    for model in models:
        if model and model not in seen:
            seen.add(model)
            unique.append(model)

    return unique
