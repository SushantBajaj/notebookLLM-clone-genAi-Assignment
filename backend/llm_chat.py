import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .rag_pipeline import retrieve_context


load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI Assistant who helps resolving the user query based on the available context provided to you from PDF file with the content and page number.

Rule:
- Only answer based on the available context from the file only.
- If the answer is not present in the context, say that the document context does not contain enough information.
- Be concise and cite the chunk numbers you used when helpful.
"""

GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


async def chat_with_llm(message: str, document: dict) -> str:
    filename = document.get("filename", "the uploaded document")
    matches = retrieve_context(document, message)
    context = format_context(matches, filename)
    prompt = build_user_prompt(message, context, filename)
    client = create_client()

    logger.info(
        "rag_context document_id=%s filename=%s retrieved_chunks=%s context_chars=%s",
        document.get("document_id"),
        filename,
        len(matches),
        len(context),
    )

    last_error: Exception | None = None
    for model in unique_models(GEMINI_MODELS):
        try:
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            )
            token_count = await count_prompt_tokens(client, model, prompt, config)
            logger.info(
                "gemini_request model=%s prompt_tokens=%s prompt_chars=%s",
                model,
                token_count,
                len(prompt),
            )

            response = await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            usage = response.usage_metadata
            logger.info(
                "gemini_response model=%s prompt_tokens=%s output_tokens=%s total_tokens=%s",
                model,
                getattr(usage, "prompt_token_count", None),
                getattr(usage, "candidates_token_count", None),
                getattr(usage, "total_token_count", None),
            )
            return response.text or "The model returned an empty response."
        except Exception as exc:
            last_error = exc
            logger.exception("gemini_model_failed model=%s error=%s", model, exc)

    raise RuntimeError(f"All Gemini model attempts failed: {last_error}") from last_error


def create_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env.")

    return genai.Client(api_key=api_key)


def build_user_prompt(message: str, context: str, filename: str) -> str:
    return f"""File: {filename}

Available context:
{context}

User query:
{message}
"""


def format_context(matches: list[dict], filename: str) -> str:
    blocks = []
    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        chunk_index = metadata.get("chunk_index", index - 1)
        blocks.append(
            f"[Chunk {chunk_index} | Source: {filename}]\n{match['text']}"
        )

    return "\n\n".join(blocks)


async def count_prompt_tokens(
    client: genai.Client,
    model: str,
    prompt: str,
    config: types.GenerateContentConfig,
) -> int | None:
    try:
        count = await client.aio.models.count_tokens(
            model=model,
            contents=prompt,
            config=config,
        )
        return getattr(count, "total_tokens", None)
    except Exception as exc:
        logger.warning("gemini_token_count_failed model=%s error=%s", model, exc)
        return None


def unique_models(models: list[str]) -> list[str]:
    seen = set()
    unique = []

    for model in models:
        if model and model not in seen:
            seen.add(model)
            unique.append(model)

    return unique
