import logging
import json
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

CRAG_MODELS = [
    os.getenv("CRAG_MODEL", "gemini-2.5-flash-lite"),
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

CRAG_WEAK_GRADES = {"weak", "bad"}
CRAG_TIMEOUT_SECONDS = int(os.getenv("CRAG_TIMEOUT_SECONDS", "20"))
QUERY_VARIANT_COUNT = int(os.getenv("QUERY_VARIANT_COUNT", "3"))
QUERY_VARIANT_RETRIEVAL_K = int(os.getenv("QUERY_VARIANT_RETRIEVAL_K", "5"))
FINAL_CONTEXT_K = int(os.getenv("FINAL_CONTEXT_K", "10"))
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "24"))
CHUNK_PREVIEW_CHARS = 700


async def chat_with_llm(message: str, documents: list[dict]) -> dict:
    crag_result = await run_basic_crag_loop(message, documents)
    matches = crag_result["matches"]
    context = format_context(matches)
    source_names = ", ".join(document.get("filename", "uploaded document") for document in documents)
    prompt = build_user_prompt(message, context, source_names)
    gateway = get_gemini_gateway()

    logger.info(
        "rag_context document_ids=%s filenames=%s retrieved_chunks=%s context_chars=%s rewritten_query=%s grade=%s corrective_retry=%s",
        [document.get("document_id") for document in documents],
        source_names,
        len(matches),
        len(context),
        crag_result["rewritten_query"],
        crag_result["grade"]["grade"],
        crag_result["corrective_retry"],
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
        "crag": format_crag_metadata(crag_result),
    }


async def run_basic_crag_loop(message: str, documents: list[dict]) -> dict:
    rewritten_query = await rewrite_query(message)
    hyde_passage = ""
    query_variants = []
    corrective_retrieval_counts = {}
    context_pool_count = 0
    rerank_metadata = {
        "used": False,
        "fallback": False,
        "candidate_count": 0,
        "selected_count": 0,
    }
    answerability = {
        "answerable": None,
        "reason": "Answerability check has not run.",
        "missing": [],
    }
    initial_matches = retrieve_document_contexts(
        documents,
        rewritten_query,
        retrieval_path="initial_rewritten_query",
    )
    logger.info(
        "crag_initial_retrieval original_query_chars=%s rewritten_query_chars=%s chunks=%s",
        len(message),
        len(rewritten_query),
        len(initial_matches),
    )

    grade = await grade_retrieved_context(message, rewritten_query, initial_matches)
    grade_value = grade["grade"]
    corrective_retry = grade_value in CRAG_WEAK_GRADES
    candidate_matches = initial_matches

    if corrective_retry:
        original_matches = retrieve_document_contexts(
            documents,
            message,
            retrieval_path="retry_original_query",
        )
        rewritten_retry_matches = retrieve_document_contexts(
            documents,
            rewritten_query,
            retrieval_path="retry_rewritten_query",
        )
        hyde_passage = await generate_hyde_passage(message, rewritten_query)
        hyde_matches = []
        if hyde_passage:
            hyde_matches = retrieve_document_contexts(
                documents,
                hyde_passage,
                retrieval_path="hyde_passage",
            )

        query_variants = await generate_query_variants(message, rewritten_query, grade)
        variant_matches = []
        for index, query_variant in enumerate(query_variants, start=1):
            variant_matches.extend(
                retrieve_document_contexts(
                    documents,
                    query_variant,
                    retrieval_path=f"query_variant_{index}",
                    limit=QUERY_VARIANT_RETRIEVAL_K,
                )
            )

        context_pool = [
            *original_matches,
            *rewritten_retry_matches,
            *hyde_matches,
            *variant_matches,
        ]
        context_pool_count = len(context_pool)
        corrective_retrieval_counts = {
            "original_query": len(original_matches),
            "rewritten_query": len(rewritten_retry_matches),
            "hyde": len(hyde_matches),
            "query_variants": len(variant_matches),
        }
        candidate_matches = dedupe_matches(
            context_pool,
            limit=RERANK_CANDIDATE_K,
        )
        logger.info(
            "crag_corrective_retry grade=%s original_chunks=%s rewritten_chunks=%s hyde_chunks=%s variant_chunks=%s variants=%s context_pool=%s deduped_candidates=%s",
            grade_value,
            len(original_matches),
            len(rewritten_retry_matches),
            len(hyde_matches),
            len(variant_matches),
            len(query_variants),
            context_pool_count,
            len(candidate_matches),
        )
    else:
        candidate_matches = dedupe_matches(candidate_matches, limit=RERANK_CANDIDATE_K)
        logger.info(
            "crag_context_accepted grade=%s candidates=%s",
            grade_value,
            len(candidate_matches),
        )

    final_matches, rerank_metadata = await rerank_context(
        original_query=message,
        rewritten_query=rewritten_query,
        matches=candidate_matches,
        limit=FINAL_CONTEXT_K,
    )
    answerability = await check_answerability(message, rewritten_query, final_matches)
    logger.info(
        "crag_final_filter candidates=%s selected=%s rerank_used=%s rerank_fallback=%s answerable=%s",
        len(candidate_matches),
        len(final_matches),
        rerank_metadata["used"],
        rerank_metadata["fallback"],
        answerability["answerable"],
    )

    return {
        "rewritten_query": rewritten_query,
        "grade": grade,
        "corrective_retry": corrective_retry,
        "hyde_passage": hyde_passage,
        "query_variants": query_variants,
        "corrective_retrieval_counts": corrective_retrieval_counts,
        "context_pool_count": context_pool_count,
        "rerank": rerank_metadata,
        "answerability": answerability,
        "matches": final_matches,
    }


async def rewrite_query(message: str) -> str:
    prompt = f"""Rewrite the user query for semantic document retrieval.

Return only the rewritten query. Preserve the user's meaning, key entities, and constraints.

User query:
{message}
"""
    config = types.GenerateContentConfig(
        temperature=0.1,
    )
    gateway = get_gemini_gateway()

    try:
        result = await gateway.generate_content(
            step="query_rewrite",
            models=preferred_crag_models(),
            contents=prompt,
            config=config,
            timeout_seconds=CRAG_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("crag_query_rewrite_failed error=%s", exc)
        return message

    rewritten_query = (result.response.text or "").strip().strip('"')
    if not rewritten_query:
        logger.warning("crag_query_rewrite_empty using_original_query=true")
        return message

    logger.info(
        "crag_query_rewrite_complete model=%s key_index=%s original_chars=%s rewritten_chars=%s",
        result.model,
        result.key_index,
        len(message),
        len(rewritten_query),
    )
    return rewritten_query


async def grade_retrieved_context(
    original_query: str,
    rewritten_query: str,
    matches: list[dict],
) -> dict:
    prompt = f"""Grade whether the retrieved document context can answer the user's query.

Return compact JSON only:
{{"grade":"good|weak|bad","rationale":"short reason"}}

Use:
- good: the context likely contains enough direct evidence to answer.
- weak: the context is related but incomplete, indirect, or missing important parts.
- bad: the context is mostly irrelevant or empty.

Original query:
{original_query}

Rewritten query:
{rewritten_query}

Retrieved context:
{format_context(matches)}
"""
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )
    gateway = get_gemini_gateway()

    try:
        result = await gateway.generate_content(
            step="retrieval_grade",
            models=preferred_crag_models(),
            contents=prompt,
            config=config,
            timeout_seconds=CRAG_TIMEOUT_SECONDS,
        )
        parsed = parse_json_object(result.response.text or "")
    except Exception as exc:
        logger.warning("crag_retrieval_grade_failed error=%s using_fallback_grade=true", exc)
        return {
            "grade": "weak" if matches else "bad",
            "rationale": "Used fallback grading because the retrieval grader did not return usable JSON.",
        }

    grade = str(parsed.get("grade", "")).strip().lower()
    if grade not in {"good", "weak", "bad"}:
        grade = "weak" if matches else "bad"

    rationale = str(parsed.get("rationale", "")).strip()
    logger.info(
        "crag_retrieval_grade_complete model=%s key_index=%s grade=%s rationale=%s",
        result.model,
        result.key_index,
        grade,
        rationale,
    )
    return {
        "grade": grade,
        "rationale": rationale or "No rationale returned by retrieval grader.",
    }


async def generate_hyde_passage(original_query: str, rewritten_query: str) -> str:
    prompt = f"""Write a short hypothetical source passage that would directly answer the user's query.

Use the style of factual document text. Do not say this is hypothetical. Do not answer conversationally.
Return only the passage, in 3-5 sentences.

Original query:
{original_query}

Retrieval query:
{rewritten_query}
"""
    config = types.GenerateContentConfig(
        temperature=0.2,
    )
    gateway = get_gemini_gateway()

    try:
        result = await gateway.generate_content(
            step="hyde_passage",
            models=preferred_crag_models(),
            contents=prompt,
            config=config,
            timeout_seconds=CRAG_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("crag_hyde_failed error=%s", exc)
        return ""

    passage = (result.response.text or "").strip()
    logger.info(
        "crag_hyde_complete model=%s key_index=%s passage_chars=%s",
        result.model,
        result.key_index,
        len(passage),
    )
    return passage


async def generate_query_variants(
    original_query: str,
    rewritten_query: str,
    grade: dict,
) -> list[str]:
    prompt = f"""Generate focused semantic retrieval query variants for the user's query.

Return compact JSON only:
{{"variants":["variant 1","variant 2","variant 3"]}}

Rules:
- Return exactly {QUERY_VARIANT_COUNT} variants.
- Keep each variant under 14 words.
- Preserve the user's intent.
- Emphasize missing details implied by the retrieval grade rationale.
- Do not include numbering.

Original query:
{original_query}

Current rewritten query:
{rewritten_query}

Retrieval grade: {grade["grade"]}
Rationale: {grade["rationale"]}
"""
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )
    gateway = get_gemini_gateway()

    try:
        result = await gateway.generate_content(
            step="query_variants",
            models=preferred_crag_models(),
            contents=prompt,
            config=config,
            timeout_seconds=CRAG_TIMEOUT_SECONDS,
        )
        parsed = parse_json_object(result.response.text or "")
    except Exception as exc:
        logger.warning("crag_query_variants_failed error=%s", exc)
        return []

    variants = parsed.get("variants", [])
    if not isinstance(variants, list):
        return []

    cleaned_variants = []
    seen = {original_query.casefold(), rewritten_query.casefold()}
    for variant in variants:
        cleaned = str(variant).strip().strip('"')
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        seen.add(key)
        cleaned_variants.append(cleaned)
        if len(cleaned_variants) >= QUERY_VARIANT_COUNT:
            break

    logger.info(
        "crag_query_variants_complete model=%s key_index=%s variants=%s",
        result.model,
        result.key_index,
        len(cleaned_variants),
    )
    return cleaned_variants


async def rerank_context(
    original_query: str,
    rewritten_query: str,
    matches: list[dict],
    limit: int,
) -> tuple[list[dict], dict]:
    metadata = {
        "used": False,
        "fallback": False,
        "candidate_count": len(matches),
        "selected_count": min(len(matches), limit),
    }
    if not matches:
        return [], metadata

    prompt = f"""Rerank retrieved chunks for answering the user's query.

Return compact JSON only:
{{"ranked_chunks":[{{"index":0,"relevance":0.95,"reason":"short reason"}}]}}

Rules:
- Rank by usefulness for answering the original query, not by writing quality.
- Include at most {limit} chunks.
- Use only indexes from the candidate list.
- Prefer chunks with direct evidence over broad background.

Original query:
{original_query}

Rewritten query:
{rewritten_query}

Candidate chunks:
{format_rerank_candidates(matches[:RERANK_CANDIDATE_K])}
"""
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )
    gateway = get_gemini_gateway()

    try:
        result = await gateway.generate_content(
            step="context_rerank",
            models=preferred_crag_models(),
            contents=prompt,
            config=config,
            timeout_seconds=CRAG_TIMEOUT_SECONDS,
        )
        parsed = parse_json_object(result.response.text or "")
        ranked_chunks = parsed.get("ranked_chunks", [])
    except Exception as exc:
        logger.warning("crag_rerank_failed error=%s using_score_order=true", exc)
        metadata["fallback"] = True
        return matches[:limit], metadata

    if not isinstance(ranked_chunks, list):
        metadata["fallback"] = True
        return matches[:limit], metadata

    selected = []
    selected_indexes = set()
    for ranked_chunk in ranked_chunks:
        if not isinstance(ranked_chunk, dict):
            continue

        index = ranked_chunk.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(matches):
            continue
        if index in selected_indexes:
            continue

        match = {
            **matches[index],
            "rerank": {
                "relevance": normalize_relevance(ranked_chunk.get("relevance")),
                "reason": str(ranked_chunk.get("reason", "")).strip(),
            },
        }
        selected.append(match)
        selected_indexes.add(index)
        if len(selected) >= limit:
            break

    if not selected:
        metadata["fallback"] = True
        return matches[:limit], metadata

    metadata["used"] = True
    metadata["selected_count"] = len(selected)
    logger.info(
        "crag_rerank_complete model=%s key_index=%s candidates=%s selected=%s",
        result.model,
        result.key_index,
        len(matches),
        len(selected),
    )
    return selected, metadata


async def check_answerability(
    original_query: str,
    rewritten_query: str,
    matches: list[dict],
) -> dict:
    fallback = {
        "answerable": None,
        "reason": "Answerability check was not available.",
        "missing": [],
    }
    if not matches:
        return {
            "answerable": False,
            "reason": "No final context chunks were selected.",
            "missing": ["relevant document context"],
        }

    prompt = f"""Check whether the selected context is enough to answer the user's query.

Return compact JSON only:
{{"answerable":true,"reason":"short reason","missing":["missing detail"]}}

Use answerable=false if important facts needed by the query are missing.

Original query:
{original_query}

Rewritten query:
{rewritten_query}

Selected context:
{format_context(matches)}
"""
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )
    gateway = get_gemini_gateway()

    try:
        result = await gateway.generate_content(
            step="answerability_check",
            models=preferred_crag_models(),
            contents=prompt,
            config=config,
            timeout_seconds=CRAG_TIMEOUT_SECONDS,
        )
        parsed = parse_json_object(result.response.text or "")
    except Exception as exc:
        logger.warning("crag_answerability_failed error=%s", exc)
        return fallback

    answerable = parsed.get("answerable")
    if not isinstance(answerable, bool):
        answerable = None

    missing = parsed.get("missing", [])
    if not isinstance(missing, list):
        missing = []

    answerability = {
        "answerable": answerable,
        "reason": str(parsed.get("reason", "")).strip() or "No reason returned.",
        "missing": [str(item).strip() for item in missing if str(item).strip()],
    }
    logger.info(
        "crag_answerability_complete model=%s key_index=%s answerable=%s missing=%s",
        result.model,
        result.key_index,
        answerability["answerable"],
        len(answerability["missing"]),
    )
    return answerability


def parse_json_object(value: str) -> dict:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    return json.loads(cleaned)


def format_rerank_candidates(matches: list[dict]) -> str:
    blocks = []
    for index, match in enumerate(matches):
        metadata = match.get("metadata", {})
        filename = match.get("filename") or metadata.get("filename", "uploaded document")
        chunk_index = metadata.get("chunk_index", 0)
        text = match["text"].replace("\n", " ").strip()
        preview = text[:CHUNK_PREVIEW_CHARS]
        if len(text) > CHUNK_PREVIEW_CHARS:
            preview += "..."
        blocks.append(
            f"[Index {index} | Source: {filename} | Chunk {chunk_index} | Score: {match['score']:.4f}]\n{preview}"
        )

    return "\n\n".join(blocks)


def normalize_relevance(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))

    return None


def preferred_crag_models() -> list[str]:
    return unique_models(CRAG_MODELS)[:1]


def retrieve_document_contexts(
    documents: list[dict],
    message: str,
    retrieval_path: str,
    limit: int = RETRIEVAL_K,
) -> list[dict]:
    matches = []

    for document in documents:
        filename = document.get("filename", "uploaded document")
        for match in retrieve_context(
            document,
            message,
            limit=limit,
            retrieval_path=retrieval_path,
        ):
            match["filename"] = filename
            matches.append(match)

    return sorted(matches, key=lambda match: match["score"])[:limit]


def dedupe_matches(matches: list[dict], limit: int) -> list[dict]:
    deduped = {}

    for match in sorted(matches, key=lambda item: item["score"]):
        metadata = match.get("metadata", {})
        key = (
            metadata.get("document_id"),
            metadata.get("chunk_index"),
        )
        if key not in deduped:
            deduped[key] = {
                **match,
                "retrieval_paths": [match.get("retrieval_path", "semantic")],
            }
            continue

        existing = deduped[key]
        retrieval_path = match.get("retrieval_path", "semantic")
        if retrieval_path not in existing["retrieval_paths"]:
            existing["retrieval_paths"].append(retrieval_path)

    return sorted(deduped.values(), key=lambda match: match["score"])[:limit]


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
                "retrieval_paths": match.get("retrieval_paths") or [match.get("retrieval_path", "semantic")],
                "rerank": match.get("rerank"),
            }
        )

    return sources


def format_crag_metadata(crag_result: dict) -> dict:
    grade = crag_result["grade"]
    return {
        "rewritten_query": crag_result["rewritten_query"],
        "retrieval_grade": grade["grade"],
        "retrieval_rationale": grade["rationale"],
        "corrective_retry": crag_result["corrective_retry"],
        "hyde_used": bool(crag_result["hyde_passage"]),
        "hyde_passage": crag_result["hyde_passage"],
        "query_variants": crag_result["query_variants"],
        "corrective_retrieval_counts": crag_result["corrective_retrieval_counts"],
        "context_pool_count": crag_result["context_pool_count"],
        "rerank": crag_result["rerank"],
        "answerability": crag_result["answerability"],
        "final_source_count": len(crag_result["matches"]),
    }


def unique_models(models: list[str]) -> list[str]:
    seen = set()
    unique = []

    for model in models:
        if model and model not in seen:
            seen.add(model)
            unique.append(model)

    return unique
