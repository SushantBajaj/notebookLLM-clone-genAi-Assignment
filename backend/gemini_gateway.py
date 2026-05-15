import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 503}


@dataclass(frozen=True)
class GeminiResult:
    response: Any
    model: str
    key_index: int


class GeminiGateway:
    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env.")

        self.api_keys = api_keys
        self._key_index = 0
        self._lock = Lock()

    async def generate_content(
        self,
        step: str,
        models: list[str],
        contents: str,
        config: types.GenerateContentConfig,
    ) -> GeminiResult:
        last_error: Exception | None = None

        for model in unique_values(models):
            for _attempt in range(len(self.api_keys)):
                key_index = self.current_key_index()
                logger.info("gemini_step step=%s model=%s key_index=%s", step, model, key_index)

                try:
                    client = self.client_for_key(key_index)
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    logger.info("gemini_step_complete step=%s model=%s key_index=%s", step, model, key_index)
                    return GeminiResult(response=response, model=model, key_index=key_index)
                except Exception as exc:
                    last_error = exc
                    status_code = get_status_code(exc)
                    if status_code in RETRYABLE_STATUS_CODES and len(self.api_keys) > 1:
                        logger.warning(
                            "gemini_step_retryable_failure step=%s model=%s key_index=%s status_code=%s error=%s",
                            step,
                            model,
                            key_index,
                            status_code,
                            exc,
                        )
                        self.rotate_key(step=step, reason=str(status_code))
                        continue

                    logger.exception(
                        "gemini_step_failed step=%s model=%s key_index=%s status_code=%s error=%s",
                        step,
                        model,
                        key_index,
                        status_code,
                        exc,
                    )
                    break

        raise RuntimeError(f"All Gemini attempts failed for step '{step}': {last_error}") from last_error

    async def count_tokens(
        self,
        step: str,
        model: str,
        contents: str,
        config: types.GenerateContentConfig,
    ) -> int | None:
        key_index = self.current_key_index()
        try:
            client = self.client_for_key(key_index)
            count = await client.aio.models.count_tokens(
                model=model,
                contents=contents,
                config=config,
            )
            return getattr(count, "total_tokens", None)
        except Exception as exc:
            logger.warning(
                "gemini_token_count_failed step=%s model=%s key_index=%s status_code=%s error=%s",
                step,
                model,
                key_index,
                get_status_code(exc),
                exc,
            )
            return None

    def current_key_index(self) -> int:
        with self._lock:
            return self._key_index

    def rotate_key(self, step: str, reason: str) -> None:
        with self._lock:
            previous_index = self._key_index
            self._key_index = (self._key_index + 1) % len(self.api_keys)
            logger.warning(
                "gemini_key_switched step=%s from_key_index=%s to_key_index=%s reason=%s",
                step,
                previous_index,
                self._key_index,
                reason,
            )

    def client_for_key(self, key_index: int) -> genai.Client:
        return genai.Client(api_key=self.api_keys[key_index])


@lru_cache(maxsize=1)
def get_gemini_gateway() -> GeminiGateway:
    return GeminiGateway(load_api_keys())


def load_api_keys() -> list[str]:
    raw_keys = os.getenv("GEMINI_API_KEYS", "")
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]

    single_key = os.getenv("GEMINI_API_KEY", "").strip()
    if single_key and single_key != "your_gemini_api_key_here":
        keys.insert(0, single_key)

    return unique_values(keys)


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []

    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)

    return unique


def get_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    text = str(exc)
    for status_code in RETRYABLE_STATUS_CODES:
        if str(status_code) in text:
            return status_code

    return None
