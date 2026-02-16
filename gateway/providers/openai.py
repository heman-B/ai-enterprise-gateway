# gateway/providers/openai.py
# OpenAI GPT API-Integration: Chat Completions API via async httpx
from __future__ import annotations

import os
import uuid

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)
from .base import BaseProvider

OPENAI_API_BASE = "https://api.openai.com/v1"


class OpenAIProvider(BaseProvider):
    """
    OpenAI GPT-Anbieter.
    Unterstützt: gpt-4o, gpt-4o-mini
    API: OpenAI Chat Completions (natives OpenAI-Format — kein Konvertierungsaufwand)
    """

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.getenv("OPENAI_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        """Bearer-Token-Header für OpenAI API."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Chat-Completion über OpenAI Chat Completions API."""
        assert self._client is not None, "Provider nicht initialisiert"

        # GPT-5+ erfordert max_completion_tokens statt max_tokens (API-Breaking-Change)
        token_param = "max_completion_tokens" if model.startswith("gpt-5") else "max_tokens"

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            token_param: request.max_tokens,
            "temperature": request.temperature,
        }

        response = await self._client.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage_data = data.get("usage", {})

        return ChatCompletionResponse(
            id=data.get("id", str(uuid.uuid4())),
            model=model,
            provider="openai",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=choice["message"]["content"],
                    ),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
        )

    async def health_check(self) -> bool:
        """OpenAI API-Erreichbarkeit prüfen."""
        if not self._api_key:
            return False
        try:
            resp = await self._client.get(
                f"{OPENAI_API_BASE}/models",
                headers=self._headers(),
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
