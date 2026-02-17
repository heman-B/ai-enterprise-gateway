# gateway/providers/anthropic.py
# Anthropic Claude API-Integration: Messages API via async httpx
from __future__ import annotations

import os
import uuid

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)
from .base import BaseProvider, is_retryable_error

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude-Anbieter (EU-konform).
    Unterstützt: claude-haiku-4-5-20251001, claude-sonnet-4-5-20250929
    API: Anthropic Messages API (kein SDK — direktes httpx für maximale Kontrolle)
    """

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        """Authentifizierungs-Header für Anthropic Messages API."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict]]:
        """
        OpenAI-Nachrichtenformat → Anthropic Messages-Format konvertieren.
        System-Nachrichten werden als separater 'system'-Parameter übergeben.
        """
        system_prompt: str | None = None
        converted: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                converted.append({"role": msg.role, "content": msg.content})
        return system_prompt, converted

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Chat-Completion über Anthropic Messages API mit automatischem Retry."""
        assert self._client is not None, "Provider nicht initialisiert — initialize() aufrufen"

        system_prompt, messages = self._convert_messages(request.messages)
        payload: dict = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "temperature": request.temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = await self._client.post(
            f"{ANTHROPIC_API_BASE}/messages",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        # Anthropic-Antwortformat in einheitliches Gateway-Format umwandeln
        content_text = data["content"][0]["text"]
        usage_data = data.get("usage", {})
        input_tokens = usage_data.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0)

        return ChatCompletionResponse(
            id=data.get("id", str(uuid.uuid4())),
            model=model,
            provider="anthropic",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content_text),
                    finish_reason=data.get("stop_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def health_check(self) -> bool:
        """Anthropic API-Erreichbarkeit prüfen (kein echter Aufruf — nur Authentifizierung)."""
        if not self._api_key:
            return False
        try:
            resp = await self._client.get(
                f"{ANTHROPIC_API_BASE}/models",
                headers=self._headers(),
                timeout=5.0,
            )
            return resp.status_code in (200, 404)  # 404 = API erreichbar, Endpunkt existiert nicht
        except Exception:
            return False
