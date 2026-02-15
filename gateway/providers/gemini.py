# gateway/providers/gemini.py
# Google Gemini API-Integration: generateContent via async httpx
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"

# Interne Modellnamen (Gemini verwendet andere Bezeichnungen als Gateway)
_MODEL_MAP = {
    "gemini-flash":     "gemini-2.0-flash",       # Gemini 2.0 Flash (default fast)
    "gemini-pro":       "gemini-1.5-pro",          # 2.0 Pro not GA yet — use 1.5 Pro
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro":   "gemini-1.5-pro",
}


class GeminiProvider(BaseProvider):
    """
    Google Gemini-Anbieter.
    Unterstützt: gemini-flash (gemini-1.5-flash), gemini-pro (gemini-1.5-pro)
    API: Gemini generateContent (eigenes Format — Konvertierung erforderlich)
    """

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.getenv("GEMINI_API_KEY", "")

    def _resolve_model(self, model: str) -> str:
        """
        Gateway-Modellnamen auf Gemini-API-Namen abbilden.
        HINWEIS: Modellnamen regelmäßig gegen cloud.google.com/vertex-ai/docs prüfen.
        """
        resolved = _MODEL_MAP.get(model)
        if not resolved:
            logger.warning("Unbekannter Gemini-Modellname '%s' — wird unverändert übergeben", model)
            return model
        return resolved

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str, list[dict]]:
        """
        OpenAI-Format → Gemini Content-Format konvertieren.
        System-Nachrichten → systemInstruction-Feld
        user/assistant → user/model roles
        """
        system_instruction = ""
        contents: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                # Gemini verwendet 'model' statt 'assistant'
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})
        return system_instruction, contents

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Chat-Completion über Gemini generateContent API."""
        assert self._client is not None, "Provider nicht initialisiert"

        gemini_model = self._resolve_model(model)
        system_instruction, contents = self._convert_messages(request.messages)

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        response = await self._client.post(
            f"{GEMINI_API_BASE}/models/{gemini_model}:generateContent",
            params={"key": self._api_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        candidate = data["candidates"][0]
        content_text = candidate["content"]["parts"][0]["text"]
        usage_meta = data.get("usageMetadata", {})

        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            model=model,
            provider="gemini",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content_text),
                    finish_reason=candidate.get("finishReason", "STOP").lower(),
                )
            ],
            usage=Usage(
                prompt_tokens=usage_meta.get("promptTokenCount", 0),
                completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                total_tokens=usage_meta.get("totalTokenCount", 0),
            ),
        )

    async def health_check(self) -> bool:
        """Gemini API-Erreichbarkeit prüfen."""
        if not self._api_key:
            return False
        try:
            resp = await self._client.get(
                f"{GEMINI_API_BASE}/models",
                params={"key": self._api_key},
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
