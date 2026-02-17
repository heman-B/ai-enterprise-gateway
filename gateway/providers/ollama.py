# gateway/providers/ollama.py
# Ollama Lokaler Anbieter: $0/Token, keine Datenweitergabe
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

# Standard-Ollama-URL (überschreibbar via Umgebungsvariable)
OLLAMA_DEFAULT_URL = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    """
    Ollama Lokaler LLM-Anbieter.
    Kosten: $0 (lokal — keine API-Gebühren, kein Datentransfer)
    Standardmodell: llama3.2
    Erfordert laufenden Ollama-Server unter $OLLAMA_URL.
    """

    def __init__(self) -> None:
        super().__init__()
        self._base_url = os.getenv("OLLAMA_URL", OLLAMA_DEFAULT_URL)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(is_retryable_error),
        reraise=True,
    )
    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Chat-Completion über lokale Ollama-Instanz (OpenAI-kompatible API)."""
        assert self._client is not None, "Provider nicht initialisiert"

        # Präfix entfernen falls vorhanden (z.B. 'ollama/llama3.2' → 'llama3.2')
        ollama_model = model.replace("ollama/", "")

        payload = {
            "model": ollama_model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        }

        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=120.0,  # Lokale Modelle können beim ersten Token langsamer sein
        )
        response.raise_for_status()
        data = response.json()

        content = data["message"]["content"]
        # Ollama liefert Token-Zahlen in eval_count / prompt_eval_count
        prompt_tokens = data.get("prompt_eval_count", max(1, len(content) // 4))
        completion_tokens = data.get("eval_count", max(1, len(content) // 4))

        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            model=model,
            provider="ollama",
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def health_check(self) -> bool:
        """Ollama-Server-Erreichbarkeit prüfen."""
        try:
            resp = await self._client.get(
                f"{self._base_url}/api/tags",
                timeout=3.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
