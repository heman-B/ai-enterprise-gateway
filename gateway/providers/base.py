# gateway/providers/base.py
# Abstrakte Basisklasse: gemeinsames Interface für alle LLM-Anbieter
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from ..models import ChatCompletionRequest, ChatCompletionResponse


class BaseProvider(ABC):
    """
    Abstrakte Basis für alle LLM-Anbieter-Implementierungen.
    Jede konkrete Klasse MUSS complete() und health_check() implementieren.
    Gemeinsamer HTTP-Client wird hier verwaltet.
    """

    def __init__(self) -> None:
        # Gemeinsamer async httpx-Client — wird in initialize() erstellt
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Async HTTP-Client mit Verbindungspool erstellen."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def shutdown(self) -> None:
        """HTTP-Client ordnungsgemäß schließen (alle Verbindungen freigeben)."""
        if self._client:
            await self._client.aclose()

    @abstractmethod
    async def complete(
        self, request: ChatCompletionRequest, model: str
    ) -> ChatCompletionResponse:
        """Chat-Completion mit dem angegebenen Modell durchführen."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verfügbarkeit des Anbieters prüfen (max. 5s Timeout)."""
        ...
