# gateway/providers/__init__.py
# Anbieter-Factory: erstellt und verwaltet alle LLM-Provider-Instanzen
from __future__ import annotations

import logging

from ..models import Provider
from .anthropic import AnthropicProvider
from .base import BaseProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory-Klasse für alle LLM-Anbieter.
    Verwaltet Initialisierung und Shutdown der HTTP-Clients.
    """

    def __init__(self) -> None:
        # Alle Provider werden instanziiert — aktiv nur wenn API-Key gesetzt
        self._providers: dict[Provider, BaseProvider] = {
            Provider.ANTHROPIC: AnthropicProvider(),
            Provider.OPENAI: OpenAIProvider(),
            Provider.GEMINI: GeminiProvider(),
            Provider.OLLAMA: OllamaProvider(),
        }

    async def initialize(self) -> None:
        """Alle Provider-HTTP-Clients initialisieren (Verbindungspool aufbauen)."""
        for provider_enum, provider in self._providers.items():
            await provider.initialize()
            logger.info("Provider %s initialisiert", provider_enum.value)

    async def shutdown(self) -> None:
        """Alle Provider-Verbindungen ordnungsgemäß schließen."""
        for provider in self._providers.values():
            await provider.shutdown()

    def get(self, provider: Provider) -> BaseProvider:
        """Provider-Instanz nach Enum-Wert abrufen."""
        if provider not in self._providers:
            raise ValueError(f"Unbekannter Provider: {provider}")
        return self._providers[provider]
