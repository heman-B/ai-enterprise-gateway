# gateway/models.py
# Pydantic v2 Datenschemas: alle Ein- und Ausgaben des Gateways
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Provider(str, Enum):
    """Unterstützte LLM-Anbieter im Gateway."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class ResidencyZone(str, Enum):
    """Datenhaltungszonen für DSGVO-Konformität."""

    EU = "eu"
    US = "us"
    ANY = "any"


class ChatMessage(BaseModel):
    """Einzelne Nachricht in einer Konversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Eingehende Chat-Completion-Anfrage (OpenAI-kompatibel)."""

    messages: list[ChatMessage]
    model: str = "auto"
    max_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    residency_zone: ResidencyZone = ResidencyZone.ANY


class Choice(BaseModel):
    """Einzelne Antwort-Option vom LLM-Anbieter."""

    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    """Token-Verbrauch und Kostenerfassung pro Anfrage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0


class ChatCompletionResponse(BaseModel):
    """Standardisierte Gateway-Antwort (OpenAI-Format)."""

    id: str
    object: str = "chat.completion"
    model: str
    provider: str
    choices: list[Choice]
    usage: Usage


class HealthResponse(BaseModel):
    """Gateway-Gesundheitsstatus mit Provider-Verfügbarkeit."""

    status: str
    providers: dict[str, Any]


class TenantKeyCreate(BaseModel):
    """Anfrage zur API-Schlüssel-Erstellung für einen Mandanten."""

    tenant_id: str
    rate_limit_per_minute: int = Field(default=100, ge=1, le=10000)
