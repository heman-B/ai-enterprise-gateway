# gateway/mcp_server.py
# MCP-Server: Exponiert das Gateway als Tool-Interface für KI-Agenten
# Ermöglicht Agenten (Claude, Cursor, Joule) die Gateway-Funktionen zu nutzen
from __future__ import annotations

import logging
from typing import Optional

from fastmcp import FastMCP

from .models import ChatCompletionRequest, ChatMessage, Provider, ResidencyZone
from .policies.cost_policy import COMPLEX_MODELS, SIMPLE_MODELS
from .router import MODEL_TO_PROVIDER, RoutingEngine

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="Enterprise AI Gateway",
    instructions=(
        "Enterprise LLM Gateway mit 5-stufiger Routing-Hierarchie. "
        "Routet Anfragen kostenoptimiert über Claude, GPT, Gemini und Ollama. "
        "Unterstützt EU-Datenhaltung, PII-Erkennung und automatisches Failover."
    ),
)

# Referenz auf die RoutingEngine — wird in main.py gesetzt
_engine: RoutingEngine | None = None


def set_engine(engine: RoutingEngine) -> None:
    """RoutingEngine-Referenz setzen (aufgerufen beim App-Startup)."""
    global _engine
    _engine = engine


@mcp.tool()
async def chat(
    message: str,
    model: str = "auto",
    residency_zone: str = "any",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Sende eine Nachricht durch das Enterprise AI Gateway.

    Das Gateway routet automatisch zum optimalen LLM-Provider basierend auf:
    - Kosten (günstigster Provider für die Anfragekomplexität)
    - Datenhaltung (EU-Daten bleiben bei EU-Endpunkten)
    - Verfügbarkeit (automatisches Failover bei Provider-Ausfällen)

    Args:
        message: Die Nachricht an das LLM
        model: Modellname oder "auto" für kostenoptimiertes Routing
        residency_zone: "eu" für EU-Datenhaltung, "any" für alle Provider
        max_tokens: Maximale Antwortlänge in Token
        temperature: Kreativitätsgrad (0.0 = deterministisch, 1.0 = kreativ)

    Returns:
        Die LLM-Antwort mit Provider- und Kosteninformationen
    """
    if _engine is None:
        return "Fehler: Gateway nicht initialisiert"

    zone = ResidencyZone.EU if residency_zone == "eu" else ResidencyZone.ANY
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=message)],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        residency_zone=zone,
    )

    response = await _engine.route(request, tenant_id="mcp-agent")

    return (
        f"{response.choices[0].message.content}\n\n"
        f"---\n"
        f"Provider: {response.provider} | Model: {response.model} | "
        f"Tokens: {response.usage.total_tokens} | "
        f"Cost: ${response.usage.cost_usd:.6f}"
    )


@mcp.tool()
async def list_models() -> str:
    """
    Zeige alle verfügbaren Modelle und ihre Provider.

    Gibt eine Übersicht aller konfigurierten LLM-Modelle zurück,
    gruppiert nach Provider (Anthropic, OpenAI, Gemini, Ollama).
    """
    by_provider: dict[str, list[str]] = {}
    for model_name, (provider, _) in MODEL_TO_PROVIDER.items():
        by_provider.setdefault(provider.value, []).append(model_name)

    lines = ["Verfügbare Modelle:\n"]
    for provider, models in sorted(by_provider.items()):
        lines.append(f"  {provider}:")
        for m in sorted(models):
            lines.append(f"    - {m}")
    lines.append(f"\nTipp: model='auto' für kostenoptimiertes Routing")
    return "\n".join(lines)


@mcp.tool()
async def health() -> str:
    """
    Prüfe den Gesundheitsstatus aller LLM-Provider.

    Zeigt den Circuit-Breaker-Status pro Provider:
    - closed = verfügbar (Anfragen werden weitergeleitet)
    - open = ausgefallen (Anfragen werden zum Fallback umgeleitet)
    """
    if _engine is None:
        return "Fehler: Gateway nicht initialisiert"

    status = await _engine.health()
    lines = ["Provider-Status:\n"]
    for provider, is_healthy in status.items():
        icon = "OK" if is_healthy else "AUSFALL"
        lines.append(f"  {provider}: {icon}")
    return "\n".join(lines)


@mcp.tool()
async def estimate_cost(
    message: str,
    model: str = "auto",
) -> str:
    """
    Schätze die Kosten einer Anfrage BEVOR sie gesendet wird.

    Nützlich für Budget-Kontrolle: Erst Kosten prüfen, dann entscheiden
    ob die Anfrage gesendet werden soll.

    Args:
        message: Die geplante Nachricht
        model: Modellname oder "auto"

    Returns:
        Geschätzte Kosten in USD mit Provider-Details
    """
    if _engine is None:
        return "Fehler: Gateway nicht initialisiert"

    token_estimate = _engine.estimate_tokens(
        [ChatMessage(role="user", content=message)]
    )
    complexity = _engine.classify_complexity(token_estimate, has_tools=False)

    model_map = SIMPLE_MODELS if complexity == "simple" else COMPLEX_MODELS
    estimated_out = max(1, int(token_estimate * 0.3))

    lines = [
        f"Kostenschätzung:",
        f"  Geschätzte Token: {token_estimate}",
        f"  Komplexität: {complexity}",
        f"  Kosten pro Provider:",
    ]

    for provider in Provider:
        provider_model = model_map.get(provider)
        if provider_model:
            cost = _engine._cost_policy.estimate_cost(
                provider, provider_model, token_estimate, estimated_out
            )
            lines.append(f"    {provider.value}/{provider_model}: ${cost:.6f}")

    return "\n".join(lines)
