# gateway/policies/fallback_policy.py
# Circuit-Breaker: 3 Fehler in 60s → Schaltkreis öffnet, Fallback-Kette aktiviert
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from ..models import Provider

logger = logging.getLogger(__name__)

# Circuit-Breaker-Konfiguration
FAILURE_THRESHOLD = 3       # Maximale Fehler im Zeitfenster bevor Schaltkreis öffnet
RECOVERY_TIMEOUT_SEC = 60   # Sekunden bis automatischer Wiederherstellungsversuch

# Fallback-Kette (in Prioritätsreihenfolge)
# Bei Ausfall des primären Providers wird diese Kette der Reihe nach durchlaufen
FALLBACK_CHAIN: list[tuple[Provider, str]] = [
    (Provider.ANTHROPIC, "claude-sonnet-4-5-20250929"),
    (Provider.OPENAI,    "gpt-4o"),
    (Provider.GEMINI,    "gemini-pro"),
    (Provider.OLLAMA,    "llama3.2"),  # Lokaler Fallback (€0, EU-konform)
]


@dataclass
class CircuitState:
    """Zustand eines einzelnen Provider-Schaltkreises."""

    failure_timestamps: list[float] = field(default_factory=list)
    last_failure_time: float = 0.0
    is_open: bool = False


class FallbackPolicy:
    """
    Circuit-Breaker-Pattern für Provider-Ausfallsicherheit.

    Logik:
    - 3 Fehler innerhalb von 60 Sekunden → Schaltkreis öffnet
    - Offener Schaltkreis → alle Anfragen sofort weitergeleitet
    - Nach 60 Sekunden: Schaltkreis schließt automatisch (Half-Open-Versuch)
    - Bei Erfolg: Fehlerzähler komplett zurückgesetzt
    """

    def __init__(self) -> None:
        self._circuits: dict[Provider, CircuitState] = {
            p: CircuitState() for p in Provider
        }

    def record_success(self, provider: Provider) -> None:
        """Erfolg registrieren: Fehlerzähler und Schaltkreis-Status zurücksetzen."""
        circuit = self._circuits[provider]
        circuit.failure_timestamps.clear()
        circuit.is_open = False

    def record_failure(self, provider: Provider) -> None:
        """
        Fehler registrieren und Circuit-Breaker-Logik anwenden.
        Veraltete Timestamps (>60s) werden vor der Zählung entfernt.
        """
        circuit = self._circuits[provider]
        now = time.monotonic()
        circuit.last_failure_time = now
        circuit.failure_timestamps.append(now)

        # Fehler außerhalb des Zeitfensters entfernen
        cutoff = now - RECOVERY_TIMEOUT_SEC
        circuit.failure_timestamps = [
            ts for ts in circuit.failure_timestamps if ts > cutoff
        ]

        # Schaltkreis öffnen wenn Schwellenwert erreicht
        if len(circuit.failure_timestamps) >= FAILURE_THRESHOLD:
            if not circuit.is_open:
                circuit.is_open = True
                logger.warning(
                    "🔴 Circuit-Breaker GEÖFFNET für %s (%d Fehler in %ds)",
                    provider.value, FAILURE_THRESHOLD, RECOVERY_TIMEOUT_SEC,
                )

    def is_circuit_closed(self, provider: Provider) -> bool:
        """
        True wenn Provider verfügbar (Schaltkreis geschlossen).
        Prüft automatische Wiederherstellung nach RECOVERY_TIMEOUT_SEC.
        """
        circuit = self._circuits[provider]
        if not circuit.is_open:
            return True

        # Automatische Wiederherstellung nach Timeout
        if time.monotonic() - circuit.last_failure_time > RECOVERY_TIMEOUT_SEC:
            circuit.is_open = False
            circuit.failure_timestamps.clear()
            logger.info("🟢 Circuit-Breaker GESCHLOSSEN für %s (automatische Wiederherstellung)",
                        provider.value)
            return True

        return False

    async def get_fallback(
        self, available_providers: list[Provider]
    ) -> tuple[Provider, str]:
        """
        Nächsten verfügbaren Provider aus Fallback-Kette wählen.
        Fallback-Reihenfolge: Claude Sonnet → GPT-4o → Gemini Pro → HTTP 503
        """
        for provider, model in FALLBACK_CHAIN:
            if provider in available_providers and self.is_circuit_closed(provider):
                logger.info("↪️  Fallback aktiviert: %s/%s", provider.value, model)
                return provider, model

        # Alle Provider ausgefallen — HTTP 503 zurückgeben
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=(
                "Alle LLM-Provider sind momentan nicht verfügbar. "
                "Bitte in 60 Sekunden erneut versuchen."
            ),
        )
