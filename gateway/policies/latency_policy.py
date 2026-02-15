# gateway/policies/latency_policy.py
# Latenz-Tracking mit gleitendem Durchschnitt (Ring-Puffer pro Anbieter)
from __future__ import annotations

from collections import defaultdict, deque

from ..models import Provider

# Fenstergröße: letzte N Anfragen für gleitenden Durchschnitt
LATENCY_WINDOW_SIZE = 20


class LatencyPolicy:
    """
    Latenz-Tracking und Tie-Breaking zwischen Providern.
    Verwendet Ring-Puffer (deque) der letzten 20 Latenzmessungen.
    Unbekannte Provider gelten als unendlich langsam (float('inf')).
    """

    def __init__(self) -> None:
        # Ring-Puffer pro Provider — automatisch begrenzt auf LATENCY_WINDOW_SIZE
        self._latencies: dict[Provider, deque[float]] = defaultdict(
            lambda: deque(maxlen=LATENCY_WINDOW_SIZE)
        )

    def record(self, provider: Provider, latency_ms: float) -> None:
        """Gemessene Latenz einer abgeschlossenen Anfrage speichern."""
        self._latencies[provider].append(latency_ms)

    def average_latency(self, provider: Provider) -> float:
        """
        Gleitenden Durchschnitt der letzten 20 Anfragen berechnen.
        Gibt float('inf') zurück wenn keine Messungen vorhanden (konservatives Verhalten).
        """
        measurements = self._latencies[provider]
        if not measurements:
            return float("inf")
        return sum(measurements) / len(measurements)

    def prefer_lower_latency(self, provider_a: Provider, provider_b: Provider) -> bool:
        """
        True wenn Provider B schneller als Provider A ist.
        Wird in Routing-Stufe 3 aufgerufen wenn Kosten-Delta <20%.
        """
        return self.average_latency(provider_b) < self.average_latency(provider_a)
