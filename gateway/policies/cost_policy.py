# gateway/policies/cost_policy.py
# Kostenoptimierungsrichtlinie: Preistabelle und Provider-Auswahl
from __future__ import annotations

from dataclasses import dataclass

from ..models import Provider

# Preistabelle: USD pro 1.000 Token (Stand: Februar 2026)
# Aktualisierung: Werte hier anpassen — kein Code-Neustart nötig wenn via Config geladen
PRICING_TABLE: dict[tuple[Provider, str], tuple[float, float]] = {
    # (Anbieter, Modell): (Eingabe $/1K Token, Ausgabe $/1K Token)
    (Provider.ANTHROPIC, "claude-haiku-4-5-20251001"):  (0.0008,  0.004),
    (Provider.ANTHROPIC, "claude-sonnet-4-5-20250929"): (0.003,   0.015),
    (Provider.OPENAI,    "gpt-5-nano"):                 (0.00005, 0.0004),  # Aug 2025, cheapest
    (Provider.OPENAI,    "gpt-4o-mini"):                (0.00015, 0.0006),
    (Provider.OPENAI,    "gpt-4o"):                     (0.0025,  0.01),
    (Provider.GEMINI,    "gemini-flash"):               (0.0001,  0.0004),   # gemini-2.5-flash-lite
    (Provider.GEMINI,    "gemini-pro"):                 (0.0003,  0.0025),   # gemini-2.5-flash
    (Provider.OLLAMA,    "llama3.2"):                   (0.0,     0.0),
}

# Modellzuordnung: Komplexität → Modell pro Anbieter
SIMPLE_MODELS: dict[Provider, str] = {
    Provider.ANTHROPIC: "claude-haiku-4-5-20251001",
    Provider.OPENAI:    "gpt-5-nano",  # Cheapest OpenAI model (Aug 2025)
    Provider.GEMINI:    "gemini-flash",
    Provider.OLLAMA:    "llama3.2",
}

COMPLEX_MODELS: dict[Provider, str] = {
    Provider.ANTHROPIC: "claude-sonnet-4-5-20250929",
    Provider.OPENAI:    "gpt-4o",
    Provider.GEMINI:    "gemini-pro",
    Provider.OLLAMA:    "llama3.2",
}

# Schwellenwert: unter diesem Kostenunterschied → Latenz bevorzugen
COST_DELTA_THRESHOLD = 0.20


@dataclass
class ProviderCost:
    """Hilfsdatenstruktur für Kostenvergleich."""

    provider: Provider
    model: str
    estimated_cost_usd: float


class CostPolicy:
    """
    Kostenoptimierungsrichtlinie für Provider-Auswahl.

    Token-Schätzung erfolgt in router.py VOR dem API-Aufruf.
    Ausgabe-Token werden als 30% der Eingabe geschätzt (Heuristik — ausreichend für Routing).
    """

    def estimate_cost(
        self, provider: Provider, model: str, tokens_in: int, tokens_out: int
    ) -> float:
        """Gesamtkosten eines API-Aufrufs in USD berechnen."""
        key = (provider, model)
        if key not in PRICING_TABLE:
            return 0.0
        price_in, price_out = PRICING_TABLE[key]
        return (tokens_in * price_in + tokens_out * price_out) / 1000

    def calculate_cost(
        self, provider: Provider, model: str, tokens_in: int, tokens_out: int
    ) -> float:
        """Tatsächliche Kosten nach abgeschlossenem API-Aufruf berechnen."""
        return self.estimate_cost(provider, model, tokens_in, tokens_out)

    def select_cheapest(
        self,
        candidates: list[Provider],
        complexity: str,
        token_estimate: int,
    ) -> tuple[Provider, str]:
        """
        Günstigsten verfügbaren Provider für die erkannte Komplexität auswählen.
        Bei identischen Kosten (z.B. Ollama): ersten Kandidaten nehmen.
        """
        model_map = SIMPLE_MODELS if complexity == "simple" else COMPLEX_MODELS
        # Ausgabe-Token konservativ als 30% der Eingabe schätzen
        estimated_out = max(1, int(token_estimate * 0.3))

        costs: list[ProviderCost] = []
        for provider in candidates:
            model = model_map.get(provider)
            if not model:
                continue
            cost = self.estimate_cost(provider, model, token_estimate, estimated_out)
            costs.append(ProviderCost(provider=provider, model=model, estimated_cost_usd=cost))

        if not costs:
            # Sicherheitsnetz: Anthropic Sonnet als letzter Ausweg
            return Provider.ANTHROPIC, "claude-sonnet-4-5-20250929"

        costs.sort(key=lambda x: x.estimated_cost_usd)
        return costs[0].provider, costs[0].model

    def get_alternative(
        self,
        candidates: list[Provider],
        complexity: str,
        token_estimate: int,
        exclude_provider: Provider,
    ) -> tuple[Provider | None, str | None]:
        """Zweitgünstigsten Provider für Latenzvergleich ermitteln (primären ausschließen)."""
        model_map = SIMPLE_MODELS if complexity == "simple" else COMPLEX_MODELS
        estimated_out = max(1, int(token_estimate * 0.3))

        costs: list[ProviderCost] = []
        for provider in candidates:
            if provider == exclude_provider:
                continue
            model = model_map.get(provider)
            if not model:
                continue
            cost = self.estimate_cost(provider, model, token_estimate, estimated_out)
            costs.append(ProviderCost(provider=provider, model=model, estimated_cost_usd=cost))

        if not costs:
            return None, None

        costs.sort(key=lambda x: x.estimated_cost_usd)
        return costs[0].provider, costs[0].model

    def cost_delta_below_threshold(
        self,
        primary: Provider,
        primary_model: str,
        alt: Provider,
        alt_model: str,
        token_estimate: int,
    ) -> bool:
        """
        True wenn Kostenunterschied zwischen primärem und alternativem Provider <20%.
        Trigger für Latenz-Tie-Breaking in Routing-Stufe 3.
        """
        estimated_out = max(1, int(token_estimate * 0.3))
        cost_primary = self.estimate_cost(primary, primary_model, token_estimate, estimated_out)
        cost_alt = self.estimate_cost(alt, alt_model, token_estimate, estimated_out)

        if cost_primary == 0 and cost_alt == 0:
            return True  # Beide kostenlos (Ollama) → Latenz entscheidet
        if cost_primary == 0:
            return False  # Primär kostenlos, Alt nicht → kein Wechsel

        delta = abs(cost_alt - cost_primary) / cost_primary
        return delta < COST_DELTA_THRESHOLD
