# gateway/policies/residency_policy.py
# DSGVO-Datenhaltungsrichtlinie: EU-Daten bleiben in der EU (nicht verhandelbar)
from __future__ import annotations

from ..models import Provider, ResidencyZone

# Provider mit EU-Datenhaltungsgarantie
# Anthropic: DSGVO-DPA verfügbar, EU-Verarbeitung möglich
# Ollama: lokal ausgeführt — Daten verlassen niemals den Host
EU_COMPLIANT_PROVIDERS: frozenset[Provider] = frozenset({
    Provider.ANTHROPIC,
    Provider.OLLAMA,
})

# Alle verfügbaren Provider (Reihenfolge für Kostenoptimierung)
ALL_PROVIDERS: list[Provider] = list(Provider)


class ResidencyPolicy:
    """
    Datenhaltungskonformitätsprüfung.
    EU-Anforderung ist PRIORITÄT 1 im Routing — wird vor allen anderen Stufen geprüft.
    Kann nicht durch Kosten- oder Latenzoptimierung umgangen werden.
    """

    def requires_eu_only(self, zone: ResidencyZone) -> bool:
        """True wenn Anfrage ausschließlich EU-Verarbeitung erfordert."""
        return zone == ResidencyZone.EU

    def filter_providers(self, eu_only: bool) -> list[Provider]:
        """
        Zulässige Provider nach Datenhaltungsanforderung filtern.
        Bei eu_only=True: nur EU_COMPLIANT_PROVIDERS zurückgeben.
        """
        if eu_only:
            return [p for p in ALL_PROVIDERS if p in EU_COMPLIANT_PROVIDERS]
        return ALL_PROVIDERS.copy()

    def is_provider_eu_compliant(self, provider: Provider) -> bool:
        """Einzelnen Provider auf EU-Konformität prüfen."""
        return provider in EU_COMPLIANT_PROVIDERS
