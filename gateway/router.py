# gateway/router.py
# Routing-Entscheidungsmaschine: 5-stufige Hierarchie
# 1. Datenhaltung  2. Kosten  3. Latenz  4. Circuit-Breaker  5. Fallback
from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid

from .cache.semantic_cache import SemanticCache
from .middleware.api_key_auth import get_tenant_budget
from .middleware.token_budget import TokenBudgetManager
from .metrics import (
    CACHE_HIT_TOTAL,
    CIRCUIT_BREAKER_OPEN,
    COST_TOTAL,
    PII_DETECTED_TOTAL,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    TOKENS_TOTAL,
)
from .middleware.audit_logger import AuditLogger
from .middleware.pii_detection import get_pii_detector
from .models import ChatCompletionRequest, ChatCompletionResponse, Provider, ResidencyZone
from .policies.cost_policy import CostPolicy
from .policies.fallback_policy import FallbackPolicy
from .policies.latency_policy import LatencyPolicy
from .policies.residency_policy import ResidencyPolicy
from .providers import ProviderFactory

logger = logging.getLogger(__name__)

# Schwellenwert: einfache vs. komplexe Anfragen (in geschätzten Token)
SIMPLE_REQUEST_TOKEN_THRESHOLD = 500

# Model-zu-Provider-Mapping für explizite Model-Anfragen
MODEL_TO_PROVIDER: dict[str, tuple[Provider, str]] = {
    # Anthropic models
    "claude-opus-4": (Provider.ANTHROPIC, "claude-opus-4"),
    "claude-sonnet-4-5-20250929": (Provider.ANTHROPIC, "claude-sonnet-4-5-20250929"),
    "claude-sonnet-4-5": (Provider.ANTHROPIC, "claude-sonnet-4-5-20250929"),
    "claude-sonnet-4": (Provider.ANTHROPIC, "claude-sonnet-4-5-20250929"),
    "claude-haiku-4-5-20251001": (Provider.ANTHROPIC, "claude-haiku-4-5-20251001"),
    "claude-haiku-4-5": (Provider.ANTHROPIC, "claude-haiku-4-5-20251001"),
    "claude-haiku-4": (Provider.ANTHROPIC, "claude-haiku-4-5-20251001"),
    # OpenAI models
    "gpt-5-nano": (Provider.OPENAI, "gpt-5-nano"),  # Released Aug 2025, cheapest
    "gpt-4o": (Provider.OPENAI, "gpt-4o"),
    "gpt-4o-mini": (Provider.OPENAI, "gpt-4o-mini"),
    "gpt-4-turbo": (Provider.OPENAI, "gpt-4-turbo"),
    "gpt-4": (Provider.OPENAI, "gpt-4"),
    "gpt-3.5-turbo": (Provider.OPENAI, "gpt-3.5-turbo"),
    # Gemini models (2.5 — 2.0 wird am 03.03.2026 eingestellt)
    "gemini-2.5-flash-lite": (Provider.GEMINI, "gemini-2.5-flash-lite"),
    "gemini-2.5-flash": (Provider.GEMINI, "gemini-2.5-flash"),
    "gemini-flash": (Provider.GEMINI, "gemini-flash"),
    "gemini-pro": (Provider.GEMINI, "gemini-pro"),
    # Legacy-Redirects
    "gemini-2.0-flash": (Provider.GEMINI, "gemini-2.0-flash"),
    "gemini-1.5-pro": (Provider.GEMINI, "gemini-1.5-pro"),
    # Ollama models
    "llama3.2": (Provider.OLLAMA, "llama3.2"),
    "llama3": (Provider.OLLAMA, "llama3"),
    "mistral": (Provider.OLLAMA, "mistral"),
}


class RoutingEngine:
    """
    Zentrales Routing-Modul mit 5-stufiger Entscheidungshierarchie.

    Reihenfolge (nicht verhandelbar):
    1. Datenhaltungsprüfung (EU erzwungen)
    2. Kostenoptimierung (günstigster Provider)
    3. Latenzoptimierung (bei <20% Kostenunterschied)
    4. Circuit-Breaker (Failover bei 3 Fehlern/60s)
    5. Fallback-Kette (Claude Sonnet → GPT-4o → Gemini Pro → 503)
    """

    def __init__(self, audit_logger: AuditLogger) -> None:
        self._audit_logger = audit_logger
        self._factory = ProviderFactory()
        self._cost_policy = CostPolicy()
        self._latency_policy = LatencyPolicy()
        self._residency_policy = ResidencyPolicy()
        self._fallback_policy = FallbackPolicy()
        self._pii_detector = get_pii_detector()
        # tiktoken für präzise Token-Schätzung (OpenAI-kompatibel)
        self._tokenizer = None
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning("tiktoken nicht verfügbar — verwende Zeichenanzahl-Approximation")
        # Semantischer Cache (nur wenn REDIS_URL gesetzt — fail-open wenn nicht)
        redis_url = os.getenv("REDIS_URL")
        self._semantic_cache: SemanticCache | None = SemanticCache(redis_url=redis_url) if redis_url else None
        if self._semantic_cache:
            logger.info("Semantischer Cache aktiviert (Redis: %s)", redis_url)
        # Token-Budget-Manager (teilt Redis-URL mit semantischem Cache)
        self._budget_manager = TokenBudgetManager(redis_url=redis_url)

    @property
    def semantic_cache(self) -> SemanticCache | None:
        return self._semantic_cache

    async def initialize(self) -> None:
        """Provider-Clients, Datenbank und Budget-Manager initialisieren."""
        await self._factory.initialize()
        await self._budget_manager.initialize()

    async def shutdown(self) -> None:
        """Alle HTTP-Verbindungen ordnungsgemäß schließen."""
        await self._factory.shutdown()
        await self._budget_manager.close()

    def estimate_tokens(self, messages: list) -> int:
        """Token-Anzahl VOR dem API-Aufruf schätzen (verhindert unnötige Kosten)."""
        text = " ".join(m.content for m in messages)
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback-Approximation: 4 Zeichen ≈ 1 Token (ausreichend für Klassifizierung)
        return max(1, len(text) // 4)

    def classify_complexity(self, token_count: int, has_tools: bool) -> str:
        """
        Anfragekomplexität klassifizieren.
        simple: <500 Token UND keine Tools → Haiku/GPT-4o-mini/Gemini Flash
        complex: >=500 Token ODER Tools → Sonnet/GPT-4o/Gemini Pro
        """
        if token_count < SIMPLE_REQUEST_TOKEN_THRESHOLD and not has_tools:
            return "simple"
        return "complex"

    def _get_provider_for_model(self, model: str) -> tuple[Provider, str] | None:
        """
        Map model string to (provider, canonical_model_name).
        Returns None if model is unknown or "auto".
        """
        if model == "auto":
            return None
        return MODEL_TO_PROVIDER.get(model)

    async def route(
        self, request: ChatCompletionRequest, tenant_id: str
    ) -> ChatCompletionResponse:
        """
        Haupt-Routing-Funktion. Implementiert die 5-stufige Hierarchie.
        Jede Stufe wird vollständig ausgeführt bevor zur nächsten gewechselt wird.
        """
        request_id = str(uuid.uuid4())

        # Schritt -1: PII-Erkennung und -Redaktion VOR jedem API-Aufruf
        pii_detected = False
        pii_types: list[str] = []
        original_messages = []

        for message in request.messages:
            # PII-Erkennung in jedem Message-Content
            entities = self._pii_detector.detect(message.content)
            if entities:
                pii_detected = True
                detected_types = list({e.type for e in entities})
                pii_types.extend(detected_types)

                # Prometheus: PII-Typen zählen
                for pii_type in detected_types:
                    PII_DETECTED_TOTAL.labels(pii_type=pii_type).inc()

                # Redaktiere PII für LLM-Provider-Aufruf
                redacted_content, _ = self._pii_detector.redact(message.content)
                original_messages.append(message.content)
                message.content = redacted_content

                logger.warning(
                    "PII erkannt in Anfrage %s: %s — redaktiert vor LLM-Aufruf",
                    request_id, detected_types
                )

        # Prompt-Hash VOR jedem API-Aufruf berechnen (Original-Prompt, vor Redaktion)
        prompt_for_hash = (
            " ".join(original_messages) if original_messages
            else " ".join(m.content for m in request.messages)
        )
        prompt_hash = hashlib.sha256(prompt_for_hash.encode()).hexdigest()

        # Semantischer Cache-Check (nach PII-Redaktion, vor Provider-Routing)
        if self._semantic_cache:
            cached = await self._semantic_cache.get(request.messages)
            if cached:
                cached_response = ChatCompletionResponse.model_validate(cached["response"])
                CACHE_HIT_TOTAL.labels(
                    provider=cached_response.provider, model=cached_response.model
                ).inc()
                REQUEST_COUNT.labels(
                    provider=cached_response.provider, model=cached_response.model,
                    tenant_id=tenant_id, status="success",
                ).inc()
                await self._audit_logger.log(
                    request_id=request_id,
                    tenant_id=tenant_id,
                    prompt_hash=prompt_hash,
                    model=cached_response.model,
                    provider=cached_response.provider,
                    tokens_in=cached.get("tokens_in", 0),
                    tokens_out=cached.get("tokens_out", 0),
                    cost_usd=cached.get("cost_usd", 0.0),
                    pii_detected=pii_detected,
                    pii_types=pii_types,
                    cache_hit=True,
                )
                logger.info(
                    "Cache-Treffer für Anfrage %s (Stufe: %s)",
                    request_id, cached.get("_cache_level", "exact"),
                )
                return cached_response

        # Schritt 0a: Token-Budget prüfen (vor API-Aufruf — kein Budget verbrauchen wenn erschöpft)
        # Systemkonten (admin, public, anonymous) sind von Budget-Durchsetzung ausgenommen
        _system_tenants = frozenset({"admin", "public", "anonymous"})
        if tenant_id not in _system_tenants:
            budget_limit = await get_tenant_budget(tenant_id)
            exceeded, used = await self._budget_manager.is_budget_exceeded(tenant_id, budget_limit)
            if exceeded:
                from fastapi import HTTPException
                logger.warning(
                    "Token-Budget erschöpft für Mandant %s: %d/%d Token verbraucht",
                    tenant_id, used, budget_limit,
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "monthly_token_budget_exceeded",
                        "budget": budget_limit,
                        "used": used,
                    },
                )

        # Schritt 0b: Token-Schätzung VOR jedem API-Aufruf (nach PII-Redaktion)
        token_estimate = self.estimate_tokens(request.messages)
        # Tool-Use-Erkennung: Erweiterbar für zukünftige Tool-Calls
        has_tools = False
        complexity = self.classify_complexity(token_estimate, has_tools)
        logger.info(
            "Anfrage %s: %d Token geschätzt, Komplexität=%s",
            request_id, token_estimate, complexity
        )

        # Stufe 1: Datenhaltungsprüfung (PRIORITÄT 1 — nicht verhandelbar)
        eu_only = self._residency_policy.requires_eu_only(request.residency_zone)
        candidate_providers = self._residency_policy.filter_providers(eu_only)

        # Production-Filter: Ollama nur lokal verfügbar (nicht auf Render/Cloud)
        is_production = os.getenv("DATABASE_URL", "").startswith("postgres")
        if is_production and Provider.OLLAMA in candidate_providers:
            candidate_providers.remove(Provider.OLLAMA)
            logger.info("Ollama deaktiviert (Production-Modus)")

        if eu_only:
            logger.info("EU-Datenhaltung aktiv: %d Provider verfügbar", len(candidate_providers))

        # Stufe 1.5: Explizite Model-Anfrage prüfen (falls vorhanden)
        explicit_provider_model = self._get_provider_for_model(request.model)
        if explicit_provider_model:
            explicit_provider, explicit_model = explicit_provider_model
            # EU-Konformitätsprüfung: Explizites Model darf EU-Anforderungen nicht verletzen
            if eu_only and explicit_provider not in candidate_providers:
                from fastapi import HTTPException
                logger.error(
                    "Model '%s' (Provider %s) nicht EU-konform — Anfrage abgelehnt",
                    request.model, explicit_provider.value
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Model '{request.model}' ist nicht EU-konform. "
                           f"Verfügbare EU-Provider: {[p.value for p in candidate_providers]}"
                )
            # Explizites Model ist gültig und EU-konform → verwenden
            logger.info("Explizite Model-Anfrage: %s (Provider %s)", request.model, explicit_provider.value)
            selected_provider, selected_model = explicit_provider, explicit_model
        else:
            # Stufe 2: Kostenoptimierung — günstigsten Provider wählen
            selected_provider, selected_model = self._cost_policy.select_cheapest(
                candidate_providers, complexity, token_estimate
            )

        # Stufe 3: Latenz-Tie-Breaking — NUR bei automatischer Routing (nicht bei explizitem Model)
        if not explicit_provider_model:
            alt_provider, alt_model = self._cost_policy.get_alternative(
                candidate_providers, complexity, token_estimate, selected_provider
            )
            if alt_provider and alt_model:
                if self._cost_policy.cost_delta_below_threshold(
                    selected_provider, selected_model, alt_provider, alt_model, token_estimate
                ):
                    if self._latency_policy.prefer_lower_latency(selected_provider, alt_provider):
                        logger.info(
                            "Latenz-Optimierung: %s → %s (ähnliche Kosten, niedrigere Latenz)",
                            selected_provider.value, alt_provider.value,
                        )
                        selected_provider, selected_model = alt_provider, alt_model

        # Stufe 4: Circuit-Breaker — bei offenem Schaltkreis Fallback aktivieren
        if not self._fallback_policy.is_circuit_closed(selected_provider):
            logger.warning(
                "Circuit-Breaker offen für %s — Fallback aktiviert", selected_provider.value
            )
            selected_provider, selected_model = await self._fallback_policy.get_fallback(
                candidate_providers
            )

        # API-Aufruf mit automatischem Fallback bei Fehler
        response = None
        last_error = None

        # Erste Versuch mit ausgewähltem Provider
        provider_client = self._factory.get(selected_provider)
        start_time = time.monotonic()
        try:
            response = await provider_client.complete(request, selected_model)
            latency_ms = (time.monotonic() - start_time) * 1000
            self._latency_policy.record(selected_provider, latency_ms)
            self._fallback_policy.record_success(selected_provider)
            # Prometheus: Latenz und Circuit-Breaker-Status
            REQUEST_LATENCY.labels(
                provider=selected_provider.value, model=selected_model
            ).observe(latency_ms)
            CIRCUIT_BREAKER_OPEN.labels(provider=selected_provider.value).set(0)
            logger.info(
                "Anfrage %s: %s/%s in %.0fms", request_id,
                selected_provider.value, selected_model, latency_ms
            )
        except Exception as exc:
            self._fallback_policy.record_failure(selected_provider)
            CIRCUIT_BREAKER_OPEN.labels(provider=selected_provider.value).set(
                0 if self._fallback_policy.is_circuit_closed(selected_provider) else 1
            )
            REQUEST_COUNT.labels(
                provider=selected_provider.value, model=selected_model,
                tenant_id=tenant_id, status="error"
            ).inc()
            logger.error("Provider %s fehlgeschlagen: %s", selected_provider.value, exc)
            last_error = exc

            # Fallback-Versuch wenn primärer Provider fehlschlägt
            try:
                fallback_provider, fallback_model = await self._fallback_policy.get_fallback(
                    candidate_providers
                )
                logger.warning(
                    "Fallback-Versuch: %s → %s", selected_provider.value, fallback_provider.value
                )
                fallback_client = self._factory.get(fallback_provider)
                start_time = time.monotonic()
                response = await fallback_client.complete(request, fallback_model)
                latency_ms = (time.monotonic() - start_time) * 1000
                self._latency_policy.record(fallback_provider, latency_ms)
                self._fallback_policy.record_success(fallback_provider)
                REQUEST_LATENCY.labels(
                    provider=fallback_provider.value, model=fallback_model
                ).observe(latency_ms)
                CIRCUIT_BREAKER_OPEN.labels(provider=fallback_provider.value).set(0)
                logger.info(
                    "✅ Fallback erfolgreich: %s/%s in %.0fms",
                    fallback_provider.value, fallback_model, latency_ms
                )
                # Fallback erfolgreich → verwende Fallback-Provider für Kosten/Audit
                selected_provider, selected_model = fallback_provider, fallback_model
            except Exception as fallback_exc:
                logger.error("Fallback fehlgeschlagen: %s", fallback_exc)
                # Beide Provider gescheitert → ursprünglichen Fehler werfen
                raise last_error

        # Wenn kein Response (sollte nicht passieren), Fehler werfen
        if response is None:
            raise last_error or Exception("Keine Antwort von Providern")

        # Kosten berechnen und in Antwort eintragen
        cost_usd = self._cost_policy.calculate_cost(
            selected_provider,
            selected_model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        response.usage.cost_usd = cost_usd

        # Prometheus: erfolgreiche Anfrage, Kosten, Token
        REQUEST_COUNT.labels(
            provider=selected_provider.value, model=selected_model,
            tenant_id=tenant_id, status="success"
        ).inc()
        COST_TOTAL.labels(
            provider=selected_provider.value, model=selected_model, tenant_id=tenant_id
        ).inc(cost_usd)
        TOKENS_TOTAL.labels(
            direction="input", provider=selected_provider.value, model=selected_model
        ).inc(response.usage.prompt_tokens)
        TOKENS_TOTAL.labels(
            direction="output", provider=selected_provider.value, model=selected_model
        ).inc(response.usage.completion_tokens)

        # Antwort im semantischen Cache speichern (für zukünftige identische/ähnliche Anfragen)
        if self._semantic_cache:
            await self._semantic_cache.set(
                request.messages,
                response,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                cost_usd,
            )

        # Stufe 5: Audit-Log — NUR SHA256-Hash des Prompts (kein Klartext)
        # prompt_hash wurde bereits früher berechnet (nach PII-Redaktion, vor Cache-Check)
        await self._audit_logger.log(
            request_id=request_id,
            tenant_id=tenant_id,
            prompt_hash=prompt_hash,
            model=selected_model,
            provider=selected_provider.value,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            cost_usd=cost_usd,
            pii_detected=pii_detected,
            pii_types=pii_types,
        )

        # Stufe 6: Monatlichen Token-Verbrauch aktualisieren (nach erfolgreichem API-Aufruf)
        if tenant_id not in _system_tenants:
            total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
            await self._budget_manager.increment_usage(tenant_id, total_tokens)

        return response

    async def health(self) -> dict:
        """Gesundheitsstatus aller konfigurierten Provider abrufen."""
        return {
            p.value: self._fallback_policy.is_circuit_closed(p)
            for p in Provider
        }
