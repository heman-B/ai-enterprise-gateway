# tests/test_router.py
# Pytest-Suite für Routing-Engine, Policies und Provider-Auswahl
# Alle externen HTTP-Aufrufe werden mit respx gemockt
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Provider,
    ResidencyZone,
    Usage,
)
from gateway.policies.cost_policy import CostPolicy
from gateway.policies.fallback_policy import FallbackPolicy
from gateway.policies.latency_policy import LatencyPolicy
from gateway.policies.residency_policy import ResidencyPolicy
from gateway.router import RoutingEngine


# ─── Hilfsfunktionen ────────────────────────────────────────────────────────


def make_request(
    content: str = "Hallo!",
    residency_zone: ResidencyZone = ResidencyZone.ANY,
    max_tokens: int = 100,
) -> ChatCompletionRequest:
    """Test-Anfrage erstellen."""
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=content)],
        residency_zone=residency_zone,
        max_tokens=max_tokens,
    )


def make_response(
    provider: str = "anthropic",
    model: str = "claude-haiku-4-5-20251001",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ChatCompletionResponse:
    """Mock-Antwort für Tests erstellen."""
    return ChatCompletionResponse(
        id="test-id-abc123",
        model=model,
        provider=provider,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content="Berlin."),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ─── CostPolicy Tests ────────────────────────────────────────────────────────


class TestCostPolicy:
    def setup_method(self):
        self.policy = CostPolicy()

    def test_haiku_cost_calculation(self):
        """Haiku: $0.0008/1K Input + $0.004/1K Output."""
        cost = self.policy.estimate_cost(
            Provider.ANTHROPIC, "claude-haiku-4-5-20251001", 1000, 500
        )
        expected = (1000 * 0.0008 + 500 * 0.004) / 1000
        assert abs(cost - expected) < 1e-9

    def test_ollama_always_free(self):
        """Ollama-Kosten müssen immer 0.0 sein."""
        cost = self.policy.estimate_cost(Provider.OLLAMA, "llama3.2", 50000, 20000)
        assert cost == 0.0

    def test_unknown_model_returns_zero(self):
        """Unbekanntes Modell → 0.0 (kein Absturz)."""
        cost = self.policy.estimate_cost(Provider.OPENAI, "gpt-9000", 1000, 500)
        assert cost == 0.0

    def test_gpt5_nano_cheapest_for_simple(self):
        """GPT-5 Nano ist günstigster Provider für einfache Anfragen (seit 2.5-Preisanpassung)."""
        provider, model = self.policy.select_cheapest(
            [Provider.ANTHROPIC, Provider.OPENAI, Provider.GEMINI],
            complexity="simple",
            token_estimate=200,
        )
        assert provider == Provider.OPENAI
        assert model == "gpt-5-nano"

    def test_ollama_wins_when_available(self):
        """Ollama ($0) schlägt alle kostenpflichtigen Provider."""
        provider, model = self.policy.select_cheapest(
            [Provider.ANTHROPIC, Provider.OPENAI, Provider.GEMINI, Provider.OLLAMA],
            complexity="complex",
            token_estimate=1000,
        )
        assert provider == Provider.OLLAMA

    def test_fallback_provider_on_empty_candidates(self):
        """Leere Kandidatenliste → Anthropic Sonnet als Sicherheitsnetz."""
        provider, model = self.policy.select_cheapest([], "complex", 500)
        assert provider == Provider.ANTHROPIC
        assert "sonnet" in model

    def test_cost_delta_haiku_vs_sonnet_exceeds_threshold(self):
        """Haiku vs. Sonnet hat >20% Kostenunterschied → kein Latenz-Switch."""
        result = self.policy.cost_delta_below_threshold(
            Provider.ANTHROPIC, "claude-haiku-4-5-20251001",
            Provider.ANTHROPIC, "claude-sonnet-4-5-20250929",
            500,
        )
        assert result is False

    def test_cost_delta_both_free_is_below_threshold(self):
        """Beide Provider kostenlos → Kostenunterschied = 0 → Latenz entscheidet."""
        result = self.policy.cost_delta_below_threshold(
            Provider.OLLAMA, "llama3.2",
            Provider.OLLAMA, "llama3.2",
            500,
        )
        assert result is True

    def test_get_alternative_excludes_primary(self):
        """get_alternative() darf primären Provider nicht zurückgeben."""
        alt_provider, alt_model = self.policy.get_alternative(
            [Provider.ANTHROPIC, Provider.OPENAI, Provider.GEMINI],
            complexity="simple",
            token_estimate=200,
            exclude_provider=Provider.GEMINI,
        )
        assert alt_provider != Provider.GEMINI
        assert alt_provider is not None


# ─── ResidencyPolicy Tests ───────────────────────────────────────────────────


class TestResidencyPolicy:
    def setup_method(self):
        self.policy = ResidencyPolicy()

    def test_eu_zone_requires_eu_only(self):
        """EU-Zone muss EU-exklusive Verarbeitung erfordern."""
        assert self.policy.requires_eu_only(ResidencyZone.EU) is True

    def test_any_zone_does_not_require_eu(self):
        """ANY-Zone erlaubt globales Routing."""
        assert self.policy.requires_eu_only(ResidencyZone.ANY) is False

    def test_us_zone_does_not_require_eu(self):
        """US-Zone erfordert keine EU-Einschränkung."""
        assert self.policy.requires_eu_only(ResidencyZone.US) is False

    def test_eu_filter_excludes_openai(self):
        """Bei EU-only: OpenAI nicht in der Liste (nicht EU-konform)."""
        providers = self.policy.filter_providers(eu_only=True)
        assert Provider.OPENAI not in providers

    def test_eu_filter_excludes_gemini(self):
        """Bei EU-only: Gemini nicht in der Liste (US-Rechenzentren)."""
        providers = self.policy.filter_providers(eu_only=True)
        assert Provider.GEMINI not in providers

    def test_eu_filter_includes_anthropic(self):
        """Bei EU-only: Anthropic erlaubt (DSGVO-DPA verfügbar)."""
        providers = self.policy.filter_providers(eu_only=True)
        assert Provider.ANTHROPIC in providers

    def test_eu_filter_includes_ollama(self):
        """Bei EU-only: Ollama immer erlaubt (lokal = keine Datenübertragung)."""
        providers = self.policy.filter_providers(eu_only=True)
        assert Provider.OLLAMA in providers

    def test_no_filter_returns_all_providers(self):
        """Ohne EU-Anforderung: alle Provider zurückgeben."""
        providers = self.policy.filter_providers(eu_only=False)
        assert len(providers) == len(Provider)

    def test_anthropic_is_eu_compliant(self):
        """Anthropic muss als EU-konform markiert sein."""
        assert self.policy.is_provider_eu_compliant(Provider.ANTHROPIC) is True

    def test_openai_is_not_eu_compliant(self):
        """OpenAI ist nicht EU-konform (keine garantierte EU-Datenhaltung)."""
        assert self.policy.is_provider_eu_compliant(Provider.OPENAI) is False


# ─── FallbackPolicy Tests ────────────────────────────────────────────────────


class TestFallbackPolicy:
    def setup_method(self):
        self.policy = FallbackPolicy()

    def test_all_circuits_initially_closed(self):
        """Alle Schaltkreise müssen initial geschlossen (verfügbar) sein."""
        for provider in Provider:
            assert self.policy.is_circuit_closed(provider) is True

    def test_circuit_opens_after_three_failures(self):
        """Schaltkreis öffnet nach genau 3 Fehlern."""
        for _ in range(3):
            self.policy.record_failure(Provider.ANTHROPIC)
        assert self.policy.is_circuit_closed(Provider.ANTHROPIC) is False

    def test_circuit_stays_closed_after_two_failures(self):
        """Schaltkreis bleibt geschlossen bei weniger als 3 Fehlern."""
        self.policy.record_failure(Provider.OPENAI)
        self.policy.record_failure(Provider.OPENAI)
        assert self.policy.is_circuit_closed(Provider.OPENAI) is True

    def test_success_resets_failure_counter(self):
        """Erfolg setzt Fehlerzähler vollständig zurück."""
        self.policy.record_failure(Provider.GEMINI)
        self.policy.record_failure(Provider.GEMINI)
        self.policy.record_success(Provider.GEMINI)
        # Nach Reset: 2 weitere Fehler reichen nicht zum Öffnen
        self.policy.record_failure(Provider.GEMINI)
        self.policy.record_failure(Provider.GEMINI)
        assert self.policy.is_circuit_closed(Provider.GEMINI) is True

    def test_independent_circuits_per_provider(self):
        """Fehler in einem Provider beeinflussen andere nicht."""
        for _ in range(3):
            self.policy.record_failure(Provider.ANTHROPIC)
        assert self.policy.is_circuit_closed(Provider.ANTHROPIC) is False
        assert self.policy.is_circuit_closed(Provider.OPENAI) is True

    @pytest.mark.asyncio
    async def test_fallback_skips_open_circuit(self):
        """Fallback überspringt Provider mit offenem Schaltkreis."""
        for _ in range(3):
            self.policy.record_failure(Provider.ANTHROPIC)

        provider, model = await self.policy.get_fallback(
            [Provider.ANTHROPIC, Provider.OPENAI, Provider.GEMINI]
        )
        assert provider == Provider.OPENAI
        assert model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_fallback_raises_503_when_all_fail(self):
        """HTTP 503 wenn alle Provider in Fallback-Kette ausgefallen sind."""
        from fastapi import HTTPException
        for p in Provider:
            for _ in range(3):
                self.policy.record_failure(p)

        with pytest.raises(HTTPException) as exc_info:
            await self.policy.get_fallback(list(Provider))
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_fallback_respects_eu_filter(self):
        """Fallback wählt nur aus erlaubten Providern (EU-Filter)."""
        # Anthropic-Schaltkreis öffnen
        for _ in range(3):
            self.policy.record_failure(Provider.ANTHROPIC)

        # Nur EU-konforme Provider erlaubt (Anthropic + Ollama)
        eu_providers = [Provider.ANTHROPIC, Provider.OLLAMA]
        provider, model = await self.policy.get_fallback(eu_providers)
        assert provider == Provider.OLLAMA


# ─── LatencyPolicy Tests ─────────────────────────────────────────────────────


class TestLatencyPolicy:
    def setup_method(self):
        self.policy = LatencyPolicy()

    def test_unknown_provider_has_infinite_latency(self):
        """Unbekannte Provider gelten als unendlich langsam."""
        assert self.policy.average_latency(Provider.ANTHROPIC) == float("inf")

    def test_average_computed_correctly(self):
        """Gleitender Durchschnitt korrekt berechnen."""
        self.policy.record(Provider.ANTHROPIC, 100.0)
        self.policy.record(Provider.ANTHROPIC, 300.0)
        assert self.policy.average_latency(Provider.ANTHROPIC) == 200.0

    def test_window_evicts_old_measurements(self):
        """Ring-Puffer: mehr als 20 Messungen — älteste werden entfernt."""
        for i in range(25):
            self.policy.record(Provider.OPENAI, float(i * 10))
        # Nach 25 Einträgen: nur letzte 20 vorhanden
        avg = self.policy.average_latency(Provider.OPENAI)
        # Letzte 20: 50, 60, ..., 240 → Durchschnitt = (50+240)/2 = 145
        assert avg == pytest.approx(145.0)

    def test_prefer_lower_latency_selects_faster_provider(self):
        """prefer_lower_latency() = True wenn Provider B schneller als A."""
        self.policy.record(Provider.ANTHROPIC, 800.0)
        self.policy.record(Provider.OPENAI, 150.0)
        assert self.policy.prefer_lower_latency(Provider.ANTHROPIC, Provider.OPENAI) is True

    def test_prefer_lower_latency_false_when_a_faster(self):
        """prefer_lower_latency() = False wenn Provider A bereits schneller ist."""
        self.policy.record(Provider.ANTHROPIC, 100.0)
        self.policy.record(Provider.OPENAI, 500.0)
        assert self.policy.prefer_lower_latency(Provider.ANTHROPIC, Provider.OPENAI) is False

    def test_prefer_lower_latency_with_unknown_b(self):
        """Unbekannter Provider B (inf) verliert gegen bekannten Provider A."""
        self.policy.record(Provider.ANTHROPIC, 200.0)
        # Provider.GEMINI hat keine Messungen → float("inf")
        assert self.policy.prefer_lower_latency(Provider.ANTHROPIC, Provider.GEMINI) is False


# ─── RoutingEngine Unit Tests ─────────────────────────────────────────────────


class TestRoutingEngine:
    def setup_method(self):
        self.audit_mock = AsyncMock()
        self.audit_mock.log = AsyncMock()
        self.engine = RoutingEngine(self.audit_mock)

    def test_estimate_tokens_returns_positive(self):
        """Token-Schätzung muss positive Ganzzahl zurückgeben."""
        messages = [ChatMessage(role="user", content="Test" * 50)]
        tokens = self.engine.estimate_tokens(messages)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_classify_short_no_tools_as_simple(self):
        """< 500 Token ohne Tools = 'simple'."""
        assert self.engine.classify_complexity(100, False) == "simple"

    def test_classify_long_as_complex(self):
        """>= 500 Token = 'complex'."""
        assert self.engine.classify_complexity(500, False) == "complex"

    def test_classify_tool_use_always_complex(self):
        """Tool-Use-Anfragen sind immer 'complex' unabhängig von Token-Anzahl."""
        assert self.engine.classify_complexity(10, has_tools=True) == "complex"

    def test_classify_boundary_499_simple(self):
        """499 Token = 'simple' (unter Schwellenwert)."""
        assert self.engine.classify_complexity(499, False) == "simple"

    def test_classify_boundary_500_complex(self):
        """500 Token = 'complex' (Schwellenwert inklusive)."""
        assert self.engine.classify_complexity(500, False) == "complex"

    @pytest.mark.asyncio
    async def test_route_calls_audit_log(self):
        """route() muss audit_logger.log() nach jedem Aufruf aufrufen."""
        # Mock-Provider einrichten
        mock_response = make_response()
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)

        self.engine._factory = MagicMock()
        self.engine._factory.get = MagicMock(return_value=mock_provider)
        self.engine._factory.initialize = AsyncMock()

        await self.engine.initialize()

        request = make_request()
        await self.engine.route(request, "test-tenant")

        # Audit-Log muss genau einmal aufgerufen worden sein
        self.audit_mock.log.assert_called_once()
        call_kwargs = self.audit_mock.log.call_args.kwargs
        assert call_kwargs["tenant_id"] == "test-tenant"
        assert "prompt_hash" in call_kwargs
        # SHA256-Hash ist 64 Zeichen (hex)
        assert len(call_kwargs["prompt_hash"]) == 64

    @pytest.mark.asyncio
    async def test_route_eu_residency_filters_providers(self):
        """EU-Anfragen dürfen OpenAI/Gemini nicht verwenden."""
        selected_providers = []

        original_select = self.engine._cost_policy.select_cheapest

        def tracking_select(candidates, complexity, token_estimate):
            selected_providers.extend(candidates)
            return original_select(candidates, complexity, token_estimate)

        self.engine._cost_policy.select_cheapest = tracking_select

        mock_response = make_response(provider="anthropic")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        self.engine._factory = MagicMock()
        self.engine._factory.get = MagicMock(return_value=mock_provider)
        self.engine._factory.initialize = AsyncMock()
        await self.engine.initialize()

        request = make_request(residency_zone=ResidencyZone.EU)
        await self.engine.route(request, "eu-tenant")

        # Keine nicht-EU-Provider in der Kandidatenliste
        assert Provider.OPENAI not in selected_providers
        assert Provider.GEMINI not in selected_providers


# ─── Explicit Model Routing Tests ──────────────────────────────────────────


class TestExplicitModelRouting:
    """Tests für explizite Model-Anfragen (DECISION-014 Fix)."""

    @pytest.fixture(autouse=True)
    async def setup(self):
        audit_logger = AsyncMock()
        self.engine = RoutingEngine(audit_logger)

    async def test_explicit_model_routes_to_correct_provider(self):
        """Explizites Model wird korrekt auf Provider gemappt."""
        mock_response = make_response(provider="openai", model="gpt-4o")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        self.engine._factory = MagicMock()
        self.engine._factory.get = MagicMock(return_value=mock_provider)
        self.engine._factory.initialize = AsyncMock()
        await self.engine.initialize()

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="gpt-4o"  # Explicit model
        )
        response = await self.engine.route(request, "test-tenant")

        # Verify correct provider was used
        self.engine._factory.get.assert_called_with(Provider.OPENAI)
        assert response.provider == "openai"
        assert response.model == "gpt-4o"

    async def test_explicit_model_skips_cost_optimization(self):
        """Explizites Model überspringt Kostenoptimierung."""
        # Anthropic Sonnet requested (expensive), but should be used despite Ollama being cheaper
        mock_response = make_response(provider="anthropic", model="claude-sonnet-4-5-20250929")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        self.engine._factory = MagicMock()
        self.engine._factory.get = MagicMock(return_value=mock_provider)
        self.engine._factory.initialize = AsyncMock()
        await self.engine.initialize()

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Short")],
            model="claude-sonnet-4-5-20250929"  # Expensive model explicitly requested
        )
        response = await self.engine.route(request, "test-tenant")

        # Verify Sonnet was used (not cheap Ollama)
        self.engine._factory.get.assert_called_with(Provider.ANTHROPIC)
        assert response.model == "claude-sonnet-4-5-20250929"

    async def test_explicit_eu_incompatible_model_rejected(self):
        """Nicht-EU-konformes Model wird bei EU-Anforderung abgelehnt."""
        from fastapi import HTTPException

        audit_logger = AsyncMock()
        self.engine = RoutingEngine(audit_logger)
        mock_provider = AsyncMock()
        self.engine._factory = MagicMock()
        self.engine._factory.get = MagicMock(return_value=mock_provider)
        self.engine._factory.initialize = AsyncMock()
        await self.engine.initialize()

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="gpt-4o",  # OpenAI not EU-compliant
            residency_zone=ResidencyZone.EU
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.engine.route(request, "eu-tenant")

        assert exc_info.value.status_code == 403
        assert "nicht EU-konform" in exc_info.value.detail

    async def test_explicit_eu_compatible_model_allowed(self):
        """EU-konformes Model wird bei EU-Anforderung akzeptiert."""
        mock_response = make_response(provider="anthropic", model="claude-haiku-4-5")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        self.engine._factory = MagicMock()
        self.engine._factory.get = MagicMock(return_value=mock_provider)
        self.engine._factory.initialize = AsyncMock()
        await self.engine.initialize()

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="claude-haiku-4-5",  # Anthropic is EU-compliant
            residency_zone=ResidencyZone.EU
        )
        response = await self.engine.route(request, "eu-tenant")

        # Verify request succeeded with Anthropic
        assert response.provider == "anthropic"

    async def test_auto_model_uses_cost_optimization(self):
        """Model='auto' führt zu Kostenoptimierung (Standard-Verhalten)."""
        mock_response = make_response(provider="ollama", model="llama3.2")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        self.engine._factory = MagicMock()
        self.engine._factory.get = MagicMock(return_value=mock_provider)
        self.engine._factory.initialize = AsyncMock()
        await self.engine.initialize()

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Short")],
            model="auto"  # Default value
        )
        response = await self.engine.route(request, "test-tenant")

        # Cost optimization should select Ollama (cheapest)
        self.engine._factory.get.assert_called_with(Provider.OLLAMA)
        assert response.provider == "ollama"
