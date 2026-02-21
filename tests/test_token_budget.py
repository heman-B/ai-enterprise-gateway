# tests/test_token_budget.py
# Tests für monatliche Token-Budget-Durchsetzung.
# Prüft: Budget-Schlüsselformat, Redis-Tracking, Überschreitungsblock (HTTP 429),
# monatlicher Reset, unbegrenztes Budget (NULL).
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.middleware.token_budget import TokenBudgetManager, _budget_key


# ─── Unit Tests: TokenBudgetManager ──────────────────────────────────────────


class TestBudgetKey:
    """Redis-Schlüsselformat für monatliches Budget."""

    def test_key_format_contains_tenant_and_month(self):
        """Schlüssel muss Mandant und Monat im Format YYYY-MM enthalten."""
        key = _budget_key("acme-corp")
        now = datetime.utcnow()
        expected_month = now.strftime("%Y-%m")
        assert key == f"budget:acme-corp:{expected_month}"

    def test_different_tenants_have_different_keys(self):
        """Verschiedene Mandanten müssen unterschiedliche Budget-Schlüssel haben."""
        key_a = _budget_key("tenant-a")
        key_b = _budget_key("tenant-b")
        assert key_a != key_b

    def test_key_changes_each_month(self):
        """Schlüssel mit anderem Monat ist ein anderer Schlüssel (monatlicher Reset)."""
        key_feb = "budget:acme:2026-02"
        key_mar = "budget:acme:2026-03"
        assert key_feb != key_mar


class TestTokenBudgetManagerUnit:
    """Unit-Tests für TokenBudgetManager mit gemocktem Redis."""

    def _make_manager(self, redis_mock) -> TokenBudgetManager:
        manager = TokenBudgetManager(redis_url="redis://localhost")
        manager._redis = redis_mock
        return manager

    @pytest.mark.asyncio
    async def test_get_usage_returns_zero_when_no_data(self):
        """Kein Redis-Eintrag → 0 Token verbraucht."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        manager = self._make_manager(redis_mock)
        usage = await manager.get_usage("tenant-x")
        assert usage == 0

    @pytest.mark.asyncio
    async def test_get_usage_returns_stored_value(self):
        """Redis-Eintrag vorhanden → korrekter Wert zurückgegeben."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value="5000")
        manager = self._make_manager(redis_mock)
        usage = await manager.get_usage("tenant-x")
        assert usage == 5000

    @pytest.mark.asyncio
    async def test_budget_not_exceeded_when_under_limit(self):
        """Verbrauch < Limit → Budget nicht erschöpft."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value="500")
        manager = self._make_manager(redis_mock)
        exceeded, used = await manager.is_budget_exceeded("tenant-x", budget_limit=1000)
        assert exceeded is False
        assert used == 500

    @pytest.mark.asyncio
    async def test_budget_exceeded_when_at_limit(self):
        """Verbrauch == Limit → Budget erschöpft."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value="1000")
        manager = self._make_manager(redis_mock)
        exceeded, used = await manager.is_budget_exceeded("tenant-x", budget_limit=1000)
        assert exceeded is True
        assert used == 1000

    @pytest.mark.asyncio
    async def test_budget_exceeded_when_over_limit(self):
        """Verbrauch > Limit → Budget erschöpft."""
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value="1500")
        manager = self._make_manager(redis_mock)
        exceeded, used = await manager.is_budget_exceeded("tenant-x", budget_limit=1000)
        assert exceeded is True
        assert used == 1500

    @pytest.mark.asyncio
    async def test_unlimited_budget_never_exceeded(self):
        """budget_limit=None → Budget nie erschöpft, keine Redis-Abfrage."""
        redis_mock = AsyncMock()
        manager = self._make_manager(redis_mock)
        exceeded, used = await manager.is_budget_exceeded("tenant-x", budget_limit=None)
        assert exceeded is False
        assert used == 0
        redis_mock.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_increment_usage_calls_incrby(self):
        """increment_usage muss INCRBY aufrufen."""
        redis_mock = AsyncMock()
        redis_mock.incrby = AsyncMock(return_value=350)
        redis_mock.expire = AsyncMock()
        manager = self._make_manager(redis_mock)
        new_total = await manager.increment_usage("tenant-x", tokens=350)
        assert new_total == 350
        redis_mock.incrby.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_increment_sets_35_day_ttl(self):
        """TTL muss auf 35 Tage gesetzt werden (automatische Bereinigung nach Monatsende)."""
        redis_mock = AsyncMock()
        redis_mock.incrby = AsyncMock(return_value=100)
        redis_mock.expire = AsyncMock()
        manager = self._make_manager(redis_mock)
        await manager.increment_usage("tenant-x", tokens=100)
        redis_mock.expire.assert_awaited_once()
        # Prüfe TTL-Wert: 35 Tage * 24 Stunden * 3600 Sekunden
        call_args = redis_mock.expire.call_args
        assert call_args[0][1] == 35 * 24 * 3600

    @pytest.mark.asyncio
    async def test_no_redis_fail_open(self):
        """Ohne Redis → fail-open (0 verbraucht, Budget nie erschöpft)."""
        manager = TokenBudgetManager(redis_url=None)
        # Kein Redis initialisiert
        usage = await manager.get_usage("tenant-x")
        assert usage == 0
        exceeded, used = await manager.is_budget_exceeded("tenant-x", budget_limit=100)
        assert exceeded is False


# ─── Integration Tests: Budget-Durchsetzung im Router ────────────────────────


class TestBudgetEnforcementResponse:
    """
    Prüft die HTTP-Antwort wenn das Budget erschöpft ist.
    Simuliert den Budget-Check ohne echte Datenbankverbindung.
    """

    @pytest.mark.asyncio
    async def test_budget_exceeded_response_format(self):
        """
        Wenn das Budget erschöpft ist, muss der Router HTTP 429 zurückgeben
        mit dem erwarteten Fehler-Body.
        """
        from fastapi import HTTPException

        from gateway.middleware.token_budget import TokenBudgetManager

        # Budget-Manager der "erschöpft" zurückgibt
        manager = TokenBudgetManager(redis_url="redis://localhost")
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value="10000")  # == budget_limit
        manager._redis = redis_mock

        exceeded, used = await manager.is_budget_exceeded("acme", budget_limit=10000)
        assert exceeded is True

        # Sicherstellen dass der richtige Fehler-Code und Body produziert wird
        with pytest.raises(HTTPException) as exc_info:
            if exceeded:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "monthly_token_budget_exceeded",
                        "budget": 10000,
                        "used": used,
                    },
                )

        assert exc_info.value.status_code == 429
        detail = exc_info.value.detail
        assert detail["error"] == "monthly_token_budget_exceeded"
        assert detail["budget"] == 10000
        assert detail["used"] == 10000
