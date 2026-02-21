# gateway/middleware/token_budget.py
# Monatliches Token-Budget pro Mandant: kumulatives Redis-Tracking + Durchsetzung.
# Schlüsselformat: budget:{tenant_id}:{YYYY-MM} → integer (verbrauchte Token diesen Monat)
# NULL-Budget = unbegrenzt (kein Tracking erforderlich).
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _budget_key(tenant_id: str) -> str:
    """Redis-Schlüssel für den aktuellen Monat: budget:{tenant}:{YYYY-MM}."""
    month = datetime.utcnow().strftime("%Y-%m")
    return f"budget:{tenant_id}:{month}"


class TokenBudgetManager:
    """
    Verwaltet monatliche Token-Budgets pro Mandant über Redis.

    - Kein Redis = fail-open (unbegrenzte Nutzung — für lokale Entwicklung)
    - Monatlicher Reset: Redis-Schlüssel läuft nach 35 Tagen ab (automatische Bereinigung)
    - Atomare Inkrementierung via INCRBY (thread-safe ohne Locks)
    """

    def __init__(self, redis_url: str | None) -> None:
        self._redis_url = redis_url
        self._redis = None

    async def initialize(self) -> None:
        """Redis-Verbindung aufbauen."""
        if not self._redis_url:
            return
        try:
            import redis.asyncio as aioredis

            self._redis = await aioredis.from_url(
                self._redis_url, decode_responses=True
            )
            logger.info("TokenBudgetManager: Redis verbunden (%s)", self._redis_url)
        except Exception as exc:
            logger.warning("TokenBudgetManager: Redis nicht verfügbar — fail-open: %s", exc)
            self._redis = None

    async def close(self) -> None:
        """Redis-Verbindung schließen."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def get_usage(self, tenant_id: str) -> int:
        """Gibt den aktuellen monatlichen Token-Verbrauch zurück (0 wenn kein Redis)."""
        if not self._redis:
            return 0
        try:
            val = await self._redis.get(_budget_key(tenant_id))
            return int(val) if val else 0
        except Exception as exc:
            logger.warning("Budget-Lesefehler für %s: %s", tenant_id, exc)
            return 0

    async def is_budget_exceeded(
        self, tenant_id: str, budget_limit: int | None
    ) -> tuple[bool, int]:
        """
        Prüft ob das Budget erschöpft ist.

        Returns:
            (exceeded: bool, current_used: int)
            False, 0 wenn budget_limit is None (unbegrenzt)
        """
        if budget_limit is None:
            return False, 0
        used = await self.get_usage(tenant_id)
        return used >= budget_limit, used

    async def increment_usage(self, tenant_id: str, tokens: int) -> int:
        """
        Erhöht den monatlichen Verbrauch atomisch.

        TTL: 35 Tage (automatische Bereinigung nach Monatsende).
        Returns: neuer Gesamtverbrauch
        """
        if not self._redis or tokens <= 0:
            return 0
        try:
            key = _budget_key(tenant_id)
            new_val = await self._redis.incrby(key, tokens)
            # TTL nur beim ersten Increment setzen (wenn Schlüssel neu)
            await self._redis.expire(key, 35 * 24 * 3600)
            return int(new_val)
        except Exception as exc:
            logger.warning("Budget-Increment-Fehler für %s: %s", tenant_id, exc)
            return 0
