# gateway/middleware/rate_limiter.py
# Redis Sliding-Window Rate-Limiter: konfigurierbar pro Mandant
from __future__ import annotations

import logging
import os
import time

import redis.asyncio as aioredis
from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Standard-Rate-Limit (überschreibbar via Umgebungsvariable oder pro-Mandant in Redis)
DEFAULT_RATE_LIMIT = int(os.getenv("DEFAULT_RATE_LIMIT_PER_MINUTE", "100"))
RATE_LIMIT_WINDOW_SEC = 60

# Endpunkte die NICHT rate-gelimitet werden
EXEMPT_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding-Window Rate-Limiter via Redis Sorted Sets.
    Pro-Mandant konfigurierbar. Standard: 100 Anfragen/Minute.

    Implementierung: Redis Sorted Set mit Unix-Timestamp als Score.
    - Alte Einträge (>60s) werden vor jeder Prüfung entfernt
    - Aktuelle Anzahl = Kardinalität des verbleibenden Sets
    - Atomic via Redis Pipeline (keine Race Conditions)
    """

    def __init__(self, app, redis_url: str | None = None) -> None:
        super().__init__(app)
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        """Redis-Verbindung lazy erstellen (erst bei erster Anfrage)."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def _get_tenant_limit(self, tenant_id: str, redis: aioredis.Redis) -> int:
        """
        Individuelles Rate-Limit für Mandanten aus Redis laden.
        Fallback auf DEFAULT_RATE_LIMIT wenn nicht konfiguriert.
        """
        limit = await redis.get(f"rate_limit:{tenant_id}")
        return int(limit) if limit else DEFAULT_RATE_LIMIT

    async def dispatch(self, request: Request, call_next) -> Response:
        """Anfrage gegen Sliding-Window Rate-Limit prüfen."""
        # Exempt paths überspringen
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", "anonymous")

        try:
            redis = await self._get_redis()
            limit = await self._get_tenant_limit(tenant_id, redis)
        except Exception as exc:
            # Redis nicht erreichbar → Anfrage durchlassen (fail-open für Verfügbarkeit)
            logger.warning("Redis nicht erreichbar — Rate-Limiting deaktiviert: %s", exc)
            return await call_next(request)

        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SEC
        key = f"requests:{tenant_id}"

        # Atomic Pipeline: alte Einträge entfernen, neue hinzufügen, Anzahl prüfen
        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)  # Veraltete Einträge löschen
        pipe.zadd(key, {f"{now:.6f}": now})               # Aktuelle Anfrage als Score
        pipe.zcard(key)                                    # Anzahl im Fenster
        pipe.expire(key, RATE_LIMIT_WINDOW_SEC + 1)        # TTL sicherstellen
        results = await pipe.execute()
        request_count: int = results[2]

        if request_count > limit:
            logger.warning(
                "Rate-Limit überschritten: Mandant=%s, Anfragen=%d, Limit=%d",
                tenant_id, request_count, limit,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Rate-Limit überschritten. Maximal {limit} Anfragen pro Minute.",
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEC)},
            )

        # Rate-Limit-Status-Header zur Antwort hinzufügen
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - request_count))
        response.headers["X-RateLimit-Window"] = str(RATE_LIMIT_WINDOW_SEC)
        return response
