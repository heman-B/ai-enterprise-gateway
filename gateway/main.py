# gateway/main.py
# FastAPI-Hauptanwendung: Middleware-Stack, Endpunkte, Lifecycle
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .middleware.api_key_auth import APIKeyAuthMiddleware, create_api_key
from .middleware.audit_logger import AuditLogger
from .middleware.rate_limiter import RateLimiterMiddleware
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    TenantKeyCreate,
)
from .router import RoutingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "./gateway.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Anwendungs-Lifecycle: Startup-Initialisierung und Shutdown-Bereinigung."""
    # Startup: Datenbankschema erstellen, Routing-Engine initialisieren
    audit_logger = AuditLogger(DATABASE_URL)
    await audit_logger.initialize()
    app.state.audit_logger = audit_logger

    app.state.router = RoutingEngine(audit_logger)
    await app.state.router.initialize()

    logger.info("✅ LLM-Gateway gestartet — bereit auf Port 8000")
    yield

    # Shutdown: HTTP-Clients und DB-Verbindungen ordnungsgemäß schließen
    await app.state.router.shutdown()
    logger.info("LLM-Gateway heruntergefahren")


app = FastAPI(
    title="Enterprise LLM Gateway",
    description="Multi-Provider LLM-Gateway mit 5-stufiger Routing-Hierarchie",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware-Reihenfolge: zuerst registriert = zuletzt ausgeführt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimiterMiddleware)
app.add_middleware(APIKeyAuthMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(request: Request) -> HealthResponse:
    """Gateway-Status und Provider-Verfügbarkeit abrufen."""
    provider_status = await request.app.state.router.health()
    return HealthResponse(status="healthy", providers=provider_status)


@app.get("/ready", tags=["Monitoring"])
async def readiness_check(request: Request) -> dict:
    """
    Readiness-Check für Render: Prüft DB + Redis Konnektivität.
    Returns HTTP 200 wenn bereit, HTTP 503 wenn nicht bereit.
    """
    import os

    from fastapi import HTTPException

    checks = {"database": False, "redis": False}

    # Database-Check
    try:
        audit_logger = request.app.state.audit_logger
        db_url = os.getenv("DATABASE_URL", "./gateway.db")

        if db_url.startswith("postgres"):
            import asyncpg

            conn = await asyncpg.connect(db_url)
            await conn.fetchval("SELECT 1")
            await conn.close()
        else:
            import aiosqlite

            async with aiosqlite.connect(db_url) as db:
                await db.execute("SELECT 1")

        checks["database"] = True
    except Exception as e:
        logger.error(f"Database readiness check failed: {e}")

    # Redis-Check
    try:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            import redis.asyncio as aioredis

            redis_client = await aioredis.from_url(redis_url)
            await redis_client.ping()
            await redis_client.close()
            checks["redis"] = True
        else:
            # Redis optional für lokale Entwicklung
            checks["redis"] = True
    except Exception as e:
        logger.error(f"Redis readiness check failed: {e}")

    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail={"status": "not ready", "checks": checks})


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["LLM"],
    summary="Chat Completion (OpenAI-kompatibel)",
)
async def chat_completions(
    request: Request,
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """
    Intelligente Chat-Completion mit automatischer Provider-Auswahl.

    Routing-Hierarchie:
    1. Datenhaltung (EU-Anforderungen)
    2. Kostenoptimierung
    3. Latenzoptimierung
    4. Circuit-Breaker-Failover
    5. Fallback-Kette
    """
    tenant_id: str = getattr(request.state, "tenant_id", "anonymous")
    return await request.app.state.router.route(payload, tenant_id)


@app.post("/admin/keys", tags=["Admin"], include_in_schema=False)
async def generate_api_key(payload: TenantKeyCreate) -> dict:
    """
    Admin-Endpunkt: neuen API-Schlüssel für Mandanten erstellen.
    Schlüssel wird EINMALIG im Klartext zurückgegeben — sicher speichern!
    """
    raw_key = await create_api_key(payload.tenant_id, payload.rate_limit_per_minute)
    return {
        "tenant_id": payload.tenant_id,
        "api_key": raw_key,
        "warning": "Schlüssel wird nicht erneut angezeigt. Sofort sicher speichern!",
    }
