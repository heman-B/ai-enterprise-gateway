# gateway/main.py
# FastAPI-Hauptanwendung: Middleware-Stack, Endpunkte, Lifecycle
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .metrics import get_metrics_response
from .middleware.api_key_auth import APIKeyAuthMiddleware, create_api_key, get_all_active_tenants
from .middleware.audit_logger import AuditLogger
from .middleware.rate_limiter import RateLimiterMiddleware
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    TenantKeyCreate,
)
from .mcp_server import mcp as mcp_server
from .mcp_server import set_engine
from .router import RoutingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "./gateway.db")

# MCP-App auf Modul-Ebene erstellen — Lifespan wird im lifespan-Context gestartet
# path="/" notwendig: FastAPI strippt den /mcp-Prefix, Sub-App muss Route bei / haben
mcp_http_app = mcp_server.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Anwendungs-Lifecycle: Startup-Initialisierung und Shutdown-Bereinigung."""
    # MCP-Lifespan einschließen: startet den StreamableHTTPSessionManager
    async with mcp_http_app.lifespan(mcp_http_app):
        # Startup: Datenbankschema erstellen, Routing-Engine initialisieren
        audit_logger = AuditLogger(DATABASE_URL)
        await audit_logger.initialize()
        app.state.audit_logger = audit_logger

        app.state.router = RoutingEngine(audit_logger)
        await app.state.router.initialize()

        # Semantischen Cache auf app.state exponieren (für /admin/cache/stats)
        app.state.semantic_cache = app.state.router.semantic_cache

        # MCP-Server mit RoutingEngine verbinden
        set_engine(app.state.router)

        logger.info("✅ LLM-Gateway gestartet — bereit auf Port 8000")
        yield

        # Shutdown: HTTP-Clients, Cache und DB-Verbindungen ordnungsgemäß schließen
        await app.state.router.shutdown()
        if app.state.semantic_cache:
            await app.state.semantic_cache.close()
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


# MCP-Server unter /mcp mounten (Streamable HTTP Transport)
app.mount("/mcp", mcp_http_app)


@app.get("/metrics", include_in_schema=False, tags=["Monitoring"])
async def prometheus_metrics() -> Response:
    """Prometheus-Metriken im Textformat (für Scraping durch Prometheus-Server)."""
    data, content_type = get_metrics_response()
    return Response(content=data, media_type=content_type)


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
    raw_key = await create_api_key(
        payload.tenant_id,
        payload.rate_limit_per_minute,
        payload.token_budget_monthly,
    )
    return {
        "tenant_id": payload.tenant_id,
        "api_key": raw_key,
        "token_budget_monthly": payload.token_budget_monthly,
        "warning": "Schlüssel wird nicht erneut angezeigt. Sofort sicher speichern!",
    }


@app.get("/admin/audit/verify", tags=["Admin"], include_in_schema=False)
async def verify_audit_chain(request: Request) -> dict:
    """
    Hash-Kette des Audit-Logs verifizieren.

    Prüft jeden Datensatz mathematisch:
    - prev_hash stimmt mit record_hash des Vorgängers überein
    - record_hash == SHA256(Felder + prev_hash)

    Jede Manipulation an einem Datensatz bricht die Kette ab diesem Punkt.

    Returns:
        {"chain_valid": true, "records_checked": N}
        {"chain_valid": false, "records_checked": N, "broken_at": "id", "reason": "..."}
    """
    audit_logger = request.app.state.audit_logger
    return await audit_logger.verify_chain()


@app.get("/admin/cache/stats", tags=["Admin"], include_in_schema=False)
async def cache_stats(request: Request) -> dict:
    """
    Semantischer Cache-Status und Einsparungsstatistiken.

    Returns:
        total_requests: Gesamtzahl der Anfragen die durch den Cache geprüft wurden
        cache_hits: Davon als Cache-Treffer bedient
        hit_rate_pct: Trefferquote in Prozent
        tokens_saved: Eingesparte LLM-Token (input + output) durch Cache-Treffer
        cost_saved_usd: Eingesparte Kosten in USD durch Cache-Treffer
    """
    cache = getattr(request.app.state, "semantic_cache", None)
    if cache is None:
        return {
            "enabled": False,
            "message": "Semantischer Cache nicht aktiv (REDIS_URL nicht gesetzt)",
        }
    stats = await cache.get_stats()
    stats["enabled"] = True
    return stats


@app.get("/admin/costs/summary", tags=["Admin"], include_in_schema=False)
async def costs_summary(request: Request) -> dict:
    """
    Kostenzusammenfassung pro Mandant für den laufenden Monat.

    Gibt für jeden aktiven Mandanten zurück:
    - tokens_used:           verbrauchte Token (aus Redis)
    - token_budget_monthly:  monatliches Limit (NULL = unbegrenzt)
    - budget_consumed_pct:   prozentualer Verbrauch (NULL wenn unbegrenzt)
    - cost_usd:              Kosten in USD (aus Audit-Log)
    - cost_eur:              Kosten in EUR (Kurs: 0.92)
    """
    import time as _time
    from datetime import datetime

    # Monatsstart berechnen (erster Tag des aktuellen Monats, UTC)
    now = datetime.utcnow()
    month_start = datetime(now.year, now.month, 1).timestamp()
    month_str = now.strftime("%Y-%m")

    # Monatliche Kosten pro Mandant aus Audit-Log
    audit_logger = request.app.state.audit_logger
    monthly_costs = await audit_logger.get_monthly_costs_by_tenant(since_timestamp=month_start)

    # Aktive Mandanten mit Budget-Limits aus DB
    tenants = await get_all_active_tenants()

    # Token-Verbrauch aus Redis (falls Budget-Manager aktiv)
    budget_manager = getattr(request.app.state.router, "_budget_manager", None)

    EUR_PER_USD = 0.92
    results = []

    for tenant_id, budget_limit in sorted(tenants.items()):
        usage = 0
        if budget_manager:
            usage = await budget_manager.get_usage(tenant_id)

        cost_usd = monthly_costs.get(tenant_id, 0.0)
        cost_eur = round(cost_usd * EUR_PER_USD, 4)

        budget_pct = (
            round(usage / budget_limit * 100, 1)
            if budget_limit and budget_limit > 0
            else None
        )

        results.append({
            "tenant_id": tenant_id,
            "tokens_used": usage,
            "token_budget_monthly": budget_limit,
            "budget_consumed_pct": budget_pct,
            "cost_usd": round(cost_usd, 4),
            "cost_eur": cost_eur,
        })

    return {
        "month": month_str,
        "eur_rate": EUR_PER_USD,
        "tenants": results,
    }


@app.post("/admin/compliance-export", tags=["Admin"], include_in_schema=False)
async def compliance_export(request: Request) -> StreamingResponse:
    """
    DSGVO-Compliance-Export als ZIP herunterladen.

    Inhalt des Archivs:
    - audit_log.jsonl         — alle Audit-Datensätze (JSON Lines)
    - integrity_manifest.json — SHA256-Hashes aller Datensätze + Gesamt-Chain-Hash
    - system_config.json      — Gateway-Version, Provider, PII-Typen, Retentionsrichtlinie

    Dieses Bundle kann direkt an Enterprise-Kunden oder Datenschutzbeauftragte gesendet werden.
    """
    audit_logger = request.app.state.audit_logger
    zip_bytes = await audit_logger.build_compliance_bundle(
        gateway_version=app.version
    )

    import time as _time

    filename = f"compliance-export-{int(_time.time())}.zip"
    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
