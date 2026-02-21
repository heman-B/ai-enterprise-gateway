# gateway/middleware/api_key_auth.py
# API-Schlüssel-Authentifizierung: pro-Mandant, SHA256-Hash in SQLite
from __future__ import annotations

import hashlib
import logging
import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "./gateway.db")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")  # Admin-Schlüssel für Bootstrap

# Endpunkte ohne Authentifizierungspflicht
# /metrics: Prometheus-Scraping benötigt keinen API-Schlüssel (Netzwerk-Level-Schutz)
PUBLIC_PATHS = frozenset({"/health", "/ready", "/docs", "/openapi.json", "/redoc", "/metrics"})
# Pfad-Präfixe ohne Authentifizierung
PUBLIC_PREFIXES: tuple[str, ...] = ()


def _is_postgres(db_url: str) -> bool:
    """Prüft, ob DATABASE_URL auf PostgreSQL zeigt."""
    return db_url.startswith("postgres://") or db_url.startswith("postgresql://")


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API-Schlüssel-Validierung für alle geschützten Endpunkte.
    Schlüssel werden als SHA256-Hash gespeichert (nie im Klartext in der DB).
    Unterstützt: X-API-Key Header oder Authorization: Bearer <key>
    """

    async def _find_tenant(self, api_key: str) -> str | None:
        """
        Mandanten-ID für gegebenen API-Schlüssel suchen.
        Vergleicht SHA256-Hash (Timing-Attack-sicher durch DB-Vergleich).
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        try:
            if _is_postgres(DATABASE_URL):
                import asyncpg

                conn = await asyncpg.connect(DATABASE_URL)
                try:
                    row = await conn.fetchrow(
                        "SELECT tenant_id FROM api_keys WHERE key_hash = $1 AND is_active = 1",
                        key_hash,
                    )
                    return row["tenant_id"] if row else None
                finally:
                    await conn.close()
            else:
                import aiosqlite

                async with aiosqlite.connect(DATABASE_URL) as db:
                    async with db.execute(
                        "SELECT tenant_id FROM api_keys WHERE key_hash = ? AND is_active = 1",
                        (key_hash,),
                    ) as cursor:
                        row = await cursor.fetchone()
                        return row[0] if row else None
        except Exception as exc:
            logger.error("Datenbankfehler bei API-Key-Prüfung: %s", exc)
            return None

    async def dispatch(self, request: Request, call_next) -> Response:
        """API-Schlüssel aus Header extrahieren und Mandanten-ID in Request-State setzen."""
        if request.url.path in PUBLIC_PATHS or request.url.path.startswith(PUBLIC_PREFIXES):
            request.state.tenant_id = "public"
            return await call_next(request)

        # X-API-Key Header oder Bearer Token extrahieren
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        if not api_key:
            # JSONResponse statt HTTPException: BaseHTTPMiddleware wraps exceptions
            # in ExceptionGroup via anyio → führt zu 500 statt 401
            return JSONResponse(
                status_code=401,
                content={"detail": "API-Schlüssel fehlt. Header 'X-API-Key' oder 'Authorization: Bearer <key>' erforderlich."},
            )

        # Admin-Key-Check: ADMIN_API_KEY aus .env hat Vorrang (für Bootstrap)
        if ADMIN_API_KEY and api_key == ADMIN_API_KEY:
            logger.info("Admin-Zugriff über ADMIN_API_KEY für %s", request.url.path)
            request.state.tenant_id = "admin"
            return await call_next(request)

        # Normaler Mandanten-Schlüssel: Datenbank-Lookup
        tenant_id = await self._find_tenant(api_key)
        if not tenant_id:
            return JSONResponse(
                status_code=401,
                content={"detail": "Ungültiger oder inaktiver API-Schlüssel."},
            )

        request.state.tenant_id = tenant_id
        return await call_next(request)


async def create_api_key(
    tenant_id: str,
    rate_limit: int = 100,
    token_budget_monthly: int | None = None,
) -> str:
    """
    Neuen API-Schlüssel für einen Mandanten erstellen.

    Sicherheitshinweise:
    - Schlüssel-Präfix 'lgw_' für einfache Erkennung bei versehentlichem Leak
    - Klartext-Schlüssel wird nur EINMALIG zurückgegeben
    - In DB wird ausschließlich SHA256-Hash gespeichert
    """
    raw_key = f"lgw_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    if _is_postgres(DATABASE_URL):
        import asyncpg

        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute(
                """
                INSERT INTO api_keys
                    (tenant_id, key_hash, is_active, rate_limit_per_minute, token_budget_monthly)
                VALUES ($1, $2, 1, $3, $4)
                ON CONFLICT (key_hash) DO UPDATE SET
                    is_active = 1,
                    rate_limit_per_minute = $3,
                    token_budget_monthly = $4
                """,
                tenant_id,
                key_hash,
                rate_limit,
                token_budget_monthly,
            )
        finally:
            await conn.close()
    else:
        import aiosqlite

        async with aiosqlite.connect(DATABASE_URL) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO api_keys
                    (tenant_id, key_hash, is_active, rate_limit_per_minute, token_budget_monthly)
                VALUES (?, ?, 1, ?, ?)
                """,
                (tenant_id, key_hash, rate_limit, token_budget_monthly),
            )
            await db.commit()

    logger.info(
        "Neuer API-Schlüssel erstellt für Mandant: %s (Budget: %s Token/Monat)",
        tenant_id, token_budget_monthly or "unbegrenzt",
    )
    return raw_key


async def get_tenant_budget(tenant_id: str) -> int | None:
    """
    Token-Budget des Mandanten aus der Datenbank lesen.

    Returns:
        token_budget_monthly (int) oder None (unbegrenzt)
    """
    try:
        if _is_postgres(DATABASE_URL):
            import asyncpg

            conn = await asyncpg.connect(DATABASE_URL)
            try:
                row = await conn.fetchrow(
                    "SELECT token_budget_monthly FROM api_keys WHERE tenant_id = $1 AND is_active = 1 LIMIT 1",
                    tenant_id,
                )
                return row["token_budget_monthly"] if row else None
            finally:
                await conn.close()
        else:
            import aiosqlite

            async with aiosqlite.connect(DATABASE_URL) as db:
                async with db.execute(
                    "SELECT token_budget_monthly FROM api_keys WHERE tenant_id = ? AND is_active = 1 LIMIT 1",
                    (tenant_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row else None
    except Exception as exc:
        logger.warning("Budget-Lookup-Fehler für %s: %s", tenant_id, exc)
        return None


async def get_all_active_tenants() -> dict[str, int | None]:
    """
    Alle aktiven Mandanten mit ihrem Token-Budget lesen.

    Returns:
        {tenant_id: token_budget_monthly}  (None = unbegrenzt)
    """
    try:
        if _is_postgres(DATABASE_URL):
            import asyncpg

            conn = await asyncpg.connect(DATABASE_URL)
            try:
                rows = await conn.fetch(
                    "SELECT DISTINCT tenant_id, token_budget_monthly FROM api_keys WHERE is_active = 1"
                )
                return {row["tenant_id"]: row["token_budget_monthly"] for row in rows}
            finally:
                await conn.close()
        else:
            import aiosqlite

            async with aiosqlite.connect(DATABASE_URL) as db:
                async with db.execute(
                    "SELECT DISTINCT tenant_id, token_budget_monthly FROM api_keys WHERE is_active = 1"
                ) as cursor:
                    rows = await cursor.fetchall()
                    return {row[0]: row[1] for row in rows}
    except Exception as exc:
        logger.warning("Mandanten-Lookup-Fehler: %s", exc)
        return {}


async def revoke_api_key(tenant_id: str) -> int:
    """Alle API-Schlüssel eines Mandanten deaktivieren. Gibt Anzahl deaktivierter Schlüssel zurück."""
    if _is_postgres(DATABASE_URL):
        import asyncpg

        conn = await asyncpg.connect(DATABASE_URL)
        try:
            result = await conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE tenant_id = $1",
                tenant_id,
            )
            # PostgreSQL execute() gibt "UPDATE N" zurück
            count = int(result.split()[-1]) if result else 0
        finally:
            await conn.close()
    else:
        import aiosqlite

        async with aiosqlite.connect(DATABASE_URL) as db:
            cursor = await db.execute(
                "UPDATE api_keys SET is_active = 0 WHERE tenant_id = ?",
                (tenant_id,),
            )
            await db.commit()
            count = cursor.rowcount

    logger.info("%d Schlüssel für Mandant %s deaktiviert", count, tenant_id)
    return count
