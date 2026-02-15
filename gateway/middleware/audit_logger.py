# gateway/middleware/audit_logger.py
# Unveränderlicher Audit-Log: SHA256-Hash des Prompts, KEIN Klartext
from __future__ import annotations

import logging
import os
import time
import uuid

import aiosqlite

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "./gateway.db")


class AuditLogger:
    """
    Append-Only Audit-Protokollierung für alle LLM-Anfragen.

    Datenschutzprinzipien:
    - Roher Prompt-Inhalt wird NIEMALS gespeichert
    - Nur SHA256(prompt) für Nachvollziehbarkeit ohne Inhaltseinsicht
    - Kosten-Tracking: jede Anfrage mit Mandant, Modell, Token, Kosten protokolliert
    - SQLite append-only: keine UPDATE/DELETE-Operationen auf audit_log
    """

    def __init__(self, db_url: str = DATABASE_URL) -> None:
        self._db_url = db_url

    async def initialize(self) -> None:
        """Datenbankschema erstellen (idempotent — sicher für mehrfachen Aufruf)."""
        async with aiosqlite.connect(self._db_url) as db:
            # Audit-Log: append-only, keine Änderungen nach dem Insert
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id           TEXT PRIMARY KEY,
                    request_id   TEXT NOT NULL,
                    tenant_id    TEXT NOT NULL,
                    timestamp    REAL NOT NULL,
                    prompt_hash  TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    provider     TEXT NOT NULL,
                    tokens_in    INTEGER NOT NULL,
                    tokens_out   INTEGER NOT NULL,
                    cost_usd     REAL NOT NULL,
                    pii_detected INTEGER NOT NULL DEFAULT 0,
                    pii_types    TEXT
                )
                """
            )
            # Index für häufige Abfragen (Mandant, Zeitraum)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, timestamp)"
            )
            # Index für PII-Tracking (Compliance-Berichte)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_pii ON audit_log(pii_detected, timestamp)"
            )

            # API-Schlüssel-Tabelle (auch von api_key_auth.py verwendet)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    tenant_id              TEXT NOT NULL,
                    key_hash               TEXT PRIMARY KEY,
                    is_active              INTEGER NOT NULL DEFAULT 1,
                    rate_limit_per_minute  INTEGER NOT NULL DEFAULT 100,
                    created_at             REAL NOT NULL DEFAULT (unixepoch('now'))
                )
                """
            )
            await db.commit()

        logger.info("Audit-Log-Datenbank initialisiert: %s", self._db_url)

    async def log(
        self,
        *,
        request_id: str,
        tenant_id: str,
        prompt_hash: str,
        model: str,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        pii_detected: bool = False,
        pii_types: list[str] | None = None,
    ) -> None:
        """
        Anfrage-Metadaten in Audit-Log schreiben.

        SICHERHEIT: prompt_hash muss SHA256(raw_prompt) sein — Aufrufer ist verantwortlich.
        Diese Methode prüft nicht den Hash-Format, vertraut dem Aufrufer (router.py).

        PII-Tracking: pii_detected (boolean) und pii_types (JSON array) für Compliance-Berichte.
        """
        pii_types_json = ",".join(sorted(set(pii_types))) if pii_types else None

        async with aiosqlite.connect(self._db_url) as db:
            await db.execute(
                """
                INSERT INTO audit_log
                    (id, request_id, tenant_id, timestamp, prompt_hash,
                     model, provider, tokens_in, tokens_out, cost_usd,
                     pii_detected, pii_types)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    request_id,
                    tenant_id,
                    time.time(),
                    prompt_hash,
                    model,
                    provider,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    1 if pii_detected else 0,
                    pii_types_json,
                ),
            )
            await db.commit()

    async def get_tenant_usage(
        self, tenant_id: str, since_timestamp: float | None = None
    ) -> dict:
        """
        Verbrauchsstatistik für einen Mandanten abrufen.
        Nützlich für Kostenberichte und Abrechnungsintegration.
        """
        since = since_timestamp or (time.time() - 86400)  # Standard: letzte 24 Stunden
        async with aiosqlite.connect(self._db_url) as db:
            async with db.execute(
                """
                SELECT
                    COUNT(*)         AS request_count,
                    SUM(tokens_in)   AS total_tokens_in,
                    SUM(tokens_out)  AS total_tokens_out,
                    SUM(cost_usd)    AS total_cost_usd,
                    SUM(pii_detected) AS pii_count,
                    model,
                    provider
                FROM audit_log
                WHERE tenant_id = ? AND timestamp >= ?
                GROUP BY model, provider
                ORDER BY total_cost_usd DESC
                """,
                (tenant_id, since),
            ) as cursor:
                rows = await cursor.fetchall()

        return {
            "tenant_id": tenant_id,
            "since": since,
            "breakdown": [
                {
                    "request_count": row[0],
                    "tokens_in": row[1],
                    "tokens_out": row[2],
                    "cost_usd": round(row[3], 6),
                    "pii_detected_count": row[4],
                    "model": row[5],
                    "provider": row[6],
                }
                for row in rows
            ],
            "total_cost_usd": round(sum(r[3] for r in rows if r[3]), 6),
            "total_pii_detections": sum(r[4] for r in rows if r[4]),
        }

    async def get_pii_compliance_report(
        self, tenant_id: str | None = None, since_timestamp: float | None = None
    ) -> dict:
        """
        PII-Compliance-Bericht für Datenschutz-Audits.

        Returns:
            - Anzahl Anfragen mit PII
            - PII-Typen und Häufigkeiten
            - Zeitreihe (PII-Erkennungen pro Tag)
        """
        since = since_timestamp or (time.time() - 2592000)  # Standard: letzte 30 Tage
        async with aiosqlite.connect(self._db_url) as db:
            # Gesamt-PII-Statistiken
            where_clause = "WHERE pii_detected = 1"
            params: tuple = ()
            if tenant_id:
                where_clause += " AND tenant_id = ?"
                params = (tenant_id,)
            if since_timestamp:
                where_clause += f" AND timestamp >= ?"
                params = params + (since,)

            async with db.execute(
                f"""
                SELECT
                    COUNT(*)           AS total_pii_requests,
                    pii_types
                FROM audit_log
                {where_clause}
                """,
                params,
            ) as cursor:
                pii_rows = await cursor.fetchall()

        # PII-Typ-Häufigkeiten extrahieren
        pii_type_counts: dict[str, int] = {}
        for row in pii_rows:
            if row[1]:  # pii_types ist nicht NULL
                types = row[1].split(",")
                for pii_type in types:
                    pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1

        return {
            "tenant_id": tenant_id or "all",
            "since": since,
            "total_requests_with_pii": pii_rows[0][0] if pii_rows else 0,
            "pii_type_breakdown": pii_type_counts,
        }
