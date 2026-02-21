# gateway/middleware/audit_logger.py
# Unveränderlicher Audit-Log: SHA256-Hash des Prompts, KEIN Klartext.
# Session 8: Hash-Kette — jeder Datensatz enthält SHA256(Felder + prev_hash).
# Manipulationsnachweis durch mathematische Verkettung, keine Datenbank-Trigger nötig.
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import time
import uuid
import zipfile

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "./gateway.db")

GENESIS_HASH = "GENESIS"  # Sentinel für den ersten Datensatz in der Kette


def _is_postgres(db_url: str) -> bool:
    """Prüft, ob DATABASE_URL auf PostgreSQL zeigt."""
    return db_url.startswith("postgres://") or db_url.startswith("postgresql://")


def _compute_record_hash(
    *,
    id: str,
    tenant_id: str,
    timestamp: float,
    prompt_hash: str,
    model: str,
    provider: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    pii_detected: int,
    prev_hash: str,
) -> str:
    """
    SHA256-Hash eines Audit-Datensatzes berechnen.

    Kanonisches Format: pipe-getrennte Felder in fester Reihenfolge.
    Jede Änderung an einem Feld macht den Hash und alle nachfolgenden ungültig.
    """
    canonical = "|".join([
        id,
        tenant_id,
        f"{timestamp:.6f}",
        prompt_hash,
        model,
        provider,
        str(tokens_in),
        str(tokens_out),
        f"{cost_usd:.8f}",
        str(pii_detected),
        prev_hash,
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLogger:
    """
    Append-Only Audit-Protokollierung für alle LLM-Anfragen mit Hash-Kette.

    Datenschutzprinzipien:
    - Roher Prompt-Inhalt wird NIEMALS gespeichert
    - Nur SHA256(prompt) für Nachvollziehbarkeit ohne Inhaltseinsicht
    - Kosten-Tracking: jede Anfrage mit Mandant, Modell, Token, Kosten protokolliert
    - Hash-Kette: jeder Datensatz enthält SHA256(Felder + prev_hash) — Manipulation erkennbar
    - Append-only: keine UPDATE/DELETE-Operationen auf audit_log
    """

    def __init__(self, db_url: str = DATABASE_URL) -> None:
        self._db_url = db_url
        # Mutex verhindert Race-Condition beim Lesen+Schreiben des letzten Hashes
        self._insert_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Datenbankschema erstellen (idempotent — sicher für mehrfachen Aufruf)."""
        if _is_postgres(self._db_url):
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                # Audit-Log: append-only mit Hash-Kette
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id           TEXT PRIMARY KEY,
                        request_id   TEXT NOT NULL,
                        tenant_id    TEXT NOT NULL,
                        timestamp    DOUBLE PRECISION NOT NULL,
                        prompt_hash  TEXT NOT NULL,
                        model        TEXT NOT NULL,
                        provider     TEXT NOT NULL,
                        tokens_in    INTEGER NOT NULL,
                        tokens_out   INTEGER NOT NULL,
                        cost_usd     DOUBLE PRECISION NOT NULL,
                        pii_detected INTEGER NOT NULL DEFAULT 0,
                        pii_types    TEXT,
                        prev_hash    TEXT,
                        record_hash  TEXT,
                        cache_hit    INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                # Hash-Ketten-Spalten nachrüsten (existierende Deployments)
                await conn.execute(
                    "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS prev_hash TEXT"
                )
                await conn.execute(
                    "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS record_hash TEXT"
                )
                # Session 9: cache_hit-Spalte (nicht Teil der Hash-Kette — reines Metadatum)
                await conn.execute(
                    "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS cache_hit INTEGER NOT NULL DEFAULT 0"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, timestamp)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_pii ON audit_log(pii_detected, timestamp)"
                )
                await conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
                )

                # API-Schlüssel-Tabelle
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        tenant_id              TEXT NOT NULL,
                        key_hash               TEXT PRIMARY KEY,
                        is_active              INTEGER NOT NULL DEFAULT 1,
                        rate_limit_per_minute  INTEGER NOT NULL DEFAULT 100,
                        token_budget_monthly   INTEGER,
                        created_at             DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())
                    )
                    """
                )
                # Spalte nachrüsten (existierende Deployments)
                await conn.execute(
                    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS token_budget_monthly INTEGER"
                )
            finally:
                await conn.close()
        else:
            # SQLite-Schema (dev)
            import aiosqlite

            async with aiosqlite.connect(self._db_url) as db:
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
                        pii_types    TEXT,
                        prev_hash    TEXT,
                        record_hash  TEXT,
                        cache_hit    INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                # Hash-Ketten-Spalten + cache_hit nachrüsten (SQLite: Exception abfangen)
                for col_def in [
                    "ALTER TABLE audit_log ADD COLUMN prev_hash TEXT",
                    "ALTER TABLE audit_log ADD COLUMN record_hash TEXT",
                    "ALTER TABLE audit_log ADD COLUMN cache_hit INTEGER NOT NULL DEFAULT 0",
                ]:
                    try:
                        await db.execute(col_def)
                    except Exception:
                        pass  # Spalte existiert bereits
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, timestamp)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_pii ON audit_log(pii_detected, timestamp)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        tenant_id              TEXT NOT NULL,
                        key_hash               TEXT PRIMARY KEY,
                        is_active              INTEGER NOT NULL DEFAULT 1,
                        rate_limit_per_minute  INTEGER NOT NULL DEFAULT 100,
                        token_budget_monthly   INTEGER,
                        created_at             REAL NOT NULL DEFAULT (unixepoch('now'))
                    )
                    """
                )
                try:
                    await db.execute(
                        "ALTER TABLE api_keys ADD COLUMN token_budget_monthly INTEGER"
                    )
                except Exception:
                    pass  # Spalte existiert bereits
                await db.commit()

        logger.info("Audit-Log-Datenbank initialisiert: %s", self._db_url)

    async def _get_last_record_hash(self, conn) -> str:
        """
        Letzten record_hash aus der Tabelle lesen (innerhalb einer gesperrten Transaktion).
        Gibt GENESIS_HASH zurück wenn die Tabelle leer ist oder kein Hash gesetzt ist.
        """
        if _is_postgres(self._db_url):
            row = await conn.fetchrow(
                "SELECT record_hash FROM audit_log ORDER BY timestamp DESC, id DESC LIMIT 1"
            )
            if row and row["record_hash"]:
                return row["record_hash"]
        else:
            async with conn.execute(
                "SELECT record_hash FROM audit_log ORDER BY timestamp DESC, id DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                if row and row[0]:
                    return row[0]
        return GENESIS_HASH

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
        cache_hit: bool = False,
    ) -> None:
        """
        Anfrage-Metadaten in Audit-Log schreiben (mit Hash-Kette).

        SICHERHEIT: prompt_hash muss SHA256(raw_prompt) sein — Aufrufer ist verantwortlich.
        Der Insert-Mutex verhindert Race-Conditions beim Lesen des vorherigen Hashes.
        cache_hit ist NICHT Teil der Hash-Kette (reines Metadatum, kein Tamper-Evidence-Feld).
        """
        pii_types_json = ",".join(sorted(set(pii_types))) if pii_types else None
        record_id = str(uuid.uuid4())
        ts = time.time()
        pii_int = 1 if pii_detected else 0
        cache_hit_int = 1 if cache_hit else 0

        async with self._insert_lock:
            if _is_postgres(self._db_url):
                import asyncpg

                conn = await asyncpg.connect(self._db_url)
                try:
                    prev_hash = await self._get_last_record_hash(conn)
                    rec_hash = _compute_record_hash(
                        id=record_id,
                        tenant_id=tenant_id,
                        timestamp=ts,
                        prompt_hash=prompt_hash,
                        model=model,
                        provider=provider,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=cost_usd,
                        pii_detected=pii_int,
                        prev_hash=prev_hash,
                    )
                    await conn.execute(
                        """
                        INSERT INTO audit_log
                            (id, request_id, tenant_id, timestamp, prompt_hash,
                             model, provider, tokens_in, tokens_out, cost_usd,
                             pii_detected, pii_types, prev_hash, record_hash, cache_hit)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        """,
                        record_id, request_id, tenant_id, ts, prompt_hash,
                        model, provider, tokens_in, tokens_out, cost_usd,
                        pii_int, pii_types_json, prev_hash, rec_hash, cache_hit_int,
                    )
                finally:
                    await conn.close()
            else:
                import aiosqlite

                async with aiosqlite.connect(self._db_url) as db:
                    prev_hash = await self._get_last_record_hash(db)
                    rec_hash = _compute_record_hash(
                        id=record_id,
                        tenant_id=tenant_id,
                        timestamp=ts,
                        prompt_hash=prompt_hash,
                        model=model,
                        provider=provider,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=cost_usd,
                        pii_detected=pii_int,
                        prev_hash=prev_hash,
                    )
                    await db.execute(
                        """
                        INSERT INTO audit_log
                            (id, request_id, tenant_id, timestamp, prompt_hash,
                             model, provider, tokens_in, tokens_out, cost_usd,
                             pii_detected, pii_types, prev_hash, record_hash, cache_hit)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record_id, request_id, tenant_id, ts, prompt_hash,
                            model, provider, tokens_in, tokens_out, cost_usd,
                            pii_int, pii_types_json, prev_hash, rec_hash, cache_hit_int,
                        ),
                    )
                    await db.commit()

    async def verify_chain(self) -> dict:
        """
        Hash-Kette vollständig verifizieren.

        Liest alle Datensätze in Zeitstempel-Reihenfolge und prüft:
        1. prev_hash des aktuellen Datensatzes == record_hash des vorherigen
        2. record_hash == SHA256(Felder + prev_hash)

        Returns:
            {"chain_valid": True, "records_checked": N}
            {"chain_valid": False, "records_checked": N, "broken_at": "record_id", "reason": "..."}
        """
        rows = await self._fetch_all_rows_ordered()

        if not rows:
            return {"chain_valid": True, "records_checked": 0}

        expected_prev = GENESIS_HASH
        for i, row in enumerate(rows):
            rec_id = row["id"]

            # Prüfung 1: prev_hash stimmt mit vorherigem record_hash überein
            actual_prev = row.get("prev_hash") or GENESIS_HASH
            if actual_prev != expected_prev:
                return {
                    "chain_valid": False,
                    "records_checked": i,
                    "broken_at": rec_id,
                    "reason": f"prev_hash mismatch: expected {expected_prev[:16]}…, got {actual_prev[:16]}…",
                }

            # Prüfung 2: record_hash ist korrekt berechnet
            expected_hash = _compute_record_hash(
                id=rec_id,
                tenant_id=row["tenant_id"],
                timestamp=row["timestamp"],
                prompt_hash=row["prompt_hash"],
                model=row["model"],
                provider=row["provider"],
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                cost_usd=row["cost_usd"],
                pii_detected=row["pii_detected"],
                prev_hash=actual_prev,
            )
            stored_hash = row.get("record_hash") or ""
            if stored_hash != expected_hash:
                return {
                    "chain_valid": False,
                    "records_checked": i,
                    "broken_at": rec_id,
                    "reason": "record_hash tampered or missing",
                }

            expected_prev = stored_hash

        return {"chain_valid": True, "records_checked": len(rows)}

    async def _fetch_all_rows_ordered(self) -> list[dict]:
        """Alle Audit-Datensätze in Einfüge-Reihenfolge (timestamp ASC, id ASC) abrufen."""
        if _is_postgres(self._db_url):
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, tenant_id, timestamp, prompt_hash, model, provider,
                           tokens_in, tokens_out, cost_usd, pii_detected, pii_types,
                           prev_hash, record_hash
                    FROM audit_log
                    ORDER BY timestamp ASC, id ASC
                    """
                )
                return [dict(r) for r in rows]
            finally:
                await conn.close()
        else:
            import aiosqlite

            async with aiosqlite.connect(self._db_url) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """
                    SELECT id, tenant_id, timestamp, prompt_hash, model, provider,
                           tokens_in, tokens_out, cost_usd, pii_detected, pii_types,
                           prev_hash, record_hash
                    FROM audit_log
                    ORDER BY timestamp ASC, id ASC
                    """
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(r) for r in rows]

    async def build_compliance_bundle(self, gateway_version: str = "1.0.0") -> bytes:
        """
        Compliance-Export-Bundle als ZIP-Bytes erstellen.

        Inhalt:
        - audit_log.jsonl         — alle Datensätze im JSON-Lines-Format
        - integrity_manifest.json — SHA256 jedes record_hash + Gesamt-Chain-Hash
        - system_config.json      — Gateway-Konfiguration + Export-Metadaten

        Dieses Bundle ist das, was ein CTO dem Enterprise-Kunden für DSGVO-Audits schickt.
        """
        rows = await self._fetch_all_rows_ordered()
        chain_status = await self.verify_chain()

        # audit_log.jsonl: ein JSON-Objekt pro Zeile
        jsonl_lines = []
        manifest_entries = []
        for row in rows:
            record = {
                "id": row["id"],
                "tenant_id": row["tenant_id"],
                "timestamp": row["timestamp"],
                "prompt_hash": row["prompt_hash"],
                "model": row["model"],
                "provider": row["provider"],
                "tokens_in": row["tokens_in"],
                "tokens_out": row["tokens_out"],
                "cost_usd": row["cost_usd"],
                "pii_detected": bool(row["pii_detected"]),
                "pii_types": row["pii_types"].split(",") if row["pii_types"] else [],
                "prev_hash": row.get("prev_hash") or GENESIS_HASH,
                "record_hash": row.get("record_hash") or "",
            }
            jsonl_lines.append(json.dumps(record, ensure_ascii=False))
            manifest_entries.append({
                "id": row["id"],
                "record_hash": row.get("record_hash") or "",
            })

        jsonl_content = "\n".join(jsonl_lines)

        # integrity_manifest.json
        overall_chain_hash = hashlib.sha256(
            "\n".join(e["record_hash"] for e in manifest_entries).encode("utf-8")
        ).hexdigest()
        manifest = {
            "export_timestamp": time.time(),
            "total_records": len(rows),
            "chain_valid": chain_status["chain_valid"],
            "overall_chain_hash": overall_chain_hash,
            "records": manifest_entries,
        }

        # system_config.json
        import importlib.metadata

        providers_configured = [
            p for p in ["anthropic", "openai", "gemini", "ollama"]
            if os.getenv(f"{p.upper()}_API_KEY") or p == "ollama"
        ]
        system_config = {
            "gateway_version": gateway_version,
            "export_timestamp": time.time(),
            "database_url_type": "postgresql" if _is_postgres(self._db_url) else "sqlite",
            "providers_configured": providers_configured,
            "pii_types_detected": [
                "IBAN", "Steuer-ID", "KFZ-Kennzeichen",
                "Handynummer", "PLZ", "Straße", "Name",
            ],
            "retention_policy": "indefinite (append-only, no deletes)",
            "hash_algorithm": "SHA256",
            "chain_genesis_sentinel": GENESIS_HASH,
            "compliance_standards": ["DSGVO", "GDPR"],
        }

        # ZIP zusammenstellen (in-memory)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("audit_log.jsonl", jsonl_content)
            zf.writestr("integrity_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            zf.writestr("system_config.json", json.dumps(system_config, indent=2, ensure_ascii=False))

        return buf.getvalue()

    async def get_tenant_usage(
        self, tenant_id: str, since_timestamp: float | None = None
    ) -> dict:
        """
        Verbrauchsstatistik für einen Mandanten abrufen.
        Nützlich für Kostenberichte und Abrechnungsintegration.
        """
        since = since_timestamp or (time.time() - 86400)  # Standard: letzte 24 Stunden

        if _is_postgres(self._db_url):
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                rows = await conn.fetch(
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
                    WHERE tenant_id = $1 AND timestamp >= $2
                    GROUP BY model, provider
                    ORDER BY total_cost_usd DESC
                    """,
                    tenant_id,
                    since,
                )
            finally:
                await conn.close()
        else:
            import aiosqlite

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

        if _is_postgres(self._db_url):
            import asyncpg

            where_clause = "WHERE pii_detected = 1"
            params: list = []
            if tenant_id:
                where_clause += " AND tenant_id = $1"
                params.append(tenant_id)
            if since_timestamp:
                where_clause += f" AND timestamp >= ${len(params) + 1}"
                params.append(since)

            conn = await asyncpg.connect(self._db_url)
            try:
                pii_rows = await conn.fetch(
                    f"""
                    SELECT
                        COUNT(*)           AS total_pii_requests,
                        pii_types
                    FROM audit_log
                    {where_clause}
                    """,
                    *params,
                )
            finally:
                await conn.close()
        else:
            import aiosqlite

            where_clause = "WHERE pii_detected = 1"
            params_tuple: tuple = ()
            if tenant_id:
                where_clause += " AND tenant_id = ?"
                params_tuple = (tenant_id,)
            if since_timestamp:
                where_clause += " AND timestamp >= ?"
                params_tuple = params_tuple + (since,)

            async with aiosqlite.connect(self._db_url) as db:
                async with db.execute(
                    f"""
                    SELECT
                        COUNT(*)           AS total_pii_requests,
                        pii_types
                    FROM audit_log
                    {where_clause}
                    """,
                    params_tuple,
                ) as cursor:
                    pii_rows = await cursor.fetchall()

        pii_type_counts: dict[str, int] = {}
        for row in pii_rows:
            if row[1]:
                types = row[1].split(",")
                for pii_type in types:
                    pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1

        return {
            "tenant_id": tenant_id or "all",
            "since": since,
            "total_requests_with_pii": pii_rows[0][0] if pii_rows else 0,
            "pii_type_breakdown": pii_type_counts,
        }

    async def get_monthly_costs_by_tenant(self, since_timestamp: float | None = None) -> dict[str, float]:
        """
        Kosten pro Mandant für den aktuellen Monat aus dem Audit-Log.

        Returns:
            {tenant_id: total_cost_usd}
        """
        since = since_timestamp or (time.time() - 31 * 86400)

        if _is_postgres(self._db_url):
            import asyncpg

            conn = await asyncpg.connect(self._db_url)
            try:
                rows = await conn.fetch(
                    """
                    SELECT tenant_id, COALESCE(SUM(cost_usd), 0) AS total_cost
                    FROM audit_log
                    WHERE timestamp >= $1
                    GROUP BY tenant_id
                    """,
                    since,
                )
                return {row["tenant_id"]: float(row["total_cost"]) for row in rows}
            finally:
                await conn.close()
        else:
            import aiosqlite

            async with aiosqlite.connect(self._db_url) as db:
                async with db.execute(
                    """
                    SELECT tenant_id, COALESCE(SUM(cost_usd), 0) AS total_cost
                    FROM audit_log
                    WHERE timestamp >= ?
                    GROUP BY tenant_id
                    """,
                    (since,),
                ) as cursor:
                    rows = await cursor.fetchall()
                    return {row[0]: float(row[1]) for row in rows}
