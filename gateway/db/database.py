# gateway/db/database.py
# Database-Abstraktion: SQLite (dev) oder PostgreSQL (production)
from __future__ import annotations

import os
from typing import Any


def is_postgres(db_url: str) -> bool:
    """Prüft, ob DATABASE_URL auf PostgreSQL zeigt."""
    return db_url.startswith("postgres://") or db_url.startswith("postgresql://")


async def get_db_connection(db_url: str):
    """
    Gibt eine Datenbankverbindung zurück (SQLite oder PostgreSQL).

    Returns:
        aiosqlite.Connection oder asyncpg.Connection (je nach DATABASE_URL)
    """
    if is_postgres(db_url):
        import asyncpg

        # asyncpg gibt Connection-Objekt direkt zurück
        return await asyncpg.connect(db_url)
    else:
        import aiosqlite

        # aiosqlite gibt Connection-Objekt zurück
        return await aiosqlite.connect(db_url)


async def execute_query(
    db_url: str,
    query: str,
    params: tuple | list | None = None,
    fetch: str | None = None,
) -> Any:
    """
    Führt SQL-Query aus (abstrahiert SQLite vs PostgreSQL).

    Args:
        db_url: Datenbank-URL
        query: SQL-Query
        params: Query-Parameter
        fetch: "one", "all", "val" oder None (für INSERT/UPDATE)

    Returns:
        - None (kein fetch)
        - dict (fetch="one")
        - list[dict] (fetch="all")
        - scalar (fetch="val")
    """
    params = params or ()

    if is_postgres(db_url):
        import asyncpg

        conn = await asyncpg.connect(db_url)
        try:
            # PostgreSQL nutzt $1, $2, ... statt ? — Query umschreiben
            pg_query = query
            for i, _ in enumerate(params, start=1):
                pg_query = pg_query.replace("?", f"${i}", 1)

            if fetch == "one":
                row = await conn.fetchrow(pg_query, *params)
                return dict(row) if row else None
            elif fetch == "all":
                rows = await conn.fetch(pg_query, *params)
                return [dict(row) for row in rows]
            elif fetch == "val":
                return await conn.fetchval(pg_query, *params)
            else:
                # INSERT/UPDATE ohne Rückgabewert
                await conn.execute(pg_query, *params)
        finally:
            await conn.close()
    else:
        import aiosqlite

        async with aiosqlite.connect(db_url) as db:
            db.row_factory = aiosqlite.Row  # dict-ähnliche Zeilen

            if fetch == "one":
                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    return dict(row) if row else None
            elif fetch == "all":
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
            elif fetch == "val":
                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row else None
            else:
                await db.execute(query, params)
                await db.commit()
