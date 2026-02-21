# gateway/cache/semantic_cache.py
# Semantischer Zwei-Stufen-Cache: SHA256-Exact-Match + Kosinus-Ähnlichkeit (OpenAI Embeddings)
# Stufe 1: SHA256(normalisierter Prompt) — kostenlos, <1ms
# Stufe 2: Kosinus-Ähnlichkeit auf OpenAI text-embedding-3-small — wenn API-Key vorhanden
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Redis-Schlüssel-Präfixe
EXACT_PREFIX = "scache:exact:"
EMB_PREFIX = "scache:emb:"
EMB_INDEX_KEY = "scache:emb_index"
STATS_KEY = "scache:stats"

SIMILARITY_THRESHOLD = 0.95   # Kosinus-Schwellenwert für semantischen Treffer
TTL_SECONDS = 3600             # Cache-Lebensdauer: 1 Stunde
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 Dimensionen, günstig


def _normalize(text: str) -> str:
    """Aggressieve Normalisierung: Kleinschreibung + Whitespace kollabieren."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Kosinus-Ähnlichkeit mit stdlib.math — kein numpy erforderlich.

    Laufzeit: O(n) wobei n = Embedding-Dimensionen (1536 für text-embedding-3-small).
    Für Gateway-Lasten (<10K unique Prompts) ausreichend effizient.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    """
    Semantischer LLM-Response-Cache mit zwei Trefferstufen.

    Stufe 1: SHA256 der normalisierten Anfrage — kostenlos, <1ms
             Trifft bei identischen und near-identical Prompts (Groß/Kleinschreibung, Leerzeichen)
    Stufe 2: Kosinus-Ähnlichkeit auf OpenAI-Embeddings — optional (benötigt OPENAI_API_KEY)
             Trifft bei semantisch ähnlichen Formulierungen (Schwellenwert: 0.95)

    Cache-Treffer liefern gespeicherte Antworten ohne Provider-API-Aufruf.
    Cache-Fehler blockieren niemals Anfragen (fail-open für Verfügbarkeit).

    Statistiken werden in Redis (STATS_KEY) atomisch inkrementiert:
      total_requests, cache_hits, tokens_saved, cost_saved_usd
    """

    def __init__(
        self,
        redis_url: str | None = None,
        openai_api_key: str | None = None,
        threshold: float = SIMILARITY_THRESHOLD,
        ttl: int = TTL_SECONDS,
    ) -> None:
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._threshold = threshold
        self._ttl = ttl
        self._redis = None
        self._http = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def _get_http(self):
        import httpx

        if self._http is None:
            self._http = httpx.AsyncClient(timeout=10.0)
        return self._http

    def _messages_to_text(self, messages: list[Any]) -> str:
        """Chat-Nachrichten zu vergleichbarem String zusammenführen."""
        parts = []
        for m in messages:
            if hasattr(m, "role") and hasattr(m, "content"):
                parts.append(f"{m.role}: {m.content}")
            elif isinstance(m, dict):
                parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        return "\n".join(parts)

    async def get(self, messages: list[Any]) -> dict | None:
        """
        Cache-Treffer suchen. Gibt Payload-Dict zurück oder None bei Fehltreffer.

        Inkrementiert total_requests bei jedem Aufruf.
        Inkrementiert cache_hits + tokens_saved + cost_saved_usd bei Treffer.
        """
        text = self._messages_to_text(messages)
        normalized = _normalize(text)
        exact_key = f"{EXACT_PREFIX}{_sha256(normalized)}"

        try:
            redis = await self._get_redis()
            await redis.hincrby(STATS_KEY, "total_requests", 1)

            # Stufe 1: Exakter SHA256-Treffer
            cached = await redis.get(exact_key)
            if cached:
                payload = json.loads(cached)
                await self._record_hit(redis, payload)
                payload["_cache_level"] = "exact"
                return payload

            # Stufe 2: Embedding-Kosinus-Ähnlichkeit (nur wenn API-Key vorhanden)
            if self._openai_api_key:
                result = await self._similarity_lookup(text, redis)
                if result:
                    result["_cache_level"] = "semantic"
                    return result

        except Exception as exc:
            # Cache-Fehler dürfen Anfragen NICHT blockieren
            logger.warning("Cache-Fehler (fail-open): %s", exc)

        return None

    async def set(
        self,
        messages: list[Any],
        response: Any,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        """Antwort mit Metadaten im Cache speichern."""
        text = self._messages_to_text(messages)
        normalized = _normalize(text)
        key_hash = _sha256(normalized)
        exact_key = f"{EXACT_PREFIX}{key_hash}"

        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else response
        )
        payload = {
            "response": response_dict,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd,
            "stored_at": time.time(),
            "hit_count": 0,
        }

        try:
            redis = await self._get_redis()
            await redis.setex(exact_key, self._ttl, json.dumps(payload))

            # Embedding für Stufe-2-Ähnlichkeitssuche speichern (asynchron, blockiert nicht)
            if self._openai_api_key:
                await self._store_embedding(text, key_hash, redis)

        except Exception as exc:
            logger.warning("Cache-Speicherfehler (ignoriert): %s", exc)

    async def get_stats(self) -> dict:
        """Cache-Statistiken für /admin/cache/stats zurückgeben."""
        try:
            redis = await self._get_redis()
            raw = await redis.hgetall(STATS_KEY)
            total = int(raw.get("total_requests", 0))
            hits = int(raw.get("cache_hits", 0))
            hit_rate = round(hits / total * 100, 2) if total > 0 else 0.0
            return {
                "total_requests": total,
                "cache_hits": hits,
                "hit_rate_pct": hit_rate,
                "tokens_saved": float(raw.get("tokens_saved", 0.0)),
                "cost_saved_usd": round(float(raw.get("cost_saved_usd", 0.0)), 6),
            }
        except Exception:
            return {
                "total_requests": 0,
                "cache_hits": 0,
                "hit_rate_pct": 0.0,
                "tokens_saved": 0.0,
                "cost_saved_usd": 0.0,
            }

    async def _record_hit(self, redis, payload: dict) -> None:
        """Cache-Treffer-Statistiken atomisch in Redis inkrementieren."""
        tokens_saved = payload.get("tokens_in", 0) + payload.get("tokens_out", 0)
        cost_saved = payload.get("cost_usd", 0.0)
        await redis.hincrby(STATS_KEY, "cache_hits", 1)
        await redis.hincrbyfloat(STATS_KEY, "tokens_saved", tokens_saved)
        await redis.hincrbyfloat(STATS_KEY, "cost_saved_usd", cost_saved)

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Embedding über OpenAI Embeddings API abrufen (text-embedding-3-small)."""
        try:
            http = await self._get_http()
            resp = await http.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": text[:8000], "model": EMBEDDING_MODEL},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as exc:
            logger.debug("Embedding-API-Fehler (Stufe-2-Cache deaktiviert): %s", exc)
            return None

    async def _store_embedding(self, text: str, key_hash: str, redis) -> None:
        """Embedding für spätere Ähnlichkeitssuche in Redis speichern."""
        embedding = await self._get_embedding(text)
        if not embedding:
            return
        emb_key = f"{EMB_PREFIX}{key_hash}"
        emb_payload = json.dumps({
            "embedding": embedding,
            "response_key": f"{EXACT_PREFIX}{key_hash}",
        })
        await redis.setex(emb_key, self._ttl, emb_payload)
        await redis.sadd(EMB_INDEX_KEY, emb_key)
        await redis.expire(EMB_INDEX_KEY, self._ttl)

    async def _similarity_lookup(self, text: str, redis) -> dict | None:
        """Semantisch ähnlichste gecachte Antwort per Kosinus-Ähnlichkeit suchen."""
        query_emb = await self._get_embedding(text)
        if not query_emb:
            return None

        emb_keys = await redis.smembers(EMB_INDEX_KEY)
        if not emb_keys:
            return None

        best_sim = 0.0
        best_response_key = None

        for emb_key in emb_keys:
            raw = await redis.get(emb_key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
                stored_emb = data.get("embedding", [])
                if stored_emb:
                    sim = _cosine_similarity(query_emb, stored_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_response_key = data.get("response_key")
            except Exception:
                continue

        if best_sim >= self._threshold and best_response_key:
            cached = await redis.get(best_response_key)
            if cached:
                payload = json.loads(cached)
                await self._record_hit(redis, payload)
                return payload

        return None

    async def close(self) -> None:
        """HTTP-Client und Redis-Verbindung ordnungsgemäß schließen."""
        if self._http:
            await self._http.aclose()
        if self._redis:
            await self._redis.aclose()
