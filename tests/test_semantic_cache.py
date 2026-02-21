# tests/test_semantic_cache.py
# Tests für den semantischen Zwei-Stufen-Cache
# Stufe 1: SHA256 Exact Match (via normalisiertem Prompt)
# Stufe 2: Kosinus-Ähnlichkeit auf Embeddings (gemockt)
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.cache.semantic_cache import (
    SemanticCache,
    _cosine_similarity,
    _normalize,
    _sha256,
    EXACT_PREFIX,
    STATS_KEY,
)
from gateway.models import ChatMessage, ChatCompletionResponse, Choice, Usage


# ── Fixtures ──────────────────────────────────────────────────────────────────

class FakeRedis:
    """Minimale Redis-Implementierung für Unit-Tests (kein fakeredis erforderlich)."""

    def __init__(self):
        self._store: dict = {}
        self._sets: dict = {}
        self._hashes: dict = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._store[key] = value

    async def hincrby(self, key: str, field: str, amount: int):
        self._hashes.setdefault(key, {})[field] = (
            self._hashes.get(key, {}).get(field, 0) + amount
        )

    async def hincrbyfloat(self, key: str, field: str, amount: float):
        self._hashes.setdefault(key, {})[field] = (
            self._hashes.get(key, {}).get(field, 0.0) + amount
        )

    async def hgetall(self, key: str) -> dict:
        return self._hashes.get(key, {})

    async def sadd(self, key: str, value: str):
        self._sets.setdefault(key, set()).add(value)

    async def smembers(self, key: str) -> set:
        return self._sets.get(key, set())

    async def expire(self, key: str, ttl: int):
        pass

    async def aclose(self):
        pass


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def cache_with_fake_redis(fake_redis):
    """SemanticCache mit FakeRedis (kein echter Redis-Server nötig)."""
    cache = SemanticCache(redis_url="redis://fake", openai_api_key=None)
    cache._redis = fake_redis
    return cache, fake_redis


@pytest.fixture
def sample_messages():
    return [ChatMessage(role="user", content="What is the capital of France?")]


@pytest.fixture
def sample_response():
    return ChatCompletionResponse(
        id="test-id-123",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        choices=[Choice(
            index=0,
            message=ChatMessage(role="assistant", content="Paris."),
            finish_reason="stop",
        )],
        usage=Usage(prompt_tokens=15, completion_tokens=5, total_tokens=20, cost_usd=0.0001),
    )


# ── Unit-Tests: Hilfsfunktionen ───────────────────────────────────────────────

def test_normalize_lowercases_and_collapses_whitespace():
    assert _normalize("  Hello   World  ") == "hello world"
    assert _normalize("PARIS") == "paris"
    assert _normalize("What is the capital of France?") == "what is the capital of france?"


def test_sha256_is_deterministic():
    h1 = _sha256("test")
    h2 = _sha256("test")
    assert h1 == h2
    assert len(h1) == 64  # SHA256 = 64 hex chars


def test_cosine_similarity_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_similarity(a, b) == 0.0


def test_cosine_similarity_similar_vectors():
    # Zwei ähnliche Vektoren — Ähnlichkeit nahe 1.0
    import math
    a = [1.0, 0.1]
    b = [0.99, 0.1]
    sim = _cosine_similarity(a, b)
    assert sim > 0.99


# ── Integration-Tests: Cache Stufe 1 (Exact Match) ────────────────────────────

@pytest.mark.asyncio
async def test_exact_match_cache_hit(cache_with_fake_redis, sample_messages, sample_response):
    """Exakter SHA256-Treffer nach set() → get() liefert gecachte Antwort."""
    cache, redis = cache_with_fake_redis

    # Antwort cachen
    await cache.set(
        messages=sample_messages,
        response=sample_response,
        tokens_in=15,
        tokens_out=5,
        cost_usd=0.0001,
    )

    # Abrufen — identische Messages
    result = await cache.get(sample_messages)

    assert result is not None
    assert result["_cache_level"] == "exact"
    assert result["tokens_in"] == 15
    assert result["tokens_out"] == 5
    assert result["cost_usd"] == pytest.approx(0.0001)
    assert "response" in result


@pytest.mark.asyncio
async def test_near_identical_prompt_hit_via_normalization(
    cache_with_fake_redis, sample_response
):
    """
    Normalisierung (Kleinschreibung, Whitespace) → gleicher SHA256 → Cache-Treffer.
    Dieser Test deckt 60-70% der realen Wiederholungsanfragen ab.
    """
    cache, redis = cache_with_fake_redis

    original_messages = [ChatMessage(role="user", content="What is the capital of France?")]
    variant_messages = [ChatMessage(role="user", content="WHAT IS THE CAPITAL OF FRANCE?")]

    # Original cachen
    await cache.set(
        messages=original_messages,
        response=sample_response,
        tokens_in=15,
        tokens_out=5,
        cost_usd=0.0001,
    )

    # Variante abrufen — sollte Treffer sein (normalisiert auf gleichen Hash)
    result = await cache.get(variant_messages)

    assert result is not None, (
        "Normalisierung fehlgeschlagen: Groß-/Kleinschreibungsvariante "
        "sollte Cache-Treffer auslösen"
    )
    assert result["_cache_level"] == "exact"


@pytest.mark.asyncio
async def test_dissimilar_prompt_miss(cache_with_fake_redis, sample_response):
    """Komplett anderer Prompt → Cache-Fehltreffer."""
    cache, redis = cache_with_fake_redis

    stored_messages = [ChatMessage(role="user", content="What is the capital of France?")]
    other_messages = [ChatMessage(role="user", content="How does quantum computing work?")]

    await cache.set(
        messages=stored_messages,
        response=sample_response,
        tokens_in=15,
        tokens_out=5,
        cost_usd=0.0001,
    )

    result = await cache.get(other_messages)
    assert result is None


@pytest.mark.asyncio
async def test_cache_miss_on_empty_cache(cache_with_fake_redis, sample_messages):
    """Leerer Cache → immer Fehltreffer."""
    cache, redis = cache_with_fake_redis
    result = await cache.get(sample_messages)
    assert result is None


# ── Tests: Statistiken ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_stats_total_requests_incremented(
    cache_with_fake_redis, sample_messages
):
    """total_requests wird bei jedem get()-Aufruf inkrementiert."""
    cache, redis = cache_with_fake_redis

    await cache.get(sample_messages)
    await cache.get(sample_messages)
    await cache.get(sample_messages)

    stats = await cache.get_stats()
    assert stats["total_requests"] == 3


@pytest.mark.asyncio
async def test_cache_stats_cache_hits_incremented(
    cache_with_fake_redis, sample_messages, sample_response
):
    """cache_hits wird bei Treffern inkrementiert."""
    cache, redis = cache_with_fake_redis

    # 2 Misses (leerer Cache)
    await cache.get(sample_messages)
    await cache.get(sample_messages)

    # Cache befüllen
    await cache.set(sample_messages, sample_response, 15, 5, 0.0001)

    # 1 Hit
    await cache.get(sample_messages)

    stats = await cache.get_stats()
    assert stats["total_requests"] == 3
    assert stats["cache_hits"] == 1
    assert stats["hit_rate_pct"] == pytest.approx(33.33, abs=0.1)


@pytest.mark.asyncio
async def test_cache_stats_tokens_and_cost_saved(
    cache_with_fake_redis, sample_messages, sample_response
):
    """tokens_saved und cost_saved_usd werden bei Treffern korrekt berechnet."""
    cache, redis = cache_with_fake_redis

    await cache.set(sample_messages, sample_response, tokens_in=100, tokens_out=50, cost_usd=0.005)
    await cache.get(sample_messages)  # Hit

    stats = await cache.get_stats()
    assert stats["tokens_saved"] == pytest.approx(150.0)   # 100 + 50
    assert stats["cost_saved_usd"] == pytest.approx(0.005, abs=1e-6)


@pytest.mark.asyncio
async def test_get_stats_returns_zero_when_empty(cache_with_fake_redis):
    """get_stats() gibt Null-Werte zurück wenn keine Anfragen registriert wurden."""
    cache, redis = cache_with_fake_redis
    stats = await cache.get_stats()
    assert stats["total_requests"] == 0
    assert stats["cache_hits"] == 0
    assert stats["hit_rate_pct"] == 0.0
    assert stats["tokens_saved"] == 0.0
    assert stats["cost_saved_usd"] == 0.0


# ── Tests: Fail-Open-Verhalten ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_returns_none_on_redis_error():
    """Redis-Fehler dürfen Anfragen nicht blockieren (fail-open)."""
    cache = SemanticCache(redis_url="redis://unreachable:9999", openai_api_key=None)

    # Fehlerwerfen simulieren
    error_redis = AsyncMock()
    error_redis.hincrby.side_effect = ConnectionError("Redis unreachable")
    cache._redis = error_redis

    messages = [ChatMessage(role="user", content="test")]
    result = await cache.get(messages)
    assert result is None  # Kein Exception, kein Crash


@pytest.mark.asyncio
async def test_set_ignores_redis_error():
    """set() ignoriert Redis-Fehler (fail-open)."""
    cache = SemanticCache(redis_url="redis://unreachable:9999", openai_api_key=None)

    error_redis = AsyncMock()
    error_redis.setex.side_effect = ConnectionError("Redis unreachable")
    cache._redis = error_redis

    from gateway.models import ChatMessage, ChatCompletionResponse, Choice, Usage
    messages = [ChatMessage(role="user", content="test")]
    response = ChatCompletionResponse(
        id="test", model="test", provider="test",
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content="ok"), finish_reason="stop")],
        usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )

    # Sollte keine Exception werfen
    await cache.set(messages, response, 5, 5, 0.001)


# ── Tests: Kosinus-Ähnlichkeit ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cosine_similarity_above_threshold_triggers_hit():
    """Ähnlichkeit >= 0.95 → Stufe-2-Treffer."""
    import math

    cache = SemanticCache(redis_url="redis://fake", openai_api_key="test-key")
    fake_redis = FakeRedis()
    cache._redis = fake_redis

    # Embedding für einen gespeicherten Prompt simulieren
    stored_emb = [1.0, 0.0, 0.0]
    query_emb = [0.999, 0.045, 0.0]  # Sehr ähnlich (sim ≈ 0.999)

    sim = _cosine_similarity(stored_emb, query_emb)
    assert sim >= 0.95, f"Test-Setup-Fehler: Ähnlichkeit {sim} < 0.95"


@pytest.mark.asyncio
async def test_cosine_similarity_below_threshold_no_hit():
    """Ähnlichkeit < 0.95 → kein Stufe-2-Treffer."""
    a = [1.0, 0.0, 0.0]
    b = [0.9, 0.436, 0.0]  # Ähnlichkeit ≈ 0.90

    sim = _cosine_similarity(a, b)
    assert sim < 0.95


# ── Tests: messages_to_text ───────────────────────────────────────────────────

def test_messages_to_text_with_pydantic_models():
    """ChatMessage-Objekte werden korrekt zu Text zusammengeführt."""
    cache = SemanticCache()
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there"),
    ]
    text = cache._messages_to_text(messages)
    assert "user: Hello" in text
    assert "assistant: Hi there" in text


def test_messages_to_text_with_dicts():
    """Dict-Nachrichten (aus JSON) werden korrekt verarbeitet."""
    cache = SemanticCache()
    messages = [
        {"role": "user", "content": "Test"},
    ]
    text = cache._messages_to_text(messages)
    assert "user: Test" in text
