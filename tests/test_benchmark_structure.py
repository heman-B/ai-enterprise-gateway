# tests/test_benchmark_structure.py
# Tests für das Benchmark-Skript: Struktur, Ausgabeformat, Qualitätsbewertung
# Verwendet gemockte Gateway-Antworten (keine echten API-Aufrufe)
import json
import pytest
import statistics
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import respx

# Benchmark-Funktionen importieren
from scripts.benchmark import (
    BenchmarkResult,
    ProviderSummary,
    compute_summary,
    run_benchmark,
    run_failover_test,
    score_response,
    write_results,
    PROVIDER_MODELS,
    PROMPTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_RESPONSE = {
    "id": "test-response-id",
    "object": "chat.completion",
    "model": "claude-haiku-4-5-20251001",
    "provider": "anthropic",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Paris is the capital of France."},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
        "cost_usd": 0.000008,
    },
}

MOCK_RESPONSE_CODE = {
    "id": "test-code-id",
    "object": "chat.completion",
    "model": "claude-haiku-4-5-20251001",
    "provider": "anthropic",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": (
                    "Here's a binary search implementation:\n\n"
                    "```python\ndef binary_search(arr: list, target: int) -> int:\n"
                    '    """Binary search. O(log n) time complexity."""\n'
                    "    left, right = 0, len(arr) - 1\n"
                    "    while left <= right:\n"
                    "        mid = (left + right) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        elif arr[mid] < target:\n"
                    "            left = mid + 1\n"
                    "        else:\n"
                    "            right = mid - 1\n"
                    "    return -1\n```\n"
                    "Time complexity: O(log n)."
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 120,
        "total_tokens": 170,
        "cost_usd": 0.000042,
    },
}


# ── Tests: Qualitätsbewertungsrubrik ─────────────────────────────────────────

def test_score_response_simple_correct_answer():
    """Korrekte kurze Antwort erhält Punkte."""
    prompt_spec = {
        "expected_keywords": ["Paris"],
        "expected_length": (3, 100),
    }
    score = score_response("Paris is the capital of France.", prompt_spec)
    assert score > 5.0, f"Erwarteter Score > 5.0, erhalten: {score}"
    assert score <= 10.0


def test_score_response_empty_answer_scores_low():
    """Leere Antwort erhält niedrigen Score."""
    prompt_spec = {
        "expected_keywords": ["Paris"],
        "expected_length": (3, 100),
    }
    score = score_response("", prompt_spec)
    assert score < 3.0


def test_score_response_code_prompt_with_code_block():
    """Code-Prompt mit Code-Block erhält vollen Code-Bonus."""
    prompt_spec = {
        "expected_keywords": ["def", "binary", "mid"],
        "expected_length": (150, 1500),
        "requires_code": True,
    }
    code_response = (
        "```python\ndef binary_search(arr, target):\n"
        "    mid = len(arr) // 2\n    return mid\n```\n"
        "Time complexity: O(log n)"
    )
    score = score_response(code_response, prompt_spec)
    assert score >= 7.0, f"Code-Antwort mit Block: erwartet >=7.0, erhalten {score}"


def test_score_response_code_prompt_without_code_block():
    """Code-Prompt ohne Code-Block erhält keinen Code-Bonus."""
    prompt_spec = {
        "expected_keywords": ["def"],
        "expected_length": (50, 1500),
        "requires_code": True,
    }
    no_code_response = "You can implement binary search by dividing the array."
    score = score_response(no_code_response, prompt_spec)
    # Ohne Code-Block: kein +3 für Code, niedrigerer Score
    score_with_code = score_response(
        "```python\ndef search(): pass\n```\n" + no_code_response, prompt_spec
    )
    assert score_with_code > score


def test_score_response_max_is_10():
    """Score überschreitet niemals 10.0."""
    prompt_spec = {
        "expected_keywords": ["Paris", "capital", "France"],
        "expected_length": (5, 1000),
        "requires_code": False,
    }
    perfect_response = "Paris is the beautiful capital city of France, located in western Europe."
    score = score_response(perfect_response, prompt_spec)
    assert score <= 10.0


# ── Tests: BenchmarkResult-Struktur ──────────────────────────────────────────

def test_benchmark_result_dataclass():
    """BenchmarkResult hat alle erforderlichen Felder."""
    result = BenchmarkResult(
        provider="anthropic",
        model="claude-haiku-4-5",
        complexity="simple",
        prompt_index=0,
        latency_ms=320.5,
        tokens_in=20,
        tokens_out=10,
        cost_usd=0.000008,
        quality_score=8.0,
        success=True,
    )
    assert result.provider == "anthropic"
    assert result.latency_ms == 320.5
    assert result.success is True
    assert result.error is None


# ── Tests: compute_summary ────────────────────────────────────────────────────

def test_compute_summary_calculates_p50_p95():
    """compute_summary berechnet korrekte Perzentile."""
    results = [
        BenchmarkResult(
            provider="anthropic", model="claude-haiku-4-5",
            complexity="simple", prompt_index=i,
            latency_ms=float(100 + i * 50),  # 100, 150, 200, 250, 300 ms
            tokens_in=20, tokens_out=10, cost_usd=0.001,
            quality_score=8.0, success=True,
        )
        for i in range(5)
    ]
    summaries = compute_summary(results)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.provider == "anthropic"
    assert s.p50_ms == 200.0  # Median von [100, 150, 200, 250, 300]
    assert s.success_rate == 100.0


def test_compute_summary_excludes_failures():
    """compute_summary ignoriert fehlgeschlagene Anfragen für Metriken."""
    results = [
        BenchmarkResult(
            provider="openai", model="gpt-4o-mini", complexity="simple",
            prompt_index=0, latency_ms=200.0, tokens_in=20, tokens_out=10,
            cost_usd=0.001, quality_score=8.0, success=True,
        ),
        BenchmarkResult(
            provider="openai", model="gpt-4o-mini", complexity="simple",
            prompt_index=1, latency_ms=0.0, tokens_in=0, tokens_out=0,
            cost_usd=0.0, quality_score=0.0, success=False, error="Timeout",
        ),
    ]
    summaries = compute_summary(results)
    assert len(summaries) == 1
    assert summaries[0].success_rate == 50.0


def test_compute_summary_sorted_by_latency():
    """Zusammenfassung ist nach p50-Latenz sortiert (aufsteigend)."""
    results = [
        BenchmarkResult(
            provider="ollama", model="llama3.2", complexity="simple",
            prompt_index=0, latency_ms=1200.0, tokens_in=20, tokens_out=50,
            cost_usd=0.0, quality_score=6.0, success=True,
        ),
        BenchmarkResult(
            provider="openai", model="gpt-4o-mini", complexity="simple",
            prompt_index=0, latency_ms=180.0, tokens_in=20, tokens_out=30,
            cost_usd=0.0002, quality_score=8.0, success=True,
        ),
    ]
    summaries = compute_summary(results)
    # OpenAI (180ms) sollte vor Ollama (1200ms) erscheinen
    assert summaries[0].provider == "openai"
    assert summaries[1].provider == "ollama"


# ── Tests: run_benchmark (mit gemocktem Gateway) ──────────────────────────────

@pytest.mark.asyncio
async def test_run_benchmark_returns_results():
    """run_benchmark liefert Ergebnisse mit korrekter Struktur."""
    with respx.mock:
        respx.post("http://test-gateway/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )

        results = await run_benchmark(
            gateway_url="http://test-gateway",
            api_key="test-key",
            providers=["anthropic"],
            n_prompts=3,
        )

    assert len(results) > 0
    for r in results:
        assert isinstance(r, BenchmarkResult)
        assert r.provider == "anthropic"
        assert r.latency_ms >= 0
        assert r.quality_score >= 0.0
        assert r.quality_score <= 10.0


@pytest.mark.asyncio
async def test_run_benchmark_handles_provider_error():
    """run_benchmark behandelt Provider-Fehler ohne Absturz."""
    with respx.mock:
        respx.post("http://failing-gateway/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Internal Server Error"})
        )

        results = await run_benchmark(
            gateway_url="http://failing-gateway",
            api_key="test-key",
            providers=["anthropic"],
            n_prompts=3,
        )

    assert len(results) > 0
    # Alle sollten Fehler haben (500-Response)
    assert all(not r.success for r in results)


@pytest.mark.asyncio
async def test_run_benchmark_multiple_providers():
    """run_benchmark läuft für mehrere Provider."""
    with respx.mock:
        respx.post("http://test-gateway/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )

        results = await run_benchmark(
            gateway_url="http://test-gateway",
            api_key="test-key",
            providers=["anthropic", "openai"],
            n_prompts=3,
        )

    providers_in_results = {r.provider for r in results}
    assert "anthropic" in providers_in_results
    assert "openai" in providers_in_results


# ── Tests: run_failover_test ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_failover_test_returns_structure():
    """run_failover_test gibt korrektes Dict zurück (Struktur-Test — HTTP gemockt)."""
    with respx.mock:
        respx.post("http://test-gateway/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSE)
        )

        result = await run_failover_test(
            gateway_url="http://test-gateway",
            api_key="test-key",
            n_requests=3,
        )

    # Struktur immer vorhanden — unabhängig vom HTTP-Erfolg
    assert "requests" in result
    assert "providers_used" in result
    assert "all_succeeded" in result
    assert "gateway_config" in result
    assert len(result["requests"]) == 3
    # Gateway-Konfiguration ist immer dokumentiert
    assert result["gateway_config"]["circuit_breaker_threshold"] == 3


# ── Tests: write_results (Dateiausgabe) ───────────────────────────────────────

def test_write_results_creates_files():
    """write_results erzeugt results.json, summary.md, failover_test.md."""
    results = [
        BenchmarkResult(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            complexity="simple",
            prompt_index=0,
            latency_ms=320.5,
            tokens_in=20,
            tokens_out=10,
            cost_usd=0.000008,
            quality_score=8.0,
            success=True,
            response_preview="Paris is the capital.",
        )
    ]
    failover = {
        "requests": [{"request": 1, "latency_ms": 200, "provider": "anthropic", "model": "claude-haiku", "success": True}],
        "providers_used": {"anthropic": 1},
        "all_succeeded": True,
        "gateway_config": {"circuit_breaker_threshold": 3, "circuit_breaker_window_sec": 60, "fallback_chain": "..."},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "benchmark_results"
        write_results(
            results=results,
            failover=failover,
            output_dir=output_dir,
            gateway_url="http://test",
            providers=["anthropic"],
            n_prompts=3,
        )

        # Alle 3 Dateien müssen existieren
        assert (output_dir / "results.json").exists()
        assert (output_dir / "summary.md").exists()
        assert (output_dir / "failover_test.md").exists()


def test_results_json_is_valid():
    """results.json ist gültiges JSON mit korrekter Struktur."""
    results = [
        BenchmarkResult(
            provider="openai",
            model="gpt-4o-mini",
            complexity="medium",
            prompt_index=0,
            latency_ms=180.0,
            tokens_in=50,
            tokens_out=100,
            cost_usd=0.00015,
            quality_score=7.5,
            success=True,
            response_preview="Microservices benefit from...",
        )
    ]
    failover = {
        "requests": [],
        "providers_used": {},
        "all_succeeded": True,
        "gateway_config": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        write_results(
            results=results,
            failover=failover,
            output_dir=output_dir,
            gateway_url="http://test",
            providers=["openai"],
            n_prompts=3,
        )

        data = json.loads((output_dir / "results.json").read_text())

    assert "metadata" in data
    assert "results" in data
    assert len(data["results"]) == 1

    r = data["results"][0]
    assert r["provider"] == "openai"
    assert r["latency_ms"] == 180.0
    assert r["success"] is True
    assert "quality_score" in r
    assert "tokens_in" in r
    assert "cost_usd" in r


def test_summary_md_contains_provider_table():
    """summary.md enthält die Provider-Vergleichstabelle."""
    results = [
        BenchmarkResult(
            provider="anthropic", model="claude-haiku-4-5",
            complexity="simple", prompt_index=0,
            latency_ms=320.0, tokens_in=20, tokens_out=10,
            cost_usd=0.000005, quality_score=8.0, success=True,
        )
    ]
    failover = {"requests": [], "providers_used": {}, "all_succeeded": True, "gateway_config": {}}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        write_results(
            results=results, failover=failover, output_dir=output_dir,
            gateway_url="http://test", providers=["anthropic"], n_prompts=3,
        )
        content = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert "anthropic" in content
    assert "p50" in content or "320" in content
    assert "Warum nicht einfach direkt OpenAI" in content
