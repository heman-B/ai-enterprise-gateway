# gateway/metrics.py
# Prometheus-Metriken: request_count, latency, cost, PII, circuit_breaker
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ── Metriken-Definitionen ──────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Gesamtanzahl LLM-Anfragen",
    ["provider", "model", "tenant_id", "status"],
)

REQUEST_LATENCY = Histogram(
    "gateway_request_latency_ms",
    "LLM-Antwortzeit in Millisekunden",
    ["provider", "model"],
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)

COST_TOTAL = Counter(
    "gateway_cost_usd_total",
    "Geschätzte Gesamtkosten in USD",
    ["provider", "model", "tenant_id"],
)

TOKENS_TOTAL = Counter(
    "gateway_tokens_total",
    "Verarbeitete Token (input + output)",
    ["direction", "provider", "model"],
)

PII_DETECTED_TOTAL = Counter(
    "gateway_pii_detected_total",
    "Erkannte PII-Entitäten nach Typ",
    ["pii_type"],
)

CIRCUIT_BREAKER_OPEN = Gauge(
    "gateway_circuit_breaker_open",
    "Circuit-Breaker-Status (1=offen/ausgefallen, 0=geschlossen/verfügbar)",
    ["provider"],
)


def get_metrics_response() -> tuple[bytes, str]:
    """Prometheus-Metriken im Textformat zurückgeben."""
    return generate_latest(), CONTENT_TYPE_LATEST
