# Enterprise AI Gateway

[![Tests](https://img.shields.io/badge/tests-154%2F154_passing-brightgreen)](./tests)
[![PII Accuracy](https://img.shields.io/badge/PII_accuracy->94%25-brightgreen)](./docs/pii_benchmark.md)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](./Dockerfile)

> **Status:** Production-ready. Live at [enterprise-ai-gateway.onrender.com](https://enterprise-ai-gateway.onrender.com)

A production-ready API gateway for routing requests across multiple LLM providers (Anthropic, OpenAI, Gemini, Ollama) with cost optimization, automatic failover, and DSGVO-compliant PII detection for German enterprise clients.

## Overview

This gateway sits between applications and LLM providers to handle:
- Multi-provider routing with automatic failover
- Cost optimization (routes simple queries to cheaper models)
- EU data residency enforcement
- German PII detection and redaction before external API calls
- Per-tenant API key management and rate limiting
- Audit logging for compliance

Built for enterprise use cases where you need reliability, cost control, and regulatory compliance.

## Features

**Smart Routing**
- Supports 4 providers: Anthropic Claude, OpenAI GPT, Google Gemini, Ollama (local)
- Cost-based model selection (40-60% savings vs always using premium models)
- Circuit breaker pattern for automatic failover
- EU residency filtering (data stays in EU when required)
- Explicit model override when needed

**DSGVO Compliance**
- Detects German PII: IBAN, Steuer-ID, KFZ-Kennzeichen, phone numbers, addresses
- Validates checksums (IBAN mod-97, Steuer-ID ISO 7064) to reduce false positives
- Automatic redaction before sending to external LLMs
- Detection latency under 20ms at p99
- Published accuracy benchmark: [>94% precision and recall](./docs/pii_benchmark.md) on synthetic DACH dataset

**Enterprise Features**
- SHA256-hashed API keys per tenant with optional monthly token budgets
- Hard token budget enforcement: HTTP 429 with `monthly_token_budget_exceeded` when exceeded, no surprise bills
- Redis-backed rate limiting (sliding window) + semantic response cache
- Hash-chained audit log: tamper-evident via SHA256 chain, one-click DSGVO export
- `GET /admin/costs/summary` — per-tenant token usage, budget %, and cost in €
- PostgreSQL for persistent storage (SQLite in dev mode)

**Semantic Response Cache**
- Two-level cache: SHA256 exact match (<1ms) + cosine similarity on embeddings (optional)
- Cache hit rate ~40% at steady state (FAQ/support workloads) → ~€50-180/month saved
- `GET /admin/cache/stats` — live hit rate, tokens saved, cost saved
- `gateway_cache_hit_total` Prometheus metric per provider/model

## Tech Stack

- **API Framework:** FastAPI (async/await throughout)
- **Providers:** Direct HTTP via httpx (no SDK dependencies)
- **Storage:** PostgreSQL (production) / SQLite (dev) + Redis
- **PII Detection:** Regex patterns + spaCy de_core_news_sm
- **Deployment:** Docker + Render (free tier), Azure Container Apps-ready

## Quick Start

```bash
git clone https://github.com/heman-B/ai-enterprise-gateway.git
cd ai-enterprise-gateway

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

# Start services
docker compose up -d

# Run tests
pytest tests/ -v
# All 154 tests should pass

# Test the gateway
curl http://localhost:8000/health
```

Gateway runs on `http://localhost:8000`

## Project Structure

```
enterprise-ai-gateway/
├── gateway/
│   ├── main.py              # FastAPI app
│   ├── router.py            # Routing logic
│   ├── providers/           # LLM provider adapters
│   ├── policies/            # Cost, latency, residency policies
│   └── middleware/          # Auth, rate limiting, PII detection
├── tests/                   # 79 test cases
│   ├── test_router.py
│   ├── test_pii_detection.py
│   └── fixtures/de/         # German synthetic test data
├── Dockerfile
└── docker-compose.yml
```

## Benchmark

Run against the live gateway to get real latency + cost numbers:

```bash
python scripts/benchmark.py --providers anthropic openai --prompts 20
```

Results written to `benchmark_results/` — `results.json` (raw data), `summary.md` (table), `failover_test.md`.

| Provider | Model | p50 (ms) | p95 (ms) | Cost/1K tokens | Quality |
|----------|-------|----------|----------|----------------|---------|
| Anthropic | claude-haiku-4-5 | ~320 | ~580 | $0.00025 | 8.2/10 |
| OpenAI | gpt-4o-mini | ~180 | ~350 | $0.00015 | 7.8/10 |
| Gemini | gemini-2.5-flash | ~410 | ~820 | $0.00030 | 8.0/10 |
| Ollama | llama3.2 | ~1200 | ~2400 | $0 (local) | 6.5/10 |

> Numbers are pre-run estimates. Run `benchmark.py` for real measurements against your deployment.

## Testing

154 tests covering:
- Multi-provider routing logic + semantic cache
- Circuit breaker behavior
- EU residency enforcement
- German PII detection with real checksum validation (>94% accuracy benchmark)
- Hash-chain audit log integrity
- Benchmark script structure and quality scoring
- Token budget enforcement (429 on exceeded, Redis tracking, monthly reset)

Run with `pytest tests/ -v`

## Performance

- PII detection: <20ms at p99
- Routing overhead: <5ms
- Cache hit: <2ms (SHA256 exact), ~20ms (embedding similarity)
- Cost savings: ~50% vs always using premium models + ~40% from cache hits

## Deployment to Render (Free Tier)

This gateway can be deployed to Render's free tier ($0/month):
- 1GB PostgreSQL database (free forever)
- 25MB Redis (Valkey, Redis-compatible, free forever)
- Web service with HTTPS (free tier)

### Prerequisites

1. **GitHub Repository:** Push your code to GitHub
2. **Render Account:** Sign up at [render.com](https://render.com)
3. **API Keys:** Have your LLM provider API keys ready

### Deploy Steps

1. **Connect to Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml` and create:
     - PostgreSQL database (`gateway-db`)
     - Redis instance (`gateway-redis`)
     - Web service (`enterprise-ai-gateway`)

2. **Set Secrets:**
   In the Render dashboard, add environment variables:
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GEMINI_API_KEY=AI...
   ```
   (ADMIN_API_KEY and DATABASE_URL are auto-generated)

3. **Wait for Deployment:**
   - First deploy takes 5-10 minutes
   - Render builds the Docker image
   - Initializes PostgreSQL schema
   - Starts the web service

4. **Test Deployment:**
   ```bash
   # Health check
   curl https://enterprise-ai-gateway.onrender.com/health

   # Readiness check (DB + Redis)
   curl https://enterprise-ai-gateway.onrender.com/ready
   ```

5. **Create API Key:**
   ```bash
   curl -X POST https://enterprise-ai-gateway.onrender.com/admin/keys \
     -H "Content-Type: application/json" \
     -d '{"tenant_id": "my-app", "rate_limit_per_minute": 100}'
   ```

### Auto-Deploy on Git Push

After initial setup, every `git push origin main` triggers a new deployment automatically.

### Cost Breakdown

**Render Free Tier:**
- Web service: $0/month (750 hours free)
- PostgreSQL (1GB): $0/month (forever free)
- Redis (25MB): $0/month (forever free)

**Total: $0/month** ✅

### Limits on Free Tier

- Web service spins down after 15 minutes of inactivity (cold start: ~30s)
- PostgreSQL: 1GB storage, 97 connections
- Redis: 25MB memory (enough for ~10K rate limit entries)

For production workloads, upgrade to Render's paid tiers ($7-25/month).

## Architecture Decision Records

Key architectural decisions documented with rationale and cost analysis:

| ADR | Decision | Why |
|-----|----------|-----|
| [ADR-001](docs/adr/ADR-001-local-pii-detection.md) | Local PII detection | Circular trust: can't send to external LLM to decide if safe to send externally |
| [ADR-002](docs/adr/ADR-002-hash-chained-audit-log.md) | Hash-chained audit log | Data egress incompatibility: audit log cannot leave the system it audits |
| [ADR-003](docs/adr/ADR-003-redis-semantic-cache.md) | Redis semantic cache | €50-180/month saved at 10K req/day with 40% hit rate |
| [ADR-004](docs/adr/ADR-004-render-vs-azure.md) | Render Free Tier | €0/month vs €63-123/month Azure; IaC ready for instant switch |
| [ADR-005](docs/adr/ADR-005-flat-rate-vs-per-token-pricing.md) | Flat-rate pricing model | CFOs need fixed invoices; per-token billing blocks enterprise procurement |

## License

MIT License - see LICENSE file for details.
