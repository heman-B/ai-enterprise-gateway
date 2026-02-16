# Enterprise AI Gateway

[![Tests](https://img.shields.io/badge/tests-79%2F79_passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](./Dockerfile)

> **Status:** Active development. Core gateway working, monitoring dashboard next.

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
- Audit trail for compliance reporting

**Enterprise Features**
- SHA256-hashed API keys per tenant
- Redis-backed rate limiting (sliding window)
- Audit logs track which models answered which requests
- PostgreSQL for persistent storage (SQLite in dev mode)

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
# All 79 tests should pass

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

## Testing

79 tests covering:
- Multi-provider routing logic
- Circuit breaker behavior
- EU residency enforcement
- German PII detection with real checksum validation
- Integration tests with live provider APIs

Run with `pytest tests/ -v`

## Performance

- PII detection: <20ms at p99
- Routing overhead: <5ms
- Cost savings: ~50% vs always using premium models (Sonnet/GPT-4)

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

## Roadmap

Next up:
- ✅ Production deployment (Render)
- Prometheus + Grafana monitoring dashboard
- GitHub Actions CI/CD pipeline
- MCP server integration

## License

MIT License - see LICENSE file for details.
