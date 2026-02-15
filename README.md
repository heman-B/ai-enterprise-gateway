# Enterprise AI Gateway

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
- **Storage:** PostgreSQL + Redis
- **PII Detection:** Regex patterns + spaCy de_core_news_sm
- **Deployment:** Docker + docker-compose, targeting Azure Container Apps

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

## Roadmap

Next up:
- Prometheus + Grafana monitoring dashboard
- Terraform for Azure Container Apps deployment
- GitHub Actions CI/CD pipeline
- MCP server integration

## License

MIT License - see LICENSE file for details.
