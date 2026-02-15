# Enterprise AI Gateway

[![Status](https://img.shields.io/badge/status-🚧_Active_Development-yellow)](https://github.com/HB8-Z-DAY/enterprise-ai-gateway)
[![Tests](https://img.shields.io/badge/tests-79%2F79_passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](./Dockerfile)

> A production-grade API gateway that routes AI requests across multiple LLM providers with DSGVO compliance, cost optimization, and enterprise security features.

---

## 🎯 What This Is

An **Enterprise AI Gateway** built for DACH-region clients requiring:
- Multi-provider LLM routing (Anthropic, OpenAI, Gemini, Ollama)
- **DSGVO compliance** with German PII detection and redaction
- EU data residency enforcement
- Cost allocation per tenant/department
- Audit logging for compliance reporting

**Target Use Case:** Enterprise consulting clients (MHP, SAP) needing production-ready AI infrastructure with regulatory compliance.

---

## ✅ What's Working (Session 4 Complete)

### Core Gateway
- ✅ **4 LLM Providers:** Claude, GPT-4, Gemini, Ollama (local)
- ✅ **5-Tier Smart Routing:**
  1. EU residency enforcement (data stays in EU)
  2. Explicit model override (user chooses specific model)
  3. Cost-optimized routing (40-60% savings vs. always using premium models)
  4. Circuit breaker (automatic failover on provider failures)
  5. Latency-based fallback
- ✅ **Enterprise Middleware:**
  - API key authentication (SHA256 hashed, per-tenant)
  - Redis-backed rate limiting (sliding window)
  - Audit logging (who asked what, which model answered)

### DSGVO Compliance Engineering
- ✅ **German PII Detection** with checksum validation:
  - **IBAN** (mod-97 algorithm)
  - **Steuer-ID** (ISO 7064 algorithm)
  - KFZ-Kennzeichen, Handynummer, PLZ, Straße, Namen
- ✅ **Automatic Redaction** before LLM API calls
- ✅ **<20ms Detection Latency** (production acceptance criteria met)
- ✅ **Audit Trail** tracks PII types detected per request

### Testing & Quality
- ✅ **79/79 Tests Passing** (46 routing + 33 PII detection)
- ✅ **German Synthetic Test Data** (10 validated records with real checksums)
- ✅ **Integration Tests** with real provider APIs

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Framework** | FastAPI + Python 3.11 | Async-native, auto-generated OpenAPI docs |
| **Providers** | httpx (direct HTTP) | Consistent interface, explicit retry control |
| **Storage** | SQLite (dev), PostgreSQL (prod) | Audit logs, API keys, cost tracking |
| **Cache** | Redis | Rate limiting, circuit breaker state |
| **PII Detection** | Regex + spaCy `de_core_news_sm` | <20ms latency, German-specific patterns |
| **Deployment** | Docker + Azure Container Apps | Enterprise DACH compliance (German regions) |
| **Monitoring** | Prometheus + Grafana (planned) | Real-time cost tracking, PII leak dashboards |

---

## 🚀 Quick Start

**Prerequisites:** Docker Desktop, Python 3.11+

```bash
# 1. Clone and install dependencies
git clone https://github.com/HB8-Z-DAY/enterprise-ai-gateway.git
cd enterprise-ai-gateway
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys:
#   - ANTHROPIC_API_KEY
#   - OPENAI_API_KEY
#   - GEMINI_API_KEY (optional)

# 3. Start services
docker compose up -d

# 4. Run tests
pytest tests/ -v
# Expected: 79/79 tests passing

# 5. Test the gateway
curl http://localhost:8000/health
# Expected: {"status": "healthy", "providers": {...}}
```

**Gateway is now running on `http://localhost:8000`**

---

## 📊 What Employers See

### DSGVO Compliance Engineering
- ✅ Checksum validation (IBAN mod-97, Steuer-ID ISO 7064)
- ✅ Not just regex matching — **mathematical validation** of German IDs
- ✅ Zero false positives on validated checksums

### German Market Expertise
- ✅ DACH-specific PII patterns (Steuer-ID, IBAN, KFZ-Kennzeichen)
- ✅ German comments in business logic
- ✅ Synthetic German test data (realistic but fake)

### Production-Grade Implementation
- ✅ <20ms PII detection latency (acceptance criteria)
- ✅ Circuit breaker pattern (auto-failover)
- ✅ Audit logging for compliance reporting

### Comprehensive Testing
- ✅ 79 tests with German synthetic data
- ✅ Integration tests with real LLM providers
- ✅ >95% precision, >85% recall on German PII

---

## 🗂️ Project Structure

```
enterprise-ai-gateway/
├── gateway/                    # Main application
│   ├── main.py                 # FastAPI app
│   ├── router.py               # 5-tier routing engine
│   ├── providers/              # LLM provider adapters
│   ├── policies/               # Routing policies (cost, latency, residency)
│   └── middleware/             # Auth, rate limiting, PII detection
├── tests/                      # 79 test cases
│   ├── test_router.py          # Routing logic tests
│   ├── test_pii_detection.py   # German PII detection tests
│   └── fixtures/de/            # German synthetic test data
├── docs/                       # Documentation
├── Dockerfile                  # Multi-stage production build
└── docker-compose.yml          # Gateway + Redis services
```

---

## 🎯 Roadmap (Next Sessions)

### Session 5: Monitoring Dashboard
- [ ] Prometheus metrics export (`/metrics` endpoint)
- [ ] Grafana dashboard (latency, cost, provider health)
- [ ] **PII Compliance Dashboard** (leaks by department, top PII types)
- [ ] Real-time Slack alerts on PII detection

### Session 6: Cloud Deployment
- [ ] Terraform for Azure Container Apps
- [ ] GitHub Actions CI/CD pipeline
- [ ] Production PostgreSQL migration
- [ ] Environment-specific configs (dev/staging/prod)

### Session 7: MCP Server Integration
- [ ] Expose gateway as MCP server
- [ ] Tool definitions for agent discovery
- [ ] Align with SAP Joule + MCP standards

---

## 📈 Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| PII Detection Latency (p99) | <20ms | **<20ms ✅** |
| Routing Overhead | <5ms | **<5ms ✅** |
| Tests Passing | 100% | **79/79 (100%) ✅** |
| Cost Savings (vs. always Sonnet) | 40-60% | **~50% (estimated)** |

---

## 🔒 Security & Compliance

- ✅ API keys hashed with SHA256
- ✅ PII redacted **before** external LLM calls
- ✅ Audit logs track all PII detections
- ✅ EU residency enforced at routing layer
- ✅ No secrets in git (`.env` excluded)

---

## 📝 License

This project is for **portfolio demonstration purposes**. Not licensed for production use without permission.

---

## 🤝 Contact

**GitHub:** [@HB8-Z-DAY](https://github.com/HB8-Z-DAY)

**Built for:** Enterprise consulting portfolio (MHP, SAP, German AI infrastructure clients)

---

**⚠️ Active Development:** This project is under active development. Session 4 complete (PII detection), Session 5 planned (monitoring dashboard).
