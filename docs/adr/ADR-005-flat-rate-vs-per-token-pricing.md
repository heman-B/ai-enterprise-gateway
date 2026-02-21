# ADR-005: Flat-Rate Pricing Model vs. Per-Token Billing

**Status:** Accepted
**Date:** 2026-02-22
**Deciders:** Engineering, Product, Finance
**Context:** Enterprise AI Gateway — pricing model for commercial deployment

---

## Context

The gateway routes requests across multiple LLM providers (Anthropic, OpenAI, Gemini, Ollama) and tracks costs per request. A pricing model must be chosen for how enterprise clients are billed for gateway access.

### The CFO Problem

Enterprise procurement runs on **annual budget cycles**. Finance teams cannot approve infrastructure spend with unbounded variability. When evaluating AI gateway vendors, the #1 CFO objection is:

> *"We can't put an unpredictable line item in our IT budget. If usage spikes in Q3, we get a surprise invoice in Q4."*

Per-token billing — the native model of all LLM APIs — fails this requirement directly. AWS, Azure, and Google all charge per-token for their managed LLM services. This creates:

- **Invoice unpredictability**: A 3× traffic spike triples the bill with no warning
- **Budget cycle mismatch**: Quarterly reviews cannot accommodate monthly token spikes
- **Procurement friction**: Legal/finance require fixed-cost contracts → per-token billing fails vendor qualification
- **Chargeback complexity**: IT must allocate costs to departments → impossible without fixed rates

### Technical Context

The gateway already:
- Tracks `cost_usd` per request in audit_log (per-token costs known)
- Has circuit-breaker + cost-optimized routing (40–60% token savings)
- Has Redis semantic cache (further 30–40% cost reduction at steady state)
- Enforces `token_budget_monthly` per tenant as a hard cap

This infrastructure enables flat-rate pricing: we absorb token cost variance internally through routing + caching.

---

## Decision

**Adopt a flat monthly rate with hard token budget caps.**

Pricing tiers (example — adjust for market):

| Tier | Price/month | Token Budget | Overage |
|------|-------------|--------------|---------|
| Starter | **€149/month** | 2M tokens | Queued (HTTP 429, auto-retry next month) |
| Professional | **€499/month** | 10M tokens | Queued |
| Enterprise | **€1,499/month** | 40M tokens | Queued |
| Custom | Negotiated | Unlimited | N/A |

**Hard cap enforcement** (not soft throttling):
- When `token_budget_monthly` is reached → HTTP 429 with `monthly_token_budget_exceeded`
- Requests queue on client side or fail gracefully — no surprise charges
- Budget resets on the 1st of each calendar month (Redis key: `budget:{tenant}:{YYYY-MM}`)

---

## Alternatives Considered

### Option A: Pure Per-Token Billing (Rejected)

Pass provider costs through with a margin (e.g. 20% markup on API costs).

| Factor | Assessment |
|--------|-----------|
| Revenue model | Simple — cost + margin |
| CFO acceptance | **❌ Fails** — unpredictable invoices block enterprise procurement |
| Engineering complexity | Low — no budget tracking needed |
| Cost risk | Low — margin covers costs by definition |
| Sales cycle | Long — legal/finance review of variable contracts takes 3–6 months |

**Rejected**: Enterprise procurement requires fixed-cost contracts. This model disqualifies us from ~60% of enterprise deals before the technical demo.

---

### Option B: Credit-Based System (Rejected)

Clients purchase token credits upfront. Credits deducted per request.

| Factor | Assessment |
|--------|-----------|
| Revenue model | Prepaid — cash flow positive |
| CFO acceptance | Partial — still requires per-token tracking visible to client |
| Engineering complexity | Medium — credit ledger, top-up flows, expiration logic |
| Cost risk | Medium — large credit purchases create long-term obligations |
| Sales cycle | Medium — requires payment processing for top-ups |

**Rejected**: Adds complexity without solving the core CFO objection. Finance still sees token consumption as variable cost. Flat subscription is simpler to budget.

---

### Option C: Flat Rate + Soft Throttling (Rejected)

Flat monthly rate, but allow overage at a penalty rate (e.g. 150% of normal cost).

| Factor | Assessment |
|--------|-----------|
| CFO acceptance | ❌ Partially fails — "overage charges" reintroduce invoice unpredictability |
| Revenue upside | Higher potential revenue from heavy users |
| Churn risk | High — surprise overage charges = top support ticket |

**Rejected**: Overage charges are the exact source of unpredictability CFOs object to. Hard caps with graceful queuing are preferable for enterprise trust.

---

## Decision Rationale

### Cost Structure at Each Tier

Worst-case API costs (all requests to expensive models, no cache hits):

| Tier | Token Budget | Claude Haiku cost* | GPT-4o-mini cost* | Margin (Haiku) |
|------|-------------|--------------------|-------------------|----------------|
| Starter | 2M | ~€0.37/month | ~€0.28/month | **€148.63/month** |
| Professional | 10M | ~€1.86/month | ~€1.40/month | **€497.14/month** |
| Enterprise | 40M | ~€7.44/month | ~€5.60/month | **€1,492.56/month** |

*At $0.00025/1K tokens (Haiku), $0.92 EUR/USD rate, 50% input/output split*

**With semantic cache (40% hit rate) and cost routing**: actual API costs drop 50–70% below worst-case. Margins expand significantly at steady state.

At Professional tier (€499/month, 10M tokens):
- Worst case API cost: €1.86/month → **97% gross margin**
- Typical API cost (with cache + routing): €0.80/month → **99.8% gross margin**

### CFO Sales Arguments

| CFO Objection | Response |
|--------------|---------|
| "We need predictable invoices" | Fixed monthly rate, guaranteed. No token overage charges. |
| "What happens when we hit the limit?" | Requests return HTTP 429. Your team sees the error, we notify you. No charges added. |
| "Can we increase the limit mid-month?" | Yes — contact sales for an immediate tier upgrade (prorated). |
| "What if we don't use our full quota?" | Unused tokens don't roll over. Buy the tier that fits your forecast. |
| "How do we chargeback to departments?" | `GET /admin/costs/summary` — per-department token usage breakdown, always current. |

### Forcing Function for Engineering Quality

Flat-rate pricing **forces internal cost efficiency** that benefits all clients:

1. **Semantic cache**: 40% fewer API calls at steady state (€50-180/month saved per 10K req/day client)
2. **Cost-optimized routing**: Routes simple queries to Haiku/GPT-4o-mini (40–60% cheaper than default)
3. **Token budget enforcement**: Hard caps prevent runaway costs from misconfigured clients

Without flat-rate pricing, these optimizations benefit only us. With flat-rate pricing, maximizing internal efficiency directly maximizes margin — creating sustainable incentive alignment.

---

## Consequences

### Positive

- **Enterprise procurement unblocked**: Fixed invoice satisfies finance approval requirements
- **Shorter sales cycles**: Legal review of fixed-price SaaS is weeks, not months
- **Predictable revenue**: Subscription ARR is bankable for growth planning
- **Margin structure**: 95–99% gross margin leaves room for support, SLA, and growth

### Negative

- **Underpriced heavy users**: If a client maximizes their quota every month with premium models and no cache hits, margins compress. Mitigation: token budget caps + routing optimization
- **Quota uncertainty at purchase**: Clients must estimate monthly token usage upfront. Mitigation: transparent `GET /admin/costs/summary` lets them track before renewal

### Neutral

- **Per-token tracking still required internally**: Necessary for cost routing, audit log, and internal margin calculation. Not exposed to clients as a billing mechanism.

---

## Implementation

The following components already implement or support this model:

| Component | Role |
|-----------|------|
| `api_keys.token_budget_monthly` | Per-tenant hard cap (NULL = unlimited) |
| `budget:{tenant}:{YYYY-MM}` (Redis) | Monthly token accumulator — resets automatically |
| `gateway/router.py` budget check | HTTP 429 when budget exhausted, before API call |
| `GET /admin/costs/summary` | Per-tenant usage, budget %, and cost in € |
| Semantic cache | Reduces internal API costs → improves margins |
| Cost-optimized routing | Routes cheap queries to cheap models → improves margins |

---

## References

- [ADR-001](ADR-001-local-pii-detection.md) — PII detection architecture
- [ADR-002](ADR-002-hash-chained-audit-log.md) — Audit log with cost tracking
- [ADR-003](ADR-003-redis-semantic-cache.md) — Semantic cache (core margin improvement tool)
- [ADR-004](ADR-004-render-vs-azure.md) — Infrastructure cost baseline
