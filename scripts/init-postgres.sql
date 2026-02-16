-- PostgreSQL Schema Initialization
-- This runs automatically on Render via DATABASE_URL
-- (Render PostgreSQL auto-creates database, we just need tables)

-- Audit Log Table (append-only)
CREATE TABLE IF NOT EXISTS audit_log (
    id           TEXT PRIMARY KEY,
    request_id   TEXT NOT NULL,
    tenant_id    TEXT NOT NULL,
    timestamp    DOUBLE PRECISION NOT NULL,
    prompt_hash  TEXT NOT NULL,
    model        TEXT NOT NULL,
    provider     TEXT NOT NULL,
    tokens_in    INTEGER NOT NULL,
    tokens_out   INTEGER NOT NULL,
    cost_usd     DOUBLE PRECISION NOT NULL,
    pii_detected INTEGER NOT NULL DEFAULT 0,
    pii_types    TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_pii ON audit_log(pii_detected, timestamp);

-- API Keys Table
CREATE TABLE IF NOT EXISTS api_keys (
    tenant_id              TEXT NOT NULL,
    key_hash               TEXT PRIMARY KEY,
    is_active              INTEGER NOT NULL DEFAULT 1,
    rate_limit_per_minute  INTEGER NOT NULL DEFAULT 100,
    created_at             DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())
);

-- Index for tenant lookup
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);
