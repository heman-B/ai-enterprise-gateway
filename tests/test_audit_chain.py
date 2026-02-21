# tests/test_audit_chain.py
# Hash-Kette + Compliance-Export Tests
# Prüft: Kette gültig nach N Einfügungen, Kette bricht bei Manipulation, Export deterministisch
import asyncio
import hashlib
import io
import json
import os
import zipfile

import pytest
import pytest_asyncio

from gateway.middleware.audit_logger import (
    GENESIS_HASH,
    AuditLogger,
    _compute_record_hash,
)

# Temporäre SQLite-Datenbank für Tests
TEST_DB = ":memory:"  # in-memory: pro Fixture-Instanz isoliert


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def logger(tmp_path):
    """Frische AuditLogger-Instanz mit eigener SQLite-Datei (kein Shared-State)."""
    db_path = str(tmp_path / "test_audit.db")
    al = AuditLogger(db_url=db_path)
    await al.initialize()
    return al


async def _insert_record(al: AuditLogger, *, tenant_id: str = "tenant_a") -> None:
    """Hilfsfunktion: einen Testdatensatz einfügen."""
    await al.log(
        request_id="req-" + tenant_id,
        tenant_id=tenant_id,
        prompt_hash=hashlib.sha256(b"test prompt").hexdigest(),
        model="claude-3-haiku",
        provider="anthropic",
        tokens_in=50,
        tokens_out=100,
        cost_usd=0.0001,
        pii_detected=False,
    )


# ──────────────────────────────────────────────────────────────
# Tests: _compute_record_hash
# ──────────────────────────────────────────────────────────────

def test_compute_record_hash_deterministic():
    """Gleiche Eingaben → gleicher Hash (deterministisch)."""
    kwargs = dict(
        id="test-id-1",
        tenant_id="tenant_a",
        timestamp=1234567890.123456,
        prompt_hash="abc123",
        model="claude-3-haiku",
        provider="anthropic",
        tokens_in=50,
        tokens_out=100,
        cost_usd=0.0001,
        pii_detected=0,
        prev_hash=GENESIS_HASH,
    )
    h1 = _compute_record_hash(**kwargs)
    h2 = _compute_record_hash(**kwargs)
    assert h1 == h2


def test_compute_record_hash_changes_on_field_mutation():
    """Jede Feldänderung erzeugt einen anderen Hash."""
    base = dict(
        id="test-id-1",
        tenant_id="tenant_a",
        timestamp=1234567890.0,
        prompt_hash="abc123",
        model="claude-3-haiku",
        provider="anthropic",
        tokens_in=50,
        tokens_out=100,
        cost_usd=0.0001,
        pii_detected=0,
        prev_hash=GENESIS_HASH,
    )
    original = _compute_record_hash(**base)

    # tokens_out geändert
    mutated = {**base, "tokens_out": 999}
    assert _compute_record_hash(**mutated) != original

    # tenant_id geändert
    mutated = {**base, "tenant_id": "attacker"}
    assert _compute_record_hash(**mutated) != original

    # prev_hash geändert
    mutated = {**base, "prev_hash": "fake_prev"}
    assert _compute_record_hash(**mutated) != original


def test_compute_record_hash_is_64_char_hex():
    """SHA256-Hash hat immer 64 Hex-Zeichen."""
    h = _compute_record_hash(
        id="x", tenant_id="t", timestamp=0.0, prompt_hash="p",
        model="m", provider="pr", tokens_in=0, tokens_out=0,
        cost_usd=0.0, pii_detected=0, prev_hash=GENESIS_HASH,
    )
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ──────────────────────────────────────────────────────────────
# Tests: Kette nach Einfügungen
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_table_chain_valid(logger):
    """Leere Tabelle → Kette ist gültig, 0 Datensätze geprüft."""
    result = await logger.verify_chain()
    assert result["chain_valid"] is True
    assert result["records_checked"] == 0


@pytest.mark.asyncio
async def test_single_insert_chain_valid(logger):
    """Einzelner Datensatz → Kette gültig."""
    await _insert_record(logger)
    result = await logger.verify_chain()
    assert result["chain_valid"] is True
    assert result["records_checked"] == 1


@pytest.mark.asyncio
async def test_multiple_inserts_chain_valid(logger):
    """10 Datensätze → Kette gültig."""
    for i in range(10):
        await al_log(logger, tenant_id=f"tenant_{i % 3}")
    result = await logger.verify_chain()
    assert result["chain_valid"] is True
    assert result["records_checked"] == 10


@pytest.mark.asyncio
async def test_first_record_uses_genesis_prev_hash(logger):
    """Erster Datensatz hat GENESIS als prev_hash."""
    await _insert_record(logger)
    rows = await logger._fetch_all_rows_ordered()
    assert len(rows) == 1
    assert rows[0]["prev_hash"] == GENESIS_HASH


@pytest.mark.asyncio
async def test_second_record_prev_hash_matches_first_record_hash(logger):
    """Zweiter Datensatz: prev_hash == record_hash des ersten."""
    await _insert_record(logger, tenant_id="t1")
    await _insert_record(logger, tenant_id="t2")
    rows = await logger._fetch_all_rows_ordered()
    assert rows[1]["prev_hash"] == rows[0]["record_hash"]


# ──────────────────────────────────────────────────────────────
# Tests: Manipulation erkennen
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tampered_record_breaks_chain(logger):
    """Manipulation an einem Datensatz → chain_valid False + broken_at zurückgegeben."""
    import aiosqlite

    for _ in range(3):
        await _insert_record(logger)

    rows = await logger._fetch_all_rows_ordered()
    tampered_id = rows[1]["id"]

    # Direkte SQL-Manipulation (simuliert Datenbankzugriff eines Angreifers)
    async with aiosqlite.connect(logger._db_url) as db:
        await db.execute(
            "UPDATE audit_log SET tokens_out = 9999 WHERE id = ?",
            (tampered_id,),
        )
        await db.commit()

    result = await logger.verify_chain()
    assert result["chain_valid"] is False
    assert result["broken_at"] == tampered_id
    assert result["records_checked"] == 1  # bricht beim zweiten Datensatz ab


@pytest.mark.asyncio
async def test_tampered_record_hash_breaks_chain(logger):
    """Direkte Manipulation des record_hash → chain_valid False."""
    import aiosqlite

    for _ in range(2):
        await _insert_record(logger)

    rows = await logger._fetch_all_rows_ordered()
    first_id = rows[0]["id"]

    async with aiosqlite.connect(logger._db_url) as db:
        await db.execute(
            "UPDATE audit_log SET record_hash = 'deadbeef' WHERE id = ?",
            (first_id,),
        )
        await db.commit()

    result = await logger.verify_chain()
    assert result["chain_valid"] is False
    assert result["broken_at"] == first_id


# ──────────────────────────────────────────────────────────────
# Tests: Compliance-Export
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compliance_export_is_valid_zip(logger):
    """Export gibt gültiges ZIP-Archiv zurück."""
    await _insert_record(logger)
    zip_bytes = await logger.build_compliance_bundle()
    assert zipfile.is_zipfile(io.BytesIO(zip_bytes))


@pytest.mark.asyncio
async def test_compliance_export_contains_required_files(logger):
    """ZIP enthält die drei Pflichtdateien."""
    await _insert_record(logger)
    zip_bytes = await logger.build_compliance_bundle()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

    assert "audit_log.jsonl" in names
    assert "integrity_manifest.json" in names
    assert "system_config.json" in names


@pytest.mark.asyncio
async def test_compliance_export_jsonl_matches_records(logger):
    """audit_log.jsonl enthält genau so viele Zeilen wie Datensätze."""
    for _ in range(5):
        await _insert_record(logger)

    zip_bytes = await logger.build_compliance_bundle()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        jsonl = zf.read("audit_log.jsonl").decode("utf-8")

    lines = [l for l in jsonl.strip().splitlines() if l]
    assert len(lines) == 5
    # Jede Zeile muss valides JSON sein
    for line in lines:
        record = json.loads(line)
        assert "id" in record
        assert "record_hash" in record
        assert "prev_hash" in record


@pytest.mark.asyncio
async def test_compliance_export_manifest_hashes_match(logger):
    """integrity_manifest.json: record_hash-Einträge stimmen mit audit_log.jsonl überein."""
    for _ in range(3):
        await _insert_record(logger)

    zip_bytes = await logger.build_compliance_bundle()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        jsonl = zf.read("audit_log.jsonl").decode("utf-8")
        manifest = json.loads(zf.read("integrity_manifest.json"))

    jsonl_records = [json.loads(l) for l in jsonl.strip().splitlines()]
    manifest_entries = {e["id"]: e["record_hash"] for e in manifest["records"]}

    for record in jsonl_records:
        assert record["id"] in manifest_entries
        assert manifest_entries[record["id"]] == record["record_hash"]


@pytest.mark.asyncio
async def test_compliance_export_overall_chain_hash_deterministic(logger):
    """overall_chain_hash ist deterministisch für denselben Datensatz."""
    await _insert_record(logger)

    zip1 = await logger.build_compliance_bundle()
    zip2 = await logger.build_compliance_bundle()

    with zipfile.ZipFile(io.BytesIO(zip1)) as zf:
        m1 = json.loads(zf.read("integrity_manifest.json"))
    with zipfile.ZipFile(io.BytesIO(zip2)) as zf:
        m2 = json.loads(zf.read("integrity_manifest.json"))

    assert m1["overall_chain_hash"] == m2["overall_chain_hash"]
    assert m1["total_records"] == m2["total_records"]


@pytest.mark.asyncio
async def test_compliance_export_system_config_contains_required_fields(logger):
    """system_config.json enthält Pflichtfelder für DSGVO-Audit."""
    zip_bytes = await logger.build_compliance_bundle(gateway_version="1.0.0")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        config = json.loads(zf.read("system_config.json"))

    assert config["gateway_version"] == "1.0.0"
    assert "compliance_standards" in config
    assert "DSGVO" in config["compliance_standards"]
    assert config["hash_algorithm"] == "SHA256"
    assert config["chain_genesis_sentinel"] == GENESIS_HASH
    assert "pii_types_detected" in config
    assert "export_timestamp" in config


@pytest.mark.asyncio
async def test_compliance_export_chain_valid_reflected_in_manifest(logger):
    """chain_valid im Manifest stimmt mit verify_chain() überein."""
    for _ in range(3):
        await _insert_record(logger)

    zip_bytes = await logger.build_compliance_bundle()
    chain_result = await logger.verify_chain()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        manifest = json.loads(zf.read("integrity_manifest.json"))

    assert manifest["chain_valid"] == chain_result["chain_valid"]


# ──────────────────────────────────────────────────────────────
# Hilfsfunktion (muss nach den Fixtures definiert sein)
# ──────────────────────────────────────────────────────────────

async def al_log(al: AuditLogger, *, tenant_id: str = "tenant_a") -> None:
    """Alias mit flexiblerem Namen für Loop-Tests."""
    await al.log(
        request_id="req-loop",
        tenant_id=tenant_id,
        prompt_hash=hashlib.sha256(b"loop prompt").hexdigest(),
        model="claude-3-haiku",
        provider="anthropic",
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.00001,
        pii_detected=False,
    )
