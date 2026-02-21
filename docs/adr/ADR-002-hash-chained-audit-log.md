# ADR-002: Hash-verkettetes Audit-Log statt Managed SIEM

**Status:** Accepted
**Datum:** 2026-02-14
**Entscheider:** Gateway-Architektur-Review

---

## Kontext

Jede LLM-Anfrage durch den Gateway muss für DSGVO-Audits nachvollziehbar
protokolliert werden. Enterprise-Kunden und Datenschutzbeauftragte erwarten:

1. **Vollständigkeit**: Kein Datensatz wurde nachträglich gelöscht
2. **Integrität**: Kein Datensatz wurde nachträglich geändert
3. **Exportierbarkeit**: Compliance-Bundle auf Anfrage lieferbar
4. **Datenschutz**: Kein Prompt-Klartext im Log (nur SHA256-Hash)

Die Frage war: Wie implementiert man eine manipulationssichere Aufzeichnung?

---

## Entscheidung

**SHA256-Hash-Kette in PostgreSQL** — in-process, kein externer Log-Dienst.

Jeder Datensatz enthält:
```
record_hash = SHA256(id | tenant_id | timestamp | prompt_hash |
                     model | provider | tokens_in | tokens_out |
                     cost_usd | pii_detected | prev_hash)
```

Jede Änderung an einem Feld ändert den record_hash und bricht damit
alle nachfolgenden prev_hash-Verknüpfungen. Manipulation ist mathematisch
nachweisbar ohne externe Verifikationsinstanz.

---

## Begründung

### Das Daten-Egress-Problem

Ein Audit-Log kann nicht an denselben externen Dienst gesendet werden, den es
überwacht. Wenn der Gateway eine Anfrage an Anthropic sendet und das Splunk-
Audit-Log auch an Anthropic geht (weil Splunk auf AWS us-east-1 läuft), dann:

1. Verlässt das Audit-Log die EU-Kontrolle
2. Enthält es möglicherweise PII-Metadaten
3. Kann es von dem überwachten System selbst kompromittiert werden

Das Audit-Log muss **im selben System** bleiben wie die Daten, die es
protokolliert.

### Kostenvergleich

| Lösung | Kosten/Monat | Daten-Egress-Problem | Implementierungsaufwand |
|--------|-------------|---------------------|-------------------------|
| In-process Hash-Kette (diese ADR) | **€0** | ✅ Keiner | 2 Tage |
| Splunk Enterprise | €100-500+ | ⚠️ US-Server | 2-4 Wochen |
| Datadog Logs | ~€0.10/GB ingested | ⚠️ US-Server | 1 Woche |
| AWS CloudTrail | ~€2/100K Events | ⚠️ Verlässt EU | 3 Tage |
| Elastic (self-hosted) | ~€30/Monat (EC2) | ✅ EU möglich | 1 Woche |

Bei dem aktuellen Deployment-Kontext (Render Free Tier) spart die In-Process-
Lösung **€100-500/Monat** gegenüber managed SIEMs.

### Mathematische Tamper-Evidence

Die Hash-Kette bietet stärkere Garantien als Datenbankzugriffskontrolle:

```
Datensatz 1: record_hash = SHA256(fields_1 + "GENESIS")
Datensatz 2: record_hash = SHA256(fields_2 + record_hash_1)
Datensatz 3: record_hash = SHA256(fields_3 + record_hash_2)
...
```

Um Datensatz 2 zu fälschen, muss ein Angreifer:
1. Den neuen record_hash für Datensatz 2 berechnen (erfordert SHA256-Preimage)
2. Alle nachfolgenden Datensätze neu berechnen
3. Den asyncio.Lock umgehen (verhindert Race-Conditions)

SHA256 gilt als kollisionsresistent (keine bekannten Preimage-Angriffe, 2026).

---

## Konsequenzen

### Positive Konsequenzen
- **Mathematisch verifizierbar**: `GET /admin/audit/verify` zeigt broken_at
  exakt an — kein "möglicherweise manipuliert"
- **€0 Zusatzkosten** — PostgreSQL bereits für API-Keys verwendet
- **Kein Daten-Egress** — Audit-Log verlässt nie das Gateway
- **One-Click-Export**: `POST /admin/compliance-export` → ZIP für
  Datenschutzbeauftragte

### Negative Konsequenzen
- **Kein Real-Time-Alerting**: Splunk/Datadog bieten ML-basierte Anomalie-
  erkennung; die Hash-Kette ist passiv
- **Keine Suche über mehrere Mandanten** hinweg (fehlt: Cross-Tenant-Analytics)
- **asyncio.Lock als Single Point of Contention**: Bei sehr hohem Durchsatz
  (>1000 req/s) wird der Lock zum Flaschenhals
- **Nur SQLite in Dev**: Konsistenz zwischen SQLite-Dev und Postgres-Prod muss
  durch Tests gewährleistet werden

### Skalierungsgrenzen
Der asyncio.Lock serialisiert alle INSERTs. Bei 1000 req/s mit ~1ms INSERT-
Zeit würde die Warteschlange wachsen. Mitigation: Batch-INSERTs oder Sharding
nach tenant_id. Aktuelles Deployment-Limit: ~100 req/s (Render Free Tier).

---

## Abgelehnte Alternativen

### Alternative A: Splunk Enterprise
**Abgelehnt wegen:** Daten-Egress (EU-Compliance), €100-500/Monat,
Implementierungsaufwand 2-4 Wochen für MVP.

### Alternative B: Datadog Logs
**Abgelehnt wegen:** Daten-Egress, keine kostenlose EU-Region,
PII im Log-Content möglicherweise nicht DSGVO-konform.

### Alternative C: Append-only Tabelle ohne Hash-Kette
**Abgelehnt wegen:** Datenbankadministrator kann Rows UPDATE/DELETE ohne
nachweisbare Spur. Hash-Kette macht Manipulation kryptographisch nachweisbar.

### Alternative D: Write-once Object Storage (S3/MinIO)
**Abgelehnt wegen:** Zusätzliche Infrastruktur, kein atomares Lesen+Schreiben
ohne externe Koordination, höhere Komplexität für Compliance-Export.

---

## Referenzen
- `gateway/middleware/audit_logger.py` — Implementierung
- `gateway/middleware/audit_logger.py:_compute_record_hash()` — Hash-Berechnung
- `GET /admin/audit/verify` — Chain-Verifikation
- `POST /admin/compliance-export` — DSGVO-Export
- NIST FIPS 180-4 — SHA-2 Standard
