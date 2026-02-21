# ADR-004: Render Free Tier statt Azure Container Apps

**Status:** Accepted (temporär — Neubewertung bei erstem Zahlungskunden)
**Datum:** 2026-02-14
**Entscheider:** Gateway-Architektur-Review

---

## Kontext

Das Gateway benötigt eine Cloud-Deployment-Plattform mit:
- HTTPS-Endpunkt für Live-Demo und API-Zugriff
- PostgreSQL-Datenbankpersistenz
- Redis für Rate-Limiting und Caching
- CI/CD-Integration (GitHub Actions)
- Ausreichende Rechenleistung für LLM-Routing + PII-Erkennung

Der Deployment-Kontext ist explizit: **Portfolio-Projekt während der Jobsuche**,
nicht produktiver Unternehmenseinsatz.

---

## Entscheidung

**Render Free Tier** für das initiale Deployment.

Stack:
- Render Web Service (Free): FastAPI Gateway
- Render PostgreSQL (Free): Audit-Log, API-Keys
- Render Redis (Free): Rate-Limiting, Semantischer Cache

---

## Begründung

### Kostenvergleich (monatlich)

| Komponente | Render Free | Azure Container Apps | AWS ECS Fargate |
|------------|-------------|---------------------|-----------------|
| Container-Runtime | **€0** | €30-60 (0.5 vCPU/1GB) | €20-40 |
| PostgreSQL | **€0** | €15-40 (Flexible Basic) | €15-30 (RDS) |
| Redis | **€0** | €13 (Cache Basic C0) | €12 (ElastiCache t3) |
| Networking | **€0** | €5-10 | €3-8 |
| **Gesamt** | **€0/Monat** | **€63-123/Monat** | **€50-88/Monat** |

Bei einer 6-monatigen Jobsuche spart Render Free Tier gegenüber Azure:
**€378-738** (6 × €63-123).

### Render Free Tier Einschränkungen (bekannt, akzeptiert)

| Einschränkung | Impact | Mitigation |
|---------------|--------|------------|
| Cold Start nach 15min Inaktivität (~30s) | Demo-Verzögerung | `GET /ready` als Pre-Warm |
| 512MB RAM | Kein spaCy `de_core_news_lg` (nur `sm`) | Regex-first PII-Detection |
| PostgreSQL: 90-Tage-Limit | Datenverlust nach 90 Tagen | Backup vor Ablauf |
| 0.1 vCPU baseline | Langsamere PII-Erkennung unter Last | Akzeptabel bei Demo-Traffic |

### Azure: Wann sinnvoll?

Azure Container Apps wird zur bevorzugten Plattform wenn:

1. **Zahlungskunder vorhanden**: SLA-Anforderungen (99.9%+) erfordern bezahlte Tier
2. **DSGVO Tier 1 erforderlich**: EU-Rechenzentrum mit Datenverarbeitungsvertrag
3. **Traffic > 1000 req/Tag**: Render Free Tier wird zum Engpass
4. **Team > 1 Person**: Azure AD Integration + RBAC für Enterprise-Deployment

### DSGVO-Implikationen

Render hostet in Frankfurt (eu-central-1) — EU-Datenresidenz ist gegeben.
Azure Deutschland/Westeuropa bietet ebenfalls EU-Residenz mit zertifiziertem
Datenverarbeitungsvertrag (DPA) nach Art. 28 DSGVO.

Für Enterprise-Kunden mit Tier-1-DSGVO-Anforderungen ist Azure die bessere
Wahl (ISO 27001, SOC 2, Azure Germany). Für das Portfolio-Deployment ist
Render ausreichend.

---

## Konsequenzen

### Positive Konsequenzen
- **€0 Betriebskosten** während der Jobsuche
- **Live-Demo verfügbar**: https://enterprise-ai-gateway.onrender.com
- **GitHub Actions CI/CD** integriert (push to main → auto-deploy)
- **Terraform-ready**: `infra/terraform/` enthält Azure Container Apps IaC
  für sofortigen Wechsel bei Bedarf

### Negative Konsequenzen
- **Cold Start**: 30s Wartezeit nach 15min Inaktivität — unprofessionell in
  Echtzeit-Demos. Mitigation: Pre-Warm-Skript vor Interviews
- **Keine Production-SLA**: 99.5% Uptime-Ziel, kein offizieller SLA
- **Datenverlust nach 90 Tagen** (PostgreSQL Free Tier Limit)
- **Skaliert nicht** über ~100 req/s (0.1 vCPU baseline)

### Migrationspfad zu Azure

```bash
# Sofortiger Switch möglich (IaC vorbereitet):
cd infra/terraform
terraform workspace select prod
terraform plan -var-file=prod.tfvars
# Review: €63-123/Monat
terraform apply -var-file=prod.tfvars

# .github/workflows/ci.yml: AZURE_DEPLOY=true setzen
# → Nächster Push deployt auf Azure Container Apps
```

---

## Abgelehnte Alternativen

### Alternative A: Azure Container Apps (sofort)
**Abgelehnt wegen:** €63-123/Monat ohne Produktionsanforderungen.
Bei Jobsuche ist ROI negativ. IaC vorbereitet für späteren Switch.

### Alternative B: AWS ECS Fargate
**Abgelehnt wegen:** €50-88/Monat, kein wesentlicher Vorteil gegenüber Azure
für DACH-Markt. Azure wird von deutschen Unternehmenskunden bevorzugt
(Microsoft-Präsenz in DE).

### Alternative C: Google Cloud Run
**Abgelehnt wegen:** Google ist kein primärer Enterprise-Anbieter für DACH.
Gemini-Integration im Gateway wäre conflict of interest bei GCP-Deployment.

### Alternative D: Self-hosted VPS (Hetzner)
**Abgelehnt wegen:** Kein managed PostgreSQL/Redis, kein HTTPS ohne manuelle
Konfiguration, kein CI/CD ohne weiteres Tooling. Operativer Aufwand zu hoch.

---

## Entscheidungsreview-Trigger

Diese Entscheidung wird neu bewertet wenn:
- [ ] Erster Zahlungskunde (→ Azure Container Apps)
- [ ] SLA-Anforderung >99.9% (→ Azure Container Apps)
- [ ] Datenschutz-Audit durch Unternehmenskunden (→ Azure Germany + DPA)
- [ ] Traffic >1000 req/Tag dauerhaft (→ Render Starter: €7/Monat)

---

## Referenzen
- `infra/terraform/` — Azure Container Apps IaC (sofort verwendbar)
- `.github/workflows/ci.yml` — CI/CD Pipeline
- `GET /ready` — Pre-Warm Endpoint
- Render Pricing: https://render.com/pricing
- Azure Container Apps Pricing: https://azure.microsoft.com/pricing/container-apps
