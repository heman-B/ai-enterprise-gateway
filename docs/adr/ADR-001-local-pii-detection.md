# ADR-001: Lokale PII-Erkennung statt externer LLM-API

**Status:** Accepted
**Datum:** 2026-02-14
**Entscheider:** Gateway-Architektur-Review

---

## Kontext

Der Gateway verarbeitet Anfragen, die möglicherweise deutsche PII enthalten
(IBAN, Steuer-ID, KFZ-Kennzeichen, Handynummern, Namen, Adressen).
Diese PII muss vor dem Weiterleiten an externe LLM-Provider erkannt und
redaktiert werden, um DSGVO-Compliance zu gewährleisten.

Zwei grundlegende Ansätze wurden evaluiert:

1. **Lokal**: Regex-Muster + spaCy NER auf dem Gateway-Server
2. **Extern**: PII-Erkennung über eine externe LLM-API (z.B. GPT-4 mit
   Structured Output, oder spezialisierte PII-APIs)

---

## Entscheidung

**Lokale PII-Erkennung** via Regex-Validierung + spaCy `de_core_news_sm`.

---

## Begründung

### Das zirkuläre Vertrauensproblem

Der entscheidende Grund für lokale Verarbeitung: Ein externes LLM kann nicht
zur PII-Erkennung verwendet werden, wenn wir noch nicht wissen, ob die Anfrage
PII enthält. Der Ablauf wäre:

```
Anfrage → [enthält PII?] → Externe API → [verarbeitet PII] → Antwort
                                ↑
                      Wir haben PII bereits übertragen
```

Das System würde PII zu einem externen Dienst senden, um zu entscheiden, ob es
PII zu einem externen Dienst senden darf. Dieser logische Zirkel macht externen
LLM-basierte PII-Erkennung für diesen Use Case strukturell ungeeignet.

### Kostenvergleich

| Ansatz | Kosten bei 10.000 req/Tag | Latenz (p99) |
|--------|--------------------------|--------------|
| Lokal (Regex + spaCy) | **€0/Monat** | <20ms |
| OpenAI GPT-4o-mini API | ~€6/Monat ($0.00002/req × 10K × 30) | 300-800ms |
| Spezialisierte PII-API (z.B. AWS Comprehend) | ~€15/Monat | 100-400ms |

Bei 10.000 req/Tag spart lokale Verarbeitung **€72-180/Jahr** bei gleichzeitig
deutlich niedrigerer Latenz.

### DSGVO-Konformität

Die DSGVO verbietet die Verarbeitung personenbezogener Daten ohne
Rechtsgrundlage. Das Senden potentiell PII-haltiger Prompts an eine externe
Erkennungs-API wäre selbst eine Datenverarbeitung, die einer Rechtgrundlage
bedürfte — ein regulatorisches Chicken-and-Egg-Problem.

---

## Konsequenzen

### Positive Konsequenzen
- **€0 Kosten** für PII-Erkennung (kein externer API-Aufruf)
- **<20ms Latenz** bei p99 (gemessen mit deutschen Testdaten)
- **Kein Daten-Egress** — PII verlässt nie den Gateway-Server
- **Deterministisch** — gleicher Input, gleicher Output (keine LLM-Variabilität)
- **Offline-fähig** — funktioniert ohne externe Abhängigkeiten

### Negative Konsequenzen
- **Precision/Recall-Kompromisse**: Regex erkennt keine kontextuellen PII
  (z.B. "Hans Müller ist krank" erkennt nicht "Hans Müller" als PII ohne NER)
- **Sprachgrenzen**: Optimiert für Deutsch; andere Sprachen benötigen
  zusätzliche Sprachmodelle
- **Maintenance**: Neue PII-Typen erfordern neue Regex-Muster und Tests

### Technische Schulden
- Presidio-Integration für Steuer-ID und GKV-Nummern nicht vollständig
  verifiziert (Stand: 2026-02-14) — offenes Research-Flag
- Konfidenz-Schwellenwerte (0.80/0.70/0.50) sind nicht empirisch kalibriert

---

## Abgelehnte Alternativen

### Alternative A: OpenAI GPT-4o-mini PII-Erkennung
**Abgelehnt wegen:** Zirkuläres Vertrauensproblem (s.o.) + Datenschutzrisiko.
Kosten: ~€6/Monat bei 10K req/Tag.

### Alternative B: AWS Comprehend (Medical + PII)
**Abgelehnt wegen:** Nicht DSGVO-konform ohne EU-Datenresidenz-Konfiguration,
Daten-Egress aus EU-Kontrolle, ~€15/Monat, Vendor Lock-in.

### Alternative C: Microsoft Presidio (vollständig)
**Teilweise implementiert**: Presidio-NER wird für kontextuelle Erkennung
verwendet. Presidio-Analyzer mit externen NER-Modellen wurde abgelehnt
(Download-Größe >500MB, Kaltstart-Latenz >2s).

---

## Referenzen
- `app/middleware/pii_detection.py` — Implementierung
- `tests/fixtures/de/` — Deutsche PII-Testdaten (Steuer-ID, IBAN, KFZ)
- DSGVO Art. 4 Nr. 1 — Definition personenbezogener Daten
- DSGVO Art. 25 — Privacy by Design
