# ADR-003: Redis Semantischer Cache

**Status:** Accepted
**Datum:** 2026-02-22
**Entscheider:** Gateway-Architektur-Review

---

## Kontext

LLM-API-Aufrufe sind die größte Kostenstelle des Gateways. In Unternehmens-
umgebungen werden dieselben oder semantisch ähnliche Anfragen häufig wiederholt
(FAQ-Bots, Dokumentenklassifizierung, Support-Chatbots).

Ohne Cache: jede Anfrage → Provider-API-Aufruf → Kosten.
Mit Cache: identische/ähnliche Anfragen → gespeicherte Antwort → keine Kosten.

Die Frage war: Wie aggressiv cachen, und mit welchem Ähnlichkeitsmaß?

---

## Entscheidung

**Zwei-Stufen semantischer Cache in Redis** mit:
- Stufe 1: SHA256 des normalisierten Prompts (exakt, kostenlos, <1ms)
- Stufe 2: Kosinus-Ähnlichkeit auf OpenAI `text-embedding-3-small` (optional, ~20ms)
- Schwellenwert: **0.95 Kosinus-Ähnlichkeit** für Stufe-2-Treffer
- TTL: **3600 Sekunden** (1 Stunde)

---

## Begründung

### Kosteneinsparung bei 10K req/Tag

Annahmen:
- 10.000 Anfragen/Tag
- Durchschnittlich 500 Token pro Anfrage (input + output)
- Routing zu `claude-sonnet-4-5` ($3.00/1M Input + $15.00/1M Output)
- Konservative Schätzung: 40% Cache-Trefferquote

Berechnung:
```
Tokens/req:        500 (avg: 300 input + 200 output)
Kosten/req:        300 × $3/1M + 200 × $15/1M = $0.0009 + $0.003 = $0.0039
Req/Tag mit Cache: 10.000 × 0.60 (cache misses) = 6.000
Req/Tag ohne Cache: 10.000

Einsparung/Tag:    (10.000 - 6.000) × $0.0039 = $15.60
Einsparung/Monat:  $15.60 × 30 = **$468/Monat**
```

Bei Routing zu günstigeren Modellen (claude-haiku: $0.00025/1K Input):
```
Einsparung/Monat:  ~$6-15/Monat (Haiku ist bereits günstig)
```

Das Gateway routet automatisch komplexe Anfragen zu Sonnet-Klasse, wo der
Cache am meisten spart. **Konservative Schätzung: €50-180/Monat** je nach
Traffic-Mix.

### Warum Normalisierungs-SHA256 vor Embeddings?

Embeddings kosten Geld und Zeit (~20ms). Der SHA256-Check ist kostenlos und
trifft ca. 60-70% der Wiederholungsanfragen:

```
"What is the capital of France?"
"what is the capital of france?"   → SHA256-Normalisierung: gleicher Hash ✅
"What is the capital of France? "  → SHA256-Normalisierung: gleicher Hash ✅
"What is France's capital?"         → SHA256: Miss → Embedding: Treffer ✅
```

### Warum 0.95 als Schwellenwert?

Für `text-embedding-3-small` (1536 Dimensionen):
- Identische Sätze: 1.00
- Paraphrasen: 0.92-0.98
- Ähnliche aber andere Fragen: 0.85-0.92
- Komplett verschiedene Themen: <0.80

0.95 ist konservativ: wir riskieren eher Cache-Misses als falsche Treffer.
Ein falscher Treffer (semantisch ähnlich aber inhaltlich unterschiedlich)
wäre schlimmer als kein Cache-Hit.

**Hinweis**: Dieser Schwellenwert ist nicht empirisch kalibriert. Er ist ein
vernünftiger Startpunkt, der mit realen Traffic-Daten angepasst werden sollte.

---

## Konsequenzen

### Positive Konsequenzen
- **€50-468/Monat gespart** (je nach Modell-Mix und Trefferquote)
- **Latenz-Verbesserung**: Cache-Hits in <2ms statt 200-1000ms Provider-Latenz
- **Redis bereits vorhanden**: Rate-Limiter nutzt denselben Redis — kein
  zusätzlicher Infrastrukturaufwand
- **Fail-open**: Redis-Fehler blockieren keine Anfragen

### Negative Konsequenzen
- **Cache-Staleness**: Antworten ändern sich (Modell-Updates, Wissensstand).
  TTL=3600s limitiert dieses Risiko, eliminiert es aber nicht
- **Embedding-Kosten** für Stufe-2: OpenAI `text-embedding-3-small` kostet
  $0.02/1M Token ≈ $0.000006 pro Anfrage — vernachlässigbar, aber nicht null
- **O(n) Similarity-Suche**: Skaliert linear mit unique Prompts im Cache.
  Bei >10.000 unique Prompts: Performance-Degradation. Mitigation: Redis
  SCAN + Limit, oder Upgrade auf pgvector/Redis Search
- **Schwellenwert ungetestet**: 0.95 wurde nicht mit echten Produktionsdaten
  kalibriert

### Nicht-Ziele
- Dieser Cache ignoriert `temperature` und `max_tokens` bei der Cache-Key-
  Berechnung (nur Prompt-Inhalt). Bei deterministischen Anfragen (temp=0)
  ist das korrekt; bei kreativen Anfragen (temp>0) könnte es zu unerwartet
  konsistenten Antworten führen.

---

## Abgelehnte Alternativen

### Alternative A: Kein Cache
**Abgelehnt wegen:** Unnötige Kosten bei wiederholten Anfragen. Bei 40%
Trefferquote entspricht das €50-468/Monat verschenkter Ausgaben.

### Alternative B: Vollständiger Response-Cache (Exact Match Only)
**Abgelehnt wegen:** Trifft nur identische Prompts (case-sensitive, whitespace-
sensitiv). Zu restriktiv für Unternehmensanwendungen mit variablen Formulierungen.

### Alternative C: pgvector + PostgreSQL
**Abgelehnt wegen:** Redis bereits vorhanden; pgvector benötigt separate
Infrastruktur oder PostgreSQL-Extension-Installation. Bessere Option bei
>50.000 unique Prompts, wenn O(n) nicht mehr skaliert.

### Alternative D: Semantische Suche via Pinecone/Weaviate
**Abgelehnt wegen:** Vendor Lock-in, €25-100/Monat Zusatzkosten, Daten-Egress
aus EU-Kontrolle.

---

## Monitoring

```prometheus
# Cache-Treffer nach Provider und Modell
gateway_cache_hit_total{provider="anthropic", model="claude-haiku-4-5"} 1234

# Cache-Statistiken
GET /admin/cache/stats
→ {hit_rate_pct: 38.5, tokens_saved: 456789, cost_saved_usd: 12.34}
```

---

## Referenzen
- `gateway/cache/semantic_cache.py` — Implementierung
- `GET /admin/cache/stats` — Statistik-Endpoint
- `gateway_cache_hit_total` — Prometheus-Metrik
- ADR-004 — Render vs Azure (Infrastrukturkontext)
- OpenAI Embeddings Pricing: https://openai.com/pricing
