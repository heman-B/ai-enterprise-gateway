# gateway/middleware/pii_detection.py
# DSGVO-konforme PII-Erkennung für deutsche Texte
# Erkennt: Steuer-ID, IBAN, KFZ, Handynummer, Namen, Adressen

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    """Erkannte PII-Entität mit Position und Typ."""

    type: str
    value_hash: str  # SHA256-Hash (NIEMALS Klartext speichern!)
    start: int
    end: int
    confidence: float


# ─── Deutsche Financial PII Patterns ────────────────────────────────────────


def validate_iban_checksum(iban: str) -> bool:
    """
    IBAN-Prüfziffer validieren (mod-97).
    DE IBAN: DE + 2 Prüfziffern + 18 Ziffern = 22 Zeichen.
    """
    iban_clean = iban.replace(" ", "").upper()
    if len(iban_clean) != 22 or not iban_clean.startswith("DE"):
        return False

    # IBAN mod-97: Verschiebe erste 4 Zeichen nach hinten, konvertiere Buchstaben zu Zahlen
    reordered = iban_clean[4:] + iban_clean[:4]
    numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in reordered)

    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def validate_steuer_id(steuer_id: str) -> bool:
    """
    Steuer-Identifikationsnummer validieren (ISO 7064 Mod 11, 10).
    11 Ziffern, eine Ziffer muss 2-3 mal vorkommen, keine mehr als 3 mal.
    """
    if not re.match(r"^\d{11}$", steuer_id):
        return False

    # Häufigkeit prüfen: Eine Ziffer muss 2-3 mal vorkommen, keine mehr als 3 mal
    digit_counts = [steuer_id.count(str(i)) for i in range(10)]
    has_double_or_triple = any(2 <= count <= 3 for count in digit_counts)
    no_more_than_three = all(count <= 3 for count in digit_counts)

    if not (has_double_or_triple and no_more_than_three):
        return False

    # ISO 7064 Mod 11, 10 Prüfziffer (letzte Ziffer)
    product = 10
    for digit in steuer_id[:-1]:
        sum_val = (int(digit) + product) % 10
        if sum_val == 0:
            sum_val = 10
        product = (sum_val * 2) % 11

    checksum = (11 - product) % 10
    return checksum == int(steuer_id[-1])


# Regex-Muster für deutsche Financial PII
IBAN_PATTERN = re.compile(r"\bDE\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{2}\b", re.IGNORECASE)

STEUER_ID_PATTERN = re.compile(r"\b\d{11}\b")  # Validierung über validate_steuer_id()

# KFZ-Kennzeichen: 1-3 Buchstaben, Bindestrich, 1-2 Buchstaben, 1-4 Ziffern
KFZ_PATTERN = re.compile(r"\b[A-ZÄÖÜ]{1,3}-[A-ZÄÖÜ]{1,2}\s?\d{1,4}\b")


# ─── Deutsche Contact PII Patterns ──────────────────────────────────────────


# Deutsche Handynummer: +49 oder 0049 oder 0, dann 1XX (Mobilfunk-Vorwahl)
# Unterstützt Formate: +49 172 3456789, +49172 3456789, 0171 1234567, etc.
HANDYNUMMER_PATTERN = re.compile(
    r"(?:\+49[\s\-]?|0049[\s\-]?|0)1\d{1,2}[\s\-]?\d{3,4}[\s\-]?\d{3,5}\b"
)

# PLZ: 5 Ziffern (10000-99999)
PLZ_PATTERN = re.compile(r"\b[0-9]{5}\b")


# ─── Deutsche Address & Name Patterns ───────────────────────────────────────


# Straßenname: Mindestens 3 Buchstaben + "straße", "weg", "platz", etc.
# Unterstützt zwei Formen:
#   - Zusammengesetzt: Hauptstraße 42, Mozartstraße 5, Rosenweg 8
#   - Zweiteilig: Berliner Straße 15, Am Ring 3
STRASSE_PATTERN = re.compile(
    r"(?:"
    r"[A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ]{2,}(?:straße|str\.|weg|platz|allee|gasse|ring|damm)"  # Hauptstraße, Rosenweg
    r"|"
    r"[A-ZÄÖÜ][a-zäöüß]+\s+(?:Straße|Str\.|Weg|Platz|Allee|Gasse|Ring|Damm)"  # Berliner Straße
    r")"
    r"\s+\d{1,4}[a-z]?\b",
    re.IGNORECASE,
)


class GermanPIIDetector:
    """
    Erkennt deutsche PII-Muster mit Checksummen-Validierung.

    Erkannte PII-Typen:
    - IBAN (mit mod-97 Validierung)
    - Steuer-ID (mit ISO 7064 Validierung)
    - KFZ-Kennzeichen
    - Handynummer
    - PLZ
    - Straßenname

    Optional (future): spaCy NER für Namen (de_core_news_sm)
    """

    def __init__(self) -> None:
        self._spacy_nlp = None
        # Versuche spaCy zu laden (optional, graceful fallback)
        try:
            import spacy

            self._spacy_nlp = spacy.load("de_core_news_sm")
            logger.info("spaCy de_core_news_sm geladen — Namen-Erkennung aktiv")
        except Exception:
            logger.warning(
                "spaCy de_core_news_sm nicht verfügbar — Namen-Erkennung deaktiviert"
            )

    def detect(self, text: str) -> list[PIIEntity]:
        """
        Erkenne alle PII-Entitäten im Text.

        Returns:
            Liste von PIIEntity-Objekten mit Typ, Hash, Position, Confidence.
        """
        entities: list[PIIEntity] = []

        # 1. IBAN (mit Checksummen-Validierung)
        for match in IBAN_PATTERN.finditer(text):
            iban = match.group(0).replace(" ", "")
            if validate_iban_checksum(iban):
                entities.append(
                    PIIEntity(
                        type="IBAN",
                        value_hash=self._hash_value(iban),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.98,
                    )
                )

        # 2. Steuer-ID (mit ISO 7064 Validierung)
        for match in STEUER_ID_PATTERN.finditer(text):
            steuer_id = match.group(0)
            if validate_steuer_id(steuer_id):
                entities.append(
                    PIIEntity(
                        type="STEUER_ID",
                        value_hash=self._hash_value(steuer_id),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.98,
                    )
                )

        # 3. KFZ-Kennzeichen
        for match in KFZ_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    type="KFZ",
                    value_hash=self._hash_value(match.group(0)),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.90,
                )
            )

        # 4. Handynummer (nicht in IBAN enthalten — verhindert Ziffernfolgen innerhalb von IBANs)
        iban_ranges = [(e.start, e.end) for e in entities if e.type == "IBAN"]
        for match in HANDYNUMMER_PATTERN.finditer(text):
            if not any(start <= match.start() < end for start, end in iban_ranges):
                entities.append(
                    PIIEntity(
                        type="HANDYNUMMER",
                        value_hash=self._hash_value(match.group(0)),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95,
                    )
                )

        # 5. PLZ (nicht in IBAN oder Handynummer enthalten — verhindert "49172" aus "+49172")
        excluded_ranges = [(e.start, e.end) for e in entities if e.type in ("IBAN", "HANDYNUMMER")]
        for match in PLZ_PATTERN.finditer(text):
            if not any(start <= match.start() < end for start, end in excluded_ranges):
                entities.append(
                    PIIEntity(
                        type="PLZ",
                        value_hash=self._hash_value(match.group(0)),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                    )
                )

        # 6. Straßenname
        for match in STRASSE_PATTERN.finditer(text):
            entities.append(
                PIIEntity(
                    type="STRASSE",
                    value_hash=self._hash_value(match.group(0)),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                )
            )

        # 7. Namen (optional via spaCy)
        if self._spacy_nlp:
            doc = self._spacy_nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PER":  # Person
                    entities.append(
                        PIIEntity(
                            type="NAME",
                            value_hash=self._hash_value(ent.text),
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.80,  # NER ist probabilistisch
                        )
                    )

        # Sortiere nach Position (für Redaktion wichtig)
        entities.sort(key=lambda e: e.start)
        return entities

    def redact(self, text: str) -> tuple[str, list[PIIEntity]]:
        """
        Redaktiere PII im Text und gebe redaktierten Text + Erkennungsbericht zurück.

        Format: [IBAN_1], [STEUER_ID_1], [NAME_1], etc.
        """
        entities = self.detect(text)

        if not entities:
            return text, []

        # Zähler pro PII-Typ
        type_counters: dict[str, int] = {}
        redacted_text = text

        # Rückwärts iterieren, um Positionen nicht zu verschieben
        for entity in reversed(entities):
            pii_type = entity.type
            type_counters[pii_type] = type_counters.get(pii_type, 0) + 1
            placeholder = f"[{pii_type}_{type_counters[pii_type]}]"

            redacted_text = (
                redacted_text[: entity.start]
                + placeholder
                + redacted_text[entity.end :]
            )

        return redacted_text, entities

    def _hash_value(self, value: str) -> str:
        """SHA256-Hash für Audit-Logging (niemals Klartext speichern)."""
        return hashlib.sha256(value.encode()).hexdigest()


# ─── Singleton Instance ─────────────────────────────────────────────────────

_detector_instance: GermanPIIDetector | None = None


def get_pii_detector() -> GermanPIIDetector:
    """Singleton-Instanz des PII-Detektors."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = GermanPIIDetector()
    return _detector_instance
