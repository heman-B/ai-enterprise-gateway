# tests/test_pii_accuracy.py
# Präzisions-/Recall-Benchmark für deutsche PII-Erkennung.
# Misst Genauigkeit pro Typ gegen annotierte Testdaten aus tests/fixtures/de/.
# Assertion: jeder Typ ≥ 90%, Gesamt (ohne NAME) ≥ 94%.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.middleware.pii_detection import get_pii_detector

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "de" / "synthetic_german_pii.json"

# PII-Typen die ohne spaCy deterministisch erkannt werden
CORE_TYPES = ["IBAN", "STEUER_ID", "KFZ", "HANDYNUMMER", "PLZ", "STRASSE"]


def _load_fixtures() -> list[dict]:
    with FIXTURES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _compute_type_metrics(
    fixtures: list[dict], pii_type: str
) -> tuple[int, int, int]:
    """Berechnet TP, FP, FN für einen PII-Typ über alle Testfälle."""
    detector = get_pii_detector()
    tp = fp = fn = 0

    for sample in fixtures:
        expected_count = sum(
            1 for e in sample["expected_entities"] if e["type"] == pii_type
        )
        detected = detector.detect(sample["text"])
        detected_count = sum(1 for e in detected if e.type == pii_type)

        tp += min(expected_count, detected_count)
        fp += max(0, detected_count - expected_count)
        fn += max(0, expected_count - detected_count)

    return tp, fp, fn


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 1.0


class TestPIIAccuracyBenchmark:
    """
    Präzisions-/Recall-Benchmark für alle unterstützten DACH-PII-Typen.

    Jeder Kerntyp (IBAN, Steuer-ID, KFZ, Handynummer, PLZ, Straße) muss
    ≥ 90% Präzision und ≥ 90% Recall erreichen. Gesamt ≥ 94%.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fixtures = _load_fixtures()
        assert len(self.fixtures) >= 10, "Mindestens 10 Testfälle erwartet"

    # ─── Per-Typ-Metriken ─────────────────────────────────────────────────────

    def test_iban_precision_and_recall(self):
        tp, fp, fn = _compute_type_metrics(self.fixtures, "IBAN")
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        assert prec >= 0.90, f"IBAN Präzision {prec:.1%} < 90% (TP={tp} FP={fp})"
        assert rec >= 0.90, f"IBAN Recall {rec:.1%} < 90% (TP={tp} FN={fn})"

    def test_steuer_id_precision_and_recall(self):
        tp, fp, fn = _compute_type_metrics(self.fixtures, "STEUER_ID")
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        assert prec >= 0.90, f"STEUER_ID Präzision {prec:.1%} < 90% (TP={tp} FP={fp})"
        assert rec >= 0.90, f"STEUER_ID Recall {rec:.1%} < 90% (TP={tp} FN={fn})"

    def test_kfz_precision_and_recall(self):
        tp, fp, fn = _compute_type_metrics(self.fixtures, "KFZ")
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        assert prec >= 0.90, f"KFZ Präzision {prec:.1%} < 90% (TP={tp} FP={fp})"
        assert rec >= 0.90, f"KFZ Recall {rec:.1%} < 90% (TP={tp} FN={fn})"

    def test_handynummer_precision_and_recall(self):
        tp, fp, fn = _compute_type_metrics(self.fixtures, "HANDYNUMMER")
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        assert prec >= 0.90, f"HANDYNUMMER Präzision {prec:.1%} < 90% (TP={tp} FP={fp})"
        assert rec >= 0.90, f"HANDYNUMMER Recall {rec:.1%} < 90% (TP={tp} FN={fn})"

    def test_plz_precision_and_recall(self):
        tp, fp, fn = _compute_type_metrics(self.fixtures, "PLZ")
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        assert prec >= 0.90, f"PLZ Präzision {prec:.1%} < 90% (TP={tp} FP={fp})"
        assert rec >= 0.90, f"PLZ Recall {rec:.1%} < 90% (TP={tp} FN={fn})"

    def test_strasse_precision_and_recall(self):
        tp, fp, fn = _compute_type_metrics(self.fixtures, "STRASSE")
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        assert prec >= 0.90, f"STRASSE Präzision {prec:.1%} < 90% (TP={tp} FP={fp})"
        assert rec >= 0.90, f"STRASSE Recall {rec:.1%} < 90% (TP={tp} FN={fn})"

    def test_name_precision_and_recall(self):
        """NAME-Erkennung erfordert spaCy — überspringe wenn nicht installiert."""
        detector = get_pii_detector()
        if detector._spacy_nlp is None:
            pytest.skip("spaCy de_core_news_sm nicht installiert — NAME-Benchmark übersprungen")
        tp, fp, fn = _compute_type_metrics(self.fixtures, "NAME")
        prec = _precision(tp, fp)
        rec = _recall(tp, fn)
        # NER ist probabilistisch — niedrigere Schwelle als regelbasierte Erkennung
        assert prec >= 0.70, f"NAME Präzision {prec:.1%} < 70%"
        assert rec >= 0.70, f"NAME Recall {rec:.1%} < 70%"

    # ─── Gesamtmetriken ───────────────────────────────────────────────────────

    def test_overall_precision_above_94_percent(self):
        """Gesamtpräzision aller Kerntypen muss ≥ 94% betragen."""
        total_tp = total_fp = 0
        for pii_type in CORE_TYPES:
            tp, fp, fn = _compute_type_metrics(self.fixtures, pii_type)
            total_tp += tp
            total_fp += fp
        overall = _precision(total_tp, total_fp)
        assert overall >= 0.94, (
            f"Gesamtpräzision {overall:.1%} < 94% "
            f"(TP={total_tp} FP={total_fp})"
        )

    def test_overall_recall_above_94_percent(self):
        """Gesamt-Recall aller Kerntypen muss ≥ 94% betragen."""
        total_tp = total_fn = 0
        for pii_type in CORE_TYPES:
            tp, fp, fn = _compute_type_metrics(self.fixtures, pii_type)
            total_tp += tp
            total_fn += fn
        overall = _recall(total_tp, total_fn)
        assert overall >= 0.94, (
            f"Gesamt-Recall {overall:.1%} < 94% "
            f"(TP={total_tp} FN={total_fn})"
        )
