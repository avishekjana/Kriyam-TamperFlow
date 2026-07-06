"""Integration tests for scripts/evaluate.py and kriyam/report.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow importing from the scripts/ folder without installing.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import _load_prediction, _pred_path, _print_tables, _score_sample, run_evaluation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANN_TAMPERED = {
    "id": "kriyam_0001",
    "image_w": 200,
    "image_h": 150,
    "is_authentic": False,
    "regions": [{"region_id": 1, "x": 10, "y": 10, "w": 50, "h": 40,
                 "tamper_types": ["splice"]}],
}

_ANN_AUTHENTIC = {
    "id": "kriyam_0002",
    "image_w": 200,
    "image_h": 150,
    "is_authentic": True,
    "regions": [],
}

_PRED_CORRECT = {
    "id": "kriyam_0001_C0",
    "model": "test_model",
    "is_authentic": False,
    "confidence": 0.92,
    "regions": [{"x": 10, "y": 10, "w": 50, "h": 40, "confidence": 0.91}],
}

_PRED_AUTHENTIC_CORRECT = {
    "id": "kriyam_0002_C0",
    "model": "test_model",
    "is_authentic": True,
    "confidence": 0.08,
    "regions": [],
}


def _make_data_dir(tmp_path: Path, annotations: list[dict]) -> Path:
    ann_dir = tmp_path / "data" / "annotations"
    ann_dir.mkdir(parents=True)
    for ann in annotations:
        (ann_dir / f"{ann['id']}.json").write_text(json.dumps(ann), encoding="utf-8")
    return tmp_path / "data"


def _make_predictions_dir(tmp_path: Path, model: str, predictions: dict[str, dict]) -> Path:
    """predictions maps '<sample_id>_<tier>' → prediction dict."""
    pred_dir = tmp_path / "predictions" / model
    pred_dir.mkdir(parents=True)
    for filename, pred in predictions.items():
        (pred_dir / f"{filename}.json").write_text(json.dumps(pred), encoding="utf-8")
    return pred_dir


# ---------------------------------------------------------------------------
# _load_prediction
# ---------------------------------------------------------------------------


def test_load_prediction_success(tmp_path: Path) -> None:
    p = tmp_path / "pred.json"
    p.write_text(json.dumps({"confidence": 0.9}), encoding="utf-8")
    assert _load_prediction(p) == {"confidence": 0.9}


def test_load_prediction_missing_returns_none(tmp_path: Path) -> None:
    assert _load_prediction(tmp_path / "ghost.json") is None


def test_load_prediction_malformed_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert _load_prediction(p) is None


# ---------------------------------------------------------------------------
# _pred_path
# ---------------------------------------------------------------------------


def test_pred_path_format(tmp_path: Path) -> None:
    path = _pred_path(tmp_path / "model", "kriyam_0042", "C2")
    assert path.name == "kriyam_0042_C2.json"


# ---------------------------------------------------------------------------
# _score_sample
# ---------------------------------------------------------------------------


def test_score_sample_correct_prediction() -> None:
    result = _score_sample(_ANN_TAMPERED, _PRED_CORRECT, "C0")
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["gt_label"] == 1
    assert result["pred_confidence"] == pytest.approx(0.91)  # max(region confidences)
    assert result["region_f1"] == pytest.approx(1.0)


def test_score_sample_authentic_no_regions() -> None:
    result = _score_sample(_ANN_AUTHENTIC, _PRED_AUTHENTIC_CORRECT, "C0")
    assert result["gt_label"] == 0
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_score_sample_keys_present() -> None:
    result = _score_sample(_ANN_TAMPERED, _PRED_CORRECT, "C0")
    for key in ("sample_id", "tier", "pred_confidence", "gt_label",
                "region_precision", "region_recall", "region_f1",
                "tp", "fp", "fn"):
        assert key in result, f"Missing key: {key}"


def test_score_sample_no_regions_confidence_is_zero() -> None:
    # Empty regions list → pred_confidence must be 0.0
    pred_empty = {"id": "kriyam_0001_C0", "regions": []}
    result = _score_sample(_ANN_TAMPERED, pred_empty, "C0")
    assert result["pred_confidence"] == pytest.approx(0.0)


def test_score_sample_pred_label_above_threshold() -> None:
    # Region confidence 0.91 >= default threshold 0.5 → pred_label=1
    result = _score_sample(_ANN_TAMPERED, _PRED_CORRECT, "C0")
    assert result["pred_label"] == 1


def test_score_sample_pred_label_below_threshold() -> None:
    # Region confidence 0.3 < threshold 0.5 → pred_label=0
    pred_low = {"id": "kriyam_0001_C0", "regions": [{"x": 10, "y": 10, "w": 50, "h": 40, "confidence": 0.3}]}
    result = _score_sample(_ANN_TAMPERED, pred_low, "C0", document_threshold=0.5)
    assert result["pred_label"] == 0


def test_score_sample_pred_label_custom_threshold() -> None:
    # Region confidence 0.3 >= custom threshold 0.2 → pred_label=1
    pred_low = {"id": "kriyam_0001_C0", "regions": [{"x": 10, "y": 10, "w": 50, "h": 40, "confidence": 0.3}]}
    result = _score_sample(_ANN_TAMPERED, pred_low, "C0", document_threshold=0.2)
    assert result["pred_label"] == 1


def test_score_sample_missing_regions_treated_as_empty() -> None:
    pred_no_regions = {k: v for k, v in _PRED_CORRECT.items() if k != "regions"}
    result = _score_sample(_ANN_TAMPERED, pred_no_regions, "C0")
    # No predicted regions → all GT regions are FN
    assert result["fn"] == 1
    assert result["tp"] == 0


# ---------------------------------------------------------------------------
# _print_table — smoke test (does not crash, produces output)
# ---------------------------------------------------------------------------


def test_print_table_smoke(capsys: pytest.CaptureFixture) -> None:
    tier_scores = {
        "C0": {
            "region_precision": 0.85,
            "region_precision_ci": (0.80, 0.90),
            "region_recall": 0.80,
            "region_recall_ci": (0.75, 0.85),
            "region_f1": 0.82,
            "region_f1_ci": (0.78, 0.87),
            "doc_auc": 0.91,
            "doc_auc_ci": (0.88, 0.94),
            "doc_f1": 0.87,
            "doc_f1_ci": (0.83, 0.91),
            "doc_fpr": 0.06,
            "doc_fpr_ci": (0.03, 0.10),
            "n_samples": 100,
        }
    }
    _print_tables(tier_scores)
    captured = capsys.readouterr()
    assert "C0" in captured.out
    assert "Region Precision" in captured.out


# ---------------------------------------------------------------------------
# run_evaluation — full integration
# ---------------------------------------------------------------------------


def test_run_evaluation_produces_scores(tmp_path: Path) -> None:
    data_dir = _make_data_dir(
        tmp_path,
        [_ANN_TAMPERED, _ANN_AUTHENTIC],
    )
    pred_dir = _make_predictions_dir(
        tmp_path,
        "test_model",
        {
            "kriyam_0001_C0": _PRED_CORRECT,
            "kriyam_0002_C0": _PRED_AUTHENTIC_CORRECT,
        },
    )
    report_path = tmp_path / "reports" / "report.html"

    tier_scores = run_evaluation(
        predictions_dir=pred_dir,
        data_dir=data_dir,
        tiers=["C0"],
        report_out=report_path,
    )

    assert "C0" in tier_scores
    assert tier_scores["C0"]["n_samples"] == 2
    assert 0.0 <= tier_scores["C0"]["doc_auc"] <= 1.0


def test_run_evaluation_writes_scores_json(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path, [_ANN_TAMPERED, _ANN_AUTHENTIC])
    pred_dir = _make_predictions_dir(
        tmp_path,
        "my_model",
        {
            "kriyam_0001_C0": _PRED_CORRECT,
            "kriyam_0002_C0": _PRED_AUTHENTIC_CORRECT,
        },
    )
    run_evaluation(
        predictions_dir=pred_dir,
        data_dir=data_dir,
        tiers=["C0"],
        report_out=tmp_path / "report.html",
    )

    scores_file = Path("results") / "my_model" / "scores.json"
    assert scores_file.is_file()
    payload = json.loads(scores_file.read_text())
    assert payload["model"] == "my_model"
    assert "C0" in payload["tiers"]


def test_run_evaluation_writes_html_report(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path, [_ANN_TAMPERED, _ANN_AUTHENTIC])
    pred_dir = _make_predictions_dir(
        tmp_path,
        "test_model",
        {
            "kriyam_0001_C0": _PRED_CORRECT,
            "kriyam_0002_C0": _PRED_AUTHENTIC_CORRECT,
        },
    )
    report_path = tmp_path / "report.html"
    run_evaluation(
        predictions_dir=pred_dir,
        data_dir=data_dir,
        tiers=["C0"],
        report_out=report_path,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "test_model" in content
    assert "C0" in content


def test_run_evaluation_skips_missing_predictions(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path, [_ANN_TAMPERED, _ANN_AUTHENTIC])
    # Only provide one of the two prediction files.
    pred_dir = _make_predictions_dir(
        tmp_path,
        "partial_model",
        {"kriyam_0001_C0": _PRED_CORRECT},
    )
    tier_scores = run_evaluation(
        predictions_dir=pred_dir,
        data_dir=data_dir,
        tiers=["C0"],
        report_out=tmp_path / "report.html",
    )
    # Only 1 of 2 predictions present → n_samples = 1
    assert tier_scores["C0"]["n_samples"] == 1



def test_run_evaluation_multiple_tiers(tmp_path: Path) -> None:
    data_dir = _make_data_dir(tmp_path, [_ANN_TAMPERED, _ANN_AUTHENTIC])
    pred_dir = _make_predictions_dir(
        tmp_path,
        "multi_tier_model",
        {
            "kriyam_0001_C0": _PRED_CORRECT,
            "kriyam_0002_C0": _PRED_AUTHENTIC_CORRECT,
            "kriyam_0001_C2": {**_PRED_CORRECT, "id": "kriyam_0001_C2", "confidence": 0.85},
            "kriyam_0002_C2": {**_PRED_AUTHENTIC_CORRECT, "id": "kriyam_0002_C2", "confidence": 0.12},
        },
    )
    tier_scores = run_evaluation(
        predictions_dir=pred_dir,
        data_dir=data_dir,
        tiers=["C0", "C2"],
        report_out=tmp_path / "report.html",
    )
    assert "C0" in tier_scores
    assert "C2" in tier_scores
    assert tier_scores["C0"]["n_samples"] == 2
    assert tier_scores["C2"]["n_samples"] == 2
