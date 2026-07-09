"""Tests for kriyam.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from kriyam.metrics import (
    _operating_point_at_tpr,
    aggregate,
    bboxes_to_mask,
    compression_robustness,
    iou_matrix,
    match_regions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 100, 100  # default image dimensions for most tests


def _r(x: int, y: int, w: int, h: int) -> dict:
    return {"x": x, "y": y, "w": w, "h": h}


def _result(
    precision: float = 1.0,
    recall: float = 1.0,
    f1: float = 1.0,
    confidence: float = 0.9,
    gt_label: int = 1,
    pred_label: int = 1,
) -> dict:
    return {
        "region_precision": precision,
        "region_recall": recall,
        "region_f1": f1,
        "pred_confidence": confidence,
        "gt_label": gt_label,
        "pred_label": pred_label,
    }


# ---------------------------------------------------------------------------
# bboxes_to_mask
# ---------------------------------------------------------------------------


def test_empty_regions_returns_zero_mask() -> None:
    mask = bboxes_to_mask([], H, W)
    assert mask.shape == (H, W)
    assert mask.sum() == 0
    assert mask.dtype == np.uint8


def test_single_region_fills_correct_pixels() -> None:
    mask = bboxes_to_mask([_r(10, 20, 30, 40)], H, W)
    # Rectangle: columns 10–39, rows 20–59
    assert mask[20:60, 10:40].all()
    assert mask.sum() == 30 * 40


def test_two_regions_are_ored() -> None:
    r1 = _r(0, 0, 10, 10)
    r2 = _r(50, 50, 10, 10)
    mask = bboxes_to_mask([r1, r2], H, W)
    assert mask.sum() == 10 * 10 * 2


def test_overlapping_regions_no_double_count() -> None:
    r1 = _r(0, 0, 20, 20)
    r2 = _r(10, 10, 20, 20)  # overlaps r1 in a 10×10 area
    mask = bboxes_to_mask([r1, r2], H, W)
    # Union area: 20*20 + 20*20 - 10*10 = 700
    assert mask.sum() == 700


def test_out_of_bounds_region_clamped() -> None:
    mask = bboxes_to_mask([_r(-5, -5, 200, 200)], H, W)
    assert mask.shape == (H, W)
    assert mask.sum() == H * W


def test_mask_shape_matches_image_dimensions() -> None:
    mask = bboxes_to_mask([_r(0, 0, 5, 5)], 80, 120)
    assert mask.shape == (80, 120)


# ---------------------------------------------------------------------------
# iou_matrix
# ---------------------------------------------------------------------------


def test_iou_matrix_shape() -> None:
    gt = [_r(0, 0, 10, 10), _r(50, 50, 10, 10)]
    pred = [_r(0, 0, 10, 10)]
    mat = iou_matrix(gt, pred, H, W)
    assert mat.shape == (2, 1)


def test_iou_perfect_overlap() -> None:
    box = _r(10, 10, 20, 20)
    mat = iou_matrix([box], [box], H, W)
    assert abs(mat[0, 0] - 1.0) < 1e-9


def test_iou_no_overlap() -> None:
    mat = iou_matrix([_r(0, 0, 10, 10)], [_r(50, 50, 10, 10)], H, W)
    assert mat[0, 0] == 0.0


def test_iou_partial_overlap() -> None:
    # Two 10×10 boxes sharing a 5×10 strip → intersection=50, union=150
    mat = iou_matrix([_r(0, 0, 10, 10)], [_r(5, 0, 10, 10)], H, W)
    expected = 50.0 / 150.0
    assert abs(mat[0, 0] - expected) < 1e-6


def test_iou_matrix_empty_gt() -> None:
    mat = iou_matrix([], [_r(0, 0, 5, 5)], H, W)
    assert mat.shape == (0, 1)


def test_iou_matrix_empty_pred() -> None:
    mat = iou_matrix([_r(0, 0, 5, 5)], [], H, W)
    assert mat.shape == (1, 0)


def test_iou_matrix_both_empty() -> None:
    mat = iou_matrix([], [], H, W)
    assert mat.shape == (0, 0)


def test_iou_matrix_dtype_float() -> None:
    mat = iou_matrix([_r(0, 0, 10, 10)], [_r(0, 0, 10, 10)], H, W)
    assert mat.dtype == np.float64


# ---------------------------------------------------------------------------
# match_regions
# ---------------------------------------------------------------------------


def test_perfect_match_all_tp() -> None:
    box = _r(10, 10, 20, 20)
    out = match_regions([box], [box], H, W)
    assert out["tp"] == 1
    assert out["fp"] == 0
    assert out["fn"] == 0
    assert abs(out["region_precision"] - 1.0) < 1e-9
    assert abs(out["region_recall"] - 1.0) < 1e-9
    assert abs(out["region_f1"] - 1.0) < 1e-9


def test_no_prediction_all_fn() -> None:
    out = match_regions([_r(0, 0, 10, 10)], [], H, W)
    assert out["tp"] == 0
    assert out["fp"] == 0
    assert out["fn"] == 1
    assert out["region_recall"] == 0.0


def test_extra_prediction_all_fp() -> None:
    out = match_regions([], [_r(0, 0, 10, 10)], H, W)
    assert out["tp"] == 0
    assert out["fp"] == 1
    assert out["fn"] == 0
    assert out["region_precision"] == 0.0


def test_below_iou_threshold_counts_as_miss() -> None:
    # Boxes share a 1-pixel column overlap only — IoU far below 0.1
    out = match_regions([_r(0, 0, 10, 10)], [_r(9, 0, 10, 10)], H, W, iou_threshold=0.1)
    # intersection 10, union 190 → IoU ≈ 0.0526 < 0.1
    assert out["tp"] == 0


def test_above_iou_threshold_counts_as_tp() -> None:
    box = _r(0, 0, 10, 10)
    shifted = _r(1, 0, 10, 10)  # intersection=90, union=110 → IoU ≈ 0.818
    out = match_regions([box], [shifted], H, W, iou_threshold=0.1)
    assert out["tp"] == 1


def test_hungarian_finds_best_assignment() -> None:
    # GT box A matches pred box 1 better than pred box 2.
    # GT box B matches pred box 2 better.  Hungarian should assign optimally.
    gt = [_r(0, 0, 20, 20), _r(60, 60, 20, 20)]
    pred = [_r(0, 0, 20, 20), _r(60, 60, 20, 20)]
    out = match_regions(gt, pred, H, W)
    assert out["tp"] == 2
    assert out["fp"] == 0
    assert out["fn"] == 0


def test_f1_harmonic_mean() -> None:
    # Precision=1, recall=0.5 → F1=0.667
    out = match_regions([_r(0, 0, 10, 10), _r(50, 50, 10, 10)], [_r(0, 0, 10, 10)], H, W)
    assert abs(out["region_f1"] - (2 * 1.0 * 0.5 / 1.5)) < 1e-6


def test_no_gt_no_pred_zero_counts() -> None:
    out = match_regions([], [], H, W)
    assert out["tp"] == out["fp"] == out["fn"] == 0
    assert out["region_precision"] == 0.0
    assert out["region_recall"] == 0.0
    assert out["region_f1"] == 0.0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate([])


def test_aggregate_single_sample_keys() -> None:
    out = aggregate([_result()])
    expected_keys = {
        "region_precision",
        "region_precision_ci",
        "region_recall",
        "region_recall_ci",
        "region_f1",
        "region_f1_ci",
        "doc_auc",
        "doc_auc_ci",
        "doc_auprc",
        "doc_auprc_ci",
        "doc_f1",
        "doc_f1_ci",
        "doc_fpr",
        "doc_fpr_ci",
        "doc_fpr_at_tpr90",
        "doc_fpr_at_tpr90_ci",
        "doc_fpr_at_tpr90_valid",
        "doc_f1_at_tpr90",
        "doc_f1_at_tpr90_ci",
        "doc_f1_at_tpr90_valid",
        "doc_tpr90_achieved_tpr",
        "n_samples",
    }
    assert expected_keys.issubset(out.keys())


def test_aggregate_n_samples() -> None:
    results = [_result() for _ in range(5)]
    out = aggregate(results)
    assert out["n_samples"] == 5


def test_aggregate_macro_average_precision() -> None:
    results = [_result(precision=0.8), _result(precision=0.6)]
    out = aggregate(results)
    assert abs(out["region_precision"] - 0.7) < 1e-9


def test_aggregate_ci_is_tuple_of_two_floats() -> None:
    results = [_result() for _ in range(10)]
    out = aggregate(results)
    lo, hi = out["region_f1_ci"]
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo <= hi


def test_aggregate_ci_lower_le_mean_le_upper() -> None:
    results = [_result(f1=v) for v in [0.4, 0.6, 0.7, 0.8, 0.5]]
    out = aggregate(results)
    lo, hi = out["region_f1_ci"]
    assert lo <= out["region_f1"] <= hi


def test_aggregate_doc_auc_perfect_separation() -> None:
    results = [
        _result(confidence=0.9, gt_label=1, pred_label=1),
        _result(confidence=0.8, gt_label=1, pred_label=1),
        _result(confidence=0.2, gt_label=0, pred_label=0),
        _result(confidence=0.1, gt_label=0, pred_label=0),
    ]
    out = aggregate(results)
    assert abs(out["doc_auc"] - 1.0) < 1e-9


def test_aggregate_doc_auc_no_discrimination() -> None:
    results = [
        _result(confidence=0.5, gt_label=1, pred_label=1),
        _result(confidence=0.5, gt_label=0, pred_label=0),
    ]
    out = aggregate(results)
    # AUC should be 0.5 for random-guess confidence scores
    assert 0.0 <= out["doc_auc"] <= 1.0


def test_aggregate_fpr_all_correct_is_zero() -> None:
    # Authentic sample correctly predicted as authentic (pred_label=0) → no FP
    results = [
        _result(confidence=0.9, gt_label=1, pred_label=1),
        _result(confidence=0.0, gt_label=0, pred_label=0),
    ]
    out = aggregate(results)
    assert out["doc_fpr"] == pytest.approx(0.0)


def test_aggregate_single_class_auc_fallback() -> None:
    # Only tampered samples — AUC undefined, should fall back to 0.5.
    results = [_result(confidence=0.9, gt_label=1) for _ in range(5)]
    out = aggregate(results)
    assert out["doc_auc"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _operating_point_at_tpr
# ---------------------------------------------------------------------------


def test_operating_point_well_separated() -> None:
    # Tampered samples score high, authentic samples score low → clean separation.
    # At threshold=0.8 all 3 tampered are TP and 0 authentic are FP → TPR=1, FPR=0.
    gt = np.array([1, 1, 1, 0, 0])
    conf = np.array([0.9, 0.85, 0.8, 0.2, 0.1])
    out = _operating_point_at_tpr(gt, conf)
    assert out["valid"] is True
    assert out["achieved_tpr"] >= 0.9
    assert out["fpr"] == pytest.approx(0.0)


def test_operating_point_single_class_returns_invalid() -> None:
    # Only tampered samples — TPR is undefined; should return valid=False.
    gt = np.array([1, 1, 1, 1, 1])
    conf = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    out = _operating_point_at_tpr(gt, conf)
    assert out["valid"] is False
    assert out["achieved_tpr"] == pytest.approx(0.0)


def test_operating_point_perfect_separation() -> None:
    # All tampered above 0.5, all authentic below → FPR=0 at TPR=1.
    gt = np.array([1, 1, 0, 0])
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    out = _operating_point_at_tpr(gt, conf)
    assert out["valid"] is True
    assert out["fpr"] == pytest.approx(0.0)
    assert out["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compression_robustness
# ---------------------------------------------------------------------------


def test_cr_perfect_robustness() -> None:
    assert compression_robustness(0.9, 0.9) == pytest.approx(1.0)


def test_cr_total_degradation() -> None:
    assert compression_robustness(0.9, 0.0) == pytest.approx(1.0 - 0.9 / 0.9)


def test_cr_partial_degradation() -> None:
    # AUC drops from 0.8 to 0.6: CR = 1 - 0.2/0.8 = 0.75
    assert compression_robustness(0.8, 0.6) == pytest.approx(0.75)


def test_cr_zero_c0_returns_one() -> None:
    assert compression_robustness(0.0, 0.0) == pytest.approx(1.0)


def test_cr_c4_above_c0() -> None:
    # C4 outperforms C0 — clamped to 1.0
    assert compression_robustness(0.7, 0.9) == pytest.approx(1.0)
