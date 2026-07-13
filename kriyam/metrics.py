"""Evaluation metrics for document tampering detection.

All region-level inputs accept plain dicts with at minimum ``x``, ``y``,
``w``, ``h`` integer keys (matching both annotation RegionDicts and
prediction region dicts).  This keeps the functions usable without
importing the loader types.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# Type alias used throughout this module.
_Region = dict[str, Any]


# ---------------------------------------------------------------------------
# 1. bboxes_to_mask
# ---------------------------------------------------------------------------


def bboxes_to_mask(regions: list[_Region], image_h: int, image_w: int) -> np.ndarray:
    """Convert a list of bounding-box regions to a binary H×W mask.

    Each region must contain integer keys ``x``, ``y``, ``w``, ``h`` describing
    a rectangle in pixel coordinates where ``(x, y)`` is the top-left corner.
    Coordinates are clamped to the image boundary before filling, so out-of-
    bounds boxes do not raise an error.

    Args:
        regions: List of region dicts.  May be empty.
        image_h: Image height in pixels.
        image_w: Image width in pixels.

    Returns:
        A ``uint8`` NumPy array of shape ``(image_h, image_w)`` where pixels
        inside at least one bounding box are ``1`` and all other pixels are
        ``0``.
    """
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    for r in regions:
        x0 = int(np.clip(r["x"], 0, image_w))
        y0 = int(np.clip(r["y"], 0, image_h))
        x1 = int(np.clip(r["x"] + r["w"], 0, image_w))
        y1 = int(np.clip(r["y"] + r["h"], 0, image_h))
        mask[y0:y1, x0:x1] = 1
    return mask


# ---------------------------------------------------------------------------
# 2. iou_matrix
# ---------------------------------------------------------------------------


def _bbox_iou(a: _Region, b: _Region, image_h: int, image_w: int) -> float:
    """Compute IoU between two single bounding boxes via mask overlap."""
    mask_a = bboxes_to_mask([a], image_h, image_w).astype(bool)
    mask_b = bboxes_to_mask([b], image_h, image_w).astype(bool)
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return intersection / union if union > 0 else 0.0


def iou_matrix(
    gt_regions: list[_Region],
    pred_regions: list[_Region],
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Build a GT×Pred matrix of pairwise bounding-box IoU scores.

    Args:
        gt_regions: Ground-truth region dicts with ``x``, ``y``, ``w``, ``h``.
        pred_regions: Predicted region dicts with ``x``, ``y``, ``w``, ``h``.
        image_h: Image height in pixels.
        image_w: Image width in pixels.

    Returns:
        A float64 NumPy array of shape ``(len(gt_regions), len(pred_regions))``.
        Returns an empty array of shape ``(0, 0)`` if either list is empty.
    """
    n_gt = len(gt_regions)
    n_pred = len(pred_regions)
    matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, gt in enumerate(gt_regions):
        for j, pred in enumerate(pred_regions):
            matrix[i, j] = _bbox_iou(gt, pred, image_h, image_w)
    return matrix


# ---------------------------------------------------------------------------
# 3. match_regions
# ---------------------------------------------------------------------------


def match_regions(
    gt_regions: list[_Region],
    pred_regions: list[_Region],
    image_h: int,
    image_w: int,
    iou_threshold: float = 0.1,
) -> dict[str, float]:
    """Match predicted regions to ground-truth regions using the Hungarian algorithm.

    Builds the IoU matrix, negates it (``linear_sum_assignment`` minimises cost),
    and applies the IoU threshold to decide which matched pairs count as true
    positives.  Unmatched GT regions are false negatives; unmatched predictions
    are false positives.

    Args:
        gt_regions: Ground-truth region dicts.
        pred_regions: Predicted region dicts.
        image_h: Image height in pixels.
        image_w: Image width in pixels.
        iou_threshold: Minimum IoU for a matched pair to be counted as a TP.
            Defaults to ``0.1`` per the benchmark spec.

    Returns:
        A dict with keys:

        - ``tp`` (int): true positives
        - ``fp`` (int): false positives
        - ``fn`` (int): false negatives
        - ``region_precision`` (float): TP / (TP + FP), or 0 if no predictions
        - ``region_recall`` (float): TP / (TP + FN), or 0 if no ground-truth regions
        - ``region_f1`` (float): harmonic mean of precision and recall
    """
    n_gt = len(gt_regions)
    n_pred = len(pred_regions)

    tp = 0
    if n_gt > 0 and n_pred > 0:
        matrix = iou_matrix(gt_regions, pred_regions, image_h, image_w)
        row_ind, col_ind = linear_sum_assignment(-matrix)
        tp = int(sum(matrix[r, c] >= iou_threshold for r, c in zip(row_ind, col_ind)))

    fp = n_pred - tp
    fn = n_gt - tp

    region_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    region_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = region_precision + region_recall
    region_f1 = 2 * region_precision * region_recall / denom if denom > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "region_precision": region_precision,
        "region_recall": region_recall,
        "region_f1": region_f1,
    }


# ---------------------------------------------------------------------------
# 4. aggregate
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: np.ndarray,
    n: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for the mean of *values*.

    Args:
        values: 1-D array of scalar values.
        n: Number of bootstrap resamples.
        ci: Coverage probability (e.g. ``0.95`` for 95 % CI).
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        A ``(lower, upper)`` tuple.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n)]
    )
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return lower, upper


def _operating_point_at_tpr(
    gt_labels: np.ndarray,
    confidences: np.ndarray,
    target_tpr: float = 0.9,
) -> dict[str, Any]:
    """Return the operating point where TPR first reaches *target_tpr*.

    Finds the smallest index in the ROC curve where ``tpr[i] >= target_tpr``
    (the highest threshold that still achieves the target recall), giving the
    lowest FPR at that operating point.

    Note on AUPRC: ``average_precision_score`` uses a random baseline equal to
    the positive class prior, not 0.5.  For this benchmark (~67 % tampered) the
    random AUPRC baseline is ≈ 0.67, not 0.50.

    Args:
        gt_labels:  1-D int array (1 = tampered, 0 = authentic).
        confidences: 1-D float array of predicted confidences.
        target_tpr: TPR level to target (default 0.9 for FPR@TPR90).

    Returns:
        Dict with keys ``threshold``, ``achieved_tpr``, ``fpr``, and ``valid``.
        ``valid`` is True iff ``achieved_tpr >= target_tpr``.
        When fewer than two classes are present, returns all-zero values with
        ``valid = False``.
    """
    if len(np.unique(gt_labels)) < 2:
        return {"threshold": 0.0, "achieved_tpr": 0.0, "fpr": 0.0, "valid": False}

    fpr_arr, tpr_arr, thresh_arr = roc_curve(gt_labels, confidences)

    idxs = np.where(tpr_arr >= target_tpr)[0]
    if len(idxs) == 0:
        idx = int(tpr_arr.argmax())
        valid = False
    else:
        idx = int(idxs[0])
        valid = True

    return {
        "threshold": float(thresh_arr[idx]),
        "achieved_tpr": float(tpr_arr[idx]),
        "fpr": float(fpr_arr[idx]),
        "valid": valid,
    }


def _operating_point_bootstrap(
    confidences: np.ndarray,
    gt_labels: np.ndarray,
    tpr_targets: tuple[float, ...] = (0.8, 0.85, 0.9),
    n: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for doc_auc, doc_auprc, and FPR at each TPR target.

    A single bootstrap loop computes all metrics simultaneously to avoid
    re-sampling the dataset once per metric.  Resamples the full dataset with
    replacement each iteration — the statistically valid approach for nonlinear
    metrics such as AUC, AUPRC, and FPR@TPR.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    n_samples = len(confidences)
    auc_boot: list[float] = []
    auprc_boot: list[float] = []
    fpr_boots: dict[float, list[float]] = {t: [] for t in tpr_targets}

    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        c_b = confidences[idx]
        g_b = gt_labels[idx]

        if len(np.unique(g_b)) < 2:
            auc_boot.append(0.5)
            auprc_boot.append(float(g_b.mean()))
            for t in tpr_targets:
                fpr_boots[t].append(0.0)
            continue

        auc_boot.append(float(roc_auc_score(g_b, c_b)))
        auprc_boot.append(float(average_precision_score(g_b, c_b)))
        for t in tpr_targets:
            op = _operating_point_at_tpr(g_b, c_b, t)
            fpr_boots[t].append(op["fpr"])

    alpha = (1.0 - ci) / 2.0

    def _pct(arr: list[float]) -> tuple[float, float]:
        a = np.array(arr)
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    result: dict[str, tuple[float, float]] = {
        "auc_ci": _pct(auc_boot),
        "auprc_ci": _pct(auprc_boot),
    }
    for t in tpr_targets:
        result[f"fpr_at_tpr{int(t * 100)}_ci"] = _pct(fpr_boots[t])
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image result dicts into macro-averaged benchmark scores.

    Each element of *results* must contain the keys returned by
    :func:`match_regions` (``region_precision``, ``region_recall``,
    ``region_f1``) plus ``gt_label`` (int, 1=tampered/0=authentic) and
    ``pred_confidence`` (float, max region confidence or 0.0 if no regions).

    Document-level AUC-ROC and AUPRC are computed with ``sklearn.metrics`` over
    all samples.  All other metrics are macro-averaged (simple mean across
    images).  Bootstrap 95 % confidence intervals (n=1 000) are reported for
    every scalar metric.

    Args:
        results: List of per-image result dicts.  Must be non-empty.

    Returns:
        A dict with the following keys (each ``*_ci`` entry is a
        ``(lower, upper)`` float tuple):

        - ``region_precision`` / ``region_precision_ci``
        - ``region_recall``    / ``region_recall_ci``
        - ``region_f1``        / ``region_f1_ci``
        - ``doc_auc``               / ``doc_auc_ci``
        - ``doc_auprc``             / ``doc_auprc_ci``
        - ``doc_fpr_at_tpr80``      / ``doc_fpr_at_tpr80_ci`` / ``doc_fpr_at_tpr80_valid``
        - ``doc_tpr80_achieved_tpr`` (float — actual TPR reached; equals target when valid)
        - ``doc_fpr_at_tpr85``      / ``doc_fpr_at_tpr85_ci`` / ``doc_fpr_at_tpr85_valid``
        - ``doc_tpr85_achieved_tpr``
        - ``doc_fpr_at_tpr90``      / ``doc_fpr_at_tpr90_ci`` / ``doc_fpr_at_tpr90_valid``
        - ``doc_tpr90_achieved_tpr``
        - ``n_samples`` (int)

    Raises:
        ValueError: If *results* is empty.
    """
    if not results:
        raise ValueError("results list must not be empty")

    rng = np.random.default_rng(seed=42)

    prec = np.array([r["region_precision"] for r in results], dtype=np.float64)
    rec = np.array([r["region_recall"] for r in results], dtype=np.float64)
    f1 = np.array([r["region_f1"] for r in results], dtype=np.float64)

    confidences = np.array([r["pred_confidence"] for r in results], dtype=np.float64)
    gt_labels = np.array([r["gt_label"] for r in results], dtype=np.int32)

    # Document-level AUC and AUPRC use pred_confidence (max region confidence or 0.0).
    # Both fall back to degenerate values when only one class is present.
    n_classes = len(np.unique(gt_labels))
    doc_auc = float(roc_auc_score(gt_labels, confidences)) if n_classes >= 2 else 0.5
    doc_auprc = (
        float(average_precision_score(gt_labels, confidences))
        if n_classes >= 2
        else float(gt_labels.mean())
    )

    op80 = _operating_point_at_tpr(gt_labels, confidences, 0.8)
    op85 = _operating_point_at_tpr(gt_labels, confidences, 0.85)
    op90 = _operating_point_at_tpr(gt_labels, confidences, 0.9)

    boot = _operating_point_bootstrap(confidences, gt_labels, rng=rng)

    return {
        "region_precision": float(prec.mean()),
        "region_precision_ci": _bootstrap_ci(prec, rng=rng),
        "region_recall": float(rec.mean()),
        "region_recall_ci": _bootstrap_ci(rec, rng=rng),
        "region_f1": float(f1.mean()),
        "region_f1_ci": _bootstrap_ci(f1, rng=rng),
        "doc_auc": doc_auc,
        "doc_auc_ci": boot["auc_ci"],
        "doc_auprc": doc_auprc,
        "doc_auprc_ci": boot["auprc_ci"],
        "doc_fpr_at_tpr80": op80["fpr"],
        "doc_fpr_at_tpr80_ci": boot["fpr_at_tpr80_ci"],
        "doc_fpr_at_tpr80_valid": op80["valid"],
        "doc_tpr80_achieved_tpr": op80["achieved_tpr"],
        "doc_fpr_at_tpr85": op85["fpr"],
        "doc_fpr_at_tpr85_ci": boot["fpr_at_tpr85_ci"],
        "doc_fpr_at_tpr85_valid": op85["valid"],
        "doc_tpr85_achieved_tpr": op85["achieved_tpr"],
        "doc_fpr_at_tpr90": op90["fpr"],
        "doc_fpr_at_tpr90_ci": boot["fpr_at_tpr90_ci"],
        "doc_fpr_at_tpr90_valid": op90["valid"],
        "doc_tpr90_achieved_tpr": op90["achieved_tpr"],
        "n_samples": len(results),
    }


# ---------------------------------------------------------------------------
# 6. compression_robustness
# ---------------------------------------------------------------------------


def compression_robustness(
    c0: float,
    c4: float,
    min_c0: float = 0.5,
    metric_name: str = "metric",
) -> float | None:
    """Compute the Compression Robustness (CR) score.

    CR = c4 / c0 (clamped to 1.0).  Returns ``None`` when ``c0 < min_c0``,
    indicating that C0 performance is too low to serve as a meaningful baseline
    for robustness measurement.

    The ``min_c0`` floor should be set to a metric-specific lower bound:

    - AUC: ``0.5`` (random baseline)
    - AUPRC: ``≈ 0.667`` (positive-class prior for this benchmark's 700/350 split)
    - Region-F1: ``0.05`` (very weak localisation still yields non-trivial CR)

    Args:
        c0: Metric value on the pristine (C0) tier.
        c4: Metric value on the heavy-photocopy (C4) tier.
        min_c0: Minimum acceptable C0 value below which ``None`` is returned.
            Defaults to ``0.5``.
        metric_name: Human-readable metric name (used in documentation only).

    Returns:
        CR score as a float in [0, 1], or ``None`` when ``c0 < min_c0``.
    """
    if c0 < min_c0:
        return None
    return min(1.0, c4 / c0)
