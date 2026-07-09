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


def _doc_metrics_bootstrap(
    confidences: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    n: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for doc_auc, doc_f1, and doc_fpr.

    Resamples the full dataset with replacement each iteration and recomputes
    the complete metric — the only statistically valid approach for metrics
    that are nonlinear functions of the sample (AUC, F1, FPR).

    Doc-F1 and FPR are derived directly from the binary *pred_labels*; no
    threshold search is performed inside the bootstrap loop.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    n_samples = len(confidences)
    auc_boot: list[float] = []
    f1_boot: list[float] = []
    fpr_boot: list[float] = []

    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        c_b = confidences[idx]
        g_b = gt_labels[idx]
        p_b = pred_labels[idx]

        # AUC — fall back to 0.5 when only one class survives the resample.
        if len(np.unique(g_b)) >= 2:
            auc_boot.append(float(roc_auc_score(g_b, c_b)))
        else:
            auc_boot.append(0.5)

        # Doc-F1 from binary labels.
        tp_b = int(((p_b == 1) & (g_b == 1)).sum())
        fp_b = int(((p_b == 1) & (g_b == 0)).sum())
        fn_b = int(((p_b == 0) & (g_b == 1)).sum())
        pr_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        rc_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        f1_boot.append(2 * pr_b * rc_b / (pr_b + rc_b) if (pr_b + rc_b) > 0 else 0.0)

        # FPR = FP / (FP + TN) from binary labels.
        tn_b = int(((p_b == 0) & (g_b == 0)).sum())
        fpr_boot.append(fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0)

    alpha = (1.0 - ci) / 2.0

    def _pct(arr: list[float]) -> tuple[float, float]:
        a = np.array(arr)
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {
        "auc_ci": _pct(auc_boot),
        "f1_ci": _pct(f1_boot),
        "fpr_ci": _pct(fpr_boot),
    }


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
        Dict with keys ``threshold``, ``achieved_tpr``, ``fpr``, ``f1``, ``valid``.
        ``valid`` is True iff ``achieved_tpr >= target_tpr``.
        When fewer than two classes are present, returns all-zero values with
        ``valid = False``.
    """
    if len(np.unique(gt_labels)) < 2:
        return {"threshold": 0.0, "achieved_tpr": 0.0, "fpr": 0.0, "f1": 0.0, "valid": False}

    fpr_arr, tpr_arr, thresh_arr = roc_curve(gt_labels, confidences)

    idxs = np.where(tpr_arr >= target_tpr)[0]
    if len(idxs) == 0:
        idx = int(tpr_arr.argmax())
        valid = False
    else:
        idx = int(idxs[0])
        valid = True

    achieved_tpr = float(tpr_arr[idx])
    fpr_val = float(fpr_arr[idx])
    threshold = float(thresh_arr[idx])

    pred_at_t = (confidences >= threshold).astype(int)
    tp = int(((pred_at_t == 1) & (gt_labels == 1)).sum())
    fp = int(((pred_at_t == 1) & (gt_labels == 0)).sum())
    fn = int(((pred_at_t == 0) & (gt_labels == 1)).sum())
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

    return {
        "threshold": threshold,
        "achieved_tpr": achieved_tpr,
        "fpr": fpr_val,
        "f1": f1,
        "valid": valid,
    }


def _operating_point_bootstrap(
    confidences: np.ndarray,
    gt_labels: np.ndarray,
    target_tpr: float = 0.9,
    n: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for doc_auprc, fpr_at_tpr90, and f1_at_tpr90.

    Mirrors the structure of :func:`_doc_metrics_bootstrap`.  Resamples the
    full dataset with replacement each iteration and recomputes each metric
    from scratch — the statistically valid approach for nonlinear metrics.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    n_samples = len(confidences)
    auprc_boot: list[float] = []
    fpr_boot: list[float] = []
    f1_boot: list[float] = []

    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        c_b = confidences[idx]
        g_b = gt_labels[idx]

        if len(np.unique(g_b)) < 2:
            auprc_boot.append(float(g_b.mean()))
            fpr_boot.append(0.0)
            f1_boot.append(0.0)
            continue

        auprc_boot.append(float(average_precision_score(g_b, c_b)))
        op = _operating_point_at_tpr(g_b, c_b, target_tpr)
        fpr_boot.append(op["fpr"])
        f1_boot.append(op["f1"])

    alpha = (1.0 - ci) / 2.0

    def _pct(arr: list[float]) -> tuple[float, float]:
        a = np.array(arr)
        return float(np.quantile(a, alpha)), float(np.quantile(a, 1.0 - alpha))

    return {
        "auprc_ci": _pct(auprc_boot),
        "fpr_at_tpr90_ci": _pct(fpr_boot),
        "f1_at_tpr90_ci": _pct(f1_boot),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image result dicts into macro-averaged benchmark scores.

    Each element of *results* must contain the keys returned by
    :func:`match_regions` (``region_precision``, ``region_recall``,
    ``region_f1``) plus ``gt_label`` (int, 1=tampered/0=authentic),
    ``pred_label`` (int, 1=tampered/0=authentic derived from predicted regions),
    and ``pred_confidence`` (float, max region confidence or 0.0 if no regions).

    Document-level AUC-ROC is computed with ``sklearn.metrics.roc_auc_score``
    over all samples.  All other metrics are macro-averaged (simple mean across
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
        - ``doc_f1``                / ``doc_f1_ci``
        - ``doc_fpr``               / ``doc_fpr_ci``
        - ``doc_fpr_at_tpr90``      / ``doc_fpr_at_tpr90_ci`` / ``doc_fpr_at_tpr90_valid``
        - ``doc_f1_at_tpr90``       / ``doc_f1_at_tpr90_ci``  / ``doc_f1_at_tpr90_valid``
        - ``doc_tpr90_achieved_tpr`` (float — actual TPR reached; equals target when valid)
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
    pred_labels = np.array([r["pred_label"] for r in results], dtype=np.int32)

    # Document-level AUC uses pred_confidence (max region confidence or 0.0).
    # Falls back to 0.5 when only one class is present (degenerate subset).
    n_classes = len(np.unique(gt_labels))
    doc_auc = float(roc_auc_score(gt_labels, confidences)) if n_classes >= 2 else 0.5

    # Doc-F1 and FPR from binary pred_labels — no threshold search.
    # pred_label is 1 if the model predicted any tampered regions, 0 otherwise.
    tp = int(((pred_labels == 1) & (gt_labels == 1)).sum())
    fp = int(((pred_labels == 1) & (gt_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (gt_labels == 1)).sum())
    tn = int(((pred_labels == 0) & (gt_labels == 0)).sum())
    doc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    doc_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    doc_f1  = 2 * doc_prec * doc_rec / (doc_prec + doc_rec) if (doc_prec + doc_rec) > 0 else 0.0
    doc_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    doc_auprc = (
        float(average_precision_score(gt_labels, confidences))
        if n_classes >= 2
        else float(gt_labels.mean())
    )

    op = _operating_point_at_tpr(gt_labels, confidences)
    doc_cis = _doc_metrics_bootstrap(confidences, gt_labels, pred_labels, rng=rng)
    op_cis = _operating_point_bootstrap(confidences, gt_labels, rng=rng)

    return {
        "region_precision": float(prec.mean()),
        "region_precision_ci": _bootstrap_ci(prec, rng=rng),
        "region_recall": float(rec.mean()),
        "region_recall_ci": _bootstrap_ci(rec, rng=rng),
        "region_f1": float(f1.mean()),
        "region_f1_ci": _bootstrap_ci(f1, rng=rng),
        "doc_auc": doc_auc,
        "doc_auc_ci": doc_cis["auc_ci"],
        "doc_auprc": doc_auprc,
        "doc_auprc_ci": op_cis["auprc_ci"],
        "doc_f1": doc_f1,
        "doc_f1_ci": doc_cis["f1_ci"],
        "doc_fpr": doc_fpr,
        "doc_fpr_ci": doc_cis["fpr_ci"],
        "doc_fpr_at_tpr90": op["fpr"],
        "doc_fpr_at_tpr90_ci": op_cis["fpr_at_tpr90_ci"],
        "doc_fpr_at_tpr90_valid": op["valid"],
        "doc_f1_at_tpr90": op["f1"],
        "doc_f1_at_tpr90_ci": op_cis["f1_at_tpr90_ci"],
        "doc_f1_at_tpr90_valid": op["valid"],
        "doc_tpr90_achieved_tpr": op["achieved_tpr"],
        "n_samples": len(results),
    }


# ---------------------------------------------------------------------------
# 6. compression_robustness
# ---------------------------------------------------------------------------


def compression_robustness(auc_c0: float, auc_c4: float) -> float:
    """Compute the Compression Robustness (CR) score.

    CR = 1 − (AUC_C0 − AUC_C4) / AUC_C0

    A CR of 1.0 means the model loses nothing across compression tiers.  Lower
    values indicate greater sensitivity to JPEG re-compression artefacts.

    Args:
        auc_c0: AUC-ROC on the pristine (C0) tier.
        auc_c4: AUC-ROC on the heavy-photocopy (C4) tier.

    Returns:
        The CR score as a float.  Returns ``1.0`` when ``auc_c0`` is zero to
        avoid division by zero (a model with 0 AUC at C0 has no signal to lose).
    """
    if auc_c0 == 0.0:
        return 1.0
    return min(1.0, 1.0 - (auc_c0 - auc_c4) / auc_c0)
