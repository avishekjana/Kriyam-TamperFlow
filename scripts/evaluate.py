#!/usr/bin/env python3
"""Evaluate a model's predictions against KriyamTamperFlow ground truth.

Usage
-----
python scripts/evaluate.py \\
    --predictions predictions/my_model/ \\
    --data-dir ./data \\
    --tiers C0 C2 C4 \\
    --report-out reports/my_model.html

The script expects:
- Annotation JSONs at ``<data-dir>/annotations/``
- Prediction JSONs at ``<predictions>/<sample_id>_<tier>.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Allow running the script directly without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from kriyam import report
from kriyam.loaders.local import load_dataset
from kriyam.metrics import aggregate, match_regions

_LOG = logging.getLogger("kriyam.evaluate")

_ALL_TIERS = ("C0", "C2", "C4")
DOCUMENT_THRESHOLD: float = 0.5
IOU_THRESHOLD: float = 0.1


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


def _load_prediction(path: Path) -> dict[str, Any] | None:
    """Load one prediction JSON file.

    Returns ``None`` and logs a warning if the file is absent or malformed.
    """
    if not path.is_file():
        _LOG.warning("Missing prediction file: %s", path)
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        _LOG.warning("Malformed prediction JSON %s: %s", path, exc)
        return None


def _pred_path(predictions_dir: Path, sample_id: str, tier: str) -> Path:
    """Return the expected prediction file path for a sample and tier."""
    return predictions_dir / f"{sample_id}_{tier}.json"


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------


def _score_sample(
    sample: dict[str, Any],
    pred: dict[str, Any],
    tier: str,
    document_threshold: float = DOCUMENT_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    """Produce a flat result dict for one (sample, tier) pair.

    Document-level prediction is derived from the predicted regions:
    - ``pred_confidence`` = max region confidence, or 0.0 if no regions.
    - ``pred_label`` = 1 if ``pred_confidence >= document_threshold`` else 0.

    Using a fixed threshold ensures every model is evaluated at the same
    operating point, making Doc-F1 and FPR directly comparable across models.
    Top-level ``confidence`` and ``is_authentic`` fields in the prediction JSON
    are ignored; region ``confidence`` is the only source of truth.

    Args:
        sample: Parsed annotation dict from :func:`load_dataset`.
        pred: Parsed prediction dict from the model's output folder.
        tier: Compression tier string (e.g. ``"C0"``).
        document_threshold: Confidence threshold for binary document prediction.
        iou_threshold: Minimum IoU for a region pair to count as a TP.

    Returns:
        A dict suitable for collection into a list and passing to
        :func:`kriyam.metrics.aggregate`.
    """
    pred_regions = list(pred.get("regions", []))
    pred_confidence = (
        max(float(r.get("confidence", 0.0)) for r in pred_regions)
        if pred_regions
        else 0.0
    )
    pred_label = int(pred_confidence >= document_threshold)

    gt_label = 0 if sample["is_authentic"] else 1

    region_result = match_regions(
        gt_regions=list(sample["regions"]),
        pred_regions=pred_regions,
        image_h=int(sample["image_h"]),
        image_w=int(sample["image_w"]),
        iou_threshold=iou_threshold,
    )

    return {
        "sample_id": sample["id"],
        "tier": tier,
        "gt_label": gt_label,
        "pred_label": pred_label,
        "pred_confidence": pred_confidence,
        **region_result,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


_TIER_LABELS = {
    "C0": "C0 — Pristine (no re-compression)",
    "C2": "C2 — Double-pass JPEG (Q=85 → Q=80)",
    "C4": "C4 — Photocopy simulation (blur + noise + Q=70)",
}


def _print_tier_table(tier: str, scores: dict[str, Any]) -> None:
    """Print a single-tier result table to stdout."""
    headers = ["Metric", "Value", "95% CI"]
    col_widths = [22, 10, 20]

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    label = _TIER_LABELS.get(tier, tier)
    print(f"\n  {label}")
    print(separator)
    print(header_row)
    print(separator)

    rows = [
        ("Region Precision", scores["region_precision"], scores["region_precision_ci"]),
        ("Region Recall",    scores["region_recall"],    scores["region_recall_ci"]),
        ("Region F1",        scores["region_f1"],        scores["region_f1_ci"]),
        ("Doc AUC-ROC",      scores["doc_auc"],          scores["doc_auc_ci"]),
        ("Doc F1",           scores["doc_f1"],           scores["doc_f1_ci"]),
        ("FPR",              scores["doc_fpr"],          scores["doc_fpr_ci"]),
    ]
    for name, value, ci in rows:
        lo, hi = ci
        ci_str = f"[{lo:.4f}, {hi:.4f}]"
        print(
            "| "
            + name.ljust(col_widths[0])
            + " | "
            + f"{value:.4f}".ljust(col_widths[1])
            + " | "
            + ci_str.ljust(col_widths[2])
            + " |"
        )

    n = scores.get("n_samples", "?")
    print(separator)
    print(f"  n = {n} samples")


def _print_tables(tier_scores: dict[str, dict[str, Any]]) -> None:
    """Print one result table per compression tier."""
    for tier in _ALL_TIERS:
        if tier in tier_scores:
            _print_tier_table(tier, tier_scores[tier])


def _write_scores(
    tier_scores: dict[str, dict[str, Any]],
    model_name: str,
    results_dir: Path,
    document_threshold: float = DOCUMENT_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> Path:
    """Serialise tier scores to ``results/<model_name>/scores.json``.

    CI tuples are converted to lists so they are valid JSON.

    Returns:
        The path of the written file.
    """
    out_dir = results_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.json"

    # Tuples are not JSON-serialisable; convert to lists.
    serialisable: dict[str, Any] = {}
    for tier, scores in tier_scores.items():
        serialisable[tier] = {
            k: list(v) if isinstance(v, tuple) else v for k, v in scores.items()
        }

    payload = {
        "model": model_name,
        "document_threshold": document_threshold,
        "iou_threshold": iou_threshold,
        "tiers": serialisable,
    }
    scores_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return scores_path


def _write_raw_results(
    tier_results: dict[str, list[dict[str, Any]]],
    model_name: str,
    results_dir: Path,
) -> Path:
    """Write per-image raw results to ``results/<model_name>/raw_results.jsonl``.

    Each line is a JSON object with the fields produced by ``_score_sample()``.
    These raw results can be re-aggregated at any document threshold without
    re-running the full evaluation — see ``scripts/report_from_cache.py``.

    Returns:
        The path of the written file.
    """
    out_dir = results_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    lines = []
    for results in tier_results.values():
        for row in results:
            lines.append(json.dumps(row))
    raw_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return raw_path


# ---------------------------------------------------------------------------
# Annotation cache management
# ---------------------------------------------------------------------------


def _ensure_annotations(annotations_dir: Path, token: str | None = None) -> None:
    """Download annotation files from HuggingFace if the local cache is empty.

    If ``annotations_dir`` already contains JSON files the function returns
    immediately — no network request is made.  On first run the annotations
    are fetched and written to ``annotations_dir`` so every subsequent run
    is fully offline.

    Args:
        annotations_dir: Expected location of the annotation JSONs
            (typically ``data/annotations/``).
        token: HuggingFace API token — not required for public datasets.

    Raises:
        SystemExit: If ``huggingface_hub`` is not installed and annotations
            are missing.
    """
    if annotations_dir.is_dir() and any(annotations_dir.glob("*.json")):
        return

    _LOG.info("No local annotations found — fetching from HuggingFace Hub …")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _LOG.error(
            "huggingface_hub is not installed and no local annotations exist.\n"
            "Fix (option 1): pip install huggingface-hub\n"
            "Fix (option 2): python scripts/download_data.py --only annotations",
        )
        sys.exit(1)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="kriyam-ai/kriyam-tamperflow",
        repo_type="dataset",
        local_dir=str(annotations_dir.parent),
        allow_patterns=["annotations/*.json"],
        token=token,
    )
    n = sum(1 for _ in annotations_dir.glob("*.json"))
    _LOG.info("Cached %d annotation(s) at %s", n, annotations_dir)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    predictions_dir: Path,
    data_dir: Path,
    tiers: list[str],
    report_out: Path,
    hf_token: str | None = None,
    document_threshold: float = DOCUMENT_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    report_template: str = "v1",
    workers: int = 2,
) -> dict[str, dict[str, Any]]:
    """Core evaluation routine.

    Loads annotations, scores every (sample, tier) pair that has a matching
    prediction file, aggregates by tier, writes ``scores.json``, and writes the
    HTML report.

    Args:
        predictions_dir: Folder containing the model's prediction JSONs.
        data_dir: Root of the benchmark data directory.
        tiers: List of tier codes to evaluate (subset of ``["C0","C2","C4"]``).
        report_out: Path for the output HTML report.

    Returns:
        Mapping from tier code to the aggregate score dict.
    """
    _ensure_annotations(data_dir / "annotations", token=hf_token)
    _LOG.info("Loading annotations from %s", data_dir / "annotations")
    samples = load_dataset(data_dir)
    if not samples:
        _LOG.error("No samples found. Check that annotations exist.")
        sys.exit(1)
    _LOG.info("Loaded %d annotation(s)", len(samples))

    # tier → list of per-image result dicts
    tier_results: dict[str, list[dict[str, Any]]] = {t: [] for t in tiers}
    skipped = 0
    total = len(samples) * len(tiers)

    with tqdm(total=total, unit="pred", desc="Scoring") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_tier: dict = {}
            for sample in samples:
                for tier in tiers:
                    path = _pred_path(predictions_dir, sample["id"], tier)
                    pred = _load_prediction(path)
                    if pred is None:
                        skipped += 1
                        pbar.update(1)
                        continue
                    fut = executor.submit(
                        _score_sample, sample, pred, tier,
                        document_threshold, iou_threshold,
                    )
                    future_to_tier[fut] = tier
            for fut in as_completed(future_to_tier):
                tier_results[future_to_tier[fut]].append(fut.result())
                pbar.update(1)

    if skipped:
        _LOG.warning("%d prediction file(s) missing or unreadable — skipped.", skipped)

    tier_scores: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        results = tier_results[tier]
        if not results:
            _LOG.warning("No scored samples for tier %s — skipping aggregation.", tier)
            continue
        authentic_in_results = sum(1 for r in results if r["gt_label"] == 0)
        if authentic_in_results == 0:
            _LOG.warning(
                "Tier %s: No authentic image predictions found (0 of %d authentic annotations). "
                "FPR will be 0.0 and Doc-AUC will fall back to 0.5 (one-class). "
                "Add prediction files for authentic images (with regions: []) to get valid FPR and Doc-AUC.",
                tier,
                sum(1 for s in samples if s["is_authentic"]),
            )
        _LOG.info("Aggregating %d result(s) for tier %s", len(results), tier)
        tier_scores[tier] = aggregate(results)

    if not tier_scores:
        _LOG.error("No results to aggregate. Check that prediction files exist.")
        sys.exit(1)

    model_name = predictions_dir.name
    results_root = Path("results")
    scores_path = _write_scores(
        tier_scores, model_name, results_root,
        document_threshold=document_threshold,
        iou_threshold=iou_threshold,
    )
    _LOG.info("Scores written to %s", scores_path)
    raw_path = _write_raw_results(tier_results, model_name, results_root)
    _LOG.info(
        "Raw per-image results written to %s "
        "(use report_from_cache.py to re-report at a different document threshold)",
        raw_path,
    )

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.generate(
        scores={"model": model_name, "document_threshold": document_threshold, "tiers": tier_scores},
        output_path=str(report_out),
        template=report_template,
    )
    _LOG.info("HTML report written to %s", report_out)

    return tier_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluate model predictions against KriyamTamperFlow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        required=True,
        metavar="DIR",
        help="Path to the model's prediction folder (e.g. predictions/my_model/).",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        metavar="DIR",
        help="Root of the benchmark data directory.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=list(_ALL_TIERS),
        default=list(_ALL_TIERS),
        metavar="TIER",
        help="Compression tiers to evaluate. Choices: C0 C2 C4.",
    )
    parser.add_argument(
        "--report-out",
        default="reports/report.html",
        metavar="PATH",
        help="Path for the output HTML report.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help=(
            "HuggingFace API token — only needed if annotations are not yet "
            "cached locally and the dataset repo is private."
        ),
    )
    parser.add_argument(
        "--report-template",
        choices=["v1", "v2", "v3"],
        default="v1",
        metavar="TEMPLATE",
        help=(
            "Report template: v1 (default, card-based layout), v2 "
            "(research-report style with full-width stacked charts), or v3 "
            "(academic paper style — serif type, booktabs tables, figure captions)."
        ),
    )
    parser.add_argument(
        "--document-threshold",
        type=float,
        default=DOCUMENT_THRESHOLD,
        metavar="FLOAT",
        help=(
            "Confidence threshold for binary document-level prediction. "
            "pred_label=1 if max(region confidences) >= threshold. "
            "Use the same value across all models for comparable Doc-F1 and FPR."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
        metavar="FLOAT",
        help=(
            "Minimum IoU for a predicted region to count as a true positive "
            "during Hungarian matching. Default is 0.1 per the benchmark spec. "
            "Changing this requires a full re-evaluation; it cannot be applied "
            "from cached raw results."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="N",
        help="Number of parallel worker threads for scoring. Default: 2.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    tier_scores = run_evaluation(
        predictions_dir=Path(args.predictions),
        data_dir=Path(args.data_dir),
        tiers=args.tiers,
        report_out=Path(args.report_out),
        hf_token=args.hf_token,
        document_threshold=args.document_threshold,
        iou_threshold=args.iou_threshold,
        report_template=args.report_template,
        workers=args.workers,
    )

    _print_tables(tier_scores)
    print()


if __name__ == "__main__":
    main()
