#!/usr/bin/env python3
"""Re-generate an evaluation report from cached raw results at a new threshold.

Instead of re-running the full evaluation (15-20 min), this script reads the
``raw_results.jsonl`` file written by ``evaluate.py``, re-applies the document
threshold, re-aggregates metrics, and writes a new HTML report in seconds.

Usage
-----
python scripts/report_from_cache.py \\
    --results results/DTD-v2/ \\
    --document-threshold 0.3 \\
    --report-out reports/DTD-v2-t030.html

The only metrics that change with the threshold are Doc-F1 and FPR.
Doc-AUC and all region metrics (Region-P/R/F1) are always threshold-free
and will be identical to the original run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kriyam import report
from kriyam.metrics import aggregate

_LOG = logging.getLogger("kriyam.report_from_cache")

DOCUMENT_THRESHOLD: float = 0.5


def _load_raw_results(raw_path: Path) -> list[dict]:
    if not raw_path.is_file():
        _LOG.error("Raw results file not found: %s", raw_path)
        _LOG.error(
            "Re-run evaluate.py first — it now writes raw_results.jsonl automatically."
        )
        sys.exit(1)
    rows = []
    for i, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _LOG.warning("Skipping malformed line %d in %s: %s", i, raw_path, exc)
    return rows


def _reaggregate(
    raw_rows: list[dict],
    document_threshold: float,
    tiers: list[str],
) -> dict[str, dict]:
    """Re-apply threshold and aggregate per tier."""
    tier_results: dict[str, list[dict]] = {t: [] for t in tiers}

    for row in raw_rows:
        tier = row.get("tier", "")
        if tier not in tier_results:
            continue
        # Re-derive pred_label from the saved pred_confidence and the new threshold.
        pred_confidence = float(row.get("pred_confidence", 0.0))
        updated = dict(row)
        updated["pred_label"] = int(pred_confidence >= document_threshold)
        tier_results[tier].append(updated)

    tier_scores: dict[str, dict] = {}
    for tier in tiers:
        results = tier_results[tier]
        if not results:
            _LOG.warning("No results for tier %s in cache — skipping.", tier)
            continue
        _LOG.info("Aggregating %d result(s) for tier %s", len(results), tier)
        tier_scores[tier] = aggregate(results)

    return tier_scores


def _write_scores(
    tier_scores: dict[str, dict],
    model_name: str,
    results_dir: Path,
    document_threshold: float,
) -> Path:
    out_dir = results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"scores_t{int(document_threshold * 100):03d}.json"
    serialisable = {
        t: {k: list(v) if isinstance(v, tuple) else v for k, v in s.items()}
        for t, s in tier_scores.items()
    }
    payload = {"model": model_name, "document_threshold": document_threshold, "tiers": serialisable}
    scores_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return scores_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="report_from_cache.py",
        description="Re-generate a report from cached raw results at a new threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        required=True,
        metavar="DIR",
        help="Path to the model's results directory (e.g. results/DTD-v2/).",
    )
    parser.add_argument(
        "--document-threshold",
        type=float,
        default=DOCUMENT_THRESHOLD,
        metavar="FLOAT",
        help="New confidence threshold for binary document-level prediction.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["C0", "C2", "C4"],
        metavar="TIER",
        help="Compression tiers to include.",
    )
    parser.add_argument(
        "--report-out",
        required=True,
        metavar="PATH",
        help="Output path for the new HTML report.",
    )
    parser.add_argument(
        "--report-template",
        choices=["v1", "v2", "v3"],
        default="v1",
        help="Report template.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    results_dir = Path(args.results)
    raw_path = results_dir / "raw_results.jsonl"

    raw_rows = _load_raw_results(raw_path)
    _LOG.info("Loaded %d raw result rows from %s", len(raw_rows), raw_path)

    tier_scores = _reaggregate(raw_rows, args.document_threshold, args.tiers)
    if not tier_scores:
        _LOG.error("No tier scores produced. Check that raw_results.jsonl is not empty.")
        sys.exit(1)

    model_name = results_dir.name
    scores_path = _write_scores(tier_scores, model_name, results_dir, args.document_threshold)
    _LOG.info("Scores written to %s", scores_path)

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report.generate(
        scores={"model": model_name, "document_threshold": args.document_threshold, "tiers": tier_scores},
        output_path=str(report_out),
        template=args.report_template,
    )
    _LOG.info("Report written to %s", report_out)

    for tier, s in tier_scores.items():
        print(f"  {tier}  Doc-AUC={s['doc_auc']:.4f}  Doc-F1={s['doc_f1']:.4f}  FPR={s['doc_fpr']:.4f}")


if __name__ == "__main__":
    main()
