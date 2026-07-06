#!/usr/bin/env python3
"""Validate a prediction folder before evaluation.

Catches submission errors early so a full evaluate.py run does not silently
produce misleading scores.  Checks every expected (sample, tier) combination
and prints a per-file pass/fail summary followed by a final count.

Usage
-----
python scripts/validate_predictions.py \\
    --predictions predictions/my_model/ \\
    --data-dir    ./data \\
    --tiers       C0 C2 C4

Exit code 0 — all files valid.
Exit code 1 — one or more issues found.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from kriyam.loaders.local import ALL_TIERS, load_annotations

# ── ANSI colour codes (suppressed automatically on non-TTY) ─────────────────
_USE_COLOUR = sys.stdout.isatty()
_GREEN = "\033[32m" if _USE_COLOUR else ""
_RED   = "\033[31m" if _USE_COLOUR else ""
_YELLOW = "\033[33m" if _USE_COLOUR else ""
_BOLD  = "\033[1m"  if _USE_COLOUR else ""
_RESET = "\033[0m"  if _USE_COLOUR else ""

_PASS = f"{_GREEN}PASS{_RESET}"
_FAIL = f"{_RED}FAIL{_RESET}"
_WARN = f"{_YELLOW}WARN{_RESET}"


# ---------------------------------------------------------------------------
# Issue dataclass
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    """A single validation finding for one prediction file."""

    sample_id: str
    tier: str
    check: str
    message: str
    fatal: bool = True  # False → warning; True → counts as a failure


@dataclass
class FileResult:
    """Aggregated validation outcome for one (sample_id, tier) pair."""

    sample_id: str
    tier: str
    pred_path: Path
    issues: list[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.fatal for i in self.issues)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: frozenset[str] = frozenset({"id", "regions"})


def _check_file_exists(path: Path, sample_id: str, tier: str) -> Issue | None:
    """Check 1 — prediction file exists."""
    if not path.is_file():
        return Issue(sample_id, tier, "file_exists",
                     f"File not found: {path}", fatal=True)
    return None


def _check_json_parseable(path: Path, sample_id: str, tier: str) -> tuple[dict | None, Issue | None]:
    """Check 2 — JSON is valid and parseable.

    Returns the parsed dict on success, or (None, Issue) on failure.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None, Issue(sample_id, tier, "json_valid",
                               "Top-level value is not a JSON object", fatal=True)
        return data, None
    except json.JSONDecodeError as exc:
        return None, Issue(sample_id, tier, "json_valid",
                           f"JSON parse error: {exc}", fatal=True)
    except OSError as exc:
        return None, Issue(sample_id, tier, "json_valid",
                           f"Cannot read file: {exc}", fatal=True)


def _check_required_fields(data: dict, sample_id: str, tier: str) -> list[Issue]:
    """Check 3 — required top-level fields are present."""
    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        return [Issue(sample_id, tier, "required_fields",
                      f"Missing field(s): {sorted(missing)}", fatal=True)]
    return []


def _check_regions(
    data: dict,
    sample_id: str,
    tier: str,
    image_w: int,
    image_h: int,
) -> list[Issue]:
    """Checks 4, 5, 6 — region fields, coordinate bounds, duplicate ids."""
    regions = data.get("regions")
    issues: list[Issue] = []

    if not isinstance(regions, list):
        issues.append(Issue(sample_id, tier, "regions_type",
                            f"regions must be a list, got {type(regions).__name__}",
                            fatal=True))
        return issues

    seen_ids: set[Any] = set()

    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            issues.append(Issue(sample_id, tier, f"region[{idx}]_type",
                                f"Region {idx} is not an object", fatal=True))
            continue

        # Check 4 — required region fields.
        for key in ("x", "y", "w", "h", "confidence"):
            if key not in region:
                issues.append(Issue(sample_id, tier, f"region[{idx}]_fields",
                                    f"Region {idx} missing field '{key}'", fatal=True))

        # Check 6 — duplicate region_id (non-fatal: id is optional in predictions)
        rid = region.get("region_id")
        if rid is not None:
            if rid in seen_ids:
                issues.append(Issue(sample_id, tier, f"region[{idx}]_dup_id",
                                    f"Duplicate region_id {rid!r}", fatal=True))
            seen_ids.add(rid)

        # Check 5 — coordinate bounds (only when geometry fields are present)
        try:
            x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
        except (KeyError, TypeError, ValueError):
            continue  # geometry already flagged by check 5

        if w <= 0 or h <= 0:
            issues.append(Issue(sample_id, tier, f"region[{idx}]_dimensions",
                                f"Region {idx} has non-positive size: w={w}, h={h}",
                                fatal=True))

        if x < 0 or y < 0:
            issues.append(Issue(sample_id, tier, f"region[{idx}]_origin",
                                f"Region {idx} has negative origin: x={x}, y={y}",
                                fatal=True))

        if x + w > image_w or y + h > image_h:
            issues.append(Issue(sample_id, tier, f"region[{idx}]_bounds",
                                f"Region {idx} ({x},{y},{w},{h}) exceeds image "
                                f"dimensions {image_w}×{image_h}",
                                fatal=True))

    return issues


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------


def validate_file(
    pred_path: Path,
    sample_id: str,
    tier: str,
    image_w: int,
    image_h: int,
) -> FileResult:
    """Run all checks against one prediction file.

    Args:
        pred_path: Expected path to the prediction JSON.
        sample_id: Annotation sample id (e.g. ``"kriyam_0042"``).
        tier: Compression tier (e.g. ``"C0"``).
        image_w: Image width from the annotation (used for bounds checking).
        image_h: Image height from the annotation.

    Returns:
        A :class:`FileResult` with all issues collected.
    """
    result = FileResult(sample_id=sample_id, tier=tier, pred_path=pred_path)

    # Check 1
    issue = _check_file_exists(pred_path, sample_id, tier)
    if issue:
        result.issues.append(issue)
        return result  # cannot proceed without the file

    # Check 2
    data, issue = _check_json_parseable(pred_path, sample_id, tier)
    if issue:
        result.issues.append(issue)
        return result  # cannot proceed without parsed data

    # Check 3
    result.issues.extend(_check_required_fields(data, sample_id, tier))

    # Checks 4–6 only make sense when required fields are present
    if not result.issues:
        result.issues.extend(_check_regions(data, sample_id, tier, image_w, image_h))

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _pred_path(predictions_dir: Path, sample_id: str, tier: str) -> Path:
    return predictions_dir / f"{sample_id}_{tier}.json"


def _print_result(result: FileResult) -> None:
    """Print one line per check outcome for *result*."""
    label = _PASS if result.valid else _FAIL
    rel = result.pred_path.name
    print(f"  [{label}] {rel}")
    for issue in result.issues:
        marker = f"{_RED}✗{_RESET}" if issue.fatal else f"{_YELLOW}⚠{_RESET}"
        print(f"         {marker} [{issue.check}] {issue.message}")


def _print_summary(results: list[FileResult]) -> None:
    """Print the final X/Y valid + issue count line."""
    total = len(results)
    n_valid = sum(1 for r in results if r.valid)
    all_issues = [i for r in results for i in r.issues if i.fatal]
    n_issues = len(all_issues)

    colour = _GREEN if n_issues == 0 else _RED
    print()
    print(
        f"{_BOLD}{colour}{n_valid}/{total} files valid. "
        f"{n_issues} issue(s) found.{_RESET}"
    )


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------


def _ensure_annotations(annotations_dir: Path, token: str | None = None) -> None:
    """Download annotation files from HuggingFace if the local cache is empty.

    Mirrors the same function in ``evaluate.py`` — annotations are fetched on
    first use and cached locally so every subsequent run is offline.
    """
    if annotations_dir.is_dir() and any(annotations_dir.glob("*.json")):
        return

    print(f"  Annotations not found locally — fetching from HuggingFace …")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            f"{_RED}Error:{_RESET} huggingface_hub is not installed and annotations "
            f"are missing.\n"
            f"  Fix: pip install huggingface-hub\n"
            f"  Or:  python scripts/download_data.py --only annotations",
            file=sys.stderr,
        )
        sys.exit(1)

    annotations_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="kriyam-ai/kriyam-tamperflow",
            repo_type="dataset",
            local_dir=str(annotations_dir.parent),
            allow_patterns=["annotations/*.json"],
            token=token,
        )
    except Exception as exc:
        print(
            f"{_RED}Error:{_RESET} Failed to download annotations: {exc}\n"
            f"  Fix: python scripts/download_data.py --only annotations",
            file=sys.stderr,
        )
        sys.exit(1)
    n = sum(1 for _ in annotations_dir.glob("*.json"))
    print(f"  Cached {n} annotation(s) at {annotations_dir}")


def validate(
    predictions_dir: Path,
    data_dir: Path,
    tiers: list[str],
    hf_token: str | None = None,
) -> list[FileResult]:
    """Validate all expected prediction files in *predictions_dir*.

    Downloads annotations from HuggingFace automatically if they are not yet
    cached locally (same behaviour as ``evaluate.py``).  Every (sample, tier)
    pair is then checked, including region bounds against the true image
    dimensions from the annotation.

    Args:
        predictions_dir: Folder containing the model's prediction JSONs.
        data_dir: Root of the benchmark data directory.
        tiers: Tier codes to check (subset of ``ALL_TIERS``).
        hf_token: HuggingFace API token — only needed for private dataset repos.

    Returns:
        A list of :class:`FileResult` objects, one per (sample, tier) pair,
        in annotation-sort order.

    Raises:
        ValueError: If *tiers* contains an unrecognised tier code.
    """
    unknown = set(tiers) - set(ALL_TIERS)
    if unknown:
        raise ValueError(
            f"Unknown tier(s): {sorted(unknown)}. Valid tiers: {list(ALL_TIERS)}"
        )

    _ensure_annotations(data_dir / "annotations", token=hf_token)

    samples = load_annotations(data_dir)
    if not samples:
        print(f"{_WARN} No annotation files found in {data_dir / 'annotations'}.")
        return []

    results: list[FileResult] = []

    for sample in samples:
        for tier in tiers:
            path = _pred_path(predictions_dir, sample["id"], tier)
            result = validate_file(
                pred_path=path,
                sample_id=sample["id"],
                tier=tier,
                image_w=int(sample["image_w"]),
                image_h=int(sample["image_h"]),
            )
            results.append(result)
            _print_result(result)

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_predictions.py",
        description="Validate a prediction folder before evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions",
        required=True,
        metavar="DIR",
        help="Path to the model's prediction folder.",
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
        choices=list(ALL_TIERS),
        default=list(ALL_TIERS),
        metavar="TIER",
        help="Tiers to check. Choices: C0 C2 C4.",
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
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.  Exits with code 1 if any issues are found."""
    args = _build_parser().parse_args(argv)

    try:
        results = validate(
            predictions_dir=Path(args.predictions),
            data_dir=Path(args.data_dir),
            tiers=args.tiers,
            hf_token=args.hf_token,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    any_issues = any(i.fatal for r in results for i in r.issues)
    sys.exit(1 if any_issues else 0)


if __name__ == "__main__":
    main()
