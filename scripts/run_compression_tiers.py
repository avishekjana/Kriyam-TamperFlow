#!/usr/bin/env python3
"""Generate C2 and C4 compression-tier variants from a folder of C0 images.

Usage
-----
python scripts/run_compression_tiers.py \\
    --input-dir  data/images/ \\
    --output-dir data/images/ \\
    --tiers C2 C4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from kriyam.compression import TIERS, apply_tier

_LOG = logging.getLogger("kriyam.run_compression_tiers")

# The only tier that acts as a source for compression variants.
_SOURCE_TIER = "C0"
_SOURCE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _output_path(source: Path, tier: str, output_dir: Path) -> Path:
    """Derive the output path for *source* at *tier*.

    Replaces the ``_C0`` suffix in the stem with ``_<tier>`` and places
    the result in *output_dir* as a ``.png`` file.

    Args:
        source: Path to a ``*_C0.{png,jpg,jpeg}`` source image.
        tier: Target tier code, e.g. ``"C2"``.
        output_dir: Directory where the output file will be written.

    Returns:
        The full output path.
    """
    new_stem = source.stem[: -len(_SOURCE_TIER)] + tier
    return output_dir / f"{new_stem}.png"


def _discover_sources(input_dir: Path) -> list[Path]:
    """Return all ``*_C0.{png,jpg,jpeg}`` files in *input_dir*, sorted by name.

    Args:
        input_dir: Directory to search.

    Returns:
        Sorted list of matching paths.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    sources: list[Path] = []
    for ext in _SOURCE_EXTENSIONS:
        sources.extend(input_dir.glob(f"*_{_SOURCE_TIER}{ext}"))
    if not sources:
        _LOG.warning(
            "No *_%s.{png,jpg,jpeg} files found in %s", _SOURCE_TIER, input_dir
        )
    return sorted(sources)


def run(
    input_dir: Path,
    output_dir: Path,
    tiers: list[str],
) -> dict[str, int]:
    """Process all C0 images in *input_dir* and write tier variants to *output_dir*.

    Args:
        input_dir: Directory containing ``*_C0.png`` source images.
        output_dir: Destination directory.  Created if it does not exist.
        tiers: Tier codes to generate (e.g. ``["C2", "C4"]``).

    Returns:
        A summary dict with keys ``"images"`` (number of C0 sources found),
        ``"variants"`` (total files written), and ``"skipped"`` (files that
        already existed and were left unchanged).

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
        ValueError: If *tiers* contains an unrecognised tier code.
    """
    unknown = set(tiers) - set(TIERS)
    if unknown:
        raise ValueError(
            f"Unknown tier(s): {sorted(unknown)}. Valid tiers: {list(TIERS)}"
        )
    if _SOURCE_TIER in tiers:
        raise ValueError(
            f"Cannot generate '{_SOURCE_TIER}' as a derivative tier — "
            f"it is the source tier."
        )

    sources = _discover_sources(input_dir)
    if not sources:
        _LOG.warning(
            "No *_%s.{png,jpg,jpeg} files found in %s", _SOURCE_TIER, input_dir
        )
        return {"images": 0, "variants": 0, "skipped": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    total_work = len(sources) * len(tiers)
    variants_written = 0
    skipped = 0

    with tqdm(total=total_work, unit="file", desc="Generating tiers") as pbar:
        for source in sources:
            outputs = {tier: _output_path(source, tier, output_dir) for tier in tiers}

            # Skip the source entirely when every requested variant already exists.
            if all(dst.exists() for dst in outputs.values()):
                _LOG.debug("All variants present, skipping: %s", source.name)
                skipped += len(tiers)
                pbar.update(len(tiers))
                continue

            for tier, dst in outputs.items():
                pbar.set_postfix(file=source.name, tier=tier, refresh=False)

                if dst.exists():
                    _LOG.debug("Skipping existing file: %s", dst)
                    skipped += 1
                    pbar.update(1)
                    continue

                try:
                    apply_tier(str(source), tier, str(dst))
                    variants_written += 1
                except Exception as exc:
                    _LOG.error("Failed to process %s → %s: %s", source.name, tier, exc)
                finally:
                    pbar.update(1)

    return {
        "images": len(sources),
        "variants": variants_written,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_compression_tiers.py",
        description="Generate C2/C4 compression-tier variants from C0 source images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Folder containing *_C0.{png,jpg,jpeg} source images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Where to write the generated variants. "
            "Defaults to the same directory as --input-dir."
        ),
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["C2", "C4"],
        choices=[t for t in TIERS if t != _SOURCE_TIER],
        metavar="TIER",
        help="Tier(s) to generate. Choices: C2, C4.",
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
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s  %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    try:
        summary = run(input_dir=input_dir, output_dir=output_dir, tiers=args.tiers)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    tiers_str = " ".join(args.tiers)
    print(
        f"\n{summary['images']} image(s) processed, "
        f"{summary['variants']} variant(s) written ({tiers_str}), "
        f"{summary['skipped']} skipped."
    )
    print(f"Output written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
