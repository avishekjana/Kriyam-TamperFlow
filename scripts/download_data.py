#!/usr/bin/env python3
"""Download Kriyam benchmark images from HuggingFace Hub into data/images/.

Annotations are NOT downloaded here — evaluate.py fetches them on-demand
and caches them in data/annotations/ automatically.

Usage
-----
python scripts/download_data.py                     # all tiers (C0, C2, C4)
python scripts/download_data.py --tiers C0          # pristine only
python scripts/download_data.py --tiers C0 C2       # two tiers
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_LOG = logging.getLogger("kriyam.download_data")

HF_REPO_ID = "kriyam-ai/kriyam-tamperflow"
TIERS: tuple[str, ...] = ("C0", "C2", "C4")


_EXPECTED_PER_TIER = 1050   # 1050 source documents × 1 image per tier
_RATE_LIMIT_WAIT = 310     # seconds to wait after a 429 (HF resets every 5 min)
_MAX_RETRIES = 5


def _count_tier(img_dir: Path, tier: str) -> int:
    if not img_dir.is_dir():
        return 0
    # C0 images may be stored as JPG/JPEG (scanned originals) or PNG; C2/C4 are always PNG.
    png = sum(1 for _ in img_dir.glob(f"*_{tier}.png"))
    if tier == "C0":
        jpg  = sum(1 for _ in img_dir.glob(f"*_{tier}.jpg"))
        jpeg = sum(1 for _ in img_dir.glob(f"*_{tier}.jpeg"))
        return png + jpg + jpeg
    return png


def _print_summary(img_dir: Path, tiers: list[str]) -> bool:
    """Print per-tier counts vs expected. Returns True when all tiers are complete."""
    tier_counts = {t: _count_tier(img_dir, t) for t in tiers}
    total = sum(tier_counts.values())
    total_expected = _EXPECTED_PER_TIER * len(tiers)

    print()
    print(f"  {'Tier':<6}  {'Downloaded':>12}  {'Expected':>9}  Status")
    print(f"  {'─'*6}  {'─'*12}  {'─'*9}  {'─'*8}")
    all_ok = True
    for tier in tiers:
        n = tier_counts[tier]
        ok = n >= _EXPECTED_PER_TIER
        if not ok:
            all_ok = False
        status = "OK" if ok else f"MISSING {_EXPECTED_PER_TIER - n}"
        print(f"  {tier:<6}  {n:>12,}  {_EXPECTED_PER_TIER:>9,}  {status}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*9}")
    print(f"  {'Total':<6}  {total:>12,}  {total_expected:>9,}")
    return all_ok


def download(
    data_dir: Path,
    tiers: list[str],
    token: str | None = None,
) -> None:
    """Download benchmark images and the master index from HuggingFace.

    Resumes automatically from already-downloaded files, and retries on 429
    rate-limit errors with a countdown before each attempt.

    Args:
        data_dir: Local data root (images land in ``data_dir/images/``).
        tiers: Compression tiers to fetch (subset of C0, C2, C4).
        token: HuggingFace API token — not required for public datasets.

    Raises:
        SystemExit: If ``huggingface_hub`` is not installed or all retries fail.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        print(
            "Error: huggingface_hub is not installed.\n"
            "Fix: pip install huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / "images"

    total_expected = _EXPECTED_PER_TIER * len(tiers)
    tier_str = ", ".join(tiers)
    print(
        f"Downloading Kriyam TamperFlow images\n"
        f"  Tiers    : {tier_str}\n"
        f"  Expected : {total_expected:,} images ({_EXPECTED_PER_TIER:,} per tier)\n"
        f"  Source   : {HF_REPO_ID}\n"
        f"  Dest     : {img_dir.resolve()}\n"
    )

    # One glob pattern per tier so users can download a subset cheaply.
    # C0 images may be JPG/JPEG (scanned originals) or PNG; C2/C4 are always PNG.
    allow_patterns = [f"images/*_{tier}.png" for tier in tiers]
    if "C0" in tiers:
        allow_patterns.append("images/*_C0.jpg")
        allow_patterns.append("images/*_C0.jpeg")
    allow_patterns.append("metadata.jsonl")

    for attempt in range(1, _MAX_RETRIES + 1):
        already = sum(_count_tier(img_dir, t) for t in tiers)
        if attempt > 1:
            print(f"\nRetry {attempt}/{_MAX_RETRIES} — resuming from {already:,} / {total_expected:,} files already cached …\n")
        try:
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                local_dir=str(data_dir),
                allow_patterns=allow_patterns,
                ignore_patterns=["annotations/*"],
                token=token,
            )
            break  # success

        except HfHubHTTPError as exc:
            if "429" not in str(exc) and "Too Many Requests" not in str(exc):
                print(f"\nError: {exc}", file=sys.stderr)
                sys.exit(1)

            cached = sum(_count_tier(img_dir, t) for t in tiers)
            print(
                f"\nRate limited by HuggingFace ({cached:,} / {total_expected:,} files downloaded so far).\n"
                f"Progress is saved — the download will resume from where it stopped.\n"
                f"Waiting {_RATE_LIMIT_WAIT // 60} min {_RATE_LIMIT_WAIT % 60} sec before retrying …"
            )
            if attempt == _MAX_RETRIES:
                print(
                    f"\nAll {_MAX_RETRIES} retries exhausted. Run this script again to continue — "
                    f"the {cached:,} files already downloaded will not be re-fetched.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Live countdown so the user sees progress.
            for remaining in range(_RATE_LIMIT_WAIT, 0, -1):
                print(f"\r  Resuming in {remaining:3d}s …", end="", flush=True)
                time.sleep(1)
            print()

    # ── Post-download summary ────────────────────────────────────────────────
    print("\nDownload complete")
    all_ok = _print_summary(img_dir, tiers)

    if not all_ok:
        print(
            "\nSome files are missing. Re-run this script — already-downloaded "
            "files will be skipped and only the missing ones fetched.",
            file=sys.stderr,
        )
    else:
        print(f"\nAll images saved to: {img_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_data.py",
        description="Download Kriyam benchmark images from HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=list(TIERS),
        default=list(TIERS),
        metavar="TIER",
        help="Compression tiers to download. Choices: C0 C2 C4.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        metavar="DIR",
        help="Local data root. Images are written to <data-dir>/images/.",
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace API token (not required for public datasets).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    download(data_dir=Path(args.data_dir), tiers=args.tiers, token=args.token)


if __name__ == "__main__":
    main()
