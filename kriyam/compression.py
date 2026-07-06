"""JPEG/PNG compression pipeline for benchmark tier simulation.

Tiers (section 3.5 of the benchmark spec):
  C0 — lossless copy; no re-compression.
  C2 — moderate digital-sharing simulation: resize to 96 % → resize back to
       original → JPEG Q=85 → JPEG Q=80 → PNG.  The downsample-upsample
       round-trip introduces Lanczos interpolation artefacts before JPEG
       encoding.  Output dimensions match C0.
  C4 — heavy photocopy/scan simulation: resize to 96 % → resize back to
       original → JPEG Q=85 → JPEG Q=75 → BMP round-trip → Gaussian blur
       (r=0.5) → Gaussian noise (σ=4) → JPEG Q=70 → PNG.  Simulates a full
       print-scan-share cycle that nearly erases DCT-based forensic signals.
       Output dimensions match C0.

Design notes:
  * Both C2 and C4 output at the original C0 dimensions.  The resize
    round-trips are done at intermediate stages to introduce interpolation
    artefacts before compression.
  * All intermediate data is kept in memory via ``BytesIO``; only the final
    PNG is written to disk.
  * Noise generation uses a fixed seed (42) so every tier is deterministic.
"""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Valid tier identifiers.
TIERS: tuple[str, ...] = ("C0", "C2", "C4")

# Both C2 and C4 use the same downsample factor before re-upsampling.
_RESIZE_SCALE: float = 0.96  # 96 % — matches real messaging-app resampling behaviour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode *img* as JPEG at *quality* and decode back into a new Image.

    The round-trip is done entirely in memory.  The image is converted to RGB
    before encoding because JPEG does not support an alpha channel; if the
    source already is RGB the conversion is a no-op.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    return Image.open(buf).copy()  # .copy() detaches from the BytesIO buffer


def _bmp_roundtrip(img: Image.Image) -> Image.Image:
    """Encode *img* as BMP and decode back; strips any metadata."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="BMP")
    buf.seek(0)
    return Image.open(buf).copy()


def _add_gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Add zero-mean Gaussian noise with standard deviation *sigma* (pixel units).

    Uses a fixed-seed RNG local to this call so the function is deterministic
    when called on identical inputs within the same process.  The result is
    clipped to [0, 255] before converting back to uint8.
    """
    rng = np.random.default_rng(seed=42)
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize *img* to *size* (width, height) using Lanczos resampling."""
    return img.resize(size, Image.LANCZOS)


def _apply_c2(img: Image.Image) -> Image.Image:
    """Moderate digital-sharing simulation.

    Pipeline: resize to 96 % → resize back to original → JPEG Q=85 →
    JPEG Q=80 → (saved as PNG by caller).

    The downsample-upsample round-trip introduces Lanczos interpolation
    artefacts that mimic resampling by messaging apps, after which two JPEG
    passes simulate repeated digital sharing.  Output dimensions match C0.
    """
    orig_size = img.size  # (width, height)
    small_size = (round(orig_size[0] * _RESIZE_SCALE), round(orig_size[1] * _RESIZE_SCALE))

    img = _resize(img.convert("RGB"), small_size)
    img = _resize(img, orig_size)
    img = _jpeg_roundtrip(img, quality=85)
    img = _jpeg_roundtrip(img, quality=80)
    return img


def _apply_c4(img: Image.Image) -> Image.Image:
    """Heavy photocopy/scan simulation.

    Pipeline: resize to 96 % → resize back to original → JPEG Q=85 →
    JPEG Q=75 → BMP round-trip → Gaussian blur (r=0.5) → Gaussian noise
    (σ=4) → JPEG Q=70 → (saved as PNG by caller).

    Each step models a distinct real-world degradation source:
      resize round-trip — interpolation artefacts from scanner/app resampling
      JPEG Q=85/75      — repeated document sharing (harder than C2's Q=80)
      BMP               — intermediate format conversion, strips metadata
      blur              — scanner optics / photocopy softness
      noise             — scanner sensor noise
      JPEG Q=70         — heavy final compression (messaging app / fax)
    """
    orig_size = img.size
    small_size = (round(orig_size[0] * _RESIZE_SCALE), round(orig_size[1] * _RESIZE_SCALE))

    img = _resize(img.convert("RGB"), small_size)
    img = _resize(img, orig_size)
    img = _jpeg_roundtrip(img, quality=85)
    img = _jpeg_roundtrip(img, quality=75)
    img = _bmp_roundtrip(img)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = _add_gaussian_noise(img, sigma=4.0)
    img = _jpeg_roundtrip(img, quality=70)
    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_tier(image_path: str, tier: str, output_path: str) -> None:
    """Apply a compression tier to *image_path* and write a PNG to *output_path*.

    Args:
        image_path: Path to the source image.  Any format readable by Pillow is
            accepted, but the canonical source is a lossless PNG (C0 image).
        tier: One of ``"C0"``, ``"C2"``, or ``"C4"`` (case-sensitive).
        output_path: Destination path.  The parent directory must already exist.
            The output is always written as a lossless PNG regardless of the
            file extension supplied.

    Raises:
        ValueError: If *tier* is not a recognised tier code.
        FileNotFoundError: If *image_path* does not exist.
    """
    if tier not in TIERS:
        raise ValueError(f"Unknown tier '{tier}'. Valid tiers: {list(TIERS)}")

    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source image not found: {src}")

    dst = Path(output_path)

    if tier == "C0":
        # Bit-exact copy — no re-encoding.
        shutil.copy2(src, dst)
        return

    img = Image.open(src)

    if tier == "C2":
        result = _apply_c2(img)
    else:  # C4
        result = _apply_c4(img)

    result.convert("RGB").save(dst, format="PNG", optimize=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kriyam.compression",
        description="Apply a benchmark compression tier to an image.",
    )
    parser.add_argument("--input", required=True, metavar="PATH", help="Source image path.")
    parser.add_argument(
        "--tier",
        required=True,
        choices=list(TIERS),
        help="Compression tier to apply (C0, C2, or C4).",
    )
    parser.add_argument("--output", required=True, metavar="PATH", help="Output PNG path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the compression CLI."""
    args = _build_parser().parse_args(argv)
    apply_tier(image_path=args.input, tier=args.tier, output_path=args.output)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
