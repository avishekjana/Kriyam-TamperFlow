"""Tests for kriyam.compression."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kriyam.compression import TIERS, apply_tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_png(tmp_path: Path, name: str = "source.png", size: tuple[int, int] = (64, 64)) -> Path:
    """Write a small solid-colour PNG and return its path.

    *size* is ``(width, height)`` matching Pillow convention.  Numpy arrays are
    created as ``(height, width, channels)`` so the two axes are swapped here.
    """
    width, height = size
    img = Image.fromarray(
        np.full((height, width, 3), fill_value=(120, 80, 200), dtype=np.uint8)
    )
    p = tmp_path / name
    img.save(p, format="PNG")
    return p


def _is_png(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(8) == _PNG_MAGIC


def _load_rgb_array(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# apply_tier — C0
# ---------------------------------------------------------------------------


def test_c0_output_is_png(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C0", str(dst))
    assert _is_png(dst)


def test_c0_is_bit_exact_copy(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C0", str(dst))
    assert src.read_bytes() == dst.read_bytes()


def test_c0_preserves_dimensions(tmp_path: Path) -> None:
    src = _make_png(tmp_path, size=(100, 80))
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C0", str(dst))
    w, h = Image.open(dst).size
    assert (w, h) == (100, 80)


# ---------------------------------------------------------------------------
# apply_tier — C2
# ---------------------------------------------------------------------------


def test_c2_output_is_png(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C2", str(dst))
    assert _is_png(dst)


def test_c2_differs_from_source(tmp_path: Path) -> None:
    # JPEG compression changes pixel values — result must not be bit-identical.
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C2", str(dst))
    assert src.read_bytes() != dst.read_bytes()


def test_c2_preserves_dimensions(tmp_path: Path) -> None:
    # C2 resizes down then back up before JPEG passes — output matches source.
    src = _make_png(tmp_path, size=(96, 64))
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C2", str(dst))
    w, h = Image.open(dst).size
    assert (w, h) == (96, 64)


def test_c2_output_is_rgb(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C2", str(dst))
    assert Image.open(dst).mode == "RGB"


def test_c2_pixel_values_within_range(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C2", str(dst))
    arr = _load_rgb_array(dst)
    assert arr.min() >= 0
    assert arr.max() <= 255


# ---------------------------------------------------------------------------
# apply_tier — C4
# ---------------------------------------------------------------------------


def test_c4_output_is_png(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C4", str(dst))
    assert _is_png(dst)


def test_c4_differs_from_c2(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst_c2 = tmp_path / "c2.png"
    dst_c4 = tmp_path / "c4.png"
    apply_tier(str(src), "C2", str(dst_c2))
    apply_tier(str(src), "C4", str(dst_c4))
    assert dst_c2.read_bytes() != dst_c4.read_bytes()


def test_c4_preserves_dimensions(tmp_path: Path) -> None:
    src = _make_png(tmp_path, size=(80, 80))
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C4", str(dst))
    w, h = Image.open(dst).size
    assert (w, h) == (80, 80)


def test_c4_output_is_rgb(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C4", str(dst))
    assert Image.open(dst).mode == "RGB"


def test_c4_pixel_values_within_range(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst = tmp_path / "out.png"
    apply_tier(str(src), "C4", str(dst))
    arr = _load_rgb_array(dst)
    assert arr.min() >= 0
    assert arr.max() <= 255


def test_c4_is_deterministic(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    dst_a = tmp_path / "a.png"
    dst_b = tmp_path / "b.png"
    apply_tier(str(src), "C4", str(dst_a))
    apply_tier(str(src), "C4", str(dst_b))
    assert dst_a.read_bytes() == dst_b.read_bytes()


def test_c4_more_degraded_than_c2(tmp_path: Path) -> None:
    # C4 should move pixel values further from the original than C2 does.
    # Both tiers output at the source dimensions, so comparison is direct.
    src = _make_png(tmp_path, size=(64, 64))
    dst_c2 = tmp_path / "c2.png"
    dst_c4 = tmp_path / "c4.png"
    apply_tier(str(src), "C2", str(dst_c2))
    apply_tier(str(src), "C4", str(dst_c4))

    original = _load_rgb_array(src).astype(np.float32)
    diff_c2 = np.abs(_load_rgb_array(dst_c2).astype(np.float32) - original).mean()
    diff_c4 = np.abs(_load_rgb_array(dst_c4).astype(np.float32) - original).mean()
    assert diff_c4 > diff_c2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_tier_raises(tmp_path: Path) -> None:
    src = _make_png(tmp_path)
    with pytest.raises(ValueError, match="Unknown tier"):
        apply_tier(str(src), "C9", str(tmp_path / "out.png"))


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        apply_tier(str(tmp_path / "ghost.png"), "C0", str(tmp_path / "out.png"))


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_c2(tmp_path: Path) -> None:
    from kriyam.compression import main

    src = _make_png(tmp_path)
    dst = tmp_path / "cli_out.png"
    main(["--input", str(src), "--tier", "C2", "--output", str(dst)])
    assert _is_png(dst)


def test_cli_invalid_tier_exits(tmp_path: Path) -> None:
    from kriyam.compression import main

    src = _make_png(tmp_path)
    with pytest.raises(SystemExit):
        main(["--input", str(src), "--tier", "C9", "--output", str(tmp_path / "out.png")])


# ---------------------------------------------------------------------------
# TIERS constant
# ---------------------------------------------------------------------------


def test_tiers_constant_contains_all_three() -> None:
    assert set(TIERS) == {"C0", "C2", "C4"}
