"""Tests for scripts/run_compression_tiers.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_compression_tiers import _discover_sources, _output_path, main, run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_c0(directory: Path, name: str = "kriyam_0001_C0.png") -> Path:
    """Write a small solid-colour C0 PNG and return its path."""
    img = Image.fromarray(
        np.full((32, 32, 3), fill_value=(100, 150, 200), dtype=np.uint8)
    )
    p = directory / name
    img.save(p, format="PNG")
    return p


def _is_png(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(8) == _PNG_MAGIC


# ---------------------------------------------------------------------------
# _output_path
# ---------------------------------------------------------------------------


def test_output_path_c2(tmp_path: Path) -> None:
    src = tmp_path / "kriyam_0001_C0.png"
    dst = _output_path(src, "C2", tmp_path)
    assert dst.name == "kriyam_0001_C2.png"
    assert dst.parent == tmp_path


def test_output_path_c4(tmp_path: Path) -> None:
    src = tmp_path / "kriyam__0042_C0.png"
    dst = _output_path(src, "C4", tmp_path)
    assert dst.name == "kriyam__0042_C4.png"


def test_output_path_different_output_dir(tmp_path: Path) -> None:
    src = tmp_path / "kriyam_0001_C0.png"
    out_dir = tmp_path / "out"
    dst = _output_path(src, "C2", out_dir)
    assert dst.parent == out_dir


# ---------------------------------------------------------------------------
# _discover_sources
# ---------------------------------------------------------------------------


def test_discover_sources_finds_c0_files(tmp_path: Path) -> None:
    _make_c0(tmp_path, "a_C0.png")
    _make_c0(tmp_path, "b_C0.png")
    sources = _discover_sources(tmp_path)
    assert len(sources) == 2
    assert all(p.name.endswith("_C0.png") for p in sources)


def test_discover_sources_ignores_non_c0(tmp_path: Path) -> None:
    _make_c0(tmp_path, "a_C0.png")
    (tmp_path / "a_C2.png").touch()
    (tmp_path / "a_C4.png").touch()
    sources = _discover_sources(tmp_path)
    assert len(sources) == 1


def test_discover_sources_sorted(tmp_path: Path) -> None:
    _make_c0(tmp_path, "z_C0.png")
    _make_c0(tmp_path, "a_C0.png")
    names = [p.name for p in _discover_sources(tmp_path)]
    assert names == sorted(names)


def test_discover_sources_empty_dir(tmp_path: Path) -> None:
    assert _discover_sources(tmp_path) == []


def test_discover_sources_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _discover_sources(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# run — basic generation
# ---------------------------------------------------------------------------


def test_run_generates_c2_and_c4(tmp_path: Path) -> None:
    _make_c0(tmp_path, "img_C0.png")
    summary = run(tmp_path, tmp_path, ["C2", "C4"])
    assert (tmp_path / "img_C2.png").is_file()
    assert (tmp_path / "img_C4.png").is_file()


def test_run_output_is_valid_png(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    run(tmp_path, tmp_path, ["C2"])
    assert _is_png(tmp_path / "kriyam_0001_C2.png")


def test_run_summary_counts(tmp_path: Path) -> None:
    _make_c0(tmp_path, "a_C0.png")
    _make_c0(tmp_path, "b_C0.png")
    summary = run(tmp_path, tmp_path, ["C2", "C4"])
    assert summary["images"] == 2
    assert summary["variants"] == 4
    assert summary["skipped"] == 0


def test_run_c2_only(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    summary = run(tmp_path, tmp_path, ["C2"])
    assert summary["variants"] == 1
    assert not (tmp_path / "kriyam_0001_C4.png").exists()


def test_run_separate_output_dir(tmp_path: Path) -> None:
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir()
    _make_c0(in_dir)
    run(in_dir, out_dir, ["C2"])
    assert (out_dir / "kriyam_0001_C2.png").is_file()
    assert not (in_dir / "kriyam_0001_C2.png").exists()


def test_run_creates_output_dir(tmp_path: Path) -> None:
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "new" / "nested" / "output"
    in_dir.mkdir()
    _make_c0(in_dir)
    run(in_dir, out_dir, ["C2"])
    assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# run — skip existing files
# ---------------------------------------------------------------------------


def test_run_skips_existing_output(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    existing = tmp_path / "kriyam_0001_C2.png"
    existing.write_bytes(b"placeholder")
    summary = run(tmp_path, tmp_path, ["C2"])
    assert summary["skipped"] == 1
    assert summary["variants"] == 0
    # Existing file must not be overwritten
    assert existing.read_bytes() == b"placeholder"


def test_run_skips_some_writes_others(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    (tmp_path / "kriyam_0001_C2.png").write_bytes(b"old")
    summary = run(tmp_path, tmp_path, ["C2", "C4"])
    assert summary["skipped"] == 1
    assert summary["variants"] == 1


# ---------------------------------------------------------------------------
# run — empty input
# ---------------------------------------------------------------------------


def test_run_empty_input_dir(tmp_path: Path) -> None:
    summary = run(tmp_path, tmp_path, ["C2"])
    assert summary["images"] == 0
    assert summary["variants"] == 0


# ---------------------------------------------------------------------------
# run — error handling
# ---------------------------------------------------------------------------


def test_run_unknown_tier_raises(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    with pytest.raises(ValueError, match="Unknown tier"):
        run(tmp_path, tmp_path, ["C9"])


def test_run_source_tier_in_tiers_raises(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    with pytest.raises(ValueError, match="source tier"):
        run(tmp_path, tmp_path, ["C0"])


def test_run_missing_input_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "ghost", tmp_path, ["C2"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_generates_files(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    main(["--input-dir", str(tmp_path), "--tiers", "C2"])
    assert (tmp_path / "kriyam_0001_C2.png").is_file()


def test_cli_default_tiers_are_c2_c4(tmp_path: Path) -> None:
    _make_c0(tmp_path)
    main(["--input-dir", str(tmp_path)])
    assert (tmp_path / "kriyam_0001_C2.png").is_file()
    assert (tmp_path / "kriyam_0001_C4.png").is_file()


def test_cli_output_dir_flag(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _make_c0(in_dir)
    main(["--input-dir", str(in_dir), "--output-dir", str(out_dir), "--tiers", "C2"])
    assert (out_dir / "kriyam_0001_C2.png").is_file()


def test_cli_missing_input_dir_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--input-dir", str(tmp_path / "ghost")])


def test_cli_unknown_tier_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--input-dir", str(tmp_path), "--tiers", "C9"])


def test_cli_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _make_c0(tmp_path)
    main(["--input-dir", str(tmp_path), "--tiers", "C2"])
    out = capsys.readouterr().out
    assert "1 image(s) processed" in out
    assert "Output written to:" in out
