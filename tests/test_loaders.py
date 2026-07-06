"""Tests for kriyam.loaders.local."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from kriyam.loaders.local import load_annotations, load_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TAMPERED_ANNOTATION = {
    "id": "kriyam_0001",
    "image_w": 1024,
    "image_h": 648,
    "is_authentic": False,
    "regions": [
        {
            "region_id": 1,
            "x": 100,
            "y": 80,
            "w": 200,
            "h": 40,
            "tamper_types": ["splice"],
        }
    ],
}

AUTHENTIC_ANNOTATION = {
    "id": "kriyam_0002",
    "image_w": 800,
    "image_h": 1100,
    "is_authentic": True,
    "regions": [],
}


def _write_data_dir(tmp_path: Path, annotations: list[dict]) -> Path:
    """Write annotation files to a temporary data directory structure."""
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True)
    for ann in annotations:
        (ann_dir / f"{ann['id']}.json").write_text(json.dumps(ann), encoding="utf-8")
    return tmp_path


def _add_images(data_dir: Path, sample_id: str, tiers: list[str]) -> None:
    """Create zero-byte placeholder image files for tier-filtering tests."""
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for tier in tiers:
        (images_dir / f"{sample_id}_{tier}.png").touch()


# ---------------------------------------------------------------------------
# load_annotations
# ---------------------------------------------------------------------------


def test_load_annotations_returns_all(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION, AUTHENTIC_ANNOTATION])
    samples = load_annotations(data_dir)
    assert len(samples) == 2


def test_load_annotations_field_types(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION])
    s = load_annotations(data_dir)[0]

    assert isinstance(s["id"], str)
    assert isinstance(s["image_w"], int)
    assert isinstance(s["image_h"], int)
    assert isinstance(s["is_authentic"], bool)
    assert isinstance(s["regions"], list)


def test_load_annotations_region_fields(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION])
    region = load_annotations(data_dir)[0]["regions"][0]

    assert region["region_id"] == 1
    assert region["x"] == 100
    assert region["tamper_types"] == ["splice"]


def test_load_annotations_authentic_has_empty_regions(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [AUTHENTIC_ANNOTATION])
    s = load_annotations(data_dir)[0]
    assert s["regions"] == []
    assert s["is_authentic"] is True



def test_load_annotations_sorted_by_id(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [AUTHENTIC_ANNOTATION, TAMPERED_ANNOTATION])
    ids = [s["id"] for s in load_annotations(data_dir)]
    assert ids == sorted(ids)


def test_load_annotations_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="annotations"):
        load_annotations(tmp_path / "nonexistent")


def test_load_annotations_missing_key_raises(tmp_path: Path) -> None:
    bad = {k: v for k, v in TAMPERED_ANNOTATION.items() if k != "is_authentic"}
    data_dir = _write_data_dir(tmp_path, [bad])
    with pytest.raises(ValueError, match="is_authentic"):
        load_annotations(data_dir)


def test_load_annotations_regions_not_list_raises(tmp_path: Path) -> None:
    bad = {**TAMPERED_ANNOTATION, "regions": "not-a-list"}
    data_dir = _write_data_dir(tmp_path, [bad])
    with pytest.raises(ValueError, match="regions"):
        load_annotations(data_dir)


def test_load_annotations_region_missing_key_raises(tmp_path: Path) -> None:
    bad_region = {"region_id": 1, "x": 0, "y": 0, "w": 10}  # missing h, tamper_types
    bad = {**TAMPERED_ANNOTATION, "regions": [bad_region]}
    data_dir = _write_data_dir(tmp_path, [bad])
    with pytest.raises(ValueError):
        load_annotations(data_dir)


# ---------------------------------------------------------------------------
# load_dataset — filtering
# ---------------------------------------------------------------------------



def test_filter_is_authentic_true(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION, AUTHENTIC_ANNOTATION])
    samples = load_dataset(data_dir, is_authentic=True)
    assert len(samples) == 1
    assert samples[0]["is_authentic"] is True


def test_filter_is_authentic_false(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION, AUTHENTIC_ANNOTATION])
    samples = load_dataset(data_dir, is_authentic=False)
    assert len(samples) == 1
    assert samples[0]["is_authentic"] is False



# ---------------------------------------------------------------------------
# load_dataset — tier filtering
# ---------------------------------------------------------------------------


def test_filter_tiers_all_present(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION])
    _add_images(data_dir, "kriyam_0001", ["C0", "C2", "C4"])
    samples = load_dataset(data_dir, tiers=["C0", "C2"])
    assert len(samples) == 1


def test_filter_tiers_partial_missing(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION])
    _add_images(data_dir, "kriyam_0001", ["C0"])  # C2 absent
    samples = load_dataset(data_dir, tiers=["C0", "C2"])
    assert samples == []


def test_filter_tiers_none_means_no_filtering(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION])
    # No images directory needed when tiers=None
    samples = load_dataset(data_dir, tiers=None)
    assert len(samples) == 1


def test_filter_invalid_tier_raises(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION])
    with pytest.raises(ValueError, match="Unknown tier"):
        load_dataset(data_dir, tiers=["C5"])


def test_filter_tiers_missing_images_dir_raises(tmp_path: Path) -> None:
    data_dir = _write_data_dir(tmp_path, [TAMPERED_ANNOTATION])
    # images/ directory was never created
    with pytest.raises(FileNotFoundError, match="images"):
        load_dataset(data_dir, tiers=["C0"])
