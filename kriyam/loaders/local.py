"""Load benchmark samples from a local data directory."""

from __future__ import annotations

import json
from pathlib import Path

from kriyam.loaders.base import RegionDict, SampleDict

# Valid compression tiers defined by the benchmark.
ALL_TIERS: tuple[str, ...] = ("C0", "C2", "C4")

# Minimum keys that every annotation file must contain.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "image_w",
        "image_h",
        "is_authentic",
        "regions",
    }
)


def _parse_region(raw: dict) -> RegionDict:
    """Parse a single region dict from an annotation JSON.

    Validates that the four geometry keys are present and passes through all
    other recognised fields.  Unknown keys are silently ignored so that future
    schema additions do not break existing code.
    """
    for key in ("region_id", "x", "y", "w", "h", "tamper_types"):
        if key not in raw:
            raise ValueError(f"Region is missing required key '{key}': {raw!r}")

    return {
        "region_id": int(raw["region_id"]),
        "x": int(raw["x"]),
        "y": int(raw["y"]),
        "w": int(raw["w"]),
        "h": int(raw["h"]),
        "tamper_types": list(raw["tamper_types"]),
    }


def _parse_annotation(path: Path) -> SampleDict:
    """Parse one annotation JSON file into a SampleDict.

    Raises ``ValueError`` if a required key is absent or if ``regions`` is not
    a list.  Raises ``json.JSONDecodeError`` if the file is not valid JSON.
    """
    with path.open(encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(
            f"Annotation '{path}' is missing required keys: {sorted(missing)}"
        )

    if not isinstance(raw["regions"], list):
        raise ValueError(f"Annotation '{path}': 'regions' must be a list")

    return {
        "id": str(raw["id"]),
        "image_w": int(raw["image_w"]),
        "image_h": int(raw["image_h"]),
        "is_authentic": bool(raw["is_authentic"]),
        "regions": [_parse_region(r) for r in raw["regions"]],
    }


def _image_stem(sample_id: str, tier: str) -> str:
    """Return the image filename stem for a given sample id and tier.

    ``sample_id`` is the annotation id, e.g. ``kriyam_0042``.
    The corresponding C0 image stem is ``kriyam_0042_C0``.
    """
    return f"{sample_id}_{tier}"


def load_annotations(data_dir: str | Path) -> list[SampleDict]:
    """Walk ``data_dir/annotations/`` and parse every ``.json`` file.

    Args:
        data_dir: Root of the benchmark data directory.  Must contain an
            ``annotations/`` sub-directory.

    Returns:
        A list of :class:`SampleDict` objects, one per annotation file, sorted
        by sample id for deterministic ordering.

    Raises:
        FileNotFoundError: If ``data_dir/annotations/`` does not exist.
        ValueError: If any annotation file fails schema validation.
    """
    annotations_dir = Path(data_dir) / "annotations"
    if not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Annotations directory not found: {annotations_dir}"
        )

    paths = sorted(annotations_dir.glob("*.json"))
    return [_parse_annotation(p) for p in paths]


def load_dataset(
    data_dir: str | Path,
    tiers: list[str] | None = None,
    is_authentic: bool | None = None,
) -> list[SampleDict]:
    """Load and filter benchmark samples from a local data directory.

    Samples are loaded from ``data_dir/annotations/``.  Optional filters narrow
    the result set; all supplied filters are applied with AND logic.

    Args:
        data_dir: Root of the benchmark data directory.  Expected layout::

            data_dir/
              annotations/   ← one .json per unique image
              images/        ← kriyam_{index:04d}_{tier}.png  (C0 may be .jpg/.jpeg)

        tiers: If provided, only return samples for which **all** listed tier
            image files exist on disk.  For example, ``tiers=["C0", "C2"]``
            returns only samples that have both a C0 image (PNG, JPG, or JPEG) and
            ``kriyam_*_C2.png`` present in ``data_dir/images/``.  Unknown tier
            values raise ``ValueError``.  When ``None``, no tier-based filtering
            is applied.

        is_authentic: If provided, keep only authentic (``True``) or tampered
            (``False``) samples.

    Returns:
        A filtered list of :class:`SampleDict` objects.

    Raises:
        FileNotFoundError: If the annotations or images directory is missing and
            tier-based filtering is requested.
        ValueError: If ``tiers`` contains an unrecognised tier code.
    """
    if tiers is not None:
        unknown = set(tiers) - set(ALL_TIERS)
        if unknown:
            raise ValueError(
                f"Unknown tier(s): {sorted(unknown)}. Valid tiers: {list(ALL_TIERS)}"
            )

    samples = load_annotations(data_dir)

    if is_authentic is not None:
        samples = [s for s in samples if s["is_authentic"] == is_authentic]

    if tiers is not None:
        images_dir = Path(data_dir) / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}"
            )
        def _image_exists(sample_id: str, tier: str) -> bool:
            stem = _image_stem(sample_id, tier)
            exts = (".png", ".jpg", ".jpeg") if tier == "C0" else (".png",)
            return any((images_dir / f"{stem}{ext}").is_file() for ext in exts)

        filtered: list[SampleDict] = []
        for sample in samples:
            if all(_image_exists(sample["id"], t) for t in tiers):
                filtered.append(sample)
        samples = filtered

    return samples
