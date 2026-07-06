"""Tests for scripts/validate_predictions.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.validate_predictions import (
    FileResult,
    Issue,
    _check_file_exists,
    _check_json_parseable,
    _check_regions,
    _check_required_fields,
    main,
    validate,
    validate_file,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANN = {
    "id": "kriyam_0001",
    "image_w": 200,
    "image_h": 150,
    "is_authentic": False,
    "regions": [{"region_id": 1, "x": 10, "y": 10, "w": 50, "h": 40,
                 "tamper_types": ["splice"]}],
}

_GOOD_PRED = {
    "id": "kriyam_0001_C0",
    "model": "test",
    "regions": [{"x": 10, "y": 10, "w": 50, "h": 40, "confidence": 0.9}],
}

_W, _H = 200, 150


def _write_ann(tmp_path: Path, ann: dict = _ANN) -> Path:
    d = tmp_path / "data" / "annotations"
    d.mkdir(parents=True)
    (d / f"{ann['id']}.json").write_text(json.dumps(ann), encoding="utf-8")
    return tmp_path / "data"


def _write_pred(directory: Path, sample_id: str, tier: str, pred: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{sample_id}_{tier}.json"
    p.write_text(json.dumps(pred), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _check_file_exists
# ---------------------------------------------------------------------------

def test_file_exists_present(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.touch()
    assert _check_file_exists(p, "id", "C0") is None


def test_file_exists_missing(tmp_path: Path) -> None:
    issue = _check_file_exists(tmp_path / "ghost.json", "id", "C0")
    assert issue is not None
    assert issue.fatal
    assert issue.check == "file_exists"


# ---------------------------------------------------------------------------
# _check_json_parseable
# ---------------------------------------------------------------------------

def test_json_parseable_valid(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    data, issue = _check_json_parseable(p, "id", "C0")
    assert data == {"a": 1}
    assert issue is None


def test_json_parseable_invalid(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text("{not json", encoding="utf-8")
    data, issue = _check_json_parseable(p, "id", "C0")
    assert data is None
    assert issue is not None and issue.fatal


def test_json_parseable_non_object(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    data, issue = _check_json_parseable(p, "id", "C0")
    assert data is None
    assert issue is not None


# ---------------------------------------------------------------------------
# _check_required_fields
# ---------------------------------------------------------------------------

def test_required_fields_all_present() -> None:
    assert _check_required_fields(_GOOD_PRED, "id", "C0") == []


def test_required_fields_missing_regions() -> None:
    bad = {"id": "kriyam_0001_C0"}
    issues = _check_required_fields(bad, "id", "C0")
    assert len(issues) == 1
    assert "regions" in issues[0].message


def test_required_fields_missing_id() -> None:
    bad = {"regions": []}
    issues = _check_required_fields(bad, "id", "C0")
    assert len(issues) == 1
    assert "id" in issues[0].message


# ---------------------------------------------------------------------------
# _check_regions
# ---------------------------------------------------------------------------

def _good_region(**overrides) -> dict:
    r = {"x": 10, "y": 10, "w": 50, "h": 40, "confidence": 0.9}
    r.update(overrides)
    return r


def test_regions_valid_list() -> None:
    assert _check_regions({"regions": [_good_region()]}, "id", "C0", _W, _H) == []


def test_regions_empty_list_valid() -> None:
    assert _check_regions({"regions": []}, "id", "C0", _W, _H) == []


def test_regions_not_list() -> None:
    issues = _check_regions({"regions": "bad"}, "id", "C0", _W, _H)
    assert any(i.check == "regions_type" for i in issues)


def test_regions_missing_field() -> None:
    r = {k: v for k, v in _good_region().items() if k != "h"}
    issues = _check_regions({"regions": [r]}, "id", "C0", _W, _H)
    assert any("h" in i.message for i in issues)


def test_regions_exceeds_image_width() -> None:
    r = _good_region(x=180, w=30)  # 180+30=210 > 200
    issues = _check_regions({"regions": [r]}, "id", "C0", _W, _H)
    assert any("bounds" in i.check for i in issues)


def test_regions_exceeds_image_height() -> None:
    r = _good_region(y=140, h=20)  # 140+20=160 > 150
    issues = _check_regions({"regions": [r]}, "id", "C0", _W, _H)
    assert any("bounds" in i.check for i in issues)


def test_regions_exactly_at_boundary_valid() -> None:
    r = _good_region(x=0, y=0, w=_W, h=_H)
    assert _check_regions({"regions": [r]}, "id", "C0", _W, _H) == []


def test_regions_negative_origin() -> None:
    issues = _check_regions({"regions": [_good_region(x=-1)]}, "id", "C0", _W, _H)
    assert any("origin" in i.check for i in issues)


def test_regions_zero_width() -> None:
    issues = _check_regions({"regions": [_good_region(w=0)]}, "id", "C0", _W, _H)
    assert any("dimensions" in i.check for i in issues)


def test_regions_duplicate_region_id() -> None:
    r1 = {**_good_region(), "region_id": 1}
    r2 = {**_good_region(), "region_id": 1}
    issues = _check_regions({"regions": [r1, r2]}, "id", "C0", _W, _H)
    assert any("dup_id" in i.check for i in issues)


def test_regions_unique_ids_valid() -> None:
    r1 = {**_good_region(), "region_id": 1}
    r2 = {**_good_region(), "region_id": 2}
    assert _check_regions({"regions": [r1, r2]}, "id", "C0", _W, _H) == []


# ---------------------------------------------------------------------------
# validate_file
# ---------------------------------------------------------------------------

def test_validate_file_all_pass(tmp_path: Path) -> None:
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(_GOOD_PRED), encoding="utf-8")
    result = validate_file(p, "kriyam_0001", "C0", _W, _H)
    assert result.valid
    assert result.issues == []


def test_validate_file_missing_returns_fatal(tmp_path: Path) -> None:
    result = validate_file(tmp_path / "missing.json", "id", "C0", _W, _H)
    assert not result.valid
    assert any(i.check == "file_exists" for i in result.issues)


def test_validate_file_stops_at_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{broken", encoding="utf-8")
    result = validate_file(p, "id", "C0", _W, _H)
    assert not result.valid
    # Only the JSON parse check should be recorded — no downstream checks
    assert all(i.check == "json_valid" for i in result.issues)


def test_validate_file_stops_at_missing_fields(tmp_path: Path) -> None:
    bad = {"id": "x"}  # missing required 'regions'
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    result = validate_file(p, "id", "C0", _W, _H)
    assert not result.valid
    assert any(i.check == "required_fields" for i in result.issues)
    # region checks must NOT run when required fields are absent
    assert not any(i.check.startswith("region") for i in result.issues)


def test_validate_file_region_bounds_issue(tmp_path: Path) -> None:
    pred = {**_GOOD_PRED, "regions": [{"x": 190, "y": 10, "w": 50, "h": 40, "confidence": 0.9}]}
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(pred), encoding="utf-8")
    result = validate_file(p, "id", "C0", _W, _H)
    assert not result.valid
    assert any("bounds" in i.check for i in result.issues)


# ---------------------------------------------------------------------------
# validate (full loop)
# ---------------------------------------------------------------------------

def test_validate_all_pass(tmp_path: Path) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "predictions" / "model"
    _write_pred(pred_dir, "kriyam_0001", "C0", _GOOD_PRED)
    results = validate(pred_dir, data_dir, ["C0"])
    assert len(results) == 1
    assert results[0].valid


def test_validate_missing_file(tmp_path: Path) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "predictions" / "model"
    pred_dir.mkdir(parents=True)
    results = validate(pred_dir, data_dir, ["C0"])
    assert len(results) == 1
    assert not results[0].valid


def test_validate_multiple_tiers(tmp_path: Path) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "predictions" / "model"
    _write_pred(pred_dir, "kriyam_0001", "C0", _GOOD_PRED)
    _write_pred(pred_dir, "kriyam_0001", "C2", _GOOD_PRED)
    results = validate(pred_dir, data_dir, ["C0", "C2"])
    assert len(results) == 2
    assert all(r.valid for r in results)


def test_validate_one_tier_missing_one_present(tmp_path: Path) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "predictions" / "model"
    _write_pred(pred_dir, "kriyam_0001", "C0", _GOOD_PRED)
    # C2 intentionally missing
    results = validate(pred_dir, data_dir, ["C0", "C2"])
    valid_flags = [r.valid for r in results]
    assert True in valid_flags
    assert False in valid_flags


def test_validate_multiple_samples(tmp_path: Path) -> None:
    ann2 = {**_ANN, "id": "kriyam_0002"}
    data_dir = tmp_path / "data"
    ann_dir = data_dir / "annotations"
    ann_dir.mkdir(parents=True)
    (ann_dir / "kriyam_0001.json").write_text(json.dumps(_ANN), encoding="utf-8")
    (ann_dir / "kriyam_0002.json").write_text(json.dumps(ann2), encoding="utf-8")

    pred_dir = tmp_path / "predictions" / "model"
    _write_pred(pred_dir, "kriyam_0001", "C0", _GOOD_PRED)
    _write_pred(pred_dir, "kriyam_0002", "C0",
                {**_GOOD_PRED, "id": "kriyam_0002_C0"})

    results = validate(pred_dir, data_dir, ["C0"])
    assert len(results) == 2
    assert all(r.valid for r in results)


def test_validate_unknown_tier_raises(tmp_path: Path) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "p"
    with pytest.raises(ValueError, match="Unknown tier"):
        validate(pred_dir, data_dir, ["C9"])


def test_validate_missing_annotations_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.validate_predictions as vp
    monkeypatch.setattr(vp, "_ensure_annotations", lambda *a, **kw: (_ for _ in ()).throw(SystemExit(1)))
    with pytest.raises(SystemExit) as exc_info:
        validate(tmp_path / "pred", tmp_path / "nodata", ["C0"])
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_all_valid_exits_zero(tmp_path: Path) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "predictions" / "model"
    _write_pred(pred_dir, "kriyam_0001", "C0", _GOOD_PRED)
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--predictions", str(pred_dir),
            "--data-dir", str(data_dir),
            "--tiers", "C0",
        ])
    assert exc_info.value.code == 0


def test_cli_issues_exits_one(tmp_path: Path) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "predictions" / "model"
    pred_dir.mkdir(parents=True)
    # Prediction file absent
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--predictions", str(pred_dir),
            "--data-dir", str(data_dir),
            "--tiers", "C0",
        ])
    assert exc_info.value.code == 1


def test_cli_missing_annotations_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.validate_predictions as vp
    monkeypatch.setattr(vp, "_ensure_annotations", lambda *a, **kw: (_ for _ in ()).throw(SystemExit(1)))
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--predictions", str(tmp_path / "pred"),
            "--data-dir", str(tmp_path / "nodata"),
        ])
    assert exc_info.value.code == 1


def test_cli_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    data_dir = _write_ann(tmp_path)
    pred_dir = tmp_path / "predictions" / "model"
    _write_pred(pred_dir, "kriyam_0001", "C0", _GOOD_PRED)
    with pytest.raises(SystemExit):
        main([
            "--predictions", str(pred_dir),
            "--data-dir", str(data_dir),
            "--tiers", "C0",
        ])
    out = capsys.readouterr().out
    assert "1/1 files valid" in out
    assert "0 issue(s) found" in out
