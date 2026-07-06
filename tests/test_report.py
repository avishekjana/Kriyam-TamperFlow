"""Tests for kriyam.report.generate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kriyam.report import generate, _as_ci, _cr as _compression_robustness, _f


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _agg(
    region_f1: float = 0.80,
    doc_auc: float = 0.90,
    doc_f1: float = 0.85,
    doc_fpr: float = 0.07,
    n: int = 100,
) -> dict[str, Any]:
    """Return a minimal aggregate dict that satisfies generate()."""
    ci = (region_f1 - 0.05, region_f1 + 0.05)
    return {
        "region_precision": region_f1,
        "region_precision_ci": ci,
        "region_recall": region_f1,
        "region_recall_ci": ci,
        "region_f1": region_f1,
        "region_f1_ci": ci,
        "doc_auc": doc_auc,
        "doc_auc_ci": (doc_auc - 0.03, doc_auc + 0.03),
        "doc_f1": doc_f1,
        "doc_f1_ci": (doc_f1 - 0.03, doc_f1 + 0.03),
        "doc_fpr": doc_fpr,
        "doc_fpr_ci": (max(0.0, doc_fpr - 0.02), doc_fpr + 0.02),
        "n_samples": n,
    }


def _scores(
    tiers: list[str] | None = None,
    model: str = "test_model",
) -> dict[str, Any]:
    tiers = tiers or ["C0", "C2", "C4"]
    payload: dict[str, Any] = {
        "model": model,
        "tiers": {
            "C0": _agg(doc_auc=0.92),
            "C2": _agg(doc_auc=0.87),
            "C4": _agg(doc_auc=0.81),
        },
    }
    payload["tiers"] = {t: v for t, v in payload["tiers"].items() if t in tiers}
    return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_as_ci_from_tuple() -> None:
    assert _as_ci((0.80, 0.90)) == (0.80, 0.90)


def test_as_ci_from_list() -> None:
    lo, hi = _as_ci([0.80, 0.90])
    assert lo == pytest.approx(0.80)
    assert hi == pytest.approx(0.90)


def test_f_formats_correctly() -> None:
    assert _f(0.12345) == "0.1235"
    assert _f(0.12345, 2) == "0.12"


def test_compression_robustness_normal() -> None:
    # CR = 1 - (0.9 - 0.72) / 0.9 = 1 - 0.2 = 0.8
    cr = _compression_robustness(0.9, 0.72)
    assert cr == pytest.approx(0.8, abs=1e-6)


def test_compression_robustness_zero_c0_returns_one() -> None:
    assert _compression_robustness(0.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# generate — file is written
# ---------------------------------------------------------------------------


def test_generate_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    generate(_scores(), str(out))
    assert out.is_file()


def test_generate_creates_parent_dir_implicit(tmp_path: Path) -> None:
    # Parent directory already exists (generate does NOT create missing parents —
    # caller is responsible; here tmp_path always exists).
    out = tmp_path / "report.html"
    generate(_scores(), str(out))
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# generate — HTML structure
# ---------------------------------------------------------------------------


def _html(tmp_path: Path, **kwargs) -> str:
    out = tmp_path / "r.html"
    generate(_scores(**kwargs), str(out))
    return out.read_text(encoding="utf-8")


def test_html_is_valid_doctype(tmp_path: Path) -> None:
    html = _html(tmp_path)
    assert html.startswith("<!DOCTYPE html>")


def test_html_contains_model_name(tmp_path: Path) -> None:
    html = _html(tmp_path, model="my_fancy_model")
    assert "my_fancy_model" in html


def test_html_contains_chart_js_cdn(tmp_path: Path) -> None:
    html = _html(tmp_path)
    assert "cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0" in html


def test_html_contains_canvas_element(tmp_path: Path) -> None:
    html = _html(tmp_path)
    assert "<canvas" in html
    # Detection metrics: three single-metric charts.
    assert "detAucChart" in html
    assert "detF1Chart" in html
    assert "detFprChart" in html
    # Localisation metrics: three single-metric charts.
    assert "locPChart" in html
    assert "locRChart" in html
    assert "locF1Chart" in html


def test_html_detection_charts_have_direction_and_remarks(tmp_path: Path) -> None:
    html = _html(tmp_path)
    assert "higher is better" in html
    assert "lower is better" in html
    # AUC degrades 0.92 → 0.81 across tiers → a "Worsens" remark must appear.
    assert "Worsens" in html
    assert 'class="chart-remark' in html


def test_html_summary_table_has_all_tiers(tmp_path: Path) -> None:
    html = _html(tmp_path)
    assert "C0" in html
    assert "C2" in html
    assert "C4" in html


def test_html_summary_table_columns(tmp_path: Path) -> None:
    html = _html(tmp_path)
    for col in ("Region-F1", "Doc-AUC", "Doc-F1", "FPR", "CR"):
        assert col in html, f"Missing column header: {col}"


def test_html_cr_stat_card_present_when_c0_and_c4(tmp_path: Path) -> None:
    html = _html(tmp_path, tiers=["C0", "C2", "C4"])
    assert "Compression Robustness" in html
    # The rendered element has class="cr-card" (CSS selector differs)
    assert 'class="cr-card"' in html


def test_html_cr_stat_card_absent_when_missing_c4(tmp_path: Path) -> None:
    html = _html(tmp_path, tiers=["C0", "C2"])
    assert 'class="cr-card"' not in html


def test_html_cr_value_in_table_for_c4(tmp_path: Path) -> None:
    html = _html(tmp_path)
    # CR for AUC 0.92 → 0.81: 1 - (0.92 - 0.81) / 0.92 ≈ 0.8804
    assert "0.88" in html


def test_html_cr_ref_for_c0_row(tmp_path: Path) -> None:
    html = _html(tmp_path)
    assert "ref" in html


def test_html_auc_values_appear_in_chart_data(tmp_path: Path) -> None:
    html = _html(tmp_path)
    # Chart data should embed the three AUC values
    assert "0.92" in html
    assert "0.87" in html
    assert "0.81" in html


def test_html_ci_values_appear_in_chart_data(tmp_path: Path) -> None:
    html = _html(tmp_path)
    # CI lower/upper for C0: (0.89, 0.95)
    assert "0.89" in html
    assert "0.95" in html


# ---------------------------------------------------------------------------
# generate — single-tier run (only C0)
# ---------------------------------------------------------------------------


def test_single_tier_no_cr_card(tmp_path: Path) -> None:
    html = _html(tmp_path, tiers=["C0"])
    assert 'class="cr-card"' not in html


def test_single_tier_null_for_missing_tiers_in_chart(tmp_path: Path) -> None:
    html = _html(tmp_path, tiers=["C0"])
    # C2 and C4 are absent, Chart.js data must include null placeholders
    assert "null" in html


# ---------------------------------------------------------------------------
# generate — CI values as lists (post-JSON-round-trip)
# ---------------------------------------------------------------------------


def test_ci_as_list_not_tuple(tmp_path: Path) -> None:
    s = _scores()
    for agg in s["tiers"].values():
        for key in ("region_f1_ci", "doc_auc_ci", "doc_f1_ci", "doc_fpr_ci",
                    "region_precision_ci", "region_recall_ci"):
            agg[key] = list(agg[key])  # simulate JSON deserialization
    out = tmp_path / "r.html"
    generate(s, str(out))
    assert out.read_text().startswith("<!DOCTYPE html>")


# ---------------------------------------------------------------------------
