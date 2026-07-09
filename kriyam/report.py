"""HTML report generation for benchmark evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CHART_JS = (
    "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"
)

_ALL_TIERS: tuple[str, ...] = ("C0", "C2", "C4")

_TIER_META: dict[str, tuple[str, str]] = {
    "C0": ("C0 — Pristine", "No re-compression. Lossless PNG. Full forensic signal intact."),
    "C2": ("C2 — Double-pass JPEG", "Two JPEG saves (Q=85 → Q=80). Simulates a typical scan-share-rescan cycle."),
    "C4": ("C4 — Photocopy simulation", "BMP round-trip → Gaussian blur → additive noise → JPEG Q=70. Nearly erases DCT-based forensic signals."),
}

_TIER_ACCENT: dict[str, str] = {
    "C0": "#16a34a",
    "C2": "#d97706",
    "C4": "#dc2626",
}

_LOCALISATION_METRICS: list[tuple[str, str, str]] = [
    ("region_precision", "region_precision_ci", "Region Precision (Region-P)"),
    ("region_recall",    "region_recall_ci",    "Region Recall (Region-R)"),
    ("region_f1",        "region_f1_ci",        "Region F1 (Region-F1)"),
]

_DETECTION_METRICS: list[tuple[str, str, str]] = [
    ("doc_auc",           "doc_auc_ci",           "Document AUC-ROC (Doc-AUC)"),
    ("doc_auprc",         "doc_auprc_ci",         "Document AUPRC (Doc-AUPRC)"),
    ("doc_f1",            "doc_f1_ci",            "Document F1 (Doc-F1)"),
    ("doc_fpr",           "doc_fpr_ci",           "False Positive Rate (FPR)"),
    ("doc_fpr_at_tpr90",  "doc_fpr_at_tpr90_ci",  "FPR @ TPR=90% (FPR@90)"),
    ("doc_f1_at_tpr90",   "doc_f1_at_tpr90_ci",   "F1 @ TPR=90% (F1@90)"),
]

# Keys whose validity may be flagged False (metric not fully achievable).
_DETECTION_VALIDITY_KEYS: dict[str, str] = {
    "doc_fpr_at_tpr90": "doc_fpr_at_tpr90_valid",
    "doc_f1_at_tpr90":  "doc_f1_at_tpr90_valid",
}

_METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "region_precision":  True,
    "region_recall":     True,
    "region_f1":         True,
    "doc_auc":           True,
    "doc_auprc":         True,
    "doc_f1":            True,
    "doc_fpr":           False,
    "doc_fpr_at_tpr90":  False,
    "doc_f1_at_tpr90":   True,
}


_CSS = """\
*, *::before, *::after { box-sizing: border-box; }

body {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  color: #1a1a1a;
  background: #f0f2f5;
  margin: 0;
  padding: 2rem 1rem 4rem;
}

.page { max-width: 1100px; margin: 0 auto; }

/* ── report header ── */
.report-header {
  background: #111827;
  border-radius: 12px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  color: #fff;
}
.report-header .bench-name {
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #9ca3af;
  margin: 0 0 0.4rem;
}
.report-header h1 {
  font-size: 1.7rem;
  font-weight: 700;
  margin: 0 0 0.5rem;
  color: #f9fafb;
}
.report-header .meta {
  font-size: 0.85rem;
  color: #6b7280;
  margin: 0;
}

/* ── sections ── */
.section {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 1.5rem 2rem;
  margin-bottom: 1.25rem;
}
.section h2 {
  font-size: 1rem;
  font-weight: 700;
  color: #111827;
  margin: 0 0 0.35rem;
}
.section .tier-desc {
  font-size: 0.82rem;
  color: #6b7280;
  margin: 0 0 1.1rem;
}
.section-intro p {
  margin: 0;
  color: #374151;
  font-size: 0.9rem;
}

/* ── eval summary slab ── */
/* ── metric tables ── */
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
thead th {
  background: #f9fafb;
  text-align: left;
  padding: 0.45rem 0.9rem;
  border-bottom: 2px solid #e5e7eb;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #6b7280;
  font-weight: 600;
}
tbody td {
  padding: 0.5rem 0.9rem;
  border-bottom: 1px solid #f3f4f6;
  vertical-align: middle;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: #fafafa; }
.metric-name { font-weight: 500; color: #374151; }
.metric-val  { font-weight: 700; color: #111827; font-variant-numeric: tabular-nums; }
.ci-range    { font-size: 0.78rem; color: #9ca3af; }
.tier-stripe { border-left: 4px solid var(--tier-color); }

/* ── merged tier table ── */
.mtt-tier-hdr {
  text-align: center !important;
  border-top: 3px solid currentColor;
  padding-top: 0.6rem !important;
}
.mtt-tier-name { display: block; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.02em; }
.mtt-tier-desc { display: block; font-size: 0.68rem; font-weight: 400; text-transform: none;
                 letter-spacing: 0; color: #9ca3af; margin-top: 0.1rem; }
.mtt-cell { text-align: center; vertical-align: middle !important; }
.mtt-val  { display: block; font-size: 1rem; font-weight: 700; color: #111827;
            font-variant-numeric: tabular-nums; }
.mtt-ci   { display: block; font-size: 0.7rem; color: #9ca3af; margin: 0.1rem 0 0.25rem; }
.mtt-spark-hdr { text-align: center !important; color: #9ca3af !important; font-weight: 500 !important; }
.mtt-spark { text-align: center; vertical-align: middle !important; padding: 0.3rem 1rem !important; }
.metric-group-header td {
  padding: 0.55rem 0.9rem 0.3rem;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #6b7280;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
  border-top: 1px solid #e5e7eb;
}
.metric-group-header.first-group td { border-top: none; }

/* ── CI stability tags ── */
.ci-tag {
  display: inline-block;
  font-size: 0.68rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  white-space: nowrap;
}
.ci-confident { background: #d1fae5; color: #065f46; }
.ci-moderate  { background: #fef3c7; color: #92400e; }
.ci-wide      { background: #fee2e2; color: #991b1b; }

/* ── n-samples footer ── */
.n-note {
  font-size: 0.78rem;
  color: #9ca3af;
  margin: 0.6rem 0 0;
  text-align: right;
}

/* ── CR section ── */
.cr-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.25rem;
}
@media (max-width: 560px) { .cr-grid { grid-template-columns: 1fr; } }

.cr-card {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  background: #f8faff;
  border-top: 4px solid #3b82f6;
}
.cr-card .cr-value {
  display: block;
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #111827;
  line-height: 1;
  margin-bottom: 0.3rem;
}
.cr-card .cr-label {
  display: block;
  font-size: 0.85rem;
  font-weight: 700;
  color: #374151;
  margin-bottom: 0.2rem;
}
.cr-card .cr-formula {
  display: block;
  font-size: 0.75rem;
  color: #9ca3af;
  font-style: italic;
}
.cr-interp {
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 0.9rem 1.1rem;
  font-size: 0.82rem;
  color: #4b5563;
  margin-top: 0;
}
.cr-interp strong { color: #111827; }

/* ── chart ── */
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-bottom: 1.25rem;
}
@media (max-width: 640px) { .charts-grid { grid-template-columns: 1fr; } }
.chart-card {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 1.25rem 1.5rem 1rem;
}
.chart-card h3 {
  font-size: 0.88rem;
  font-weight: 700;
  color: #111827;
  margin: 0 0 0.2rem;
}
.chart-card .chart-sub {
  font-size: 0.78rem;
  color: #6b7280;
  margin: 0 0 0.9rem;
}
.charts-grid-3 {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1.25rem;
}
@media (max-width: 560px) { .charts-grid-3 { grid-template-columns: 1fr; } }
.charts-row-3 {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 1.25rem;
}
@media (max-width: 660px) { .charts-row-3 { grid-template-columns: 1fr; } }
.span-full { grid-column: 1 / -1; }
.hb-badge {
  font-size: 0.6rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
  vertical-align: middle;
  white-space: nowrap;
  background: #eef2ff;
  color: #4338ca;
}
.chart-remark {
  font-size: 0.75rem;
  margin: 0.75rem 0 0;
  padding: 0.5rem 0.7rem;
  border-radius: 6px;
  line-height: 1.45;
}
.chart-remark .remark-arrow { font-weight: 700; margin-right: 0.15rem; }
.chart-remark.good    { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
.chart-remark.bad     { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.chart-remark.neutral { background: #f9fafb; color: #6b7280; border: 1px solid #e5e7eb; }

/* ── share bar ── */
.share-bar {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-bottom: 1.25rem;
}
.share-bar span {
  font-size: 0.82rem;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.share-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.45rem 1rem;
  border-radius: 7px;
  border: 1.5px solid #e5e7eb;
  background: #fff;
  font-size: 0.82rem;
  font-weight: 600;
  color: #374151;
  cursor: pointer;
  text-decoration: none;
  transition: background 0.15s, border-color 0.15s;
}
.share-btn:hover { background: #f9fafb; border-color: #d1d5db; }
.share-btn.primary {
  background: #111827;
  border-color: #111827;
  color: #fff;
}
.share-btn.primary:hover { background: #1f2937; }
.share-btn svg { flex-shrink: 0; }
.toast {
  position: fixed;
  bottom: 1.5rem;
  left: 50%;
  transform: translateX(-50%) translateY(8px);
  background: #111827;
  color: #fff;
  font-size: 0.85rem;
  padding: 0.55rem 1.2rem;
  border-radius: 8px;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s, transform 0.2s;
  z-index: 999;
}
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

/* ── footer ── */
.site-footer {
  margin-top: 2.5rem;
  padding: 1.75rem 2rem;
  background: #111827;
  border-radius: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
  align-items: center;
  justify-content: space-between;
}
.footer-brand {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}
.footer-brand .brand-name {
  font-size: 0.95rem;
  font-weight: 700;
  color: #f9fafb;
}
.footer-brand .brand-sub {
  font-size: 0.78rem;
  color: #6b7280;
}
.footer-links {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.footer-link {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 0.9rem;
  border-radius: 7px;
  border: 1.5px solid #374151;
  font-size: 0.8rem;
  font-weight: 600;
  color: #d1d5db;
  text-decoration: none;
  transition: border-color 0.15s, color 0.15s;
}
.footer-link:hover { border-color: #9ca3af; color: #fff; }
.footer-link.hf { border-color: #f59e0b55; color: #fbbf24; }
.footer-link.hf:hover { border-color: #f59e0b; }

/* ── reading guide ── */
.guide { margin-bottom: 1.25rem; }
.guide h2 { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6b7280; margin: 0 0 1.1rem; }
.guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 600px) { .guide-grid { grid-template-columns: 1fr; } }
.guide-block h3 { font-size: 0.82rem; font-weight: 700; color: #111827; margin: 0 0 0.55rem; }
.guide-block dl { margin: 0; }
.guide-block dt { font-size: 0.78rem; font-weight: 600; color: #374151; margin-top: 0.55rem; }
.guide-block dd { font-size: 0.78rem; color: #6b7280; margin: 0.1rem 0 0 0; line-height: 1.5; }
.guide-block p  { font-size: 0.78rem; color: #6b7280; margin: 0.4rem 0 0; line-height: 1.5; }
.guide-tier { display: flex; gap: 0.5rem; flex-direction: column; }
.guide-tier-row { display: flex; gap: 0.65rem; align-items: flex-start; font-size: 0.78rem; }
.guide-tier-badge { font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.55rem; border-radius: 4px; white-space: nowrap; flex-shrink: 0; margin-top: 0.15rem; }
.guide-tier-c0 { background: #dcfce7; color: #166534; }
.guide-tier-c2 { background: #fef3c7; color: #92400e; }
.guide-tier-c4 { background: #fee2e2; color: #991b1b; }
.guide-tier-text { color: #6b7280; line-height: 1.5; }
.guide-ci-table { width: 100%; border-collapse: collapse; font-size: 0.76rem; margin-top: 0.3rem; }
.guide-ci-table th { text-align: left; font-weight: 600; color: #374151; padding: 0.3rem 0.5rem 0.3rem 0; border-bottom: 1px solid #e5e7eb; }
.guide-ci-table td { padding: 0.35rem 0.5rem 0.35rem 0; border-bottom: 1px solid #f3f4f6; color: #6b7280; vertical-align: middle; }
.guide-ci-table td:last-child { color: #374151; }
.guide-cr-table { width: 100%; border-collapse: collapse; font-size: 0.76rem; margin-top: 0.3rem; }
.guide-cr-table th { text-align: left; font-weight: 600; color: #374151; padding: 0.3rem 0.5rem 0.3rem 0; border-bottom: 1px solid #e5e7eb; }
.guide-cr-table td { padding: 0.35rem 0.5rem 0.35rem 0; border-bottom: 1px solid #f3f4f6; color: #6b7280; }
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def _as_ci(v: Any) -> tuple[float, float]:
    return float(v[0]), float(v[1])


def _ci_tag(lo: float, hi: float) -> tuple[str, str]:
    """Return (label, css-class) based on CI width.

    Thresholds (absolute width):
      ≤ 0.05  → Confident   (narrow — score is stable)
      ≤ 0.15  → Moderate    (medium)
      > 0.15  → Less stable (wide — score varies with sample)
    """
    width = hi - lo
    if width <= 0.05:
        return "Confident", "ci-confident"
    elif width <= 0.15:
        return "Moderate", "ci-moderate"
    return "Less stable", "ci-wide"


def _cr(c0: float, c4: float) -> float:
    """CR = 1 − (val_C0 − val_C4) / val_C0; returns 1.0 when val_C0 == 0."""
    if c0 == 0.0:
        return 1.0
    return min(1.0, 1.0 - (c0 - c4) / c0)


def _cr_label(cr: float) -> str:
    if cr >= 0.9:
        return "Excellent — almost no degradation"
    if cr >= 0.7:
        return "Moderate degradation"
    if cr >= 0.5:
        return "Significant degradation"
    return "Severe degradation — model heavily relies on compression artifacts"


def _cr_remark(cr: float) -> str:
    """Return a chart-remark block summarising the CR value."""
    if cr >= 0.9:
        cls  = "good"
        body = f"CR = {cr:.3f} — Strong robustness. Performance is nearly unaffected by aggressive JPEG re-compression."
    elif cr >= 0.7:
        cls  = "neutral"
        body = f"CR = {cr:.3f} — Moderate robustness. Some dependence on compression artifact fingerprints."
    elif cr >= 0.5:
        cls  = "bad"
        body = f"CR = {cr:.3f} — Significant degradation. Model performance drops noticeably as forensic signals erode."
    else:
        cls  = "bad"
        body = f"CR = {cr:.3f} — Severe degradation. Model relies heavily on DCT/compression artifact signals."
    return f'<p class="chart-remark {cls}">{body}</p>'


def _missing_authentic_warning(tier_scores: dict[str, Any]) -> str:
    """Return an HTML warning banner when authentic predictions are entirely absent.

    Detected by doc_auc == 0.5 (one-class fallback) AND doc_fpr == 0.0 across
    all scored tiers — this combination is otherwise statistically improbable.
    """
    scored = [s for s in tier_scores.values() if s]
    if not scored:
        return ""
    all_missing = all(
        abs(float(s.get("doc_auc", 1.0)) - 0.5) < 1e-6
        and float(s.get("doc_fpr", 1.0)) == 0.0
        for s in scored
    )
    if not all_missing:
        return ""
    return (
        '<div style="background:#fef3c7;border-left:4px solid #d97706;'
        'padding:0.65rem 0.9rem;margin:0.75rem 0 1rem;border-radius:0 4px 4px 0;'
        'font-size:8.5pt;font-family:Helvetica,Arial,sans-serif;color:#78350f">'
        "<strong>&#9888;&ensp;Missing authentic predictions</strong> &mdash; "
        "FPR is 0.0 and Doc-AUC is 0.5 (one-class fallback) across all tiers, "
        "which indicates that no prediction files were found for authentic (non-tampered) images. "
        "Add a <code>regions: []</code> prediction file for every authentic image "
        "so FPR and Doc-AUC can be computed correctly."
        "</div>"
    )


# ---------------------------------------------------------------------------
# HTML blocks
# ---------------------------------------------------------------------------



def _metric_rows(
    metrics: list[tuple[str, str, str]],
    scores: dict[str, Any],
    accent: str,
    validity_map: dict[str, bool] | None = None,
) -> tuple[str, bool]:
    """Return ``(html_rows, any_invalid)``."""
    rows: list[str] = []
    any_invalid = False
    for val_key, ci_key, label in metrics:
        raw = scores.get(val_key)
        if raw is None:
            continue
        val = float(raw)
        ci_raw = scores.get(ci_key)
        lo, hi = _as_ci(ci_raw) if ci_raw is not None else (val, val)
        tag_label, tag_cls = _ci_tag(lo, hi)

        valid = True if (validity_map is None) else validity_map.get(val_key, True)
        val_display = _f(val) + ("" if valid else " †")
        if not valid:
            any_invalid = True

        rows.append(
            f"<tr class='tier-stripe' style='--tier-color:{accent}'>"
            f"<td class='metric-name'>{label}</td>"
            f"<td class='metric-val'>{val_display}</td>"
            f"<td class='ci-range'>[{_f(lo, 3)}, {_f(hi, 3)}]</td>"
            f"<td><span class='ci-tag {tag_cls}'>{tag_label}</span></td>"
            f"</tr>"
        )
    return "".join(rows), any_invalid


def _tier_section(tier: str, scores: dict[str, Any]) -> str:
    title, desc = _TIER_META.get(tier, (tier, ""))
    accent = _TIER_ACCENT.get(tier, "#6b7280")
    n = scores.get("n_samples", "?")

    validity_map = {
        vk: bool(scores.get(fk, True))
        for vk, fk in _DETECTION_VALIDITY_KEYS.items()
    }
    localisation_rows, _ = _metric_rows(_LOCALISATION_METRICS, scores, accent)
    detection_rows, any_invalid = _metric_rows(_DETECTION_METRICS, scores, accent, validity_map)

    achieved = float(scores.get("doc_tpr90_achieved_tpr", 0.9))
    footnote = (
        f"<p class='n-note'>† TPR ≥ 0.90 not reachable — "
        f"shown at max achievable TPR ({achieved:.2f}).</p>"
        if any_invalid else ""
    )

    return f"""
<div class="section">
  <h2 style="color:{accent}">{title}</h2>
  <p class="tier-desc">{desc}</p>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Score</th>
        <th>95% CI</th>
        <th>Stability</th>
      </tr>
    </thead>
    <tbody>
      <tr class="metric-group-header first-group">
        <td colspan="4">Localisation Metrics</td>
      </tr>
      {localisation_rows}
      <tr class="metric-group-header">
        <td colspan="4">Document-Level Classification</td>
      </tr>
      {detection_rows}
    </tbody>
  </table>
  <p class="n-note">n = {n} evaluated samples</p>
  {footnote}
</div>"""


def _sparkline_svg(
    values: list[float],
    higher_is_better: bool,
    width: int = 88,
    height: int = 34,
) -> str:
    """Return an inline SVG sparkline for a metric's C0→C2→C4 values."""
    n = len(values)
    if n < 2:
        return ""

    vmin, vmax = min(values), max(values)
    span = vmax - vmin if vmax != vmin else 1.0

    pad = 5
    iw = width - 2 * pad
    ih = height - 2 * pad

    xs = [pad + i * iw / (n - 1) for i in range(n)]
    ys = [pad + ih * (1.0 - (v - vmin) / span) for v in values]  # SVG y-axis is inverted

    # Trend color: compare first vs last value
    delta = values[-1] - values[0]
    eps = span * 0.02
    if abs(delta) <= eps or span < 1e-9:
        stroke = "#9ca3af"   # gray — stable
        fill_dot = "#9ca3af"
    elif (delta > 0) == higher_is_better:
        stroke = "#16a34a"   # green — good direction
        fill_dot = "#16a34a"
    else:
        stroke = "#dc2626"   # red — bad direction
        fill_dot = "#dc2626"

    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="{fill_dot}"/>'
        for x, y in zip(xs, ys)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
        f' xmlns="http://www.w3.org/2000/svg" style="display:block;margin:auto">'
        f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="2"'
        f' stroke-linejoin="round" stroke-linecap="round"/>'
        f"{dots}"
        f"</svg>"
    )


def _merged_tier_table(tier_scores: dict[str, Any]) -> str:
    """Single table with metrics as rows and C0/C2/C4 as columns."""
    present = [t for t in _ALL_TIERS if t in tier_scores]
    if not present:
        return ""

    show_spark = len(present) >= 2
    n_cols = 1 + len(present) + (1 if show_spark else 0)

    # Column headers — tier name + short description + optional Trend column
    hdr_cells = "<th>Metric</th>"
    for tier in present:
        title, _ = _TIER_META.get(tier, (tier, ""))
        accent = _TIER_ACCENT.get(tier, "#6b7280")
        short_name = title.split(" — ")[0]
        short_desc = title.split(" — ", 1)[1] if " — " in title else ""
        hdr_cells += (
            f"<th class='mtt-tier-hdr' style='color:{accent}'>"
            f"<span class='mtt-tier-name'>{short_name}</span>"
            f"<span class='mtt-tier-desc'>{short_desc}</span>"
            f"</th>"
        )
    if show_spark:
        hdr_cells += "<th class='mtt-spark-hdr'>Trend</th>"

    def _tier_cell(val_key: str, ci_key: str, tier: str) -> str:
        s = tier_scores[tier]
        raw = s.get(val_key)
        if raw is None:
            return "<td class='mtt-cell'><span class='mtt-val'>—</span></td>"
        val = float(raw)
        ci_raw = s.get(ci_key)
        lo, hi = _as_ci(ci_raw) if ci_raw is not None else (val, val)
        tag_label, tag_cls = _ci_tag(lo, hi)
        valid = bool(s.get(_DETECTION_VALIDITY_KEYS.get(val_key, ""), True))
        val_display = _f(val) + ("" if valid else " †")
        return (
            f"<td class='mtt-cell'>"
            f"<span class='mtt-val'>{val_display}</span>"
            f"<span class='mtt-ci'>[{_f(lo, 3)}, {_f(hi, 3)}]</span>"
            f"<span class='ci-tag {tag_cls}'>{tag_label}</span>"
            f"</td>"
        )

    def _metric_row(label: str, val_key: str, ci_key: str) -> str:
        cells = f"<td class='metric-name'>{label}</td>"
        for tier in present:
            cells += _tier_cell(val_key, ci_key, tier)
        if show_spark:
            vals = [float(tier_scores[t][val_key]) for t in present]
            hib = _METRIC_HIGHER_IS_BETTER.get(val_key, True)
            svg = _sparkline_svg(vals, hib)
            cells += f"<td class='mtt-spark'>{svg}</td>"
        return f"<tr>{cells}</tr>"

    def _group_row(label: str, first: bool = False) -> str:
        cls = "metric-group-header first-group" if first else "metric-group-header"
        return f"<tr class='{cls}'><td colspan='{n_cols}'>{label}</td></tr>"

    loc_rows = "".join(
        _metric_row(label, vk, ck) for vk, ck, label in _LOCALISATION_METRICS
    )
    det_rows = "".join(
        _metric_row(label, vk, ck) for vk, ck, label in _DETECTION_METRICS
    )

    n_note = " &nbsp;·&nbsp; ".join(
        f"{t}: {tier_scores[t]['n_samples']:,} samples" for t in present
    )

    return f"""
<div class="section" style="padding-left:0;padding-right:0">
  <table>
    <thead>
      <tr>{hdr_cells}</tr>
    </thead>
    <tbody>
      {_group_row("Localisation Metrics", first=True)}
      {loc_rows}
      {_group_row("Document-Level Classification")}
      {det_rows}
    </tbody>
  </table>
  <p class="n-note" style="padding-right:1.5rem">{n_note}</p>
</div>"""


def _cr_section(tier_scores: dict[str, Any]) -> str:
    c0 = tier_scores.get("C0")
    c4 = tier_scores.get("C4")
    if c0 is None or c4 is None:
        return ""

    cr_auc = _cr(float(c0["doc_auc"]), float(c4["doc_auc"]))
    cr_f1  = _cr(float(c0["region_f1"]), float(c4["region_f1"]))

    return f"""
<div class="section">
  <h2>Compression Robustness</h2>
  <p class="tier-desc">
    How much does performance degrade when forensic signals are erased by JPEG re-compression?
    A score of 1.0 means no degradation; lower values indicate the model depends heavily on
    compression artifact fingerprints. CR is clamped to 1.0 — values where C4 performance
    exceeds C0 are treated as no degradation (statistical variation).
  </p>

  <div class="cr-grid">
    <div class="cr-card">
      <span class="cr-value">{_f(cr_auc, 3)}</span>
      <span class="cr-label">CR<sub>DocAUC</sub> — Detection Robustness</span>
      <span class="cr-formula">1 &minus; (AUC<sub>C0</sub> &minus; AUC<sub>C4</sub>) / AUC<sub>C0</sub></span>
      {_cr_remark(cr_auc)}
    </div>
    <div class="cr-card">
      <span class="cr-value">{_f(cr_f1, 3)}</span>
      <span class="cr-label">CR<sub>RegionF1</sub> — Localisation Robustness</span>
      <span class="cr-formula">1 &minus; (F1<sub>C0</sub> &minus; F1<sub>C4</sub>) / F1<sub>C0</sub></span>
      {_cr_remark(cr_f1)}
    </div>
  </div>

  <div class="cr-interp">
    <strong>Interpretation guide</strong> &mdash;
    ≥ 0.9 = almost unaffected &nbsp;|&nbsp;
    0.7 – 0.9 = moderate &nbsp;|&nbsp;
    0.5 – 0.7 = significant &nbsp;|&nbsp;
    &lt; 0.5 = severe
  </div>
</div>"""


def _multi_chart(
    canvas_id: str,
    series: list[tuple[str, str, str]],  # (val_key, label, color)
    tier_scores: dict[str, Any],
    y_scale: str = "fixed",  # "fixed" → 0..1 | "auto" → fit to data range
) -> str:
    """Return the JS block for a Chart.js line chart with multiple series.

    CI shading is omitted in multi-series charts to avoid overlapping fills;
    CI values remain visible in the metric tables.
    Use ``y_scale="auto"`` when metric values are very small so Chart.js
    zooms the axis to the actual data range.
    """
    def _v(tier: str, key: str) -> str:
        s = tier_scores.get(tier)
        if s is None:
            return "null"
        v = s.get(key)
        return "null" if v is None else str(round(float(v), 6))

    labels_js = json.dumps(["C0", "C2", "C4"])

    datasets_js_parts: list[str] = []
    for val_key, label, color in series:
        data = [_v(t, val_key) for t in _ALL_TIERS]
        data_js = f"[{', '.join(data)}]"
        datasets_js_parts.append(f"""        {{
          label: {json.dumps(label)},
          data: {data_js},
          borderColor: '{color}',
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.3,
          pointRadius: 5,
          pointBackgroundColor: '{color}',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          borderWidth: 2.5,
        }}""")

    datasets_js = ",\n".join(datasets_js_parts)

    return f"""
<script>
(function () {{
  new Chart(document.getElementById('{canvas_id}'), {{
    type: 'line',
    data: {{
      labels: {labels_js},
      datasets: [
{datasets_js}
      ]
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ font: {{ size: 12 }} }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y != null ? ctx.parsed.y.toFixed(4) : 'N/A'}}`
          }}
        }}
      }},
      scales: {{
        y: {{
          {("min: 0, max: 1, ticks: { stepSize: 0.2 }," if y_scale == "fixed" else "beginAtZero: true,")}
          grid: {{ color: '#f3f4f6' }}
        }},
        x: {{ grid: {{ color: '#f3f4f6' }} }}
      }}
    }}
  }});
}})();
</script>"""


def _trend_remark(
    tier_scores: dict[str, Any],
    val_key: str,
    higher_is_better: bool,
) -> str:
    """Return an HTML remark on whether a metric improves or worsens C0 → C4.

    Compares the C0 and C4 values, accounting for the direction of goodness
    (FPR is lower-is-better; all others higher-is-better).  Falls back to a
    neutral note when either endpoint tier is missing.
    """
    c0 = tier_scores.get("C0")
    c4 = tier_scores.get("C4")
    if not c0 or not c4:
        return (
            '<p class="chart-remark neutral">'
            '<span class="remark-arrow">–</span>'
            "Needs both C0 and C4 to assess the trend.</p>"
        )

    if c0.get(val_key) is None or c4.get(val_key) is None:
        return (
            '<p class="chart-remark neutral">'
            '<span class="remark-arrow">–</span>'
            "Metric not available in this result set.</p>"
        )
    v0 = float(c0[val_key])
    v4 = float(c4[val_key])
    delta = v4 - v0
    eps = 0.005  # treat sub-0.005 moves as noise

    # arrow tracks the value's direction; cls/verb track whether that is good or bad
    if abs(delta) <= eps:
        cls, arrow, verb = "neutral", "▬", "Stable"
    elif delta > 0:
        arrow = "▲"
        cls, verb = ("good", "Improves") if higher_is_better else ("bad", "Worsens")
    else:
        arrow = "▼"
        cls, verb = ("bad", "Worsens") if higher_is_better else ("good", "Improves")

    if cls == "neutral":
        body = f"Stable across tiers ({v0:.3f} → {v4:.3f})."
    else:
        body = (
            f"{verb} by {abs(delta):.3f} ({v0:.3f} → {v4:.3f}) "
            f"as compression increases C0 → C4."
        )
    return f'<p class="chart-remark {cls}"><span class="remark-arrow">{arrow}</span>{body}</p>'


def _detection_chart_card(
    title: str,
    canvas_id: str,
    val_key: str,
    color: str,
    higher_is_better: bool,
    y_scale: str,
    tier_scores: dict[str, Any],
    extra_cls: str = "",
) -> tuple[str, str]:
    """Build one single-metric chart card with direction badge and trend remark.

    Returns a ``(html, js)`` pair.  Pass ``extra_cls`` to add CSS classes to
    the outer ``chart-card`` div (e.g. ``"span-full"`` to span a grid row).
    """
    direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
    js = _multi_chart(canvas_id, [(val_key, title, color)], tier_scores, y_scale=y_scale)
    remark = _trend_remark(tier_scores, val_key, higher_is_better)
    cls = f"chart-card {extra_cls}".strip()
    html = f"""  <div class="{cls}">
    <h3>{title} <span class="hb-badge">{direction}</span></h3>
    <p class="chart-sub">Across C0 → C2 → C4</p>
    <canvas id="{canvas_id}" height="200"></canvas>
    {remark}
  </div>"""
    return html, js


def _degradation_charts(tier_scores: dict[str, Any]) -> str:
    detection_charts = [
        _detection_chart_card("Doc-AUC", "detAucChart", "doc_auc", "#3b82f6", True,  "fixed", tier_scores),
        _detection_chart_card("Doc-F1",  "detF1Chart",  "doc_f1",  "#8b5cf6", True,  "fixed", tier_scores),
        _detection_chart_card("FPR",     "detFprChart", "doc_fpr", "#f97316", False, "auto",  tier_scores),
    ]
    detection_cards_html = "\n".join(html for html, _ in detection_charts)
    detection_js = "\n".join(js for _, js in detection_charts)

    localisation_charts = [
        _detection_chart_card("Region Precision", "locPChart",  "region_precision", "#06b6d4", True, "auto", tier_scores),
        _detection_chart_card("Region Recall",    "locRChart",  "region_recall",    "#f59e0b", True, "auto", tier_scores),
        _detection_chart_card("Region F1",        "locF1Chart", "region_f1",        "#10b981", True, "auto", tier_scores),
    ]
    localisation_cards_html = "\n".join(html for html, _ in localisation_charts)
    localisation_js = "\n".join(js for _, js in localisation_charts)

    tprf_charts = [
        _detection_chart_card("Doc AUPRC",    "detAuprcChart",  "doc_auprc",        "#0ea5e9", True,  "fixed", tier_scores),
        _detection_chart_card("FPR @ TPR=90", "detFprT90Chart", "doc_fpr_at_tpr90", "#f43f5e", False, "auto",  tier_scores),
        _detection_chart_card("F1 @ TPR=90",  "detF1T90Chart",  "doc_f1_at_tpr90",  "#a855f7", True,  "fixed", tier_scores),
    ]
    tprf_cards_html = "\n".join(html for html, _ in tprf_charts)
    tprf_js = "\n".join(js for _, js in tprf_charts)

    return f"""
<div class="section" style="padding:0.75rem 2rem">
  <h2 style="font-size:0.75rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6b7280;margin:0 0 0.35rem">
    Score Degradation across Compression Tiers
  </h2>
  <p style="font-size:0.82rem;color:#6b7280;margin:0;line-height:1.5">
    How each metric degrades as JPEG re-compression erases forensic signals.
    Each chart shows its goodness direction and a C0 → C4 trend remark.
  </p>
</div>
<div class="section" style="padding:0 2rem 0.25rem">
  <h3 style="font-size:0.82rem;font-weight:700;color:#374151;margin:0">Localisation Metrics</h3>
</div>
<div class="charts-row-3">
{localisation_cards_html}
</div>
<div class="section" style="padding:0 2rem 0.25rem">
  <h3 style="font-size:0.82rem;font-weight:700;color:#374151;margin:0">Document-Level Classification Metrics</h3>
</div>
<div class="charts-row-3">
{detection_cards_html}
</div>
<div class="section" style="padding:0 2rem 0.25rem">
  <h3 style="font-size:0.82rem;font-weight:700;color:#374151;margin:0">Threshold-Free &amp; Anchored Detection Metrics</h3>
</div>
<div class="charts-row-3">
{tprf_cards_html}
</div>
{localisation_js}
{detection_js}
{tprf_js}"""


_GH_URL   = "https://github.com/Kriyam-ai/kriyam-tamperflow"
_HF_URL   = "https://huggingface.co/datasets/kriyam-ai/kriyam-tamperflow"
_LABS_URL = "https://labs.kriyam.ai/"

_ICON_GITHUB = (
    "<svg width='16' height='16' viewBox='0 0 24 24' fill='currentColor'>"
    "<path d='M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58"
    "0-.28-.01-1.03-.02-2.03-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76"
    "-1.09-.75.08-.73.08-.73 1.21.08 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.5 1"
    ".11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22"
    "-.14-.3-.54-1.52.1-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02.005"
    " 2.04.138 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.64 1.66.24 2.88.12 3.18.77.84"
    " 1.23 1.91 1.23 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.81 1.1.81 2.22 0 1.6"
    "-.01 2.9-.01 3.29 0 .32.21.69.82.57C20.57 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z'/>"
    "</svg>"
)

_ICON_SHARE = (
    "<svg width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor'"
    " stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
    "<circle cx='18' cy='5' r='3'/><circle cx='6' cy='12' r='3'/>"
    "<circle cx='18' cy='19' r='3'/>"
    "<line x1='8.59' y1='13.51' x2='15.42' y2='17.49'/>"
    "<line x1='15.41' y1='6.51' x2='8.59' y2='10.49'/>"
    "</svg>"
)
_ICON_COPY = (
    "<svg width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='currentColor'"
    " stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
    "<rect x='9' y='9' width='13' height='13' rx='2'/>"
    "<path d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'/>"
    "</svg>"
)


def _share_bar(model_name: str, tier_scores: dict[str, Any]) -> str:
    """Return the share action bar shown above the footer."""
    # Build a plain-text summary for clipboard sharing
    lines = [f"Kriyam TamperFlow — {model_name}"]
    for tier in _ALL_TIERS:
        s = tier_scores.get(tier)
        if s is None:
            continue
        lines.append(
            f"  {tier}: Region-F1={s['region_f1']:.4f}  Doc-AUC={s['doc_auc']:.4f}"
            f"  FPR={s['doc_fpr']:.4f}"
        )
    c0, c4 = tier_scores.get("C0"), tier_scores.get("C4")
    if c0 and c4:
        cr_auc = _cr(float(c0["doc_auc"]), float(c4["doc_auc"]))
        cr_f1  = _cr(float(c0["region_f1"]), float(c4["region_f1"]))
        lines.append(f"  CR_DocAUC={cr_auc:.3f}  CR_RegionF1={cr_f1:.3f}")
    lines.append(_GH_URL)
    copy_text = "\\n".join(lines).replace("'", "\\'")

    title_js = json.dumps(f"Kriyam TamperFlow — {model_name} results")

    return f"""
<div class="share-bar">
  <span>Share</span>
  <button class="share-btn primary" onclick="shareReport({title_js})">
    {_ICON_SHARE} Share report
  </button>
  <button class="share-btn" onclick="copyText('{copy_text}')">
    {_ICON_COPY} Copy results
  </button>
</div>
<div class="toast" id="toast"></div>
<script>
function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}}
function copyText(text) {{
  navigator.clipboard.writeText(text.replace(/\\\\n/g, '\\n'))
    .then(() => showToast('Results copied to clipboard'))
    .catch(() => showToast('Copy failed — try Ctrl+A, Ctrl+C'));
}}
function shareReport(title) {{
  if (navigator.share) {{
    navigator.share({{ title, url: window.location.href }})
      .catch(() => {{}});
  }} else {{
    navigator.clipboard.writeText(window.location.href)
      .then(() => showToast('Report link copied to clipboard'))
      .catch(() => showToast('Open this file in a browser to share'));
  }}
}}
</script>"""


def _footer() -> str:
    """Return the page footer with GitHub and HuggingFace links."""
    return f"""
<footer class="site-footer">
  <div class="footer-brand">
    <span class="brand-name">Kriyam TamperFlow</span>
    <span class="brand-sub">Document tampering detection benchmark for Indian documents</span>
  </div>
  <div class="footer-links">
    <a class="footer-link" href="{_GH_URL}" target="_blank" rel="noopener">
      {_ICON_GITHUB} Star on GitHub
    </a>
    <a class="footer-link hf" href="{_HF_URL}" target="_blank" rel="noopener">
      🤗 Follow on Hugging Face
    </a>
    <a class="footer-link" href="{_GH_URL}/issues" target="_blank" rel="noopener">
      Report an issue
    </a>
    <a class="footer-link" href="{_LABS_URL}" target="_blank" rel="noopener">
      Kriyam AI Labs
    </a>
  </div>
</footer>"""


# ---------------------------------------------------------------------------
# Reading guide (shared across templates)
# ---------------------------------------------------------------------------


def _reading_guide() -> str:
    """Return an HTML 'How to read this report' guide section."""
    return """
<div class="guide section">
  <h2>How to Read This Report</h2>
  <div class="guide-grid">

    <!-- Compression tiers -->
    <div class="guide-block">
      <h3>Compression Tiers</h3>
      <p>Each source document is evaluated at three JPEG re-compression levels.
         Scores are reported independently per tier so you can see how a model degrades
         as forensic signals are erased.</p>
      <div class="guide-tier" style="margin-top:0.7rem">
        <div class="guide-tier-row">
          <span class="guide-tier-badge guide-tier-c0">C0</span>
          <span class="guide-tier-text"><strong>Pristine.</strong> No re-compression. Lossless PNG.
            All artifact signals intact. Represents the model's upper-bound performance.</span>
        </div>
        <div class="guide-tier-row">
          <span class="guide-tier-badge guide-tier-c2">C2</span>
          <span class="guide-tier-text"><strong>Double-pass JPEG</strong> (Q=85 → Q=80).
            Simulates a typical scan → email → resave workflow. Most real-world documents fall here.</span>
        </div>
        <div class="guide-tier-row">
          <span class="guide-tier-badge guide-tier-c4">C4</span>
          <span class="guide-tier-text"><strong>Photocopy simulation</strong> — blur + noise + JPEG Q=70.
            Represents heavily degraded documents (faxed IDs, photocopied forms). Hardest tier.</span>
        </div>
      </div>
    </div>

    <!-- Localisation metrics -->
    <div class="guide-block">
      <h3>Localisation Metrics</h3>
      <p>Measure how accurately the model finds the <em>location</em> of tampered regions.
         Regions are matched using Hungarian assignment with IoU&nbsp;≥&nbsp;0.1.</p>
      <dl>
        <dt>Region Precision (Region-P)</dt>
        <dd>Of all predicted regions, what fraction overlapped a real tampered region?
            High precision = few false alarms.</dd>
        <dt>Region Recall (Region-R)</dt>
        <dd>Of all real tampered regions, what fraction did the model find?
            High recall = few missed forgeries.</dd>
        <dt>Region F1 (Region-F1)</dt>
        <dd>Harmonic mean of precision and recall. The primary localisation score.
            Balances both false alarms and missed detections.</dd>
      </dl>
    </div>

    <!-- Document-level classification metrics -->
    <div class="guide-block">
      <h3>Document-Level Classification</h3>
      <p>Measure whether the model correctly classifies each document as authentic or tampered,
         without needing to localise the exact region.</p>
      <dl>
        <dt>Doc-AUC</dt>
        <dd>Area under the ROC curve for document-level classification. Threshold-free.
            0.5&nbsp;= random; 1.0&nbsp;= perfect. Uses max(region confidence) as the document score.</dd>
        <dt>Doc-F1</dt>
        <dd>Binary classification F1 at document level. Derived from
            pred_label = 1 if confidence ≥ document threshold (default 0.5).</dd>
        <dt>FPR — False Positive Rate</dt>
        <dd>Fraction of authentic documents incorrectly flagged as tampered.
            FPR = FP&nbsp;/&nbsp;(FP&nbsp;+&nbsp;TN). Lower is better.</dd>
      </dl>
    </div>

    <!-- Confidence interval -->
    <div class="guide-block">
      <h3>Confidence Interval &amp; Stability</h3>
      <p>A 95% CI shows how much a score might change if the benchmark were run on a
         slightly different set of documents. Computed via bootstrap resampling
         (1,000 iterations): the dataset is resampled with replacement and the metric
         recomputed each time; the middle 95% of those scores forms the interval.</p>
      <table class="guide-ci-table" style="margin-top:0.6rem">
        <thead>
          <tr><th>Tag</th><th>CI width</th><th>Meaning</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="ci-tag ci-confident">Confident</span></td>
            <td>≤ 0.05</td>
            <td>Score is stable — would change little with a different sample.</td>
          </tr>
          <tr>
            <td><span class="ci-tag ci-moderate">Moderate</span></td>
            <td>0.05 – 0.15</td>
            <td>Some variability; interpret with mild caution.</td>
          </tr>
          <tr>
            <td><span class="ci-tag ci-wide">Less stable</span></td>
            <td>&gt; 0.15</td>
            <td>Score is sensitive to which documents are sampled — treat with caution.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Compression Robustness -->
    <div class="guide-block" style="grid-column: 1 / -1">
      <h3>Compression Robustness (CR)</h3>
      <p>Summarises how much a model's performance degrades from the pristine tier (C0) to the
         photocopy-simulation tier (C4). Computed as:
         <strong>CR = 1 − (score<sub>C0</sub> − score<sub>C4</sub>) / score<sub>C0</sub></strong>.
         Reported separately for Doc-AUC (detection) and Region-F1 (localisation).
         CR is clamped to 1.0 — if C4 outperforms C0, it is treated as no degradation (statistical variation).</p>
      <table class="guide-cr-table" style="margin-top:0.6rem;max-width:520px">
        <thead>
          <tr><th>CR score</th><th>Interpretation</th></tr>
        </thead>
        <tbody>
          <tr><td>≥ 0.9</td><td>Excellent — almost no degradation across compression tiers.</td></tr>
          <tr><td>0.7 – 0.9</td><td>Moderate — noticeable drop but model still functional.</td></tr>
          <tr><td>0.5 – 0.7</td><td>Significant — model relies heavily on JPEG artifact signals.</td></tr>
          <tr><td>&lt; 0.5</td><td>Severe — performance collapses under real-world compression.</td></tr>
        </tbody>
      </table>
    </div>

  </div>
</div>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _v1_html(model_name: str, tier_scores: dict[str, Any], document_threshold: float) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kriyam TamperFlow &mdash; {model_name}</title>
  <style>
{_CSS}
  </style>
  <script src="{_CHART_JS}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
</head>
<body>
<div class="page">

  <div class="report-header">
    <p class="bench-name">Kriyam TamperFlow &mdash; Evaluation Report</p>
    <h1>Model: {model_name}</h1>
    <p class="meta">
      Benchmark v1.0 &nbsp;&middot;&nbsp;
      1,050 documents (700 tampered, 350 authentic) &nbsp;&middot;&nbsp;
      Document threshold: {document_threshold:.2f}
    </p>
  </div>

  {_missing_authentic_warning(tier_scores)}

  {_merged_tier_table(tier_scores)}

  {_degradation_charts(tier_scores)}

  {_cr_section(tier_scores)}

  {_reading_guide()}

  {_share_bar(model_name, tier_scores)}

  {_footer()}

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Template v2 — research-report style
# ---------------------------------------------------------------------------

_CSS_V2 = """\
*, *::before, *::after { box-sizing: border-box; }

body {
  font-family: system-ui, -apple-system, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: #111827;
  background: #fff;
  margin: 0;
  padding: 2.5rem 1rem 5rem;
}

.v2-page { max-width: 820px; margin: 0 auto; }

/* ── section labels ── */
.v2-sec {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #9ca3af;
  border-bottom: 1.5px solid #e5e7eb;
  padding-bottom: 0.45rem;
  margin: 0 0 1.25rem;
}

/* ── header ── */
.v2-header { margin-bottom: 2rem; }
.v2-header .v2-bench {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #9ca3af;
  margin: 0 0 0.25rem;
}
.v2-header h1 {
  font-size: 1.55rem;
  font-weight: 800;
  color: #111827;
  margin: 0 0 0.2rem;
  letter-spacing: -0.02em;
}
.v2-header .v2-meta { font-size: 0.78rem; color: #9ca3af; margin: 0; }

/* ── overview cards ── */
.v2-overview {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 2.75rem;
}
@media (max-width: 580px) { .v2-overview { grid-template-columns: repeat(2, 1fr); } }
.v2-stat {
  padding: 1.1rem 1.2rem 1rem;
  border-right: 1px solid #e5e7eb;
}
.v2-stat:last-child { border-right: none; }
.v2-stat-lbl {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: #9ca3af;
  margin-bottom: 0.2rem;
}
.v2-stat-val {
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #111827;
  line-height: 1.05;
  margin-bottom: 0.2rem;
}
.v2-stat-note { font-size: 0.72rem; color: #6b7280; line-height: 1.35; }

/* ── chart sections ── */
.v2-chart-sec { margin-bottom: 2.5rem; }
.v2-chart-wrap {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 1.1rem 1.4rem 0.9rem;
}

/* ── tier breakdown ── */
.v2-tier-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.9rem;
  margin-bottom: 2.5rem;
}
@media (max-width: 580px) { .v2-tier-grid { grid-template-columns: 1fr; } }
.v2-tier-card {
  border: 1px solid #e5e7eb;
  border-top: 3px solid var(--acc);
  border-radius: 8px;
  padding: 1rem 1.15rem 0.85rem;
}
.v2-tier-card h3 {
  font-size: 0.88rem;
  font-weight: 800;
  color: #111827;
  margin: 0 0 0.1rem;
}
.v2-tier-card .v2-tdesc {
  font-size: 0.68rem;
  color: #9ca3af;
  margin: 0 0 0.85rem;
}
.v2-mrow {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.26rem 0;
  border-bottom: 1px solid #f3f4f6;
  font-size: 0.8rem;
  gap: 0.5rem;
}
.v2-mrow:last-child { border-bottom: none; }
.v2-mrow-name { color: #6b7280; white-space: nowrap; }
.v2-mrow-val { font-weight: 700; color: #111827; font-variant-numeric: tabular-nums; text-align: right; }

/* ── CR ── */
.v2-cr-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.9rem;
  margin-bottom: 2.5rem;
}
@media (max-width: 520px) { .v2-cr-grid { grid-template-columns: 1fr; } }
.v2-cr-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 1.1rem 1.25rem;
}
.v2-cr-num {
  font-size: 2rem;
  font-weight: 800;
  color: #111827;
  letter-spacing: -0.03em;
  line-height: 1;
  margin-bottom: 0.25rem;
}
.v2-cr-name { font-size: 0.8rem; font-weight: 700; color: #374151; margin-bottom: 0.1rem; }
.v2-cr-interp { font-size: 0.75rem; color: #6b7280; }
.v2-cr-formula { font-size: 0.68rem; color: #9ca3af; font-style: italic; margin-top: 0.45rem; }

/* ── CI stability tags ── */
.ci-tag {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 600;
  padding: 0.1rem 0.4rem;
  border-radius: 4px;
  white-space: nowrap;
}
.ci-confident { background: #d1fae5; color: #065f46; }
.ci-moderate  { background: #fef3c7; color: #92400e; }
.ci-wide      { background: #fee2e2; color: #991b1b; }

/* ── footer ── */
.v2-footer {
  margin-top: 3rem;
  padding-top: 1.1rem;
  border-top: 1.5px solid #e5e7eb;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
  justify-content: space-between;
}
.v2-footer-brand { font-size: 0.75rem; color: #9ca3af; }
.v2-footer-links { display: flex; gap: 0.45rem; flex-wrap: wrap; }
.v2-footer-link {
  font-size: 0.72rem;
  color: #6b7280;
  text-decoration: none;
  padding: 0.25rem 0.6rem;
  border: 1px solid #e5e7eb;
  border-radius: 5px;
  transition: border-color 0.15s, color 0.15s;
}
.v2-footer-link:hover { border-color: #9ca3af; color: #111827; }

/* ── reading guide (shared with v1) ── */
.guide { margin-bottom: 1.25rem; }
.guide h2 { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #9ca3af; border-bottom: 1.5px solid #e5e7eb; padding-bottom: 0.45rem; margin: 0 0 1.1rem; }
.guide-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
@media (max-width: 600px) { .guide-grid { grid-template-columns: 1fr; } }
.guide-block h3 { font-size: 0.8rem; font-weight: 700; color: #111827; margin: 0 0 0.5rem; }
.guide-block dl { margin: 0; }
.guide-block dt { font-size: 0.76rem; font-weight: 600; color: #374151; margin-top: 0.5rem; }
.guide-block dd { font-size: 0.76rem; color: #6b7280; margin: 0.1rem 0 0 0; line-height: 1.5; }
.guide-block p  { font-size: 0.76rem; color: #6b7280; margin: 0.4rem 0 0; line-height: 1.5; }
.guide-tier { display: flex; gap: 0.5rem; flex-direction: column; }
.guide-tier-row { display: flex; gap: 0.65rem; align-items: flex-start; font-size: 0.76rem; }
.guide-tier-badge { font-size: 0.63rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 4px; white-space: nowrap; flex-shrink: 0; margin-top: 0.15rem; }
.guide-tier-c0 { background: #dcfce7; color: #166534; }
.guide-tier-c2 { background: #fef3c7; color: #92400e; }
.guide-tier-c4 { background: #fee2e2; color: #991b1b; }
.guide-tier-text { color: #6b7280; line-height: 1.5; }
.guide-ci-table, .guide-cr-table { width: 100%; border-collapse: collapse; font-size: 0.74rem; margin-top: 0.3rem; }
.guide-ci-table th, .guide-cr-table th { text-align: left; font-weight: 600; color: #374151; padding: 0.3rem 0.5rem 0.3rem 0; border-bottom: 1px solid #e5e7eb; }
.guide-ci-table td, .guide-cr-table td { padding: 0.32rem 0.5rem 0.32rem 0; border-bottom: 1px solid #f3f4f6; color: #6b7280; vertical-align: middle; }
"""


def _v2_overview(tier_scores: dict[str, Any]) -> str:
    c0 = tier_scores.get("C0", {})
    c4 = tier_scores.get("C4", {})

    doc_auc   = float(c0.get("doc_auc",   0.0))
    region_f1 = float(c0.get("region_f1", 0.0))
    fpr       = float(c0.get("doc_fpr",   0.0))
    cr_auc    = _cr(doc_auc, float(c4.get("doc_auc", 0.0))) if c4 else 1.0

    auc_note  = "above random" if doc_auc > 0.5 else "at or below random (0.5)"
    fpr_note  = f"{fpr * 100:.0f}% of authentic docs flagged"
    cr_note   = _cr_label(cr_auc)

    stats = [
        ("Doc-AUC (C0)",   f"{doc_auc:.3f}",   auc_note),
        ("Region-F1 (C0)", f"{region_f1:.4f}",  "localisation score at pristine tier"),
        ("FPR (C0)",       f"{fpr:.3f}",        fpr_note),
        ("CR — DocAUC",    f"{cr_auc:.3f}",     cr_note),
    ]
    cards = "".join(
        f"<div class='v2-stat'>"
        f"<div class='v2-stat-lbl'>{lbl}</div>"
        f"<div class='v2-stat-val'>{val}</div>"
        f"<div class='v2-stat-note'>{note}</div>"
        f"</div>"
        for lbl, val, note in stats
    )
    return f"<div class='v2-overview'>{cards}</div>"


def _v2_chart_section(
    section_label: str,
    canvas_id: str,
    series: list[tuple[str, str, str]],
    tier_scores: dict[str, Any],
    y_scale: str = "fixed",
    note: str = "",
) -> str:
    js = _multi_chart(canvas_id, series, tier_scores, y_scale=y_scale)
    note_html = f"<p style='font-size:0.7rem;color:#9ca3af;margin:0.5rem 0 0'>{note}</p>" if note else ""
    return f"""<div class="v2-chart-sec">
  <p class="v2-sec">{section_label}</p>
  <div class="v2-chart-wrap">
    <canvas id="{canvas_id}" height="210"></canvas>
    {note_html}
  </div>
</div>
{js}"""


def _v2_tier_breakdown(tier_scores: dict[str, Any]) -> str:
    _SHORT  = {"C0": "Pristine — no re-compression", "C2": "Double-pass JPEG (Q=85→80)", "C4": "Photocopy simulation (Q=70)"}
    _ACCENT = {"C0": "#16a34a", "C2": "#d97706", "C4": "#dc2626"}
    # (val_key, ci_key, display_name)
    _METRIC_KEYS = [
        ("region_precision",  "region_precision_ci",  "Region-P"),
        ("region_recall",     "region_recall_ci",     "Region-R"),
        ("region_f1",         "region_f1_ci",         "Region-F1"),
        ("doc_auc",           "doc_auc_ci",           "Doc-AUC"),
        ("doc_auprc",         "doc_auprc_ci",         "Doc-AUPRC"),
        ("doc_f1",            "doc_f1_ci",            "Doc-F1"),
        ("doc_fpr",           "doc_fpr_ci",           "FPR"),
        ("doc_fpr_at_tpr90",  "doc_fpr_at_tpr90_ci",  "FPR@90"),
        ("doc_f1_at_tpr90",   "doc_f1_at_tpr90_ci",   "F1@90"),
    ]

    cards: list[str] = []
    for tier in _ALL_TIERS:
        s = tier_scores.get(tier)
        if s is None:
            continue
        acc  = _ACCENT.get(tier, "#6b7280")
        desc = _SHORT.get(tier, tier)
        n    = s.get("n_samples", "?")

        rows_html = ""
        for val_key, ci_key, name in _METRIC_KEYS:
            raw = s.get(val_key)
            if raw is None:
                continue
            val = _f(float(raw))
            ci_raw = s.get(ci_key)
            lo, hi = _as_ci(ci_raw) if ci_raw is not None else (float(raw), float(raw))
            tag_label, tag_cls = _ci_tag(lo, hi)
            ci_range = f"[{_f(lo, 3)}, {_f(hi, 3)}]"
            valid = bool(s.get(_DETECTION_VALIDITY_KEYS.get(val_key, ""), True))
            val_display = val + ("" if valid else " †")
            rows_html += (
                f"<div class='v2-mrow' title='95% CI: {ci_range}'>"
                f"<span class='v2-mrow-name'>{name}</span>"
                f"<span class='v2-mrow-val'>"
                f"{val_display}&nbsp;<span class='ci-tag {tag_cls}'>{tag_label}</span>"
                f"</span>"
                f"</div>"
            )
        # n row (no CI)
        rows_html += (
            f"<div class='v2-mrow'>"
            f"<span class='v2-mrow-name'>n</span>"
            f"<span class='v2-mrow-val'>{n}</span>"
            f"</div>"
        )

        cards.append(
            f"<div class='v2-tier-card' style='--acc:{acc}'>"
            f"<h3>{tier}</h3>"
            f"<p class='v2-tdesc'>{desc}</p>"
            f"{rows_html}"
            f"</div>"
        )

    return f"<div class='v2-tier-grid'>{''.join(cards)}</div>"


def _v2_cr(tier_scores: dict[str, Any]) -> str:
    c0 = tier_scores.get("C0")
    c4 = tier_scores.get("C4")
    if not c0 or not c4:
        return ""

    cr_auc = _cr(float(c0["doc_auc"]),   float(c4["doc_auc"]))
    cr_f1  = _cr(float(c0["region_f1"]), float(c4["region_f1"]))

    def _card(val: float, name: str, formula: str) -> str:
        return (
            f"<div class='v2-cr-card'>"
            f"<div class='v2-cr-num'>{_f(val, 3)}</div>"
            f"<div class='v2-cr-name'>{name}</div>"
            f"<div class='v2-cr-interp'>{_cr_label(val)}</div>"
            f"<div class='v2-cr-formula'>{formula}</div>"
            f"</div>"
        )

    return (
        f"<div class='v2-cr-grid'>"
        + _card(cr_auc, "CR<sub>DocAUC</sub> &mdash; Detection Robustness",
                "1 &minus; (AUC<sub>C0</sub> &minus; AUC<sub>C4</sub>) / AUC<sub>C0</sub>")
        + _card(cr_f1,  "CR<sub>RegionF1</sub> &mdash; Localisation Robustness",
                "1 &minus; (F1<sub>C0</sub> &minus; F1<sub>C4</sub>) / F1<sub>C0</sub>")
        + "</div>"
    )


def _v2_footer() -> str:
    return (
        f"<div class='v2-footer'>"
        f"<span class='v2-footer-brand'>Kriyam TamperFlow &mdash; Benchmark v1.0</span>"
        f"<div class='v2-footer-links'>"
        f"<a class='v2-footer-link' href='{_GH_URL}' target='_blank' rel='noopener'>GitHub</a>"
        f"<a class='v2-footer-link' href='{_HF_URL}' target='_blank' rel='noopener'>🤗 Dataset</a>"
        f"<a class='v2-footer-link' href='{_GH_URL}/issues' target='_blank' rel='noopener'>Report issue</a>"
        f"</div>"
        f"</div>"
    )


def _v2_html(model_name: str, tier_scores: dict[str, Any], document_threshold: float) -> str:
    loc_series = [
        ("region_precision", "Region-P",  "#06b6d4"),
        ("region_recall",    "Region-R",  "#f59e0b"),
        ("region_f1",        "Region-F1", "#10b981"),
    ]
    det_series = [
        ("doc_auc",          "Doc-AUC",    "#3b82f6"),
        ("doc_auprc",        "Doc-AUPRC",  "#0ea5e9"),
        ("doc_f1",           "Doc-F1",     "#8b5cf6"),
        ("doc_fpr",          "FPR",        "#f97316"),
        ("doc_fpr_at_tpr90", "FPR@TPR=90", "#f43f5e"),
        ("doc_f1_at_tpr90",  "F1@TPR=90",  "#a855f7"),
    ]

    loc_chart = _v2_chart_section(
        "Localisation Metrics across Compression Tiers",
        "v2LocChart", loc_series, tier_scores,
        y_scale="auto",
        note="Y-axis auto-scaled to data range — values may be very small.",
    )
    det_chart = _v2_chart_section(
        "Detection Metrics across Compression Tiers",
        "v2DetChart", det_series, tier_scores,
        y_scale="fixed",
        note="Lower FPR and FPR@TPR=90 is better; all others higher is better.",
    )

    total_n = sum(
        int(tier_scores[t]["n_samples"]) for t in _ALL_TIERS if t in tier_scores
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kriyam TamperFlow &mdash; {model_name}</title>
  <style>
{_CSS_V2}
  </style>
  <script src="{_CHART_JS}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
</head>
<body>
<div class="v2-page">

  <div class="v2-header">
    <p class="v2-bench">Kriyam TamperFlow &mdash; Evaluation Report</p>
    <h1>{model_name}</h1>
    <p class="v2-meta">
      1,050 documents (700 tampered, 350 authentic) &nbsp;&middot;&nbsp;
      {total_n} predictions evaluated &nbsp;&middot;&nbsp;
      Document threshold: {document_threshold:.2f}
    </p>
  </div>

  {_missing_authentic_warning(tier_scores)}

  <p class="v2-sec">Overview</p>
  {_v2_overview(tier_scores)}

  {loc_chart}

  {det_chart}

  <p class="v2-sec">Per-Tier Breakdown</p>
  {_v2_tier_breakdown(tier_scores)}

  <p class="v2-sec">Compression Robustness</p>
  {_v2_cr(tier_scores)}

  {_reading_guide()}

  {_v2_footer()}

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Template v3 — academic / research-paper style
# ---------------------------------------------------------------------------

_CSS_V3 = """\
*, *::before, *::after { box-sizing: border-box; }

body {
  font-family: Georgia, 'Times New Roman', 'Palatino Linotype', serif;
  font-size: 10.5pt;
  line-height: 1.65;
  color: #111;
  background: #dde1e7;
  margin: 0;
  padding: 2rem 1rem 5rem;
}

.v3-page {
  max-width: 740px;
  margin: 0 auto;
  background: #fff;
  padding: 3rem 3.75rem 3.5rem;
  box-shadow: 0 2px 20px rgba(0,0,0,0.15);
}
@media (max-width: 640px) { .v3-page { padding: 1.5rem 1.25rem; } }

/* ── Title block ── */
.v3-title-block { text-align: center; margin-bottom: 1.5rem; }
.v3-paper-title {
  font-size: 18pt;
  font-weight: bold;
  line-height: 1.2;
  margin: 0 0 0.3rem;
  color: #111;
}
.v3-report-subtitle {
  font-size: 11pt;
  font-style: italic;
  color: #666;
  margin: 0 0 0.55rem;
  font-weight: normal;
}
.v3-model-line {
  font-size: 11pt;
  font-style: italic;
  color: #333;
  margin: 0 0 0.5rem;
}
.v3-meta-line {
  font-size: 8pt;
  font-family: Helvetica, Arial, sans-serif;
  color: #aaa;
  margin: 0;
}

/* ── Rules ── */
.v3-rule-double { border: none; border-top: 3px double #111; margin: 1.3rem 0; }
.v3-rule-thin   { border: none; border-top: 0.75pt solid #ccc; margin: 1.5rem 0; }

/* ── Abstract ── */
.v3-abstract {
  margin: 0 0.75rem;
  font-size: 9.5pt;
  text-align: justify;
  hyphens: auto;
}
.v3-abstract-kw {
  font-weight: bold;
  font-variant: small-caps;
  letter-spacing: 0.04em;
}

/* ── Section headings ── */
.v3-h2 {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 8.5pt;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #333;
  margin: 2rem 0 0.75rem;
  padding-bottom: 0.3rem;
  border-bottom: 0.75pt solid #ccc;
}

/* ── Booktabs tables ── */
.v3-tbl-wrap { overflow-x: auto; margin: 0.5rem 0 0.25rem; }
.v3-tbl {
  width: 100%;
  border-collapse: collapse;
  font-size: 8.5pt;
  font-family: Helvetica, Arial, sans-serif;
}
.v3-tbl thead th {
  text-align: right;
  padding: 0.32rem 0.65rem;
  font-weight: 700;
  border-top: 2pt solid #111;
  border-bottom: 1pt solid #111;
  white-space: nowrap;
}
.v3-tbl thead th:first-child { text-align: left; }
.v3-tbl tbody td {
  text-align: right;
  padding: 0.28rem 0.65rem;
  vertical-align: top;
}
.v3-tbl tbody td:first-child {
  text-align: left;
  font-weight: 600;
  white-space: nowrap;
}
.v3-tbl tbody tr.v3-row-last td { border-bottom: 2pt solid #111; }
.v3-tbl tbody tr.v3-row-cr td   { border-top: 0.75pt dashed #ccc; color: #555; }
.v3-tbl tbody tr:hover td       { background: #fafafa; }
.tbl-val { display: block; }
.tbl-ci  { display: block; font-size: 7pt; color: #aaa; white-space: nowrap; }
.v3-tbl-note {
  font-size: 8pt;
  font-style: italic;
  color: #666;
  margin: 0.35rem 0 0;
  line-height: 1.45;
}
.v3-tbl-note strong { font-style: normal; }

/* ── Figures / charts ── */
.v3-figure {
  margin: 1rem 0 0.25rem;
  border: 0.75pt solid #ddd;
  border-radius: 3px;
  padding: 1rem 1.25rem 0.7rem;
  background: #fcfcfc;
}
.v3-fig-title {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 9pt;
  font-weight: 700;
  color: #222;
  margin: 0 0 0.7rem;
  letter-spacing: 0.01em;
}
.v3-fig-note {
  font-size: 8pt;
  font-style: italic;
  color: #666;
  margin-top: 0.5rem;
  line-height: 1.45;
}
.v3-fig-note strong { font-style: normal; }

/* ── CR boxes ── */
.v3-cr-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin: 0.5rem 0 0.25rem;
}
@media (max-width: 480px) { .v3-cr-grid { grid-template-columns: 1fr; } }
.v3-cr-box {
  border: 0.75pt solid #ddd;
  border-radius: 3px;
  padding: 0.85rem 1rem;
  font-family: Helvetica, Arial, sans-serif;
}
.v3-cr-num     { font-size: 1.9rem; font-weight: 800; color: #111; letter-spacing: -0.03em; line-height: 1; display: block; }
.v3-cr-name    { font-size: 8pt;  color: #444; display: block; margin-top: 0.3rem; }
.v3-cr-interp  { font-size: 7.5pt; color: #888; font-style: italic; margin-top: 0.2rem; display: block; }
.v3-cr-formula { font-size: 7pt;  color: #bbb; font-style: italic; margin-top: 0.35rem; display: block; }

/* ── Appendix reading guide ── */
.v3-appendix-h {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 8.5pt;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #aaa;
  margin: 2.5rem 0 0.75rem;
  padding-bottom: 0.3rem;
  border-bottom: 0.75pt solid #eee;
}
.v3-guide-cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem 2.5rem;
  font-size: 8.5pt;
}
@media (max-width: 540px) { .v3-guide-cols { grid-template-columns: 1fr; } }
.v3-guide-block h4 { font-size: 8.5pt; font-weight: bold; margin: 0 0 0.3rem; }
.v3-guide-block p  { font-size: 8pt; color: #555; margin: 0 0 0.3rem; line-height: 1.5; }
.v3-guide-block ul { font-size: 8pt; color: #555; margin: 0.3rem 0 0 1rem; padding: 0; line-height: 1.5; }
.v3-guide-block li { margin-bottom: 0.2rem; }
.v3-guide-block dl { margin: 0; }
.v3-guide-block dt { font-size: 8pt; font-weight: 600; margin-top: 0.4rem; }
.v3-guide-block dd { font-size: 8pt; color: #555; margin: 0.1rem 0 0 0.5rem; line-height: 1.45; }
.v3-guide-ci {
  width: 100%;
  border-collapse: collapse;
  font-size: 8pt;
  margin-top: 0.35rem;
  font-family: Helvetica, Arial, sans-serif;
}
.v3-guide-ci th {
  text-align: left;
  font-weight: 600;
  padding: 0.22rem 0.4rem 0.22rem 0;
  border-top: 1.5pt solid #ccc;
  border-bottom: 0.75pt solid #ccc;
}
.v3-guide-ci td {
  padding: 0.22rem 0.4rem 0.22rem 0;
  border-bottom: 0.5pt solid #eee;
  color: #555;
}
.v3-guide-ci tr:last-child td { border-bottom: 1.5pt solid #ccc; }

/* ── Footer ── */
.v3-footer {
  margin-top: 2.5rem;
  padding-top: 0.6rem;
  border-top: 0.75pt solid #ccc;
  font-size: 7.5pt;
  font-family: Helvetica, Arial, sans-serif;
  color: #bbb;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.3rem;
}
.v3-footer a { color: #888; text-decoration: none; }
.v3-footer a:hover { color: #444; }
"""


def _v3_abstract(model_name: str, tier_scores: dict[str, Any], document_threshold: float) -> str:
    total_n = sum(int(tier_scores[t]["n_samples"]) for t in _ALL_TIERS if t in tier_scores)
    tiers_present = [t for t in _ALL_TIERS if t in tier_scores]
    tiers_str = ", ".join(tiers_present)
    c0 = tier_scores.get("C0", {})
    c4 = tier_scores.get("C4", {})
    auc_c0 = float(c0["doc_auc"])   if c0 else None
    f1_c0  = float(c0["region_f1"]) if c0 else None
    auc_c4 = float(c4["doc_auc"])   if c4 else None

    if auc_c0 is not None and auc_c4 is not None:
        cr  = _cr(auc_c0, auc_c4)
        perf = (
            f" At the pristine tier (C0), the model attains a document-level AUC-ROC of "
            f"{_f(auc_c0)} and a Region-F1 of {_f(f1_c0 or 0.0)}."
            f" The detection Compression Robustness score is {_f(cr, 3)}"
            f" ({_cr_label(cr)})."
        )
    elif auc_c0 is not None:
        perf = (
            f" At the pristine tier (C0), the model attains a document-level"
            f" AUC-ROC of {_f(auc_c0)} and a Region-F1 of {_f(f1_c0 or 0.0)}."
        )
    else:
        perf = ""

    return f"""
<div class="v3-abstract">
  <p><span class="v3-abstract-kw">Summary.</span>
  This report presents evaluation results for <em>{model_name}</em> on the
  Kriyam TamperFlow, a document tampering detection benchmark
  targeting Indian origin documents
  under varying JPEG re-compression conditions.
  The evaluation covers {total_n}&nbsp;predictions across
  {len(tiers_present)}&nbsp;compression tier{"s" if len(tiers_present) > 1 else ""}
  ({tiers_str}), using pixel-level region matching with Hungarian assignment
  (IoU&nbsp;≥&nbsp;0.1) and a fixed document confidence threshold
  of&nbsp;{document_threshold:.2f}.{perf}
  All point estimates are accompanied by 95% bootstrap confidence intervals
  (1,000 resamples).</p>
</div>"""


def _v3_results_table(tier_scores: dict[str, Any]) -> str:
    """Booktabs-style per-tier results table."""
    cols: list[tuple[str, str, str]] = [
        ("region_precision",  "region_precision_ci",  "Reg-P"),
        ("region_recall",     "region_recall_ci",     "Reg-R"),
        ("region_f1",         "region_f1_ci",         "Reg-F1"),
        ("doc_auc",           "doc_auc_ci",           "Doc-AUC"),
        ("doc_auprc",         "doc_auprc_ci",         "AUPRC"),
        ("doc_f1",            "doc_f1_ci",            "Doc-F1"),
        ("doc_fpr",           "doc_fpr_ci",           "FPR ↓"),
        ("doc_fpr_at_tpr90",  "doc_fpr_at_tpr90_ci",  "FPR@90 ↓"),
        ("doc_f1_at_tpr90",   "doc_f1_at_tpr90_ci",   "F1@90"),
    ]
    header_cells = "<th>Tier</th>" + "".join(f"<th>{lbl}</th>" for _, _, lbl in cols)

    tiers_present = [t for t in _ALL_TIERS if t in tier_scores]
    rows: list[str] = []
    for i, tier in enumerate(tiers_present):
        s = tier_scores[tier]
        is_last = i == len(tiers_present) - 1
        row_cls = " class='v3-row-last'" if is_last else ""
        cells = [
            f"<td><span class='tbl-val'>{tier}</span>"
            f"<span class='tbl-ci'>n&nbsp;=&nbsp;{int(s['n_samples'])}</span></td>"
        ]
        for val_key, ci_key, _ in cols:
            raw = s.get(val_key)
            if raw is None:
                cells.append("<td><span class='tbl-val'>—</span></td>")
                continue
            val = _f(float(raw))
            ci_raw = s.get(ci_key)
            lo, hi = _as_ci(ci_raw) if ci_raw is not None else (float(raw), float(raw))
            valid = bool(s.get(_DETECTION_VALIDITY_KEYS.get(val_key, ""), True))
            val_display = val + ("" if valid else " †")
            cells.append(
                f"<td><span class='tbl-val'>{val_display}</span>"
                f"<span class='tbl-ci'>[{_f(lo, 3)},&thinsp;{_f(hi, 3)}]</span></td>"
            )
        rows.append(f"<tr{row_cls}>{''.join(cells)}</tr>")

    return f"""
<div class="v3-tbl-wrap">
  <table class="v3-tbl">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
<p class="v3-tbl-note"><strong>Table 1.</strong> Per-tier evaluation results.
Values shown as point estimate with 95% bootstrap CI [lo,&thinsp;hi] (1,000 resamples) in grey below.
FPR ↓ and FPR@90 ↓ lower is better; all other metrics higher is better.
† TPR ≥ 0.90 not reachable — shown at max achievable TPR.
Compression Robustness is reported separately in Section&nbsp;3.</p>"""


def _v3_figure_chart(
    canvas_id: str,
    fig_num: int,
    title: str,
    caption: str,
    series: list[tuple[str, str, str]],
    tier_scores: dict[str, Any],
    y_scale: str = "fixed",
) -> str:
    js = _multi_chart(canvas_id, series, tier_scores, y_scale=y_scale)
    return f"""
<div class="v3-figure">
  <p class="v3-fig-title">{title}</p>
  <canvas id="{canvas_id}" height="200"></canvas>
  <p class="v3-fig-note"><strong>Figure {fig_num}.</strong> {caption}</p>
</div>
{js}"""


def _v3_cr_section(tier_scores: dict[str, Any]) -> str:
    c0 = tier_scores.get("C0")
    c4 = tier_scores.get("C4")
    if not c0 or not c4:
        return (
            "<p style='font-size:9pt;color:#888;font-style:italic'>"
            "Compression Robustness requires both C0 and C4 tiers to be evaluated.</p>"
        )

    cr_auc = _cr(float(c0["doc_auc"]),   float(c4["doc_auc"]))
    cr_f1  = _cr(float(c0["region_f1"]), float(c4["region_f1"]))

    def _box(val: float, name: str, formula: str) -> str:
        return (
            f"<div class='v3-cr-box'>"
            f"<span class='v3-cr-num'>{_f(val, 3)}</span>"
            f"<span class='v3-cr-name'>{name}</span>"
            f"<span class='v3-cr-interp'>{_cr_label(val)}</span>"
            f"<span class='v3-cr-formula'>{formula}</span>"
            f"</div>"
        )

    return (
        f"<div class='v3-cr-grid'>"
        + _box(cr_auc,
               "CR<sub>DocAUC</sub> &mdash; Detection Robustness",
               "1 &minus; (AUC<sub>C0</sub> &minus; AUC<sub>C4</sub>) / AUC<sub>C0</sub>")
        + _box(cr_f1,
               "CR<sub>RegionF1</sub> &mdash; Localisation Robustness",
               "1 &minus; (F1<sub>C0</sub> &minus; F1<sub>C4</sub>) / F1<sub>C0</sub>")
        + "</div>"
        + "<p class='v3-tbl-note'><strong>Table 2.</strong> Compression Robustness scores. "
          "CR&nbsp;=&nbsp;1&nbsp;&minus;&nbsp;(score<sub>C0</sub>&nbsp;&minus;&nbsp;score<sub>C4</sub>)&thinsp;/"
          "&thinsp;score<sub>C0</sub>. Clamped at 1.0; values where C4&nbsp;&gt;&nbsp;C0 "
          "are treated as no degradation (statistical variation).</p>"
    )


def _v3_reading_guide() -> str:
    return """
<h3 class="v3-appendix-h">Appendix &mdash; Glossary &amp; Interpretation Guide</h3>
<div class="v3-guide-cols">

  <div class="v3-guide-block">
    <h4>Compression Tiers</h4>
    <p>Each source document is evaluated at three JPEG re-compression depths simulating
       real-world forensic signal degradation.</p>
    <ul>
      <li><strong>C0 — Pristine.</strong> No re-compression. Lossless PNG.
          Full artifact signal intact. Upper-bound performance.</li>
      <li><strong>C2 — Double-pass JPEG</strong> (Q=85&thinsp;→&thinsp;Q=80).
          Typical scan–share–resave cycle. Most real-world documents.</li>
      <li><strong>C4 — Photocopy simulation.</strong> BMP round-trip, Gaussian
          blur, additive noise, JPEG Q=70. Hardest tier; DCT artifact
          signal nearly erased.</li>
    </ul>
  </div>

  <div class="v3-guide-block">
    <h4>Localisation Metrics</h4>
    <p>Measure how accurately the model locates tampered <em>regions</em>.
       Matching uses Hungarian assignment with IoU&nbsp;≥&nbsp;0.1.</p>
    <dl>
      <dt>Reg-P (Region Precision)</dt>
      <dd>Fraction of predicted regions overlapping a ground-truth region.
          High = few false alarms.</dd>
      <dt>Reg-R (Region Recall)</dt>
      <dd>Fraction of ground-truth regions the model found.
          High = few missed forgeries.</dd>
      <dt>Reg-F1</dt>
      <dd>Harmonic mean of Reg-P and Reg-R.
          Primary localisation score.</dd>
    </dl>
  </div>

  <div class="v3-guide-block">
    <h4>Detection Metrics</h4>
    <p>Measure document-level classification (authentic vs. tampered),
       independent of region localisation accuracy.</p>
    <dl>
      <dt>Doc-AUC</dt>
      <dd>Area under ROC curve. Threshold-free; uses max(region confidence)
          as the document score. 0.5&nbsp;=&nbsp;random, 1.0&nbsp;=&nbsp;perfect.</dd>
      <dt>Doc-F1</dt>
      <dd>Binary classification F1. Derived from pred_label&nbsp;=&nbsp;1
          if confidence&nbsp;≥&nbsp;threshold (default 0.50).</dd>
      <dt>FPR (↓ lower is better)</dt>
      <dd>Fraction of authentic documents flagged as tampered.
          FP&nbsp;/&nbsp;(FP&nbsp;+&nbsp;TN).</dd>
    </dl>
  </div>

  <div class="v3-guide-block">
    <h4>Confidence Intervals &amp; CR</h4>
    <p>95% bootstrap CIs: resample the dataset with replacement 1,000 times,
       recompute the metric each time; report the central 95% of results.</p>
    <table class="v3-guide-ci">
      <thead><tr><th>CI width</th><th>Stability</th></tr></thead>
      <tbody>
        <tr><td>≤ 0.05</td><td>Confident — stable across samples</td></tr>
        <tr><td>0.05–0.15</td><td>Moderate — interpret with care</td></tr>
        <tr><td>&gt; 0.15</td><td>Less stable — sensitive to sample</td></tr>
      </tbody>
    </table>
    <p style="margin-top:0.5rem">
      <strong>CR</strong>&nbsp;=&nbsp;1&nbsp;&minus;&nbsp;(score<sub>C0</sub>&nbsp;&minus;&nbsp;score<sub>C4</sub>)&thinsp;/&thinsp;score<sub>C0</sub>.
      Range: ≥0.9&nbsp;=&nbsp;excellent, 0.7–0.9&nbsp;=&nbsp;moderate,
      0.5–0.7&nbsp;=&nbsp;significant, &lt;0.5&nbsp;=&nbsp;severe degradation.
    </p>
  </div>

</div>"""


def _v3_html(model_name: str, tier_scores: dict[str, Any], document_threshold: float) -> str:
    total_n = sum(int(tier_scores[t]["n_samples"]) for t in _ALL_TIERS if t in tier_scores)

    det_series = [
        ("doc_auc",          "Doc-AUC",    "#1d4ed8"),
        ("doc_auprc",        "AUPRC",      "#0369a1"),
        ("doc_f1",           "Doc-F1",     "#6d28d9"),
        ("doc_fpr",          "FPR",        "#b91c1c"),
        ("doc_fpr_at_tpr90", "FPR@TPR=90", "#be123c"),
        ("doc_f1_at_tpr90",  "F1@TPR=90",  "#7e22ce"),
    ]
    loc_series = [
        ("region_precision", "Region-P",  "#0e7490"),
        ("region_recall",    "Region-R",  "#b45309"),
        ("region_f1",        "Region-F1", "#15803d"),
    ]

    det_fig = _v3_figure_chart(
        "v3DetChart", 1,
        "Detection Metrics",
        "Doc-AUC, AUPRC, Doc-F1, FPR, FPR@TPR=90, and F1@TPR=90 across compression tiers "
        "C0&thinsp;→&thinsp;C2&thinsp;→&thinsp;C4. "
        "FPR and FPR@TPR=90 lower is better; all others higher is better.",
        det_series, tier_scores, y_scale="fixed",
    )
    loc_fig = _v3_figure_chart(
        "v3LocChart", 2,
        "Localisation Metrics",
        "Region-P, Region-R, and Region-F1 across compression tiers. "
        "Y-axis is auto-scaled to the data range.",
        loc_series, tier_scores, y_scale="auto",
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kriyam TamperFlow &mdash; {model_name}</title>
  <style>
{_CSS_V3}
  </style>
  <script src="{_CHART_JS}" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
</head>
<body>
<div class="v3-page">

  <div class="v3-title-block">
    <h1 class="v3-paper-title">Kriyam TamperFlow</h1>
    <p class="v3-report-subtitle">Evaluation Report</p>
    <p class="v3-model-line">Model: {model_name}</p>
    <p class="v3-meta-line">
      Benchmark v1.0 &nbsp;&middot;&nbsp;
      {total_n} predictions evaluated &nbsp;&middot;&nbsp;
      Document threshold: {document_threshold:.2f}
    </p>
  </div>

  <hr class="v3-rule-double">

  {_v3_abstract(model_name, tier_scores, document_threshold)}
  {_missing_authentic_warning(tier_scores)}

  <hr class="v3-rule-double">

  <h2 class="v3-h2">1.&ensp;Evaluation Results</h2>
  <p style="font-size:9pt;color:#666;margin:0 0 0.75rem;font-style:italic">
    Six metrics reported independently for each compression tier: three localisation metrics
    (Reg-P, Reg-R, Reg-F1) measuring tampered-region detection accuracy, and three detection
    metrics (Doc-AUC, Doc-F1, FPR) measuring document-level classification.
    Each value is accompanied by a 95% bootstrap confidence interval shown in grey below.
  </p>
  {_v3_results_table(tier_scores)}

  <h2 class="v3-h2">2.&ensp;Score Degradation across Compression Tiers</h2>
  <p style="font-size:9pt;color:#666;margin:0 0 0.75rem;font-style:italic">
    Figures 1 and 2 show how detection and localisation performance changes as JPEG
    re-compression depth increases from C0 (pristine, full forensic signal) to C4
    (photocopy simulation, signal nearly erased).
  </p>
  {det_fig}
  {loc_fig}

  <h2 class="v3-h2">3.&ensp;Compression Robustness</h2>
  <p style="font-size:9pt;color:#666;margin:0 0 0.75rem;font-style:italic">
    CR quantifies how much a model&rsquo;s performance degrades from C0 to C4.
    A value of 1.0 indicates no degradation; lower values indicate dependence on
    compression artifact fingerprints that are erased by re-compression.
  </p>
  {_v3_cr_section(tier_scores)}

  {_v3_reading_guide()}

  <div class="v3-footer">
    <span>Kriyam TamperFlow &mdash; Benchmark v1.0</span>
    <span>
      <a href="{_GH_URL}" target="_blank" rel="noopener">GitHub</a>
      &nbsp;&middot;&nbsp;
      <a href="{_HF_URL}" target="_blank" rel="noopener">Dataset on Hugging&nbsp;Face</a>
    </span>
  </div>

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(scores: dict[str, Any], output_path: str, template: str = "v1") -> None:
    """Write a self-contained HTML evaluation report.

    Args:
        scores: Dict with ``"model"`` (str), ``"tiers"`` (tier → aggregate dict),
            and optionally ``"document_threshold"`` (float).
        output_path: Destination path for the HTML file.
        template: ``"v1"`` (card layout), ``"v2"`` (research-report), or
            ``"v3"`` (academic paper style with serif type and booktabs tables).
    """
    model_name: str    = scores.get("model", "unknown")
    tier_scores: dict  = scores.get("tiers", {})
    document_threshold = float(scores.get("document_threshold", 0.5))

    if template == "v2":
        html = _v2_html(model_name, tier_scores, document_threshold)
    elif template == "v3":
        html = _v3_html(model_name, tier_scores, document_threshold)
    else:
        html = _v1_html(model_name, tier_scores, document_threshold)

    Path(output_path).write_text(html, encoding="utf-8")
