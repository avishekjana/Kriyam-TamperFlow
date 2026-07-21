# Kriyam TamperFlow — Leaderboard

Results for baseline models evaluated on **Kriyam TamperFlow v1.0** (1,050 documents: 700 tampered, 350 authentic) across all three compression tiers (C0, C2, C4). Metric definitions and formulas: **[METRICS.md](METRICS.md)**.

Full HTML reports (per-image results, degradation charts, bootstrap CIs) are in [`reports/`](reports/):

| Model | Report |
|---|---|
| ADCD-v6 | [reports/ADCD-v6-report-v5.html](reports/ADCD-v6-report-v5.html) |
| CAFTB-v6 | [reports/CAFTB-v6-report-v5.html](reports/CAFTB-v6-report-v5.html) |
| DTD-v6 | [reports/DTD-v6-report-v5.html](reports/DTD-v6-report-v5.html) |
| FFDN-v6 | [reports/FFDN-v6-report-v5.html](reports/FFDN-v6-report-v5.html) |
| MVSS-v4 | [reports/MVSS-v4-report-v5.html](reports/MVSS-v4-report-v5.html) |

---

## Ranking

Per the benchmark protocol, models are ranked primarily by **Region-F1 at the C4 tier** (the realistic-processing stress test), with Doc-AUC and Compression Robustness reported as secondary criteria.

| Rank | Model | Region-F1 @ C4 (95% CI) | Doc-AUC @ C4 (95% CI) | CR<sub>RegionF1</sub> | CR<sub>DocAUC</sub> | CR<sub>AUPRC</sub> |
|---|---|---|---|---|---|---|
| 1 | **MVSS-v4** | 0.084 [0.073, 0.096] | 0.689 [0.661, 0.718] | 0.814 | 0.986 | 0.994 |
| 2 | **ADCD-v6** | 0.053 [0.045, 0.062] | 0.767 [0.738, 0.795] | 0.671 | 1.000 | 1.000 |
| 3 | **FFDN-v6** | 0.050 [0.041, 0.059] | 0.786 [0.756, 0.815] | 0.688 | 0.934 | 0.954 |
| 4 | **CAFTB-v6** | 0.047 [0.042, 0.053] | 0.514 [0.477, 0.554] | 0.809 | 0.980 | 1.000 |
| 5 | **DTD-v6** | 0.020 [0.014, 0.025] | 0.717 [0.688, 0.744] | 0.303 | 0.935 | 0.933 |

**Reading the ranking:** all five baselines localise poorly on this benchmark — Region-F1 @ C4 sits between 0.02 and 0.08 for every model, far below what would be needed for operational deployment. Rank 1 vs. rank 2 (MVSS-v4 vs. ADCD-v6) is a statistically distinguishable gap — the 95% CIs don't overlap. Ranks 2–4 (ADCD-v6, FFDN-v6, CAFTB-v6) have overlapping Region-F1 CIs, so the ordering among them is not statistically significant at the 95% level; treat them as a tied middle tier rather than a strict ordering.

Note the detection/localisation split: CAFTB-v6 has the weakest document-level ranking (Doc-AUC ≈ 0.51, barely above random) yet a comparatively strong CR<sub>RegionF1</sub> (0.809) — its region localisation, while weak in absolute terms, degrades the least under compression. DTD-v6 shows the opposite pattern — reasonable Doc-AUC (0.72) but the steepest region-localisation collapse under compression (CR<sub>RegionF1</sub> = 0.303), consistent with its reliance on DCT-based artifacts that C4 processing erases.

---

## Per-tier scores

### C0 — Pristine

| Model | Region-P | Region-R | Region-F1 | Doc-AUC | Doc-AUPRC | FPR@TPR80 | FPR@TPR85 | FPR@TPR90 |
|---|---|---|---|---|---|---|---|---|
| MVSS-v4 | 0.185 | 0.100 | 0.104 | 0.699 | 0.773 | 1.000 | 1.000 | 1.000 |
| ADCD-v6 | 0.080 | 0.118 | 0.079 | 0.717 | 0.827 | 0.491 | 0.600 | 0.709 |
| FFDN-v6 | 0.075 | 0.092 | 0.073 | 0.842 | 0.909 | 0.226 | 0.346 | 1.000 |
| CAFTB-v6 | 0.044 | 0.154 | 0.059 | 0.524 | 0.675 | 0.726 | 0.760 | 0.789 |
| DTD-v6 | 0.061 | 0.095 | 0.066 | 0.767 | 0.876 | 0.546 | 1.000 | 1.000 |

### C2 — Double-pass JPEG

| Model | Region-P | Region-R | Region-F1 | Doc-AUC | Doc-AUPRC | FPR@TPR80 | FPR@TPR85 | FPR@TPR90 |
|---|---|---|---|---|---|---|---|---|
| MVSS-v4 | 0.186 | 0.098 | 0.103 | 0.712 | 0.781 | 1.000 | 1.000 | 1.000 |
| ADCD-v6 | 0.064 | 0.088 | 0.060 | 0.716 | 0.841 | 0.531 | 0.623 | 0.780 |
| FFDN-v6 | 0.068 | 0.061 | 0.055 | 0.826 | 0.903 | 0.317 | 0.391 | 0.506 |
| CAFTB-v6 | 0.046 | 0.149 | 0.058 | 0.450 | 0.640 | 0.820 | 0.837 | 0.851 |
| DTD-v6 | 0.039 | 0.013 | 0.017 | 0.660 | 0.798 | 1.000 | 1.000 | 1.000 |

### C4 — Photocopy simulation

| Model | Region-P | Region-R | Region-F1 | Doc-AUC | Doc-AUPRC | FPR@TPR80 | FPR@TPR85 | FPR@TPR90 |
|---|---|---|---|---|---|---|---|---|
| MVSS-v4 | 0.159 | 0.086 | 0.084 | 0.689 | 0.769 | 1.000 | 1.000 | 1.000 |
| ADCD-v6 | 0.061 | 0.073 | 0.053 | 0.767 | 0.864 | 0.406 | 0.480 | 0.654 |
| FFDN-v6 | 0.053 | 0.059 | 0.050 | 0.786 | 0.867 | 0.374 | 0.400 | 0.466 |
| CAFTB-v6 | 0.037 | 0.125 | 0.047 | 0.514 | 0.678 | 0.694 | 1.000 | 1.000 |
| DTD-v6 | 0.037 | 0.016 | 0.020 | 0.717 | 0.817 | 1.000 | 1.000 | 1.000 |

All values are point estimates from `aggregate()`; 95% bootstrap CIs (n=1,000) for every cell are in the individual HTML reports linked above.

> **FPR@TPR90** is reported for completeness but excluded from the ranking above, per the benchmark protocol — three of five models (MVSS-v4, DTD-v6, and CAFTB-v6 at C4) saturate at FPR=1.0 at this operating point, making cross-model comparison at TPR=90% uninformative.

---

## Compression Robustness detail

CR = `min(1.0, Metric_C4 / Metric_C0)`, computed per model:

| Model | CR<sub>RegionF1</sub> | CR<sub>DocAUC</sub> | CR<sub>AUPRC</sub> |
|---|---|---|---|
| ADCD-v6 | 0.671 (Significant) | 1.000 (Excellent) | 1.000 (Excellent) |
| CAFTB-v6 | 0.809 (Moderate) | 0.980 (Excellent) | 1.000 (Excellent) |
| DTD-v6 | 0.303 (Severe) | 0.935 (Excellent) | 0.933 (Excellent) |
| FFDN-v6 | 0.688 (Significant) | 0.934 (Excellent) | 0.954 (Excellent) |
| MVSS-v4 | 0.814 (Moderate) | 0.986 (Excellent) | 0.994 (Excellent) |

All five models are far more robust in document-level ranking (CR<sub>DocAUC</sub>, CR<sub>AUPRC</sub> ≥ 0.93) than in region localisation (CR<sub>RegionF1</sub> ranges from 0.30 to 0.81) — consistent with the benchmark's core finding that compression artifact loss hurts *where* a model looks far more than *whether* it flags a document at all.

---

## Submit your model

See [Submit to the leaderboard](README.md#submit-to-the-leaderboard) in the main README.
