# Metrics Reference — Kriyam TamperFlow

This document explains every metric produced by the benchmark evaluator, including the exact computation, the design rationale, and how to interpret the numbers in the context of document tampering detection on Indic documents.

All metrics are implemented in [`kriyam/metrics.py`](kriyam/metrics.py).

---

## Table of Contents

1. [Evaluation tasks](#evaluation-tasks)
2. [Region-level metrics](#region-level-metrics)
   - [Hungarian matching and IoU threshold](#hungarian-matching-and-iou-threshold)
   - [Region Precision](#region-precision)
   - [Region Recall](#region-recall)
   - [Region F1](#region-f1)
3. [Document-level metrics](#document-level-metrics)
   - [Document AUC-ROC](#document-auc-roc-doc-auc)
   - [Document AUPRC](#document-auprc-doc-auprc)
   - [FPR at fixed recall](#fpr-at-fixed-recall-fprtpr)
4. [Confidence intervals](#confidence-intervals)
5. [Compression Robustness](#compression-robustness-cr)
6. [Per-tier reporting](#per-tier-reporting)
7. [Quick reference table](#quick-reference-table)

---

## Evaluation tasks

The benchmark evaluates two distinct tasks for every image:

| Task | Question | Primary metric |
|------|----------|----------------|
| **Document classification** | Is this document authentic or tampered? | Doc-AUC |
| **Region localisation** | Where exactly is the tampering? | Region-F1 |

These tasks are kept separate because a model can be strong at one and weak at the other. For example, a model that flags every document as tampered will have perfect recall but terrible precision and a high FPR; Doc-AUC captures this through its threshold-free scoring.

---

## Region-level metrics

Region-level metrics measure how well a model localises tampered areas within a document. Ground-truth regions are axis-aligned bounding boxes annotated in the C0 (pristine) coordinate space; the same boxes are used for C2 and C4 evaluation because re-compression does not change image dimensions.

### Hungarian matching and IoU threshold

A document may have multiple tampered regions, and the model may predict multiple regions. Naively counting any overlap would allow a model to "flood" the image with many large boxes and achieve high recall with no penalty.

The evaluator avoids this by solving an **optimal assignment problem**:

1. Build an **IoU matrix** of shape `(num_gt_regions × num_pred_regions)`. Each cell `[i, j]` is the Intersection-over-Union between ground-truth region `i` and predicted region `j`.

   ```
   IoU(A, B) = |A ∩ B| / |A ∪ B|
   ```

   Pixel masks are used for the overlap computation, so partially-overlapping boxes are handled correctly even at the sub-pixel level.

2. Run **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) on the negated IoU matrix. This finds the one-to-one matching of GT↔prediction pairs that maximises total IoU — each ground-truth region is matched to at most one predicted region and vice versa.

3. Apply the **IoU threshold of 0.1**. A matched pair (GT region, predicted region) counts as a **True Positive (TP)** only when `IoU ≥ 0.1`. This low threshold is intentional: document bounding boxes from human annotation have inherent slack, and requiring tight overlap would penalise models that correctly identify the general area of manipulation even if the box boundary differs slightly from the annotator's.

4. Classify remaining regions:
   - **Unmatched GT regions** → False Negatives (FN): tampering the model missed entirely.
   - **Unmatched predicted regions** → False Positives (FP): boxes the model fired on where no ground-truth tampering exists. This penalises models that indiscriminately mark large portions of the document.

> **Why 0.1 and not 0.5?** Standard object-detection benchmarks (COCO) use IoU 0.5–0.95. Document forensics is different: an annotator drawing a box around a replaced name field cannot be expected to align perfectly with another annotator or with a model's heatmap boundary. At 0.1, a predicted box is credited as a TP as long as it substantially overlaps the ground-truth area. The benchmark is measuring *detection of the tampered region*, not precise boundary estimation — that is a separate forensic task not yet in scope for v1.0.

---

### Region Precision

**"Of all the regions my model flagged, what fraction were actually tampered?"**

```
Region Precision = TP / (TP + FP)
```

- Defined as **0.0** when the model predicts no regions (`TP + FP = 0`).
- A model that flags every pixel/region achieves perfect recall but low precision. Precision penalises false alarms on authentic parts of a tampered document or on authentic documents entirely.
- Macro-averaged across all images in the evaluated set.

**In this benchmark:** authentic documents (`is_authentic: true`) have zero ground-truth regions; any predicted region on an authentic document counts directly as a FP and reduces precision.

---

### Region Recall

**"Of all the ground-truth tampered regions, what fraction did my model find?"**

```
Region Recall = TP / (TP + FN)
```

- Defined as **0.0** when the document has no ground-truth regions (`TP + FN = 0`).
- For authentic documents (no tampered regions), recall is always 0.0 by convention — there is nothing to detect. Their contribution to the aggregate comes through FP penalisation in precision and FPR.
- Macro-averaged across all images.

---

### Region F1

**"A single score balancing precision and recall for region localisation."**

```
Region F1 = 2 × Precision × Recall / (Precision + Recall)
```

- Defined as **0.0** when both precision and recall are zero.
- Computed **per image** first, then macro-averaged across all images. This means each document contributes equally to the final score regardless of how many regions it contains. A document with 5 regions does not outweigh a document with 1 region.
- This is the **primary localisation metric** used in the compression robustness calculation and the degradation chart.

---

## Document-level metrics

Document-level metrics treat each image as a binary classification problem: **tampered (1)** or **authentic (0)**. The ground truth comes from the `is_authentic` field in the annotation; the model's prediction is the `confidence` score in the prediction file.

---

### Document AUC-ROC (Doc-AUC)

**"How well does my model rank tampered documents above authentic ones, regardless of any threshold?"**

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures the probability that a randomly chosen tampered document receives a higher confidence score than a randomly chosen authentic document.

```
Doc-AUC = P(score(tampered) > score(authentic))
```

- Ranges from 0.0 to 1.0. **0.5 = random; 1.0 = perfect separation.**
- Computed using `sklearn.metrics.roc_auc_score` over all images in the evaluation set.
- **Threshold-free**: does not require choosing a decision boundary. This makes it the most stable metric when the authentic/tampered ratio varies between evaluation subsets.
- Falls back to **0.5** (random performance) when the evaluation set contains only one class — this can happen when slicing by `--doc-class` or `--tamper-type` on small subsets.

**Why AUC matters for this benchmark:** The compression tier stress-test is fundamentally about whether forensic signals survive re-compression. AUC directly measures the ranking quality of the model's confidence scores. A model whose confidence scores become indistinguishable between tampered and authentic under C4 compression will collapse toward 0.5 — this is the exact degradation the benchmark is designed to expose.

---

### Document AUPRC (Doc-AUPRC)

**"How well does my model rank tampered documents above authentic ones, viewed through precision instead of the full ROC space?"**

AUPRC (Area Under the Precision-Recall Curve) is reported alongside Doc-AUC because it is more informative under class imbalance — it is more sensitive to how well a model handles the minority (authentic) class than AUC is.

```
Doc-AUPRC = average_precision_score(gt_labels, confidences)
```

- Computed using `sklearn.metrics.average_precision_score` over all images in the evaluation set.
- **Important — the random baseline is not 0.5.** `average_precision_score` is anchored to the positive-class prior. With 700 tampered and 350 authentic documents, a random ranker scores Doc-AUPRC ≈ **0.667** (the tampered base rate), not 0.5. Always compare a model's Doc-AUPRC against this baseline, not against the AUC convention.
- Falls back to the tampered base rate (`gt_labels.mean()`) when the evaluation set contains only one class.
- **Threshold-free**, like Doc-AUC — no decision boundary is chosen anywhere in its computation.

**Why AUPRC matters for this benchmark:** it gives a direct sanity check for whether a model's ranking carries signal beyond the class prior, and is the more sensitive of the two ranking metrics when a model's scores cluster near the decision boundary — a common symptom of forensic signal loss at heavier compression tiers.

---

### FPR at fixed recall (FPR@TPR)

**"What false-alarm rate would this model incur if it had to catch at least X% of tampered documents?"**

Rather than committing to a single fixed decision threshold (which reflects how a model happened to be calibrated on its original training distribution, not how it would behave tuned for document verification), the benchmark reports the false positive rate required to reach specific recall (TPR) targets, read directly off each model's ROC curve:

```
FPR@TPR{t} = fpr at the first ROC point where tpr >= t
```

- Reported at **t = 80% and 85%** recall (`FPR@TPR80`, `FPR@TPR85`) — these are the two operating points used in comparative model rankings.
- **t = 90%** (`FPR@TPR90`) is also computed and included in every `scores.json`/report for completeness, but is **excluded from comparative analysis**: at this operating point, only one model in evaluation avoids saturating at FPR = 1.0, and its bootstrap CIs are consistently wide across all three compression tiers, making cross-model comparison unreliable at t=90.
- Each operating point carries a `valid` flag: `true` when some threshold on that model's confidence scores actually reaches the target TPR, `false` when it doesn't. A reported FPR of **1.0** with `valid = false` is a real, informative result — it means no threshold reaches the target recall short of flagging every document as tampered — not a measurement failure.
- **Threshold-free** in the sense that no single deployment threshold is chosen by the benchmark; the reported FPR is the best achievable value at the requested recall, computed independently per tier.
- Implementation: `_operating_point_at_tpr()` in `kriyam/metrics.py`, which locates the smallest ROC-curve index where `tpr >= target_tpr` (the highest threshold that still clears the target, giving the lowest FPR at that recall).

---

## Confidence intervals

Every scalar metric is accompanied by a **95% bootstrap confidence interval**. This quantifies how stable the estimate is given the finite evaluation set size.

**How bootstrap CI works in this benchmark:**

1. Treat the `n` per-image scores (e.g., Region-F1 values) as the population.
2. Draw `n` values **with replacement** from this population — one resample.
3. Compute the mean of the resample.
4. Repeat 1,000 times to get a distribution of resample means.
5. Report the 2.5th and 97.5th percentiles as the lower and upper bounds of the 95% CI.

The random seed is fixed at **42** for reproducibility — identical inputs always produce identical CIs.

**Interpreting CIs:** A wide CI (e.g., `0.61 ± 0.09`) means the metric is sensitive to which specific documents happen to be in the evaluation set. This is common when evaluating on small subsets (by `--doc-class` or `--tamper-type`). Overlapping CIs between two models mean the difference is not statistically significant at the 95% level.

**Why bootstrap and not parametric?** Metric distributions for document forensics are highly non-normal — many documents have Region-F1 of exactly 0.0 (model missed entirely) or 1.0 (model found everything), with few values in between. Bootstrap is assumption-free and handles this bimodality correctly; parametric CIs (e.g., based on the normal distribution) would be misleading.

---

## Compression Robustness (CR)

CR scores answer the central research question of this benchmark: **how much does a model degrade as JPEG re-compression erases forensic artifact signals?**

Three CR scores are computed — for detection (two ranking metrics) and for localisation:

```
CR = min(1.0, Metric_C4 / Metric_C0)
```

### CR for detection (CR_DocAUC, CR_AUPRC)

```
CR_DocAUC = min(1.0, AUC_C4   / AUC_C0)
CR_AUPRC  = min(1.0, AUPRC_C4 / AUPRC_C0)
```

### CR for localisation (CR_RegionF1)

```
CR_RegionF1 = min(1.0, RegionF1_C4 / RegionF1_C0)
```

**Properties:**
- Ranges from **0.0 to 1.0** by construction — the ratio is explicitly clamped at 1.0.
- **1.0** = no degradation (including the case where the model performs *better* at C4 than C0 — `Metric_C4 > Metric_C0` can happen by chance on small evaluation sets, or occasionally when a model's heuristics fire more reliably on the noise patterns that compression introduces; this is clamped to 1.0 rather than reported above it).
- **0.0** = the model's C4 performance is zero. Complete collapse.
- **Undefined below a metric-specific floor**: CR is only meaningful when C0 performance itself clears a minimum bar — a model that fails at C0 shouldn't get credit for "no degradation" simply because it fails equally at every tier. Each metric therefore has a `min_c0` floor below which the CR score is reported as **N/A** (not computed) rather than as a misleadingly high ratio:
  - **Doc-AUC**: floor `0.5` (the random-classifier baseline)
  - **Doc-AUPRC**: floor `≈0.667` (the tampered-class prior for this benchmark's 700/350 split — see [Document AUPRC](#document-auprc-doc-auprc))
  - **Region-F1**: floor `0.05` (very weak localisation still yields a non-trivial CR)
- Implementation: `compression_robustness()` in `kriyam/metrics.py`.

**Interpretation guide:**

| CR score | Label | Meaning |
|----------|-------|---------|
| ≥ 0.90 | Excellent | Almost no degradation — model does not rely on JPEG artifacts |
| 0.70 – 0.89 | Moderate | Noticeable drop; some compression-artifact dependency |
| 0.50 – 0.69 | Significant | Model struggles on real-world re-compressed documents |
| < 0.50 | Severe | Model has collapsed — effectively random on scanned material |

**Why C0 and C4, not C0 and C2?** C4 (photocopy simulation) represents the worst practical case — after four lossy operations (JPEG Q=75, Gaussian blur, additive noise, JPEG Q=70), virtually all DCT-coefficient fingerprints are erased. A model that survives C4 is robust in practice. Using C4 as the lower anchor makes CR a meaningful stress-test rather than a mild sensitivity check. C2 performance is still reported independently and is useful for understanding the real-world (single scan-share cycle) scenario.

---

## Per-tier reporting

All metrics above are computed **three times** — once per compression tier (C0, C2, C4) — producing three independent result tables. This structure is the core experimental design of the benchmark.

| Tier | Expected model behaviour |
|------|--------------------------|
| **C0** — Pristine, lossless PNG | Highest performance. Full DCT artifact signal available. Models that rely on compression fingerprints perform best here. |
| **C2** — Double JPEG pass (Q=85 → Q=80) | Moderate drop expected. Simulates a document scanned, saved, and emailed once. Most real-world Indic documents are at this tier or worse. |
| **C4** — Photocopy simulation + noise + Q=70 | Steepest drop expected. Simulates repeated photocopying or fax-transmission cycles. DCT-based models approach random at this tier. |

The degradation from C0 → C2 → C4 is the **primary experimental finding** the benchmark is designed to measure. Models with high CR scores (robust across tiers) are more deployable in real Indic document processing pipelines.

---

## Quick reference table

| Symbol | Full name | Range | Better | Tier-specific | Implementation |
|--------|-----------|-------|--------|---------------|----------------|
| Region-P | Region Precision | [0, 1] | Higher | Yes | `match_regions()` |
| Region-R | Region Recall | [0, 1] | Higher | Yes | `match_regions()` |
| Region-F1 | Region F1 | [0, 1] | Higher | Yes | `match_regions()` |
| Doc-AUC | Document AUC-ROC | [0, 1] | Higher | Yes | `aggregate()` via `roc_auc_score` |
| Doc-AUPRC | Document AUPRC | [0, 1] (random ≈ 0.667) | Higher | Yes | `aggregate()` via `average_precision_score` |
| FPR@TPR80 | FPR at 80% recall | [0, 1] | Lower | Yes | `aggregate()` via `_operating_point_at_tpr()` |
| FPR@TPR85 | FPR at 85% recall | [0, 1] | Lower | Yes | `aggregate()` via `_operating_point_at_tpr()` |
| FPR@TPR90 | FPR at 90% recall (reported, not used in comparative rankings) | [0, 1] | Lower | Yes | `aggregate()` via `_operating_point_at_tpr()` |
| CR_DocAUC | Compression Robustness (detection, AUC) | [0, 1] or N/A | Higher | No (C0 vs C4) | `compression_robustness()` |
| CR_AUPRC | Compression Robustness (detection, AUPRC) | [0, 1] or N/A | Higher | No (C0 vs C4) | `compression_robustness()` |
| CR_RegionF1 | Compression Robustness (localisation) | [0, 1] or N/A | Higher | No (C0 vs C4) | `compression_robustness()` |

All per-tier metrics are reported with **95% bootstrap CI** (n = 1,000 resamples, seed = 42). Models are ranked primarily by **Region-F1 at C4**, with Doc-AUC and CR reported as secondary criteria.
