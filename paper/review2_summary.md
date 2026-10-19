# RectifyCLIP — Consolidated Review Summary

Compiled for an external slide deck. Every number traces to `experiments/logs/day1-7.md`,
`results/tables/`, or `paper/draft.md`. Nothing here is recomputed or re-rounded.

---

## 1. Project pitch

RectifyCLIP is a controlled empirical study testing whether classical, homography-based
perspective rectification (plain OpenCV) can recover the zero-shot architectural-style
classification accuracy that a frozen CLIP model loses when a building facade photo is
taken at an angle. It matters because frozen vision-language models like CLIP are
increasingly used "as-is" on built-environment tasks, real facade photos are almost
never shot straight-on, and prior work (Tarkhan et al., 2025) flags viewing-angle
distortion as an error source but never tests or fixes it. The novelty is the isolation
of one variable — perspective distortion, with and without rectification — on an
unmodified foundation model, not a new technique.

## 2. Research question (verbatim from the execution plan)

> Does classical (OpenCV-based) perspective rectification recover the zero-shot
> architectural-style classification accuracy that a frozen CLIP model loses under
> controlled synthetic perspective distortion of building facade images?

## 3. Methodology (plain language)

- **Dataset:** MonuMAI — 1,514 expert-labelled Spanish monument facade photos, 4
  architectural styles (Hispanic-Muslim 327, Gothic 359, Renaissance 312, Baroque 516).
  All 1,514 used as the evaluation set (no train/test split — the model is never trained).
- **Model:** CLIP ViT-B/32 with `laion2b_s34b_b79k` open pretrained weights, used purely
  zero-shot (no fine-tuning). Prediction = cosine similarity between one image embedding
  and four text-prompt embeddings, one per style. Default prompt: `"a photo of {} architecture"`.
- **Distortion:** each of the four image corners is moved by an independent random offset,
  bounded as a fraction of image size — mild = 5%, medium = 10%, severe = 20%. Offsets
  build a true 8-degree-of-freedom projective homography via `cv2.getPerspectiveTransform`
  + `cv2.warpPerspective` (no rotation, scale, or lens distortion mixed in). Fully seeded
  and byte-reproducible.
- **Rectification:** *oracle* — the exact known distortion homography is inverted
  (`H⁻¹`) and applied. This deliberately answers "if geometry could be perfectly
  corrected, would CLIP recover?" Border gaps left by the warp are filled with the mean
  colour of the valid region (chosen over black-fill or cropping to avoid a
  severity-correlated confound). Round-trip pixel error verified at MAE 2.0–10.3 / 255.
- **Experimental design:** full grid of 1,514 images × 3 severities × 3 conditions
  (clean / distorted / rectified) = **13,626 rows** (`results/raw/day5_full_grid.csv`),
  ~18 min on local CPU.
- **Metrics:** top-1 accuracy; recovery % = (acc_rectified − acc_distorted) /
  (acc_clean − acc_distorted) × 100; mean CLIP confidence on the true class.
- **Statistical testing:** McNemar's exact paired test on the discordant images
  (cases where distorted and rectified disagree on correctness), since both conditions
  are evaluated on the identical 1,514 images. A raw percentage-point gap cannot separate
  a real effect from sampling noise; the paired test can.

## 4. Headline numbers

### 4a. Clean baseline
Zero-shot CLIP, default prompt: **61.43% accuracy (930 / 1,514)**. Chance = 25%.
Per-class: Hispanic-Muslim 0.783, Gothic 0.705, Renaissance 0.785, Baroque 0.341
(320 / 516 Baroque images misclassified as Renaissance even with no distortion).

### 4b. Accuracy by severity × condition (`results/tables/day5_headline.md`)
| Severity | Clean | Distorted | Rectified |
|---|---|---|---|
| mild   | 0.614 (930) | 0.608 (920) | 0.602 (911) |
| medium | 0.614 (930) | 0.578 (875) | 0.605 (916) |
| severe | 0.614 (930) | 0.554 (839) | 0.614 (929) |

### 4c. Recovery % + McNemar (`results/tables/day6_recovery.md`, `day5_mcnemar.md`)
| Severity | Recovery % | Numerator / Denominator (images) | McNemar b / c | p-value | Meaningful? |
|---|---|---|---|---|---|
| mild   | −90.0% | −9 / 10 | 81 / 72 | 0.5179 | **No — inconclusive.** Distortion barely hurts at mild (10-image gap); the −90% is a tiny numerator over a tiny denominator and would swing 10–20 pts if 1–2 images flipped. Statistically indistinguishable from zero effect. |
| medium | 74.5% | 41 / 55 | 81 / 122 | 0.0049 | **Yes — significant.** |
| severe | 98.9% | 90 / 91 | 96 / 186 | 0.0000 (<0.0001) | **Yes — significant.** Rectified (929) essentially equals clean (930). |

### 4d. Per-class recovery, pooled across severities (`results/tables/day6_per_class.md`)
| Style | Clean | Distorted | Rectified | Recovery % |
|---|---|---|---|---|
| Hispanic-Muslim | 0.783 | 0.735 | 0.783 | 100.0% |
| **Gothic** | 0.705 | 0.670 | **0.653** | **−51.4%** |
| Renaissance | 0.785 | 0.845 | 0.801 | 73.2% |
| Baroque | 0.341 | 0.258 | 0.346 | 105.5% |

**Gothic is the only class with negative recovery — rectification actively hurts it**,
and rectified accuracy is below distorted at *every* severity (0.691→0.680 mild,
0.660→0.641 medium, 0.660→0.638 severe), not an averaging artifact.

### 4e. Renaissance attractor (`experiments/logs/day6.md` Task 5, `day6_per_class.md`)
- **Favorable framing (pooled recovery %):** Renaissance recovery % = **+73.2%**, and its
  own accuracy is *higher under distortion than clean* at every severity
  (0.845 pooled vs 0.785 clean).
- **Unfavorable framing (raw per-image count):** Renaissance is the class **most often
  hurt** by rectification — 90 hurt vs 49 helped, **net −41**, helped:hurt ratio **0.54**.
- Of the 258 images rectification *broke* (correct-when-distorted → wrong-after),
  **148 (57.4%) are re-labelled specifically "Renaissance"**. Of the 380 images
  rectification *fixed*, **313 (82.4%) had been wrongly called "Renaissance" under
  distortion.** CLIP defaults to "Renaissance" for ambiguous perturbed facades; both
  distortion and rectification push images in and out of that basin.

### 4f. Hispanic-Muslim hypothesis — guessed, then refuted (`day4.md`, `day6.md` Task 5)
- **Guess (Day 4):** Hispanic-Muslim images had the highest round-trip pixel
  reconstruction error (MAE 7.5–10.3 / 255, vs 2.0–4.0 for Gothic/Baroque), attributed
  to fine lattice/arabesque texture; the hypothesis was that higher pixel error would
  mean worse *classification* recovery.
- **Refuted:** Hispanic-Muslim is under-represented among "hurt" cases (39 / 258 = 15.1%,
  below its 21.6% dataset share) and has the second-best net recovery of any class:
  **net +47, helped:hurt ratio 2.21.** Pixel-reconstruction fidelity and
  classification-level recovery do not correlate.

## 5. Ranked findings

1. **The "Renaissance attractor" — CLIP systematically defaults to "Renaissance" for
   ambiguous geometrically-perturbed facades, and this drives most of rectification's
   apparent benefit.** 148/258 broken predictions and 313/380 fixed predictions run
   through a wrong "Renaissance" label. *Backed by:* `experiments/logs/day6.md` Task 5;
   `results/tables/day6_per_class.md`; `figures/day6_failure_gallery.png`;
   `figures/day2_confusion_clean.png` (320/516 clean Baroque→Renaissance).
2. **Rectification is not uniformly helpful — it has a consistent negative effect on
   Gothic (−51.4% pooled recovery), at every severity.** *Backed by:*
   `results/tables/day6_per_class.md` (rectified 0.653 < distorted 0.670 pooled;
   below distorted at mild/medium/severe individually).
3. **Where distortion actually bites, oracle rectification recovers most of the loss and
   the effect is statistically robust: 74.5% recovery at medium (p=0.005), 98.9% at
   severe (p<0.0001); mild is inconclusive (p=0.52).** *Backed by:*
   `results/tables/day6_recovery.md`; `results/tables/day5_mcnemar.md`;
   `results/tables/day5_headline.md`.
4. **Renaissance classification accuracy *increases* under distortion (0.845 vs 0.785
   clean, pooled, at every severity) — corrupting the geometry makes CLIP better at
   true Renaissance facades**, a side effect of the same attractor bias. *Backed by:*
   `results/tables/day6_per_class.md` (per-severity 0.827 / 0.840 / 0.869).
5. **The pixel-error hypothesis is refuted: the class with the worst round-trip pixel
   reconstruction (Hispanic-Muslim, MAE 7.5–10.3) has the second-best classification
   recovery (net +47, ratio 2.21).** *Backed by:* `experiments/logs/day4.md` MAE table;
   `experiments/logs/day6.md` Task 5; `results/tables/day6_per_class.md`.

## 6. Figure manifest (verified present in `figures/`)

| Filename | What it shows | Best supports | Suggested slide |
|---|---|---|---|
| `day2_confusion_clean.png` | 4×4 confusion matrix of zero-shot CLIP on clean images; Baroque is heavily read as Renaissance (320/516). | Finding 1 (attractor is present even before distortion) | Results — baseline / Renaissance-attractor slide |
| `day3_distortion_gallery.png` | 6 images × 4 columns (original + mild/medium/severe distorted) — what each severity level looks like. | Methodology (distortion protocol) | Methodology slide |
| `day4_rectification_gallery.png` | 6 images: original \| mild distorted \| mild rectified \| pixel-diff — oracle rectification visually restores geometry (MAE 2–10/255). | Methodology (rectification protocol); Limitation (oracle) | Methodology slide |
| `day6_severity_curve.png` | Accuracy vs severity, 3 lines: distorted declines monotonically (0.614→0.554), rectified tracks the clean line at every severity. | Finding 3 | Results — headline slide |
| `day6_confidence.png` | Mean CLIP confidence on the true class vs severity: distorted confidence falls steadily (0.561→0.519); rectified stays at clean level, slightly exceeding it at severe (0.5690 vs 0.5674). | Finding 3 (supporting) | Results — confidence / supporting slide |
| `day6_failure_gallery.png` | 8 uniformly-sampled (distorted, rectified) pairs where rectification broke a correct prediction; the rectified panels look cleanly restored — the failure is a subtle boundary effect, not a visible artifact. | Findings 1 and 2 | Results — Renaissance-attractor slide |

## 7. Limitations (from `paper/draft.md` §5)

- **Oracle, not blind, rectification.** Every result inverts the *exact known*
  homography. A real system does not have ground truth and would need blind rectification
  (line / vanishing-point estimation). These numbers are an **upper bound**, not a
  deployable result; blind rectification is future work.
- **Single dataset, model, task, prompt.** MonuMAI only (4 Spanish-monument styles),
  CLIP ViT-B/32 / `laion2b_s34b_b79k` only, 4-way style classification only, one primary
  prompt. Generalization to other backbones, datasets, or tasks is untested.
- **The Renaissance-attractor mechanism is unresolved.** The count data is strong
  (148/258, 313/380) but no targeted experiment was run to tell apart a prompt-wording
  artifact, a CLIP pretraining-frequency prior, and genuine cross-style visual ambiguity.
- **Metric ambiguity / honest nulls.** For Renaissance the clean-anchored recovery-%
  (+73.2%) and the raw per-image count (net −41) tell opposite stories — both are
  reported. Mild severity is inconclusive, not evidence of harm. "Severe" is a ceiling
  on distortion magnitude, not a constant (independent per-corner offsets).

## 8. One-sentence takeaway

Classical perspective rectification does recover most of the architectural-style accuracy
a frozen CLIP loses to moderate and severe viewing-angle distortion, but the benefit is
uneven — it actively hurts one style (Gothic) and is largely explained by correcting a
single previously-undocumented CLIP bias (a systematic "Renaissance" default), not by
uniformly restoring geometric legibility.
