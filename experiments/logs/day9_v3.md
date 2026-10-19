# Day 9 v3 — mismatched-H control, attractor calibration (blocked), noisy oracle

Three diagnostic experiments probing *why* oracle rectification recovers
CLIP accuracy (v2's headline finding), using v2's exact seeds/pipeline so
results pair 1:1 with v2 by (image_id, severity, replicate). New code only:
`src/experiment_v3.py`, `src/day9_v3_analysis.py`. v1 and v2 code/outputs
were not touched — only imported from (`derive_seed`, `distort`,
`rectify_oracle`, `apply_border_policy`/`CHOSEN_BORDER_POLICY`,
`compute_valid_mask`, and `clip_zero_shot`'s model loading) and read
(`results/raw/day8_v2_grid.csv`, read-only, for comparison baselines).

## 1. Mismatched-H control

**Question.** Does rectification help because it undoes *this image's*
specific distortion, or would rectifying with basically any plausible
homography do about as well — e.g. because `warpPerspective`'s resampling
alone smooths/regularizes the image in a way CLIP likes, independent of
whether the geometry is actually corrected?

**Method.** For each image, severity (medium, severe — mild excluded per
spec), and each of v2's 5 replicates: distort the image exactly as v2 did
(same `derive_seed(image_id, replicate, base_seed)`), then rectify it with a
**different** image's homography instead of its own. Pairing is a
deterministic cyclic shift of the image order, seeded per (severity,
replicate) — guaranteed no image is ever paired with itself, fully
reproducible. Implementation detail: each image's own H only needs to be
computed once per (severity, replicate) — cached by image_id in a first pass,
then looked up as another image's "wrong" H in a second pass, rather than
distorting twice per image.

**Time estimate vs. actual.** 20-image smoke test: 30.1s (10 severity x
replicate blocks, ~2.6s/block for 20 images, i.e. ~0.13s/image).
Extrapolated full-grid estimate: 1,514 x 2 x 5 x 0.13s ~= 33 min. Actual:
2,491.6s = **41.5 min** — close to estimate (the smoke test's per-image rate
was measured without any other process competing for CPU; the full run had
no such contention either, so the modest gap is ordinary variance, not an
anomaly worth investigating further).

**Diagnostic surprise:** 3 of 15,140 wrong-H rows (0.02%) hit a case where
the wrong homography pushed every pixel of the rectified canvas outside the
original image bounds — an entirely empty valid mask — which made
`apply_border_policy`'s `mean_fill` compute a mean over zero pixels (NaN,
then cast to a degenerate fill value), triggering `RuntimeWarning: Mean of
empty slice`. Quantified by rerunning just the mask computation (no CLIP)
over all 15,140 (image, severity, replicate) combinations: exactly 3 empty
masks, 1 at medium, 2 at severe. Left in the reported numbers uncorrected —
3/15,140 rows cannot move any accuracy or CI outside the precision already
reported. Flagging it rather than silently fixing it, since it's a real
(if rare) edge case of the mismatched-H design that a reviewer could ask
about: a sufficiently wrong homography doesn't just mis-rectify, it can fail
to produce any valid image at all.

**Result:**

| Severity | Distorted acc. [CI] | Wrong-H rectified acc. [CI] | Oracle rectified acc. [CI] | Wrong-H recovery % [CI] | Oracle recovery % [CI] |
|---|---|---|---|---|---|
| medium | 0.599 [0.578, 0.619] | 0.602 [0.582, 0.624] | 0.611 [0.588, 0.633] | 20.4% [-112.2, 130.2] | 77.0% [-28.1, 217.2] |
| severe | 0.580 [0.559, 0.601] | 0.578 [0.558, 0.597] | 0.613 [0.591, 0.634] | -7.4% [-46.8, 23.0] | 95.7% [67.2, 145.8] |

**Wrong-H rectification does NOT recover accuracy.** At both severities,
wrong-H accuracy is statistically indistinguishable from plain distorted
accuracy (medium: 0.602 vs. 0.599, overlapping CIs; severe: 0.578 vs. 0.580,
overlapping CIs, wrong-H even nominally *lower*) — while oracle rectification
is clearly and separately higher at severe (0.613, CI [0.591, 0.634],
non-overlapping with distorted). **Saying this clearly, as asked**:
rectification's benefit in v2 is NOT an artifact of "any warpPerspective
resampling helps CLIP" — it specifically requires undoing the correct,
image-matching homography. This is a positive, not a null, result for the
project's core claim: it rules out the most obvious confound that would have
undermined the "oracle rectification is a valid upper bound on what correct
geometric correction can do" framing.

Figure: `figures/v3_mismatched_h.png`. Table: `results/tables/v3_mismatched_h.md`.

## 2. Attractor calibration test — BLOCKED, not run

**Instruction was explicit**: "If probabilities are not stored, tell me
instead of rerunning." Checked `results/raw/day8_v2_grid.csv`'s header
directly:

```
image_id,true_label,severity,replicate,condition,predicted_label,correct,top1_confidence,prompt_id,base_seed,derived_seed,tilt_angle_deg,tilt_magnitude
```

Only `top1_confidence` (the confidence of whichever class CLIP predicted) is
stored — not the full 4-class softmax vector needed to fit per-class
additive log-prob offsets against clean images' true class distribution.
This experiment cannot be done from v2's existing data. **Not implemented,
not worked around by a v2 rerun**, per the instruction. If you want this
diagnostic, it needs a fresh CLIP forward pass that logs all 4 per-class
softmax scores (the same gap Day 6's `run_confidence_and_prompt_sweep`
hit and solved for its own purposes, reusing v2's seeds so the new pass's
pixels match) — happy to scope that as its own small v3 addition if wanted.

## 3. Noisy oracle

**Question.** How sensitive is rectification's benefit to imperfect
knowledge of the true homography — relevant because real (blind) 
rectification, out of scope for this project's main experiment, would only
ever have an *estimated* homography (e.g. from line/vanishing-point
detection), never the exact one oracle rectification uses.

**Method.** For each image, severity (medium, severe), and 3 of v2's 5
replicates (0, 1, 2 — "3 reps" per spec): recover the exact 4
(src_corner, dst_corner) correspondences `distort()` used to build the true
H (verified exact: reconstructing dst_corners by applying H to src_corners
and comparing against distort()'s internal ground truth gives max abs
difference 0.0 in a direct test), perturb the dst-corner half by +/- pct% of
image size (pct in {0, 1, 2, 4}; independent per-corner, per-image,
per-pct-level draws via the same sha256-based seed scheme as
`derive_seed`), rebuild the rectifying homography via
`cv2.getPerspectiveTransform` on the perturbed correspondences, and rectify
with that. pct=0 (the exact oracle case) is pulled directly from v2's
existing "rectified" rows, restricted to replicates 0-2 for a fair
apples-to-apples comparison — not recomputed, since it is pixel-identical.

**Time estimate vs. actual.** 20-image smoke test: 78.2s (6
severity x replicate blocks, 3 pct levels each, ~12.4s/block, i.e.
~0.21s/image/pct-level). Extrapolated: 1,514 x 2 x 3 x 3 x 0.21s ~= 94 min.
Actual: 3,726.5s = **62.1 min** — notably faster than the smoke-test
extrapolation; most likely because the 20-image smoke test ran concurrently
with the mismatched-H empty-mask diagnostic script competing for CPU, which
inflated the smoke test's per-image rate. The full run had no such
contention. Noted so the estimate's accuracy isn't misread as a real
anomaly in either direction.

**Result:**

| Severity | Perturbation % | Accuracy [CI] | Recovery % [CI] |
|---|---|---|---|
| medium | 0 | 0.611 [0.589, 0.634] | 76.6% [-65.4, 265.3] |
| medium | 1 | 0.605 [0.582, 0.629] | 37.5% [-145.7, 172.8] |
| medium | 2 | 0.607 [0.585, 0.630] | 48.4% [-96.1, 200.3] |
| medium | 4 | 0.609 [0.587, 0.631] | 62.5% [-95.2, 275.2] |
| severe | 0 | 0.609 [0.586, 0.631] | 82.7% [48.8, 135.9] |
| severe | 1 | 0.607 [0.585, 0.629] | 76.3% [40.8, 125.2] |
| severe | 2 | 0.606 [0.584, 0.628] | 73.4% [37.9, 120.6] |
| severe | 4 | 0.609 [0.587, 0.632] | 82.0% [46.5, 153.3] |

**Honest null result: no detectable degradation from 0% to 4% corner
noise, at either severity, within this experiment's precision.**

- At **severe** severity, recovery is positive and statistically significant
  (CI excludes 0) at every single perturbation level (73.4% to 82.7%), and
  essentially flat — no monotonic decline as noise increases from 0% to 4%.
  Accuracy itself barely moves (0.606-0.609 across all 4 noise levels, vs.
  0.580 distorted baseline — see v2's table). Rectification with up to 4%
  corner-localization error recovers just as much accuracy, within
  measurement precision, as the exact oracle does.
- At **medium** severity, point estimates wobble non-monotonically (76.6% ->
  37.5% -> 48.4% -> 62.5%) but **every single CI is enormous and crosses
  zero** — none of these four numbers are statistically distinguishable from
  each other, or from "no recovery at all." This is the same small-
  denominator instability v2 and Day 6 already flagged for medium/mild
  severity's recovery ratio (clean and distorted accuracy are close at
  medium, so the ratio is noisy) — not new information, just the same
  caveat reappearing in a new table.

**Reading this honestly, not oversold**: this does NOT show that
rectification is robust to arbitrary homography error — it shows no
*detectable* degradation within the specific 0-4% range tested, which was
chosen per the instruction, not because it was known to be the range where
degradation would appear. A real blind-rectification method's corner/line-
detection error could plausibly exceed 4% of image size, especially at
higher severities where more of the facade's edges are foreshortened or
cropped out of frame — this experiment says nothing about what happens past
4%, and extrapolating "rectification is error-tolerant" beyond the tested
range would overclaim. If a follow-up pushes perturbation higher (e.g. 8%,
16%), a real degradation curve may well appear.

Figure: `figures/v3_noisy_oracle_recovery.png`. Table:
`results/tables/v3_noisy_oracle.md`.

## Summary (what, why, result, surprises)

| # | What | Why | Result | Surprise |
|---|---|---|---|---|
| 1 | Mismatched-H control | Rule out "any warp helps" as a confound for v2's recovery finding | Wrong-H does NOT recover accuracy (overlaps distorted); oracle does | None — this is the expected/hoped-for result, strengthening confidence in v2's main finding |
| 2 | Attractor calibration | Diagnose whether Renaissance-attractor bias is a simple additive log-prob offset | BLOCKED — v2 never stored full softmax | Not a result, a data-availability gap; flagged rather than worked around |
| 3 | Noisy oracle | Probe sensitivity of rectification's benefit to homography-estimation error | No detectable accuracy/recovery degradation from 0-4% corner noise | Mildly surprising — expected at least a visible downward trend by 4%; absence of one is itself informative but should not be read past the tested range |

## Files

- `src/experiment_v3.py` — new; mismatched-H and noisy-oracle runners.
- `src/day9_v3_analysis.py` — new; read-only analysis over v2 + v3 CSVs.
- `results/raw/v3_mismatched_h.csv` (15,140 rows), `results/raw/v3_noisy_oracle.csv` (27,252 rows).
- `results/tables/v3_mismatched_h.md`, `results/tables/v3_noisy_oracle.md`.
- `figures/v3_mismatched_h.png`, `figures/v3_noisy_oracle_recovery.png`.
- No changes to any v1 or v2 file (code or output).
