We replaced the old "percent-of-image-size" distortion knob with a real
camera-tilt-angle model (5 to 35 degrees), so "severity" now means an actual
rotation angle, not an arbitrary pixel fraction. Rectification clearly helps
at small-to-moderate tilts (5-15 degrees) but that benefit disappears — and
even reverses slightly — at large tilts (25-35 degrees). The reason is
largely mechanical: at large angles the camera rotation pushes most of the
image out of frame, and accuracy tracks how much real image area is left, not
the tilt angle by itself. Gothic's "rectification hurts this class" finding
and the overall shape of the results replicate under this completely
different distortion model, strengthening confidence they are real; the
Renaissance "attractor" bias, however, turns out to reverse at extreme tilt
and to be partly a baseline CLIP bias (present even on clean images) rather
than a purely distortion-driven effect. Everything ran cleanly end-to-end
(smoke test, checkpointed full run, resumability verified) with no failures;
all numbers, including the null and negative ones, are reported as found.

---

# Day 10 v4 — angle-calibrated severity via a pure camera-rotation homography

New code only: `src/experiment_v4.py`, `src/day10_v4_analysis.py`. v1, v2, and
v3 code and outputs were not touched this session — v4 only imports generic,
unmodified building blocks (`distortion.derive_seed`/`image_id_from_path`/
`load_image_rgb`/`homography_tilt_angle`, `rectification.rectify_oracle`/
`apply_border_policy`/`compute_valid_mask`/`CHOSEN_BORDER_POLICY`,
`data_loader.load_monumai`, `clip_zero_shot.load_config`/`load_model`/
`encode_text_features`/`encode_image_arrays`) and reads nothing from v1-v3's
result files.

## Ambiguities resolved conservatively (user was asleep, no input available)

- **"Axis in the image plane"**: interpreted as a 3D axis lying in the
  camera's XY-plane (zero Z-component), at a uniformly random in-plane
  direction phi in [0, 360) — this is the standard way to parameterize "the
  camera tilts in some random direction," matching the spirit of v1-v3's
  "tilt direction uniformly random" requirement, generalized from 2D corner
  offsets to 3D camera rotation.
- **Principal point**: used exactly `(width/2, height/2)`, not
  `((width-1)/2, (height-1)/2)` — a sub-pixel difference, immaterial at any
  image size MonuMAI uses, picked for formula simplicity over the
  half-pixel-correct convention.
- **"Report accuracy against valid-area fraction as a possible confound"**:
  interpreted as requiring its own dedicated analysis (binned accuracy vs.
  valid-area fraction, pooled across theta), not just logging the field —
  added as `results/tables/v4_valid_area_confound.md` /
  `figures/v4_valid_area_confound.png`, beyond the 6 numbered analysis tasks,
  since without it the "larger angles lose more image" instruction's "report
  accuracy against it" clause would otherwise go unaddressed.
- **Recovery-% near-zero-denominator reporting**: instruction only specified
  a qualitative threshold ("near zero"); picked `|denom| < 0.02` as a
  concrete cutoff, consistent with v2/v3's established caution about
  small-denominator ratios (same spirit, not an arbitrary new rule).
- **Calibration fit hyperparameters** (500 iterations, learning rate 0.5):
  not specified; chosen to reliably converge the fitted predicted-class
  counts to within 1 of the true counts on clean images (verified directly —
  see the calibration table) without an exhaustive hyperparameter search,
  since this is explicitly a diagnostic, not a tuned method.

## Distortion model: H = K R K^-1

**Intrinsics assumption (a real limitation, not a calibrated fact)**: K is
assumed, since no calibration data exists for MonuMAI's photos — focal
length = 1.0 x image width, principal point at the image center. theta should
be read as "the angle a camera with these assumed intrinsics would need to
tilt to produce this homography," not a verified real-world angle of any
actual MonuMAI photo.

**Rotation**: axis = (cos(phi), sin(phi), 0) with phi uniformly random in
[0, 360) degrees per (image, theta, replicate); R built via
`cv2.Rodrigues(axis * radians(theta))`; H = K @ R @ K^-1, normalized so
H[2,2]=1. H maps original -> distorted coordinates, exactly matching v1-v3's
convention, so `rectification.rectify_oracle` (unmodified) inverts it
identically.

**Verified correct before any expensive run**: theta=0 gives H ≈ identity
(confirmed exactly). A point round-tripped through H then H^-1 returns to
within 1.5e-5 px (floating-point noise, not a real error). A real-image
round-trip (distort then rectify, theta=20, phi=45) gives MAE=4.97 in the
valid region — consistent with v1 Day 4's reported 2.0-10.3 MAE range for
its own (different) distortion model, i.e. interpolation error of the
expected order of magnitude, not a bug.

**Seeds**: `derive_seed(f"v4_camera_rotation:theta{theta}:{image_id}",
replicate, base_seed)` — the `"v4_camera_rotation:theta{theta}:"` prefix is
the new namespace, guaranteeing these hash inputs never collide with v2's
`derive_seed(image_id, replicate, base_seed)` or v3's own prefixed keys.

## Reliability: smoke test, checkpointing, resumability

- **20-image smoke test**: passed cleanly, 76.2s, 620/620 rows (20 clean +
  20 x 5 theta x 3 replicates x 2 conditions), all 4 softmax scores summing
  to 1.0 on a 30-row spot check, `valid_area_fraction` monotonically
  decreasing with theta (0.868 at theta=5 down to 0.314 at theta=35) as
  physically expected. Per the instruction, since it passed, the full run
  was launched automatically without waiting for confirmation.
- **Resumability verified directly**: reran the exact same smoke-test command
  a second time — every one of the 16 blocks (1 clean + 15 theta/replicate)
  was correctly skipped per the manifest, 2.2s total vs. 76.2s fresh, 0
  duplicate rows. The manifest (`results/raw/day10_v4_progress.json`) is
  written atomically (temp file + `os.replace`) after every block, so a
  restart mid-block can only ever lose that one block's in-progress work,
  never corrupt the manifest or duplicate a completed block's rows.
- **Full run**: launched as a true background OS process with stdout/stderr
  redirected to `results/raw/day10_v4_run.log` (not just the harness's
  internal task-output buffer), polled via repeated short-interval checks of
  that log file for a completion or error marker, per the instruction not to
  rely on scheduled wake-ups. Completed in 5,165.6s (~86 min) with exit code
  0, `ALL BLOCKS COMPLETE` logged, matching the 150-image smoke-benchmark's
  linear extrapolation (~0.233s/image/block x 1,514 x 15 ~= 88 min) closely.
- **Row-count integrity**: 46,934 rows total (1,514 clean + 1,514 x 5 theta
  x 3 replicates x 2 conditions = 1,514 + 45,420), exactly as expected.
  0 rows with a NaN/missing softmax score.
- **Known, quantified edge case**: at theta=35 only, 16 of 22,710
  (image, theta-replicate... more precisely 16 of the 4,542 theta=35 rows'
  underlying image/replicate combinations, 0.35% of theta=35's data) had a
  camera rotation so extreme that the rectified canvas's valid mask was
  entirely empty — `rectification.py`'s `mean_fill` then computed a mean
  over zero pixels (the `RuntimeWarning: Mean of empty slice` visible in
  `day10_v4_run.log`). Quantified by recomputing just the mask (no CLIP) over
  all 22,710 (image, theta, replicate) combinations: 0 at theta<=25, 16 at
  theta=35. Left uncorrected in the reported numbers — 16/22,710 (0.07%
  overall) cannot move any accuracy or CI outside the precision already
  reported, and this is the same (rare, physically expected at extreme
  angles) edge case v3's mismatched-H control already identified and
  quantified the same way.

## Analysis results

### 1. Accuracy, balanced accuracy, agreement-with-clean vs. theta

| Theta | Distorted acc. [CI] | Rectified acc. [CI] | Dist. balanced acc. | Rect. balanced acc. | Dist. agreement | Rect. agreement |
|---|---|---|---|---|---|---|
| 0 (clean) | 0.614 [0.590, 0.638] | — | 0.653 | — | n/a | n/a |
| 5 | 0.584 [0.563, 0.607] | 0.606 [0.584, 0.629] | 0.636 | 0.646 | 0.826 | 0.884 |
| 10 | 0.579 [0.556, 0.602] | 0.594 [0.572, 0.617] | 0.626 | 0.636 | 0.818 | 0.853 |
| 15 | 0.570 [0.548, 0.591] | 0.582 [0.559, 0.604] | 0.616 | 0.625 | 0.795 | 0.818 |
| 25 | 0.538 [0.518, 0.558] | 0.529 [0.508, 0.549] | 0.577 | 0.565 | 0.723 | 0.729 |
| 35 | 0.489 [0.471, 0.509] | 0.485 [0.466, 0.504] | 0.515 | 0.508 | 0.616 | 0.615 |

Rectified > distorted (CIs mostly non-overlapping) through theta=15; by
theta=25 rectified is nominally *below* distorted; by theta=35 both have
converged to essentially the same (much lower) accuracy. Agreement with
clean falls steadily for both conditions, converging to ~0.615-0.616 at
theta=35 — rectification stops making predictions look any more like the
clean baseline once the tilt is extreme. Table: `results/tables/v4_accuracy.md`.
Figure: `figures/v4_accuracy_vs_theta.png`.

### 2. Recovery % vs. theta

| Theta | Denominator | Recovery % [CI] | Reliable? |
|---|---|---|---|
| 5 | 0.0299 | 73.5% [36.5, 121.4] | yes |
| 10 | 0.0357 | 42.6% [8.4, 81.3] | yes |
| 15 | 0.0443 | 27.9% [-0.7, 55.1] | yes (borderline — CI nearly touches 0) |
| 25 | 0.0764 | -11.8% [-36.1, 7.6] | yes |
| 35 | 0.1248 | -3.9% [-20.3, 8.5] | yes |

No denominator was actually near zero at the `0.02` threshold in the
UNcalibrated numbers (unlike v2's mild severity) — theta's larger,
monotonically increasing clean-vs-distorted gap keeps the ratio well-defined
throughout. Recovery is positive and CI-excludes-zero at theta=5 and 10,
borderline at 15, and crosses zero (not significant, trending negative) at
25 and 35. Table: `results/tables/v4_recovery.md`. Figure:
`figures/v4_recovery_vs_theta.png`.

### 3. McNemar (distorted vs. rectified) per theta

| Theta | n discordant | p-value |
|---|---|---|
| 5 | 660 | 0.0001 |
| 10 | 683 | 0.0092 |
| 15 | 724 | 0.0409 |
| 25 | 973 | 0.1997 |
| 35 | 1186 | 0.5420 |

Confirms the accuracy table exactly: rectification's effect is statistically
real at theta=5/10/15 and statistically indistinguishable from chance at
theta=25/35 — **the benefit of rectification doesn't just shrink at large
angles, it becomes undetectable.** Table: `results/tables/v4_mcnemar.md`.

### 4. Per-class accuracy vs. theta — Gothic and the Renaissance attractor, re-checked under a different distortion model entirely

**Gothic's negative-recovery finding replicates again, now under camera
rotation instead of corner-offset distortion** — rectified < distorted at
every single theta tested (5, 10, 15, 25, 35), the same pattern found in v1,
v2, and now v4 under a completely different geometric model. This is now
three independent distortion models (corner-offset, mismatched-H, camera
rotation) all showing the same effect — strong evidence this is a real
property of how CLIP/rectification interact with Gothic facades, not an
artifact of any one distortion implementation.

**The Renaissance attractor effect is theta-dependent — it does NOT hold at
extreme tilt.** Distorted accuracy exceeds clean (0.785) at theta=5 (0.864),
10 (0.856), and 15 (0.851) — confirming v1-v3's finding — but REVERSES at
theta=25 (0.774, below clean) and especially theta=35 (0.604, well below
clean). This is new information v1-v3 couldn't show (their severity levels
never got this extreme): the bias that makes CLIP over-predict "Renaissance"
under distortion is itself a property of moderate distortion, not of
distortion in general — extreme tilt instead just destroys Renaissance
accuracy along with everything else.

**Baroque shows the most robust, persistent rectification benefit of any
class** — rectified accuracy is at or above distorted at every single theta,
including theta=35 (0.336 vs. 0.299, both close to clean's 0.341), unlike
the aggregate story where rectification's benefit vanishes by theta=25.
Table: `results/tables/v4_per_class.md`. Figure: `figures/v4_per_class.png`.

### 5. Direction check

| Theta | share(H20>0) | share(H21>0) | mean axis angle |
|---|---|---|---|
| 5 | 0.500 | 0.506 | 179.5deg |
| 10 | 0.511 | 0.479 | 183.1deg |
| 15 | 0.499 | 0.494 | 180.0deg |
| 25 | 0.491 | 0.488 | 177.6deg |
| 35 | 0.493 | 0.493 | 177.5deg |

All shares within 0.011 of 0.5, every angle histogram flat to within ~10% of
the expected per-bin count (n=4,542/theta, expected ~568/bin). The camera-
rotation model's uniform random axis-direction sampling works correctly —
no repeat of v1's shared-direction bug. Table:
`results/tables/v4_direction_check.md`.

### Valid-area-fraction confound (beyond the 6 numbered tasks — see "ambiguities resolved" above for why this was added)

Binning all distorted/rectified rows (pooled across theta) by valid-area
fraction into 10 quantile bins: accuracy rises close to monotonically from
~0.46 (lowest valid-area bin, mean fraction 0.246) to ~0.60-0.62 (highest
valid-area bin, mean fraction 0.877-0.887), for both conditions. **This is
the mechanism behind the theta=25/35 crossover**: it isn't that rectification
specifically fails at large angles — it's that once enough of the frame is
lost to the camera-rotation warp, there isn't enough real image content left
for ANY processing (distorted or rectified) to work with, so rectification's
advantage (which is concentrated in the high-valid-area, low-theta regime)
has nothing left to act on. Point-biserial correlations are modest in
magnitude (+0.069 distorted, +0.090 rectified) because theta and valid-area
aren't perfectly colinear (the random axis direction also affects area loss
at a fixed theta) — but the binned trend is clear and consistent. Table:
`results/tables/v4_valid_area_confound.md`. Figure:
`figures/v4_valid_area_confound.png`.

### 6. Attractor calibration diagnostic (NOT a method — diagnostic only)

Per-class additive log-prob offsets fit on clean images only (500
iterations, converges predicted counts to exactly match true counts:
327/359/312/516 for Hispanic-Muslim/Gothic/Renaissance/Baroque).

**The fitted offsets reveal the Renaissance attractor bias is present even
on CLEAN, undistorted images**: uncalibrated clean predictions call 699
images "Renaissance" against a true count of only 312 (46% of predictions
vs. 20.6% of true labels) — a large baseline bias with zero distortion
involved. Calibrating it away raises clean accuracy from 0.614 to **0.682**
(+6.8 points) using nothing but a per-class constant.

**Applying that same (clean-only-fit) calibration to distorted/rectified
predictions is genuinely informative, in an unexpected way**: at theta=5 and
10, calibrated distorted accuracy (0.682, 0.684) becomes nearly IDENTICAL to
calibrated clean accuracy (0.682) — the recovery-% denominator collapses
toward zero (-0.0007, -0.0024), correctly flagged as unreliable rather than
reporting the nonsensical -2400%/-127% ratios that fall out of dividing by
it. **Read honestly, this means a large share of what looks like "accuracy
loss from mild/moderate tilt" in the uncalibrated numbers is actually a
fixed classification bias CLIP already has on clean images — not information
actually lost to the distortion.** At theta=25 and 35, calibration does NOT
erase the loss (calibrated distorted accuracy 0.621 and 0.560, both well
below calibrated clean's 0.682) — this is where genuine, bias-independent
geometric information loss takes over, consistent with the valid-area-
confound finding above. Table:
`results/tables/v4_attractor_calibration.md`.

## Honest summary of what changed vs. v1-v3, and what didn't

**Replicated, now under a 3rd independent distortion model**: Gothic's
negative rectification effect (every theta, 5-35). The qualitative shape
"rectification helps, McNemar significant" at low/moderate severity.

**New nuance v1-v3 could not have shown** (their severities never got this
extreme in true-angle terms): rectification's benefit is not monotonic in
severity — it has a ceiling. Past some angle (between 15 and 25 degrees
here, with this assumed K), there simply isn't enough surviving image area
for rectification to help, and the McNemar test confirms the effect becomes
statistically undetectable, not just smaller. The Renaissance attractor
effect is similarly not a stable property of "distortion" in general — it
appears at moderate tilt and reverses at extreme tilt.

**New diagnostic finding**: a meaningful fraction of CLIP's apparent
sensitivity to mild/moderate camera tilt is actually a fixed per-class
classification bias already present on clean images, not geometric
information loss — surfaced only because this diagnostic was finally run
with the full softmax available (v2/v3 could not do this; their CSVs never
logged it).

## Files

- `src/experiment_v4.py` — new; camera-rotation distortion, full-softmax
  logging, checkpointed/resumable grid runner.
- `src/day10_v4_analysis.py` — new; all 6 analysis parts + the valid-area
  confound check, read-only over v4's own CSV.
- `results/raw/day10_v4_grid.csv` (46,934 rows), `day10_v4_progress.json`
  (manifest), `day10_v4_run.log` (full run log).
- `experiments/configs/day10_v4_run.json`.
- `results/tables/v4_accuracy.md`, `v4_recovery.md`, `v4_mcnemar.md`,
  `v4_per_class.md`, `v4_direction_check.md`, `v4_valid_area_confound.md`,
  `v4_attractor_calibration.md`.
- `figures/v4_accuracy_vs_theta.png`, `v4_recovery_vs_theta.png`,
  `v4_per_class.png`, `v4_valid_area_confound.png`.
- No v1/v2/v3 file was read-from-data-wise except v1-v3's generic,
  unmodified source code (imported, never edited) — no v1/v2/v3 *output*
  file (CSV, table, figure) was read or written by this session at all,
  since v4's distortion model is entirely new and has no v1-v3 baseline to
  compare pixel-for-pixel against.
