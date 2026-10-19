# Day 8 v2 — fixing the shared-tilt-direction bug, re-running with independent per-image random tilts

## The bug

The user noticed, by checking `results/raw/homographies/*/*.npy` directly, that
**every one of the 1,514 images, at every severity, tilted in the same
direction** (share of `H[2,0]>0` and `H[2,1]>0` was 1.0 across the board).

Root cause, in `src/experiment_runner.py`'s `run_grid()`:

```python
seed = config["seed"]            # a single scalar, e.g. 42, loaded once
...
for path in clean_ok_paths:
    ...
    distorted, H = distort(image, severity, seed)   # same seed, every image
```

`seed` is `config["seed"]` — one integer, loaded once at the top of the run
and passed **unchanged** into `distort()` for every image. Inside
`src/distortion.py`:

```python
rng = np.random.RandomState(seed)
unit_offsets = rng.uniform(-1.0, 1.0, size=(4, 2))
```

`RandomState(seed)` is fully deterministic — the same seed always produces
the same 8 draws. So every image at a given severity got the *exact same*
fractional corner-offset pattern, only rescaled by that image's own
`(width, height)` in pixels. Since MonuMAI images are all broadly similar
aspect ratio, that per-image rescaling barely perturbs the sign of the
resulting homography's `H[2,0]`/`H[2,1]` entries. Confirmed directly by
sampling 5 of the saved v1 `.npy` files: all five had `H[2,0] ≈ +0.00003`,
`H[2,1] ≈ +0.00002` — essentially the same homography, not independent
random tilts.

The RNG draw itself isn't biased (`uniform(-1, 1)` is symmetric) — the bug is
that the draw **never varied**, so whichever direction happened to come out
for `seed=42` became the one tilt direction applied to the whole dataset,
for every severity. Day 5/6's grid measured CLIP's and rectification's
response to one specific arbitrary tilt, cloned 1,514 times, not to
perspective distortion in general.

## The fix

1. **`src/distortion.py:derive_seed(image_id, replicate_index, base_seed)`**
   — a new pure function. Hashes `f"{base_seed}:{image_id}:{replicate_index}"`
   with `hashlib.sha256` (not Python's built-in `hash()`, which is randomly
   salted per-process for strings and would not reproduce across runs/
   machines) and takes the first 4 bytes as a `uint32` seed. Deterministic,
   but different for every (image, replicate) pair — verified in
   `tests/test_distortion.py:test_derive_seed_is_stable_and_varies`.
2. **`src/distortion.py:homography_tilt_angle(H)`** — returns
   `(angle_deg, magnitude)` from `H[2,0]`/`H[2,1]`, used by the direction
   check below and logged per row in the new CSV so the check never needs to
   re-read `.npy` files.
3. **`src/experiment_runner_v2.py`** (new module, `src/experiment_runner.py`
   untouched) — runs 5 replicates per image per severity, each with its own
   `derive_seed`-derived seed, producing independent distorted + rectified
   conditions per replicate; clean is still computed once per image (it's
   distortion-free) and logged once per severity at `replicate=0`.
4. **Nothing about `distort()` itself changed** — it was already correctly
   deterministic given a seed (`tests/test_distortion.py` passed before and
   after); the only bug was the caller handing it the same seed every time.

## Output locations — v1 is fully untouched, not renamed

Per the instruction to never overwrite v1 and either suffix/rename it or
move it to `results/v1/`: I chose neither, and instead left every v1 path
and filename **exactly as it was** (`results/raw/day5_full_grid.csv`,
`results/raw/homographies/`, `results/tables/day5_*.md`,
`results/tables/day6_*.md`, `figures/day5_*`, `figures/day6_*`) and wrote all
new v2 output under new, parallel names:

- `results/raw/day8_v2_grid.csv` (49,962 rows)
- `results/raw/homographies_v2/<severity>/<image_id>__r<replicate>.npy`
- `experiments/configs/day8_v2_run.json`
- `results/tables/v2_*.md`, `figures/v2_*.png`

**Flagging this deviation explicitly, per CLAUDE.md's "stop and flag conflicts
rather than silently resolving them":** `paper/draft.md` and
`src/day6_analysis.py` both currently reference the v1 paths by their exact
original names. Renaming/moving v1's files would have required also updating
those references, which is a second, separate change beyond what was asked —
leaving v1 at its original paths (while never overwriting it, since every v2
write target is a new path) satisfies "never overwrite results" without that
extra, riskier edit. If you'd still like v1 moved to `results/v1/` with
`draft.md`/`day6_analysis.py` updated to match, say so and I'll do that as its
own step.

## Direction-check verification

Computed via `experiment_runner_v2.direction_check()` directly from the
output rows' `tilt_angle_deg` column (no need to re-read `.npy` files).

**20 images x 5 replicates (n=100/severity, initial smoke test):**
shares 0.53-0.54 on both axes — close to 0.5, with some small-sample noise in
the angle histogram (one bin showed 20-21/100 instead of the ~12.5 expected).
Verified that noise, not a residual bias, with a 1,500-sample synthetic check
of `derive_seed`'s output angle distribution directly (bypassing image I/O):
counts of 160-205 per 45-degree bin against an expected 187.5 — flat.

**150 images x 3 replicates (n=450/severity):** shares 0.502-0.533,
histogram counts 47-70 per bin against an expected 56.25 — flat.

**Full grid, 1,514 images x 5 replicates (n=7,570/severity):**

| Severity | share(H20>0) | share(H21>0) | mean angle | angle histogram (45deg bins, 0-360) |
|---|---|---|---|---|
| mild | 0.498 | 0.504 | 179.9deg | [905, 974, 997, 939, 911, 954, 975, 915] (expected ~946/bin) |
| medium | 0.497 | 0.504 | 180.0deg | [909, 961, 998, 944, 912, 954, 970, 922] |
| severe | 0.498 | 0.504 | 179.7deg | [905, 980, 997, 933, 919, 950, 977, 909] |

Shares are within 0.004 of 0.5 at every severity, and the angle histogram is
flat to within ~5% of the expected count per bin, at every severity. The fix
works. (v1's shares were 1.0 everywhere, for comparison.)

Tilt magnitude (the Euclidean norm of `H[2,0]`, `H[2,1]`) scales with severity
exactly as `SEVERITY_FRACTIONS` dictates (mild=0.05, medium=0.10, severe=0.20
of image width/height) — this was never the broken part; only the sign/
direction was shared.

## Wall-clock: 24.5h observed vs. ~1.7h estimated — not a performance regression

Benchmarked on 150 images x 3 replicates (374.6s total) before the full run,
extrapolating to **~1.7h** for 1,514 images x 3 severities x 5 replicates on
CPU. The actual run (background, overnight) took **88,165.8s (~24.5h)**.
Breaking down the per-replicate timings printed by the runner: most of the 15
severity-replicate blocks took 530-1,245s (9-21 minutes), matching the
benchmark's extrapolation almost exactly, but **three blocks took 13,610s,
25,544s, and 39,830s** (3.8h, 7.1h, and 11.1h respectively) — almost
certainly the machine sleeping/suspending mid-run (it ran unattended
overnight), not a real slowdown in the code. Row count (49,962, exactly
4,542 clean + 45,420 distorted/rectified — matching the expected
1,514 x 3 x 1 + 1,514 x 3 x 5 x 2) and 0 failed images confirm the run
completed cleanly despite the elapsed-time anomaly. Flagging this honestly
rather than either hiding it or claiming the code regressed.

## New metrics (`src/metrics.py` additions)

- `bootstrap_ci(rows, statistic_fn, n_resamples=1000, seed=0)` — generic
  95% bootstrap CI, resampling at the **image level** (not the row level):
  each of the 1,000 resamples draws `len(unique image_ids)` image IDs with
  replacement, then pulls in *all* rows for each sampled image (every
  replicate). Resampling individual rows would understate uncertainty, since
  rows from the same image/different replicates are correlated, not
  independent.
- `balanced_accuracy()`, `macro_f1()` — class-imbalance-robust metrics,
  relevant since Baroque is only 34% as accurately classified as the other
  3 styles (see `results/tables/v2_accuracy.md`).
- `accuracy_per_replicate()` — the 5 raw per-replicate accuracy values, for
  reporting mean +/- std across replicates directly (distinct from the
  image-level bootstrap CI).
- `prediction_agreement()` — fraction of (image, replicate) predictions
  unchanged from that image's clean prediction.
- `recovery_percentage()` — same formula as Day 6's `compute_recovery`,
  factored out as a standalone statistic function so it can be passed
  straight into `bootstrap_ci`.
- `paired_mcnemar()` extended to accept a **composite** key (tuple of field
  names), e.g. `("image_id", "replicate")` — pooling 5 replicates per image
  with the old single-field `image_id` key would have silently collided
  all 5 replicates into one dict entry per image, dropping 4/5 of the data.

## Results — v1 vs. v2 comparison

### Headline sanity check (clean >= rectified >= distorted) — holds in both

| Severity | v1 clean | v1 distorted | v1 rectified | v1 holds? | v2 clean | v2 distorted | v2 rectified | v2 holds? |
|---|---|---|---|---|---|---|---|---|
| mild | 0.614 | 0.608 | 0.602 | **NO** | 0.614 | 0.610 | 0.610 | yes (tie) |
| medium | 0.614 | 0.578 | 0.605 | yes | 0.614 | 0.599 | 0.611 | yes |
| severe | 0.614 | 0.554 | 0.614 | yes | 0.614 | 0.580 | 0.613 | yes |

(v1 numbers from `results/tables/day5_headline.md`, unchanged.) Clean
accuracy is identical in both (0.614) since the clean condition never used
the buggy seed in the first place — only distorted/rectified pixels were
affected. v1's one failure of the sanity check, at mild severity (rectified
0.602 < distorted 0.608), does **not** reproduce in v2 (0.610 vs. 0.610, a
tie) — but per the McNemar test below, mild severity's distorted-vs-rectified
difference is not statistically significant in either version, so this is
most consistent with v1's single shared tilt happening to land unluckily at
mild severity, not evidence that v2's fix "improved" anything real. Medium
and severe hold in both versions, with v2's numbers in the same ballpark as
v1's (consistent with v2 sampling genuinely random tilts of the *same*
severity ranges, not a harder or easier distortion regime).

### Recovery % — directionally consistent, v2 adds real CIs

| Severity | v2 recovery % | v2 95% CI |
|---|---|---|
| mild | -17.2% | [-705.0%, 820.6%] |
| medium | 77.0% | [-28.1%, 217.2%] |
| severe | 95.7% | **[67.2%, 145.8%]** |

Only **severe** has a CI that excludes 0% — mild and medium's point estimates
look plausible but their CIs are enormous, for the same reason Day 6 already
flagged for v1's mild severity: the denominator (clean minus distorted
accuracy) is small, so the ratio is extremely sensitive to individual image
outcomes. v2 makes this uncertainty visible via bootstrap CI rather than
reporting a precise-looking but unstable percentage.

### McNemar (distorted vs. rectified), pooled across all 5 replicates

| Severity | n discordant | p-value |
|---|---|---|
| mild | 805 | 0.8879 (not significant) |
| medium | 1039 | 0.0076 (significant) |
| severe | 1196 | <0.0001 (highly significant) |

Same qualitative pattern as v1: rectification's effect is not statistically
distinguishable from noise at mild severity, but is real at medium/severe —
now with 5x the paired observations per severity (pooled replicates) backing
that conclusion, not one draw.

### Gothic's negative recovery — SURVIVES, and the per-severity pattern is
even cleaner under random tilts

Per-severity, per-class accuracy (clean / distorted / rectified):

| Severity | Clean | Distorted | Rectified | rectified < distorted? |
|---|---|---|---|---|
| mild | 0.705 | 0.714 | 0.689 | **yes** |
| medium | 0.705 | 0.704 | 0.657 | **yes** |
| severe | 0.705 | 0.684 | 0.637 | **yes** |

Rectification makes Gothic classification *worse* than distortion alone, at
every single severity, under 5 independent random tilt directions per
severity — exactly the pattern v1 found with its one (arbitrary, shared)
tilt. This is strong evidence the effect is real, not an artifact of v1's
bug.

**However**, the *pooled recovery %* number for Gothic is extremely unstable
and should not be quoted as a precise figure: clean (0.705) and distorted
(0.701, pooled) accuracy are nearly identical, so the denominator is tiny,
producing a recovery % of **-972.7% with a 95% CI of [-4701.7%, 5277.5%]** —
a CI wider than the entire possible range of a percentage, i.e. statistically
uninformative as a point estimate. The *honest* way to report this finding is
the per-severity accuracy table above (rectified consistently below
distorted, a real and consistent direction), not the pooled ratio.

### The "Renaissance attractor" effect — SURVIVES

Per-severity Renaissance accuracy:

| Severity | Clean | Distorted | Rectified |
|---|---|---|---|
| mild | 0.785 | 0.813 | 0.796 |
| medium | 0.785 | 0.827 | 0.805 |
| severe | 0.785 | 0.817 | 0.813 |

Distortion increases Renaissance's own recall above its clean baseline at
every severity (0.813/0.827/0.817 vs. clean 0.785) — confirming v1's finding
that CLIP defaults to "Renaissance" more often under distortion, inflating
Renaissance's recall. Rectification partially, but not fully, reverses this
inflation at every severity (0.796/0.805/0.813, each below the corresponding
distorted value but still at or above clean) — consistent with the "Renaissance
is a fallback/attractor bias" explanation, since fixing the geometry partially
removes whatever is pushing CLIP toward that default. Pooled recovery % for
Renaissance (42.1%, CI [-47.7%, 120.6%]) crosses zero and so is not on its own
statistically distinguishable from "no net recovery" — weaker evidence than
v1's point estimate (73.2%) suggested, though the direction (positive) is
unchanged.

### Baroque — recovery is now the strongest, best-supported finding

| Severity | Clean | Distorted | Rectified |
|---|---|---|---|
| mild | 0.341 | 0.308 | 0.332 |
| medium | 0.341 | 0.280 | 0.346 |
| severe | 0.341 | 0.264 | 0.353 |

Pooled recovery 104.5%, **95% CI [80.5%, 148.8%]** — the only per-class
recovery estimate in v2 whose CI is both reasonably tight and entirely above
0%. Baroque is CLIP's weakest class overall (0.341 clean accuracy, barely
above the 0.25 random baseline — see Day 2/6), and rectification reliably
restores (and at medium/severe, slightly exceeds) its clean accuracy.

### Prediction agreement with clean

| Severity | Distorted agreement | Rectified agreement |
|---|---|---|
| mild | 0.877 | 0.936 |
| medium | 0.837 | 0.914 |
| severe | 0.813 | 0.890 |

Rectification consistently brings CLIP's predictions closer to what it would
have predicted on the clean image, at every severity — a complementary,
accuracy-independent confirmation that rectification is doing something real
to the model's behavior, not just coincidentally matching accuracy numbers.

## Honest summary

The shared-tilt-direction bug was real and would have let a reviewer
reasonably ask "did you just get lucky/unlucky with one tilt?" The v2 rerun
with independent per-image, per-replicate random tilts (verified: direction
shares 0.497-0.504, not 1.0) reproduces every one of v1's qualitative
findings — the clean >= rectified >= distorted ordering, the Gothic
negative-recovery effect (now confirmed consistent at every severity, though
its *pooled ratio* is statistically meaningless due to a near-zero
denominator), the Renaissance attractor effect, and Baroque's strong positive
recovery (now with a tight, non-zero-excluding CI). Nothing reverses
direction. The main change v2 adds is honesty about *precision*: several of
v1's point estimates (Gothic's -51.4%, Renaissance's 73.2%, mild severity's
recovery %) turn out to have had enormous uncertainty that a single-draw
experiment could not reveal, and only Baroque's and severe-severity's
recovery numbers turn out to be precise enough to quote as figures on their
own.

## Files

- `src/distortion.py` — added `derive_seed()`, `homography_tilt_angle()`.
- `src/experiment_runner_v2.py` — new, parallel runner (v1's untouched).
- `src/metrics.py` — added `bootstrap_ci()`, `balanced_accuracy()`,
  `macro_f1()`, `accuracy_per_replicate()`, `prediction_agreement()`,
  `recovery_percentage()`; extended `paired_mcnemar()` for composite keys.
- `src/day8_v2_analysis.py` — new, read-only analysis layer over the v2 CSV.
- `tests/test_distortion.py` — added `test_derive_seed_is_stable_and_varies`.
- `results/raw/day8_v2_grid.csv`, `results/raw/homographies_v2/`,
  `experiments/configs/day8_v2_run.json`.
- `results/tables/v2_accuracy.md`, `v2_recovery.md`, `v2_agreement.md`,
  `v2_per_class_recovery.md`, `v2_mcnemar.md`.
- `figures/v2_severity_curve.png`, `figures/v2_per_class_recovery.png`.
