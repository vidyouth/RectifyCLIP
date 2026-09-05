# Day 5 — Full experimental grid

Date: 2026-09-05

## Objective (from execution plan)

Run the actual experiment. Every (image, severity, condition) triple gets one
row in one big table — this is the source of truth for the rest of the paper.

## Hardware

**Local CPU only**, no Colab, no GPU. Confirmed via `torch.cuda.is_available()
== False` (Day 1 log) and by simply not using Colab at any point. The plan's
Colab-fallback path was not needed — see timing below.

## What was built

- `src/experiment_runner.py`: `run_grid(config, ...)` loops over all 1,514
  images; for each image, encodes the "clean" condition once (reused across
  all 3 severities, since it doesn't change with severity — a compute
  optimization, not a change to the plan's row schema: the CSV still gets 3
  separate clean rows per image, one per severity, exactly as specified); for
  each severity, generates the distorted and rectified (oracle,
  `mean_fill` border policy from Day 4) versions in memory, batch-encodes
  both with CLIP, and logs one row per condition.
- Row schema matches the plan exactly: `image_id, true_label, severity,
  condition, predicted_label, correct, top1_confidence, prompt_id, seed`.
  `prompt_id` reuses Day 2's `PROMPT_TEMPLATE_SLUGS` mapping
  (`a_photo_of_architecture` for the config's default template) rather than
  inventing a new identifier scheme, so prompt IDs are consistent across
  Day 2/5/6 tables.
- No new config fields were introduced — `configs/default.yaml` is read
  as-is (model, prompt, class names, seed, dataset root); severity fractions
  come from `src/distortion.py`'s existing `SEVERITY_FRACTIONS`, and the
  border policy from Day 4's `CHOSEN_BORDER_POLICY`, both imported rather
  than re-specified.
- `src/metrics.py` gained `paired_mcnemar` (see "Sanity check" below) — added
  because Day 5's own success criterion requires properly characterizing
  whether the headline pattern holds, which a raw accuracy comparison alone
  cannot do; this is analysis of Day 5's own data, not a new experimental
  condition, so it doesn't encroach on Day 6.

## Smoke test (run before committing to the full grid, as instructed)

Ran on the first 15 images: completed in 11.1s, produced exactly 135 rows
(15 × 3 × 3), 0 failures, schema correct, headline numbers sane
(clean/distorted/rectified all in a plausible 0.87-1.00 range for this tiny
subset). A second, 40-image timing run gave a clean breakdown: setup 2.10s
(fixed, one-time), clean-condition 2.14s (40 images), and 6.7-6.9s per
severity (40 images, covering both distort+rectify CPU work and 2 batched
CLIP encodes). Extrapolating to 1,514 images: setup ~2s + clean ~81s +
3 × severity ~260s ≈ **~14-15 minutes total** — comfortably under the "many
hours" threshold that would have required stopping to ask before launching.
Proceeded directly to the full run per the plan's own conditional
instruction.

## Bug found during the first full-grid attempt (fixed before finalizing results)

The first full run completed without crashing but produced only **13,509**
data rows instead of the expected 13,626 — 117 rows short. `cv2.imread`
(used by `src/distortion.py`'s `load_image_rgb`, needed for the
distort/rectify path) silently failed (returned `None`, logged as an OpenCV
warning, not an exception) on 13 specific MonuMAI files whose filenames
contain Spanish accented characters, e.g.
`Gothic/iglesia_nuestra_señora_de_la_o_chipiona.jpg`. This is a known
Windows-specific limitation: `cv2.imread` opens paths through a narrow
(non-Unicode) file API on Windows, so any path with non-ASCII characters
fails silently.

This was **not treated as a genuine "image failed to load" case** for the
final results, because the same 13 files load without any problem via PIL
(used by the "clean" condition's `encode_image_features`) — Day 2's clean
baseline already successfully classified all 1,514 images, these 13
included, with zero failures. That PIL succeeds where `cv2.imread` fails on
the identical files is conclusive: the files are valid, undamaged images: the
failure was a code bug (an OpenCV/Windows path-encoding issue), not a data
problem. Reporting these 13 images as "failed to load" in the final results
would have been dishonest — the correct fix was to fix the code, not to
document a fake data limitation.

**Fix**: `load_image_rgb` now reads via `np.fromfile(path) +
cv2.imdecode(...)` instead of `cv2.imread(path)` directly — the standard
workaround for this exact Windows/OpenCV limitation, since `np.fromfile`
handles Unicode paths correctly and `cv2.imdecode` only ever sees raw bytes,
never a path. Verified the fix loads the previously-failing file correctly,
re-ran both existing test suites (`tests/test_distortion.py`,
`tests/test_rectification.py`) to confirm no regression — both passed with
**identical** numbers to before the fix (same MAE values, same hashes),
confirming the fix only affects the previously-broken Unicode-path files, not
any other behavior. Re-ran the full grid from scratch afterward.

## Final run: 0 failures, exact expected row count

```
Loaded 1514 images from data/monumai
Clean condition encoded for 1514 images (0 failed to load).
severity=mild: 1514 images processed in 332.4s
severity=medium: 1514 images processed in 336.1s
severity=severe: 1514 images processed in 305.0s
Saved results/raw/day5_full_grid.csv (13626 rows)
Total wall-clock time: 1078.6s
Failed images: 0
```

**13,626 rows** = 1,514 images × 3 severities × 3 conditions, exactly as
expected. **Total wall-clock: 1,078.6s (~18.0 minutes)**, entirely on local
CPU. 4,542 homography `.npy` files (1,514 × 3 severities) were saved under
`results/raw/homographies/<severity>/<image_id>.npy`, populating the
mechanism Day 3 built and Day 4 exercised on a small scale.

**Internal consistency check**: clean-condition accuracy in this run is
930/1514 = **0.6143** at every severity — exactly matching Day 2's clean
baseline (930/1514, same figure, computed via an entirely separate code path
run 3 days apart). This is expected (identical model, prompt, images, no
randomness in either pipeline) but is exactly the kind of cross-check that
would have caught a subtle bug (e.g. an accidentally-different prompt or
class-name ordering) had one existed, so it's worth stating explicitly:
verified, not assumed.

## Headline results (`results/tables/day5_headline.md`)

| Severity | Clean | Distorted | Rectified | clean ≥ rectified ≥ distorted? |
|---|---|---|---|---|
| mild | 0.614 | 0.608 | 0.602 | **NO** |
| medium | 0.614 | 0.578 | 0.605 | yes |
| severe | 0.614 | 0.554 | 0.614 | yes |

(Counts: clean 930/1514 at every severity; mild distorted 920/1514, rectified
911/1514; medium distorted 875/1514, rectified 916/1514; severe distorted
839/1514, rectified 929/1514 — severe rectified is statistically
indistinguishable from clean, 929 vs 930.)

## Sanity check: does clean ≥ rectified ≥ distorted hold? — a real, investigated finding, not a bug

**At medium and severe severity, yes, clearly.** Rectification recovers a
substantial fraction of the accuracy lost to distortion at both levels
(medium: 57.8% → 60.5%; severe: 55.4% → 61.4%, nearly back to the clean
61.4% ceiling).

**At mild severity, the raw numbers show a small reversal**: rectified
(60.2%) is slightly *below* distorted (60.8%), by 9 images out of 1,514. This
is exactly the kind of outcome the plan anticipated needing "a clear, honest
explanation" for rather than silently accepting or hiding — so it was
investigated rather than reported as-is.

**Investigation**: a 9-image, 0.6-percentage-point gap could easily be
ordinary sampling noise rather than a real effect, and comparing two raw
accuracy numbers can't distinguish the two. Because distorted and rectified
are evaluated on the *identical* 1,514 images (a paired design), McNemar's
exact test can: it looks only at the "discordant" images — cases where the
two conditions disagree on correctness — and asks whether the direction of
disagreement is lopsided enough to be unlikely under pure chance.
`src/metrics.py`'s new `paired_mcnemar` function computed this
(`results/tables/day5_mcnemar.md`):

| Severity | distorted-only-correct (b) | rectified-only-correct (c) | n discordant | p-value | significant (p<0.05)? |
|---|---|---|---|---|---|
| mild | 81 | 72 | 153 | 0.518 | **no** |
| medium | 81 | 122 | 203 | 0.0049 | yes |
| severe | 96 | 186 | 282 | <0.0001 | yes |

**Conclusion**: at medium and severe severity, rectification's benefit is
statistically robust (p = 0.005 and p < 0.0001 respectively) — far more
images are corrected by rectification than are broken by it (122 vs 81 at
medium; 186 vs 96 at severe). At mild severity, the 81-vs-72 split is
statistically indistinguishable from a coin flip (p = 0.52) — meaning **the
apparent "rectification hurts at mild severity" pattern in the raw
percentages is not a demonstrated real effect**; it is consistent with
ordinary sampling noise around a true difference of roughly zero.

This is being reported as: **the plan's expected sanity pattern holds with
statistical confidence at medium and severe severity, and is neither
confirmed nor contradicted at mild severity** (the data are simply not
informative enough at that small an effect size to say either way) — not "a
violation was found and explained away," and not "the pattern holds
everywhere." A plausible reading, consistent with this data without
overclaiming: at mild severity the distortion itself barely hurts CLIP
(61.4% → 60.8%, a much smaller drop than at medium/severe), leaving very
little room for rectification to recover anything, so whatever the two-step
warp's own interpolation cost (Day 4's measured MAE 2-10 out of 255) does to
the image roughly cancels out against whatever tiny geometric benefit
rectification provides at that severity — neither effect dominates enough to
show up clearly at n=1,514.

## What was deliberately not done today (flagged for Day 6, not expanded here)

- No per-class breakdown was computed (Day 6 task). If the mild-severity
  near-tie turns out to hide an per-class asymmetry (e.g. rectification
  clearly helping one class while hurting another, cancelling out in the
  aggregate), that would only show up in Day 6's per-class table — worth
  checking there.
- No new experimental conditions or severities were added — the grid is
  exactly clean × {mild, medium, severe} × {distorted, rectified}, as
  specified.
- The paper was not touched today, per the plan's explicit "do not start
  writing the paper today" instruction.

## Reproducibility artifacts

- `results/raw/day5_full_grid.csv` — all 13,626 rows.
- `experiments/configs/day5_run.json` — seed (42), prompt template, class
  names, dataset root, model/weights, severity fractions, border policy, git
  commit hash of HEAD at run time, run timestamp, row/failure counts, total
  wall-clock time.
  - **Caveat worth stating plainly**: per this project's git workflow (I do
    not commit — the user reviews and commits everything), the recorded
    git commit hash (`99b4450f...`, the Day 4 commit) reflects the last
    *committed* state, not the exact Day 5 code in the working tree at run
    time (this run used today's not-yet-committed `experiment_runner.py`,
    the updated `distortion.py` Unicode fix, and the new `metrics.py`
    function). This is a known limitation of hash-based provenance under an
    uncommitted-working-tree workflow, not something silently glossed over —
    once today's changes are committed, the run could be re-verified against
    the new commit hash if needed.
- `results/tables/day5_headline.md`, `results/tables/day5_mcnemar.md` — both
  regenerated directly from `day5_full_grid.csv`, never hand-edited.
- `results/raw/homographies/{mild,medium,severe}/*.npy` — 4,542 files, one
  per (image, severity), from this run.

## Success criteria check

Met, with an honestly-reported nuance rather than a forced clean result: the
grid ran to completion (13,626/13,626 expected rows, 0 failures after fixing
a real code bug rather than hiding it as a data issue), and the headline
sanity check holds with statistical significance at medium and severe
severity. The mild-severity near-tie was investigated with an appropriate
statistical test rather than either ignored or "explained away," and is
reported as inconclusive at that severity level, not as a violation.
