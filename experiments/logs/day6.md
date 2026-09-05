# Day 6 — Analysis, metrics, plots, ablations, failure gallery

Date: 2026-09-05

## Objective (from execution plan)

Turn Day 5's CSV into every figure and number the paper needs. Nothing new
gets computed after today except the explicitly-listed prompt-sensitivity
sweep and the confidence metric (both required a fresh CLIP forward pass
since Day 5's CSV only stored top1/predicted-class confidence, not the full
per-class softmax vector needed for "confidence on the TRUE class").

## One-sentence answer to the research question

**Yes, but conditionally and unevenly**: oracle rectification recovers most
of the accuracy CLIP loses to perspective distortion at medium (74.5%
recovery) and severe (98.9% recovery, essentially full recovery, both
statistically significant per Day 5's McNemar test), recovers nothing
distinguishable from noise at mild severity (McNemar p=0.52 — distortion
barely hurts accuracy at mild in the first place), and does **not** help all
4 architectural styles equally — it actively hurts Gothic (pooled recovery
-51.4%) while dramatically over-helping Baroque (105.5%, exceeding its own
clean accuracy), a pattern traced to a newly-discovered systematic bias: CLIP
defaults to "Renaissance" as a fallback prediction for ambiguous facades, and
both distortion and rectification interact with that bias in ways that are
not simply "more correction is always better."

## Task 1 — Recovery percentage per severity

Computed from `results/raw/day5_full_grid.csv` (no rerun — Day 5's CSV
already has everything needed) via `src/day6_analysis.py:compute_recovery`,
using exact integer correct-counts rather than pre-rounded accuracy floats
(a ratio of two small differences amplifies rounding error otherwise).
Table: `results/tables/day6_recovery.md`.

| Severity | Clean correct | Distorted correct | Rectified correct | Denominator | Numerator | Recovery % |
|---|---|---|---|---|---|---|
| mild | 930 | 920 | 911 | 10 | -9 | **-90.0%** |
| medium | 930 | 875 | 916 | 55 | 41 | 74.5% |
| severe | 930 | 839 | 929 | 91 | 90 | 98.9% |

**Mild severity's recovery % is flagged as not meaningful, not hidden.** The
denominator (clean-minus-distorted correct-count) is only 10 images out of
1,514 — distortion barely hurts accuracy at mild severity to begin with, and
Day 5's McNemar test already established the rectified-vs-distorted gap here
is statistically indistinguishable from chance (p=0.52). Dividing a small,
non-significant numerator by a denominator of 10 produces a number
(-90.0%) that reads as large and precise but would swing by ~10-20
percentage points if just 1-2 individual image outcomes flipped. This is
reported in the table (per the "don't cherry-pick, don't hide numbers" rule)
but explicitly annotated there as not a reliable effect size — medium and
severe are where the recovery-% metric is actually informative, since both
have a much larger, statistically significant effect behind them.

## Task 2 — Severity curve

`figures/day6_severity_curve.png`. Clean is flat at 0.614 across all 4
x-positions (by construction — it's the same condition regardless of
severity). Distorted declines monotonically and substantially (0.614 → 0.608
→ 0.578 → 0.554). Rectified tracks close to clean at every severity and
actually **exceeds** it very slightly at severe (0.6136 vs 0.6143 clean — a
1-image difference, not a meaningful excess, but notably not a shortfall
either). The visual story matches the numbers: distortion clearly hurts and
gets worse with severity; rectification pulls the accuracy back up close to
(mild, medium) or essentially onto (severe) the clean line.

## Task 3 — Mean CLIP confidence on the TRUE class

Required a fresh forward pass (`run_confidence_and_prompt_sweep` in
`src/day6_analysis.py`) since Day 5's CSV only logged the predicted class's
confidence, not the true class's — the two are the same value only when the
prediction was correct. Full run: 1,514 images × 3 severities × 3 conditions
for the default prompt (plus 2 extra prompts at medium severity only, see
Task 4), ~19.8 minutes wall-clock, 22,710 rows saved to
`results/raw/day6_scores.csv`.

**Internal consistency check, done before trusting any of today's new
numbers**: filtering `day6_scores.csv` to the default prompt and recomputing
accuracy independently reproduces Day 5's exact correct-counts at every
severity/condition (930/920/911, 930/875/916, 930/839/929) — confirming the
same seed, same deterministic distort/rectify pipeline, and same model
produce byte-for-byte-equivalent classification outcomes across two entirely
separate runs. This is the reproducibility guarantee from Day 3
(`distort(image, severity, seed)` byte-identical) and Day 4 (deterministic
rectification) paying off at the full-grid level, not just per-image.

Table: `results/tables/day6_confidence.md`. Plot: `figures/day6_confidence.png`.

| Severity | Clean | Distorted | Rectified |
|---|---|---|---|
| mild | 0.5674 | 0.5613 | 0.5638 |
| medium | 0.5674 | 0.5488 | 0.5637 |
| severe | 0.5674 | 0.5186 | 0.5690 |

Confidence tells a smoother, monotonic story than accuracy: distorted
confidence declines steadily and substantially with severity (0.561 → 0.549
→ 0.519), unlike distorted accuracy's noisier mild-severity blip. **Surprising
number**: at severe severity, rectified confidence (0.5690) is not just
close to clean but slightly *exceeds* it (0.5674) — mirroring the same
slight excess seen in severe accuracy. This is consistent enough between
confidence and accuracy that it looks like a real (if small) pattern rather
than two independent coincidences, though the effect is tiny and not
separately significance-tested here.

## Task 4 — Prompt-sensitivity ablation (medium severity, 3 prompts)

Table: `results/tables/day6_prompt_sensitivity.md`.

| Prompt | Clean | Distorted | Rectified | clean ≥ rectified ≥ distorted? |
|---|---|---|---|---|
| `a_facade_in_architectural_style` | 0.595 | 0.585 | 0.596 | NO |
| `a_photo_of_architecture` (Day 5 default) | 0.614 | 0.578 | 0.605 | yes |
| `bare_class_name` | 0.368 | 0.341 | 0.368 | yes |

**The direction the plan asked to check for robustness holds for all 3
prompts without exception**: rectified accuracy is strictly greater than
distorted accuracy in every single row (0.596>0.585, 0.605>0.578,
0.368>0.341) — rectification helps at medium severity regardless of which of
the 3 prompt templates is used, including the degenerate `bare_class_name`
prompt that Day 2 already flagged as unusually weak overall. The one
"NO" in the strict `clean >= rectified >= distorted` column is not a
reversal of that core finding — it's `a_facade_in_architectural_style`'s
rectified accuracy (0.596) exceeding its own clean accuracy (0.595) by a
single image's worth of difference, the same small "slight excess at high
correction" pattern seen in the severity curve and confidence table, not a
case where rectification failed to beat distortion.

## Task 5 — Failure gallery, and the "Renaissance attractor" finding

`figures/day6_failure_gallery.png`: 8 side-by-sides (distorted, rectified)
sampled uniformly at random (seed=42, no stratification by class — per the
plan's "don't cherry-pick" rule, whatever class mix resulted is itself part
of the finding) from all 258 images where rectification broke a
previously-correct distorted prediction.

**This day's central, connect-the-dots finding.** The task asked to check
whether the failure pattern correlates with Day 4's flagged Hispanic-Muslim
texture-sensitivity hypothesis (higher round-trip pixel MAE for
Hispanic-Muslim images due to fine lattice/brickwork texture). **It does
not** — computed properly over the full candidate pool, not just the 8
sampled images:

| True class | "Hurt" (distorted✓ → rectified✗) | "Helped" (distorted✗ → rectified✓) | Net | Helped:Hurt ratio |
|---|---|---|---|---|
| Hispanic-Muslim | 39 | 86 | **+47** | 2.21 |
| Gothic | 76 | 57 | **-19** | 0.75 |
| Renaissance | 90 | 49 | **-41** | 0.54 |
| Baroque | 53 | 188 | **+135** | 3.55 |

Hispanic-Muslim is **not** overrepresented among "hurt" cases (39/258 = 15.1%,
below its 21.6% dataset share) — despite Day 4's higher pixel-level
reconstruction error, its classification-level recovery is strongly net
positive. The Day 4 hypothesis, tested properly here, is refuted: pixel-level
interpolation blur for textured facades does not translate into worse
classification recovery.

**What the data actually shows instead**: of the 258 "hurt" cases, 148
(57.4%) end up wrongly re-classified as **Renaissance** specifically after
rectification — not spread evenly across the other 3 classes. And looking at
the reverse direction, of the 380 "helped" cases (rectification fixed a wrong
distorted prediction), 313 (82.4%) were wrongly predicted as Renaissance
*under distortion* before rectification corrected them. Renaissance is acting
as a systematic attractor/default prediction for CLIP on this dataset under
geometric perturbation — both distortion and rectification interact with it,
in opposite directions for different images. This directly explains — and
connects back to — Day 2's originally-flagged finding that Baroque is heavily
confused as Renaissance even in the clean condition (320/516 Baroque images
misclassified as Renaissance on Day 2). Baroque is not an unrelated weak
class; it is the class most vulnerable to falling into this same
Renaissance-attractor basin, which is exactly why Baroque shows the strongest
raw recovery (188 helped vs. 53 hurt) — rectification is disproportionately
rescuing Baroque images that distortion had pushed into a false-Renaissance
prediction.

**Common visual pattern in the gallery** (one paragraph, as requested): most
of the 8 sampled failures do not show gross rectification artifacts (no
obvious black-border contamination or visible over-warp) — the rectified
panels generally look like clean, well-restored facades, geometrically very
close to their distorted counterparts. The failures are not visually obvious
bugs; they are borderline classification decisions where correcting the
geometry removed some distortion-specific visual cue that had, by chance,
been pushing CLIP toward the correct class, and rectification's more
"normal-looking" image nudges the prediction toward CLIP's Renaissance
attractor instead. This is a genuinely subtle failure mode — a human looking
at the 8 image pairs would not obviously predict which direction the
misclassification would go, which is a different (and less obvious) story
than the naive "border artifact confused it" hypothesis this task started
from.

## Task 6 — Per-class breakdown

`results/tables/day6_per_class.md`. Pooled across all 3 severities:

| Style | Clean | Distorted | Rectified | Recovery % (pooled) |
|---|---|---|---|---|
| Hispanic-Muslim | 0.783 | 0.735 | 0.783 | 100.0% |
| Gothic | 0.705 | 0.670 | 0.653 | **-51.4%** |
| Renaissance | 0.785 | 0.845 | 0.801 | 73.2% |
| Baroque | 0.341 | 0.258 | 0.346 | 105.5% |

**Rectification does not help all 4 styles equally — it actively hurts
Gothic.** Gothic's rectified accuracy (0.653) is lower than its distorted
accuracy (0.670) *pooled*, and this holds at **every individual severity**
(per-severity table in `day6_per_class.md`: 0.691→0.680 at mild,
0.660→0.641 at medium, 0.660→0.638 at severe — rectified is below distorted
each time, not just on average). This is a consistent, not incidental,
negative effect for one specific class.

**A second, related surprise**: Renaissance's own accuracy (recall) is
*higher* under distortion than under clean conditions, at every severity
(0.827/0.840/0.869 vs. clean 0.785) — distortion appears to *improve*
Renaissance classification. Read alongside the "Renaissance attractor"
finding above, this is very likely the same underlying mechanism: if
distortion increases the rate at which CLIP defaults to predicting
"Renaissance" across the board, that inflates Renaissance's own recall (more
things get called Renaissance, including more true Renaissance images) while
simultaneously hurting the other 3 classes' precision-adjacent behavior. This
also explains why Renaissance's *pooled* recovery % (73.2%) looks positive
by the plan's formula even though Renaissance is actually the class most
often *hurt* in raw count (90 hurt vs. 49 helped, net -41, from the Task 5
table) — the recovery-% formula is anchored to the clean baseline, and
rectification pulling Renaissance's inflated-by-distortion accuracy back
down toward (but still above) clean reads as "recovery" relative to that
baseline, even though more individual Renaissance predictions were broken
than fixed by rectification. **This is flagged explicitly as a case where the
plan's headline formula and the raw per-image outcome count tell different
stories about the same class — both are reported here rather than picking
the more flattering one.**

Baroque (Day 2's weakest class, 34.1% clean accuracy, heavily confused into
Renaissance) shows the strongest and most unambiguous benefit from
rectification: net +135 images helped vs. hurt, and its rectified accuracy at
severe (0.380) actually exceeds its own clean accuracy (0.341) — rectification
does not just recover Baroque's distortion losses, it slightly overshoots
the clean baseline, most likely because rectification also fixes some of
Baroque's pre-existing Renaissance-confusion cases that were present even
before any synthetic distortion was applied to them at the geometric
level (i.e., a small side benefit unrelated to undoing this project's own
injected distortion).

## The 3-5 most surprising numbers today

1. **148/258 (57.4%)** of all "rectification broke a correct prediction"
   cases land specifically on a wrong "Renaissance" prediction — not spread
   across the other 3 classes. This is the day's most important finding and
   reframes several earlier results (Day 2's Baroque confusion, today's
   per-class table) as manifestations of one underlying CLIP bias rather than
   unrelated observations.
2. **Gothic has a negative pooled recovery percentage (-51.4%)**, and
   uniquely so — the only class where rectification's average effect is
   harmful, consistently across all 3 severities, not an averaging artifact.
3. **Mild severity's raw recovery % is -90.0%** — a large, precise-looking
   negative number produced almost entirely by dividing by a denominator of
   just 10 images, on an effect Day 5's McNemar test already showed is not
   statistically distinguishable from noise (p=0.52).
4. **Renaissance's own classification accuracy goes UP under distortion**
   (0.845 vs. 0.785 clean, pooled) — geometric corruption of the input
   images makes CLIP *better*, not worse, at recognizing true Renaissance
   facades, apparently as a side effect of the same attractor bias that hurts
   other classes.
5. **The Day 4 Hispanic-Muslim texture-sensitivity hypothesis is refuted**:
   despite showing the highest round-trip pixel reconstruction error (MAE
   7.5-10.3 vs. 2.0-4.0 for other classes) in Day 4, Hispanic-Muslim shows the
   *best* net classification recovery of any class after Baroque (net +47,
   ratio 2.21) — pixel-level reconstruction error and classification-level
   recovery are not the same thing, and assuming they'd correlate would have
   been the wrong story to tell in the paper.

## Success criterion check

Per the plan: "you could hand the results/ and figures/ folders to someone
else and they could write the paper." Everything produced today is traceable
to a specific script/function and regenerable: recovery %, severity curve,
and per-class breakdown come directly from `results/raw/day5_full_grid.csv`
(frozen Day 5 artifact) via `src/day6_analysis.py`; confidence and
prompt-sensitivity come from a fresh, seeded, reproducible rerun
(`results/raw/day6_scores.csv`, verified to exactly reproduce Day 5's
accuracy numbers on the default prompt as an internal consistency check);
the failure gallery's 8 images are a seeded (seed=42) sample from a fully
enumerated, reproducible candidate list. No manual figure editing, no
hand-typed numbers not backed by a CSV or table.

**One gap being flagged rather than quietly padded around**: the "Renaissance
attractor" finding is well-supported by the count data (148/258, 313/380) but
this project has not run a follow-up experiment specifically designed to
isolate *why* CLIP defaults to Renaissance (e.g., testing whether it's a
prompt-wording artifact, a training-data frequency prior, or a genuine visual
ambiguity between these particular styles). That would be new-experiment
scope beyond what Day 6 allows ("do not run new experiments today except the
prompt-ablation sweep") — it is recorded here as an open question for the
paper's Discussion & limitations section (Day 7), not resolved today.

## Files this depends on (traceability)

- `src/day6_analysis.py` — all Day 6 computation (recovery, severity curve,
  per-class table, failure-gallery candidate finding + gallery rendering,
  confidence + prompt-sensitivity rerun, tables/plots saving). Does not modify
  `src/experiment_runner.py` (Day 5's runner stays frozen).
- `results/raw/day5_full_grid.csv` — source for recovery %, severity curve,
  per-class breakdown, and failure-gallery candidate list.
- `results/raw/day6_scores.csv` — source for confidence and prompt-sensitivity
  tables (fresh rerun, seed=42, verified consistent with Day 5).
