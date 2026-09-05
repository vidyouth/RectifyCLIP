# RectifyCLIP — paper draft

> Status: complete draft, all 6 sections filled in (Day 7). Every number below
> traces to a file under `results/` or `figures/`, and every such file is
> itself regenerable from `results/raw/day5_full_grid.csv` and
> `results/raw/day6_scores.csv` via `src/day6_analysis.py` (see
> `reproduce.sh`). This is a draft written to be turned into a formatted
> submission, not a finished camera-ready paper — citation details flagged
> below as unverified should be checked against the original sources before
> submission, per the execution plan's own warning that exact citation
> details were not independently re-verified here.

## Abstract

Frozen vision-language models like CLIP are increasingly applied "as-is" to
built-environment tasks, yet real facade photographs are rarely captured
straight-on, and prior work has flagged viewing-angle distortion as a
plausible error source without testing it directly. We present a controlled
study asking whether classical, homography-based perspective rectification
recovers the zero-shot architectural-style classification accuracy that a
frozen CLIP (ViT-B/32) loses under synthetic perspective distortion of 1,514
MonuMAI facade images across 4 styles. Using oracle rectification (the exact
inverse of a known distortion homography), we find recovery is real but
uneven: statistically robust at medium (74.5% recovery, p=0.005) and severe
(98.9% recovery, p<0.0001) distortion, but statistically indistinguishable
from no effect at mild distortion (p=0.52), where distortion itself barely
hurts accuracy. Critically, rectification does not help all 4 classes
equally — it has a consistently negative effect on Gothic facades
(-51.4% pooled recovery) even as it strongly benefits others. We trace this,
and the outsized recovery for the weakest class (Baroque, 105.5%), to a
previously undocumented "Renaissance attractor" bias: CLIP systematically
defaults to a Renaissance prediction for ambiguous distorted facades, and
most of rectification's apparent benefit is disproportionately driven by
correcting this one specific failure mode. A competing hypothesis — that
classification recovery would track a class's pixel-level reconstruction
error — was tested and refuted. These results suggest classical rectification
is a genuinely useful but class-dependent and mechanism-specific remedy, not
a uniform fix, for perspective sensitivity in frozen zero-shot classifiers.

## 1. Introduction

Frozen, zero-shot vision-language models such as CLIP (Radford et al., 2021)
are attractive for built-environment analysis tasks precisely because they
require no task-specific fine-tuning: a practitioner can point CLIP at a
photograph of a building and a text prompt naming candidate styles or
materials and get a usable prediction with no labeled training set of their
own. Tarkhan et al. (2025) demonstrate this directly for facade material
mapping using zero-shot segmentation, and explicitly note that real-world
facade photographs are rarely captured from a perfectly frontal viewpoint —
viewing-angle distortion is flagged in their discussion as a plausible source
of error. However, to our knowledge this specific failure mode (projective
perspective distortion, as opposed to the blur/noise/weather corruptions
studied in ImageNet-C-style robustness benchmarks) has not been directly
measured for CLIP-based architectural classification, nor has any correction
strategy been evaluated against it.

This paper closes that specific, narrow gap. We do not propose a new method:
the contribution is a controlled empirical isolation of one variable —
perspective distortion, with and without classical geometric rectification —
on a frozen, off-the-shelf CLIP model, holding the model, prompts (mostly),
dataset, and task fixed. Concretely, we ask: **does classical
(OpenCV-based) perspective rectification recover the zero-shot
architectural-style classification accuracy that a frozen CLIP model loses
under controlled synthetic perspective distortion of building facade
images?** We answer this with oracle rectification — assuming perfect
knowledge of the distorting homography, which isolates "is this recoverable
in principle" from "can a real system recover it without ground truth,"
the latter being future work (Section 6).

Our answer is not a simple yes: rectification helps substantially at medium
and severe distortion, is inconclusive at mild distortion, and — most
notably — does not help uniformly across the 4 architectural styles in our
dataset, revealing a systematic classification bias in CLIP that we believe
is itself a useful, previously undocumented observation about how this
frozen model behaves under geometric perturbation.

## 2. Related work

**CLIP and its robustness to visual corruption.** CLIP (Radford et al., 2021)
established that a model trained with contrastive image-text pairs at scale
can perform strong zero-shot image classification via natural-language
prompts, with no task-specific training. Follow-up work has studied CLIP's
robustness to various test-time corruptions and distribution shift; BATCLIP
(Sanyal et al., arXiv:2412.02837, 2024) studies bimodal (image + text)
test-time adaptation for CLIP under corruption, which we take as the closest
prior work on CLIP corruption robustness generally, though — to our
knowledge — not specifically on projective/perspective geometric distortion
of the kind isolated here. *(Citation detail not independently re-verified;
confirm exact venue/arXiv status before submission.)*

**Zero-shot facade analysis.** Tarkhan et al. (Mapping facade materials
utilizing zero-shot segmentation…, Scientific Reports 15, 5492, 2025, DOI
10.1038/s41598-025-86307-1) apply zero-shot segmentation to facade material
mapping and explicitly flag viewing-angle distortion as an unaddressed
limitation — this is the specific gap motivating the present study, as
described in Section 1. *(DOI as supplied in the project's execution plan;
not independently re-resolved here — verify before submission.)*

**Classical rectification for downstream vision models.** Prior work has
paired classical geometric rectification with learned downstream models
in other domains — we point to GLNet and OpenFACADES as representative of
this line of work (rectification or geometric normalization as a
preprocessing step feeding a downstream neural model), though full citation
details for both were not independently verified for this draft and should
be resolved before submission; we flag this explicitly per the plan's own
caution about not inventing citation details. Our study differs from that
general line of work in specifically pairing classical, homography-based
*oracle* rectification with a *frozen, zero-shot* (not fine-tuned)
foundation model, and in explicitly measuring the recovery as a function of
distortion severity and per-class, rather than reporting only an aggregate
before/after number.

**Homography estimation and geometric rectification.** Our distortion and
rectification protocol (Section 3) follows standard projective-geometry
formalism; see Hartley & Zisserman, *Multiple View Geometry in Computer
Vision*, 2nd ed. (2004), for the underlying theory of homographies and their
inversion.

**Dataset.** We use MonuMAI (Lamas et al. — *MonuMAI: Dataset, Deep Learning
Pipeline and Citizen Science Based App for Monument Recognition*; exact
venue/year not independently re-verified here, see README.md), an
expert-labeled dataset of 1,514 Spanish monument facade photographs across 4
architectural styles (Hispanic-Muslim, Gothic, Renaissance, Baroque),
originally built for a detection task (object-level style-element bounding
boxes) that this study does not use — only whole-image style labels.

## 3. Method

### 3.1 Model, dataset, and task

All experiments use a single frozen model: CLIP ViT-B/32 with
`laion2b_s34b_b79k` pretrained weights (via `open_clip`), used purely
zero-shot — no fine-tuning, no gradient updates. This backbone was chosen for
being cheap enough to run the full experimental grid on CPU, well-studied
(the same architecture family as the original CLIP paper), and openly
licensed/reproducible (see `experiments/logs/day1.md`). The task throughout
is 4-way zero-shot architectural-style classification (Hispanic-Muslim,
Gothic, Renaissance, Baroque) on all 1,514 images of MonuMAI (Section 2),
using cosine similarity between one CLIP image embedding and 4 CLIP text
embeddings (one prompt per class, `"a photo of {} architecture"` as the
default/primary prompt — see Section 3.4 and 4.6 for the prompt-sensitivity
check). Preprocessing is entirely `open_clip`'s own default ViT-B/32
transform; no custom resizing or cropping was added at any point.

### 3.2 Distortion protocol

*(See the existing "Distortion protocol" subsection below, written on Day 3
— unchanged here.)*

### 3.3 Rectification protocol

*(See the existing "Rectification protocol" subsection below, written on
Day 4 — unchanged here.)*

### 3.4 Experimental grid and metrics

The full experimental grid (`src/experiment_runner.py`, `results/raw/
day5_full_grid.csv`) evaluates every one of the 1,514 images under 3
severities (mild, medium, severe) × 3 conditions (clean, distorted,
rectified), logging one row per (image, severity, condition) — including 3
separate clean rows per image, one per severity, so each severity's
clean/distorted/rectified comparison is self-contained. This gives
1,514 × 3 × 3 = 13,626 rows and, because the "clean" condition does not
depend on severity, only 1,514 × 7 = 10,598 actual CLIP image encodes
(clean once + distorted/rectified per severity), a compute optimization that
does not change any logged value. All randomness (corner-offset sampling,
image selection for galleries) is seeded (seed=42 throughout), and every
`distort()` call is independently verified byte-identical across repeated
calls and fresh process invocations (`tests/test_distortion.py`,
`experiments/logs/day3.md`).

Four metrics are reported: **top-1 accuracy** per condition/severity;
**recovery percentage**, `(acc_rectified − acc_distorted) / (acc_clean −
acc_distorted) × 100`, the paper's headline effect-size metric; **McNemar's
exact test** (`src/metrics.py:paired_mcnemar`) on the distorted-vs-rectified
discordant pairs, used because accuracy is evaluated on the *same* 1,514
images under both conditions (a paired design) — a raw percentage-point gap
alone cannot distinguish a real effect from sampling noise, but a paired
significance test can; and **mean CLIP confidence on the true class**
(distinct from top-1/predicted-class confidence, which equals the true-class
confidence only when the prediction is correct).

One methods-level robustness note: during the Day 5 full-grid run, 13 MonuMAI
image files with Spanish-accented filenames (e.g.
`iglesia_nuestra_señora_de_la_o_chipiona.jpg`) initially failed to load via
OpenCV's `cv2.imread` on Windows — a known Windows-specific limitation where
`cv2.imread` opens paths through a narrow, non-Unicode file API and fails
silently on non-ASCII paths. This was caught before finalizing results
(these same files loaded successfully via PIL for the clean condition,
confirming the files were valid and the failure was a code bug, not a data
problem) and fixed by reading via `np.fromfile` + `cv2.imdecode` instead,
which handles Unicode paths correctly. The fix was verified to change no
other results (regression tests before/after gave identical numbers) and the
grid was re-run to completion with 0 failures. See
`experiments/logs/day5.md` for the full account.

## 4. Results

### 4.1 Clean baseline

Zero-shot CLIP (default prompt) achieves **61.43% accuracy (930/1,514)** on
clean, undistorted MonuMAI images — well above the 25% chance rate for this
4-way task, but far from ceiling. Per-class accuracy is markedly uneven:
Hispanic-Muslim 78.3%, Gothic 70.5%, Renaissance 78.5%, and Baroque only
34.1% — with Baroque's errors concentrated specifically as Renaissance
misclassifications (320/516 Baroque images, 62%; confusion matrix in
`figures/day2_confusion_clean.png`). This pre-existing Baroque→Renaissance
confusion, observed before any synthetic distortion is applied, turns out to
be the first symptom of a bias examined at length in Section 4.5.

### 4.2 Severity curve and recovery percentage

Figure `day6_severity_curve.png` and Table (`results/tables/day5_headline.md`,
`results/tables/day6_recovery.md`):

| Severity | Clean | Distorted | Rectified | Recovery % | McNemar p-value |
|---|---|---|---|---|---|
| mild | 61.4% | 60.8% | 60.2% | −90.0% (see caveat) | 0.52 (not significant) |
| medium | 61.4% | 57.8% | 60.5% | 74.5% | 0.005 |
| severe | 61.4% | 55.4% | 61.4% | 98.9% | <0.0001 |

Distortion accuracy declines monotonically and substantially with severity
(61.4% → 60.8% → 57.8% → 55.4%); rectification tracks close to the clean
line at every severity and is essentially indistinguishable from it at
severe (61.36% vs. 61.43% clean — 929 vs. 930 correct out of 1,514).

**The mild-severity recovery figure requires explicit caution, not a face-value
reading.** At mild severity, distortion itself barely hurts accuracy (930→920
correct, a 10-image gap), and the apparent rectified-vs-distorted reversal
(920→911, i.e. rectified looking *worse* than distorted) amounts to only 9
images. McNemar's exact test on this specific comparison gives p=0.52 — this
9-image gap is statistically indistinguishable from chance. Reporting the
raw recovery percentage this produces (−90.0%) without this context would
overstate the precision the data supports: dividing a statistically
non-significant 9-image numerator by a 10-image denominator produces a
number that looks large and precise but would swing by 10-20 percentage
points if just 1-2 individual image outcomes flipped. **We report mild
severity's effect as statistically inconclusive — neither confirming nor
contradicting a recovery/harm effect — rather than as evidence that
"rectification hurts at mild severity."** Medium and severe, by contrast, both
show large, statistically significant recovery effects (p=0.005, p<0.0001)
and are where the recovery-percentage metric is actually informative.

### 4.3 Per-class breakdown: rectification does not help all classes equally

Table `results/tables/day6_per_class.md`, pooled across all 3 severities:

| Style | Clean | Distorted | Rectified | Recovery % |
|---|---|---|---|---|
| Hispanic-Muslim | 78.3% | 73.5% | 78.3% | 100.0% |
| **Gothic** | 70.5% | 67.0% | **65.3%** | **−51.4%** |
| Renaissance | 78.5% | 84.5% | 80.1% | 73.2% |
| Baroque | 34.1% | 25.8% | 34.6% | 105.5% |

Gothic is the sole class with a *negative* pooled recovery percentage, and
this is not an averaging artifact: rectified accuracy is below distorted
accuracy at **every individual severity** for Gothic (mild 69.1%→68.0%,
medium 66.0%→64.1%, severe 66.0%→63.8%). Rectification has a small but
consistent, direction-stable *harmful* effect specifically on Gothic
facades. This directly complicates any simple "rectification works"
headline and is reported as a real, investigated finding, not a footnote.

Baroque — the weakest class at baseline (Section 4.1) — shows the strongest
benefit, its rectified accuracy (34.6%) slightly exceeding its own clean
accuracy (34.1%). Renaissance is the most paradoxical: its own accuracy is
*higher under distortion than under clean conditions* at every severity
(84.5% vs. 78.5% pooled) — synthetic geometric corruption apparently makes
CLIP *better*, not worse, at recognizing genuine Renaissance facades. Section
4.5 argues this, Gothic's harm, and Baroque's outsized benefit share one
underlying cause.

### 4.4 The Renaissance attractor (novel finding)

Examining every individual image where rectification changed a prediction's
correctness (not just the aggregate accuracy numbers) reveals a striking,
consistent pattern. Define "hurt" as an image correctly classified when
distorted but incorrectly classified after rectification, and "helped" as
the reverse.

| True class | Hurt | Helped | Net | Helped:Hurt ratio |
|---|---|---|---|---|
| Hispanic-Muslim | 39 | 86 | +47 | 2.21 |
| Gothic | 76 | 57 | −19 | 0.75 |
| Renaissance | 90 | 49 | −41 | 0.54 |
| Baroque | 53 | 188 | +135 | 3.55 |

Of the 258 total "hurt" images, **148 (57.4%) are wrongly reclassified
specifically as "Renaissance"** after rectification — not spread evenly
across the other 3 classes. Conversely, of the 380 "helped" images, **313
(82.4%) were wrongly predicted as Renaissance under distortion** before
rectification corrected them. We term this a **"Renaissance attractor"**:
CLIP has a systematic tendency, activated or amplified under geometric
perturbation, to default to a Renaissance prediction for ambiguous facades —
and both distortion and rectification interact with this same bias, in
opposite directions for different images.

This directly explains two results reported elsewhere in this paper: the
pre-existing Baroque→Renaissance confusion observed in the *clean* condition
(Section 4.1) is not an unrelated quirk but the same underlying bias
appearing without any synthetic distortion at all; and Baroque's outsized
105.5% recovery (Section 4.3) reflects rectification disproportionately
rescuing Baroque images that distortion had pushed into a false-Renaissance
prediction (188 helped vs. 53 hurt). It also resolves the apparent paradox
of Section 4.3's Renaissance numbers: Renaissance's pooled accuracy looks
like it "recovers" positively (73.2%) by the plan's clean-anchored formula,
even though Renaissance is actually the class *most often hurt* in raw image
count (net −41) — the recovery-percentage metric and the raw per-image
outcome count are both correct, but tell different stories about the same
class, and we report both explicitly rather than only the more favorable
number.

Qualitatively (`figures/day6_failure_gallery.png`, 8 uniformly-sampled
"hurt" cases, not cherry-picked, per the plan's explicit rule against
selective figures): most failures show no obvious rectification artifact —
no visible over-warp or border contamination — the rectified panels
generally look like well-restored, geometrically correct facades. The
failure mode is a subtle classification boundary effect, not a visible bug:
correcting the geometry apparently removes some distortion-specific visual
cue that had, by chance, been pushing CLIP toward the correct class, and the
more "normal-looking" corrected image nudges the prediction toward CLIP's
Renaissance default instead. A root-cause investigation of *why* CLIP
defaults to Renaissance specifically (a prompt-wording artifact? a
pretraining-data frequency prior? genuine visual ambiguity between these
particular architectural styles?) was not undertaken — that would be new
experimental scope beyond this week's plan — and is named explicitly as
future work (Section 6).

### 4.5 A tested and refuted hypothesis: pixel-level texture sensitivity

Before the per-class breakdown above was available, an earlier stage of this
project (rectification correctness testing, Section 3.3) observed that
Hispanic-Muslim images consistently showed higher round-trip pixel
reconstruction error (mean absolute error 7.5-10.3 out of 255) than
Gothic/Baroque images (2.0-4.0), hypothesized to result from fine, repetitive
lattice/arabesque texture being more sensitive to the sub-pixel blur
introduced by double bilinear interpolation (distortion's forward warp, then
rectification's inverse warp). This raised a natural hypothesis: that
classes with higher pixel-level reconstruction error would show worse
classification-level recovery.

**This hypothesis was explicitly tested and is refuted by the data.**
Hispanic-Muslim is *not* overrepresented among "hurt" cases (39/258 = 15.1%,
below its 21.6% share of the dataset) and shows the second-best net
classification recovery of any class (+47, ratio 2.21) — better than both
Gothic and Renaissance, both of which have *negative* net recovery despite
lower pixel-level reconstruction error. Pixel-level reconstruction fidelity
and classification-level recovery are not the same thing, and are not even
positively correlated across classes in this data; we report this as a
negative result about our own working hypothesis rather than silently
dropping it, since it directly informs the more accurate story told in
Section 4.4.

### 4.6 Prompt sensitivity

To check whether the headline finding is an artifact of one specific prompt
wording, the medium-severity grid was rerun for 2 additional prompt
templates (`results/tables/day6_prompt_sensitivity.md`):

| Prompt | Clean | Distorted | Rectified |
|---|---|---|---|
| `"{}"` (bare class name) | 36.8% | 34.1% | 36.8% |
| `"a photo of {} architecture"` (default) | 61.4% | 57.8% | 60.5% |
| `"a facade in {} architectural style"` | 59.5% | 58.5% | 59.6% |

Rectified accuracy exceeds distorted accuracy at medium severity for **all 3
prompts without exception**, including the degenerate bare-class-name
prompt — the direction of the core finding (rectification helps at medium
severity) is robust to prompt choice, even though the exact magnitude
clearly is not (the bare-class-name prompt is dramatically weaker overall;
see below). We do not claim the exact recovery percentages generalize across
prompts, only the direction, per the plan's explicit goal for this ablation.

Separately, and unrelated to the rectification question, the bare
class-name prompt was found on Day 2 to catastrophically fail specifically
on Hispanic-Muslim (0/327 correct, 0.0% accuracy) — plausibly because,
without the word "architecture" to frame it, CLIP's text encoder interprets
"Hispanic-Muslim" via unrelated demographic/religious associations from its
pretraining data rather than as an architectural style descriptor. This
remains a hypothesis, not confirmed, but is a striking illustration of why
prompt engineering matters for this kind of task and why no single prompt
was chosen as "the" prompt without this kind of check.

### 4.7 Failure gallery

See Section 4.4 above and `figures/day6_failure_gallery.png` for the 8
sampled failure cases and their common qualitative pattern.

## 5. Discussion & limitations

**Oracle vs. blind rectification.** All rectification results in this paper
use *oracle* rectification — inverting the exact, known distortion
homography. This isolates a clean scientific question ("if geometry could be
perfectly corrected, would CLIP recover?") but is not directly deployable: a
real system encountering an arbitrarily-tilted facade photograph does not
know the ground-truth homography and would need *blind* rectification (e.g.
vanishing-point estimation via `cv2.HoughLinesP` line detection, then
homography-from-lines). This project deliberately did not implement blind
rectification — it was scoped as a Day 7 stretch goal only, and this week's
plan explicitly prioritized a rigorous oracle-rectification study over a
rushed blind-rectification implementation. The oracle results in this paper
should be read as an upper bound on what any real rectification pipeline
could achieve, not as a demonstrated deployable result.

**Single dataset, single model, single task.** All results are specific to
MonuMAI (1,514 images, 4 Spanish-monument architectural styles), CLIP
ViT-B/32 with `laion2b_s34b_b79k` weights, and 4-way zero-shot
classification with one primary prompt template. We do not know whether the
severity-dependent recovery pattern, the per-class unevenness, or
specifically the Renaissance-attractor bias would replicate with a different
CLIP backbone (e.g. ViT-L/14), a different pretraining checkpoint, a
different dataset (different architectural styles, different geographic/
photographic distribution), or a different downstream task (e.g. material
classification, as in Tarkhan et al., rather than style classification).

**The Renaissance attractor's root cause is unresolved.** Section 4.4's
central finding — that CLIP systematically defaults to Renaissance under
geometric perturbation, and that this drives most of the apparent
rectification benefit — is well-supported by the count data (148/258,
313/380) but this project did not run a targeted follow-up experiment to
isolate *why*. Plausible candidate explanations include a prompt-wording
artifact specific to how "Renaissance" is phrased relative to the other 3
class names, a training-data frequency prior in CLIP's pretraining corpus
(if "Renaissance architecture" is a more common web image-text pairing than
the other 3 styles), or a genuine visual ambiguity between Renaissance and
the other styles under geometric perturbation specifically (as opposed to
under clean viewing conditions, where Baroque→Renaissance confusion is
present but weaker). Distinguishing between these would require new,
targeted experiments (e.g. varying only the Renaissance class name/prompt
while holding the image set fixed) not undertaken this week.

**Null-result honesty.** We explicitly do not claim rectification "works" as
a blanket statement. It is statistically robust and substantial at medium
and severe distortion, inconclusive at mild distortion, and actively harmful
for one specific class (Gothic) at every severity tested. A reviewer should
take from this paper that classical rectification is a real but
class-dependent and severity-dependent remedy for a frozen zero-shot
classifier's perspective sensitivity — not a uniform fix, and its apparent
aggregate benefit is disproportionately attributable to correcting one
specific, previously undocumented classification bias (Section 4.4) rather
than uniformly improving geometric legibility across all classes.

## 6. Future work

1. **Blind rectification**: replace the oracle homography with a real
   estimation pipeline (line detection via `cv2.HoughLinesP`,
   vanishing-point estimation, homography-from-lines) and re-measure
   recovery against the same clean/distorted baselines — this determines
   whether the oracle-rectification benefit reported here survives contact
   with a realistic, no-ground-truth deployment.
2. **Investigate the Renaissance-attractor mechanism directly**: targeted
   prompt-ablation and/or embedding-space analysis to distinguish a
   prompt-wording artifact from a pretraining-frequency prior from a genuine
   visual-similarity effect.
3. **A second dataset**: re-run this same protocol on a dataset with a
   different architectural-style distribution (e.g. WikiChurches) to check
   whether the per-class unevenness and the attractor bias are MonuMAI-
   specific or a more general property of CLIP's style-classification
   behavior.

## References

- Radford, A. et al. *Learning Transferable Visual Models from Natural
  Language Supervision*. ICML 2021.
- Tarkhan et al. *Mapping facade materials utilizing zero-shot
  segmentation…*. Scientific Reports 15, 5492 (2025).
  DOI: 10.1038/s41598-025-86307-1.
- Sanyal et al. *BATCLIP: Bimodal Online Test-Time Adaptation for CLIP*.
  arXiv:2412.02837 (2024). *(Not independently re-verified — confirm before
  submission.)*
- Hartley, R. & Zisserman, A. *Multiple View Geometry in Computer Vision*,
  2nd ed. (2004).
- Lamas et al. MonuMAI dataset paper. *(Exact venue/year not independently
  re-verified — confirm before submission; see README.md.)*
- GLNet, OpenFACADES — classical rectification for downstream vision models.
  *(Full citation details not independently verified for this draft —
  resolve before submission.)*

---

## Method (existing subsections, unchanged from Days 3-4)

### Distortion protocol

Perspective distortion is applied as a single controlled projective warp,
isolating this one variable from any other kind of image corruption (no
rotation, scale, or lens distortion is mixed in). For an image of width `w`
and height `h`, the four image corners are each perturbed independently by a
random offset `(dx, dy)` drawn uniformly from `[-severity_fraction * w,
severity_fraction * w]` (horizontally) and `[-severity_fraction * h,
severity_fraction * h]` (vertically), where `severity_fraction` is 0.05
(mild), 0.10 (medium), or 0.20 (severe). Perturbing all 4 corners
independently — rather than deriving one corner from the other three, as an
affine transform would — is what gives the resulting transform its full 8
degrees of freedom as a genuine projective (homography) warp, matching what
an off-axis camera tilt produces geometrically. The homography `H` mapping
the original corners to the perturbed ones is computed with
`cv2.getPerspectiveTransform`, and the image is warped with
`cv2.warpPerspective`, using OpenCV's default bilinear interpolation and
black (zero-value) border fill for any newly-exposed regions.

Severities were chosen to span a physically plausible range of real
handheld-camera framing error: mild (5%) approximates a small, largely
unintentional tilt typical of an ordinary phone snapshot; medium (10%) is a
clearly-angled shot, such as one taken specifically to fit an entire facade
into frame from a non-frontal position; severe (20%) sits near the edge of
legibility, where the geometry is visibly and substantially skewed but a
human viewer can still recognize the building and its style. Figure
`day3_distortion_gallery.png` shows this progression across 6 sample images.

Every distortion is fully reproducible: given the same image, severity, and
integer seed, `distort()` returns byte-identical pixels and an identical `H`
on every call, on any machine running the pinned dependency versions in
`requirements.txt` (verified in `tests/test_distortion.py` and
`experiments/logs/day3.md`). `H` is saved to disk as a `.npy` file alongside
each experiment's results, keyed by image ID and severity, so any distorted
image used in this study can be traced back to the exact transform that
produced it — and, from Day 4 onward, inverted exactly for oracle
rectification.

### Rectification protocol

Rectification in this study is *oracle* rectification: given a distorted
image and the exact homography `H` that produced it, the image is warped back
by `H^{-1}`, i.e. `rectify_oracle(image, H) = warpPerspective(image,
inverse(H))`. This is a deliberate methodological choice, not a limitation
being glossed over: the question this project asks is whether CLIP's
accuracy loss under perspective distortion is *recoverable in principle* by
geometric correction, which oracle rectification tests directly and exactly.
A real deployment would not have access to the true `H` and would need
*blind* rectification (e.g. vanishing-point or line-based estimation) —
explicitly out of scope here and left as future work (Section:
Discussion & limitations).

Because `H` maps the original image's rectangle onto a perturbed
quadrilateral that generally does not exactly cover the output canvas,
rectifying a distorted image leaves some pixels with no corresponding real
source content — regions where either the original distortion step or the
rectification step itself sampled outside the valid image area. Three
border-handling policies were evaluated on sample images before choosing
one for the full experiment: leaving these regions black (OpenCV's default
warp behavior); cropping the image down to the largest axis-aligned
rectangle entirely inside the valid region; and filling the invalid region
with the mean color of the valid region. Cropping was rejected because the
amount cropped away differs per image and grows with distortion severity,
which would confound the rectification condition with an implicit,
severity-correlated change in effective zoom/field-of-view — introducing a
second variable alongside the one this project isolates. Leaving borders
black was rejected because a hard, high-contrast rectangular artifact has no
analog in the clean condition or in real photography and could plausibly
influence CLIP's prediction through the artifact itself rather than through
the (correctly restored) architectural content. The chosen policy,
**mean-color fill**, keeps every image's dimensions and field of view
identical across the clean/distorted/rectified conditions and across
severities, while replacing the stark black artifact with a low-contrast,
background-toned patch. See `experiments/logs/day4.md` for the full
side-by-side comparison and reasoning.

Oracle rectification's correctness was verified directly: for real MonuMAI
images across all 3 severities, `rectify_oracle(distort(I, H), H)` was
compared pixel-by-pixel against the original `I`, restricted to the valid
(non-border) region. Mean absolute pixel difference (0-255 scale) ranged from
2.0 to 10.3 across test images and severities — consistent with the blur
expected from two successive bilinear-interpolation warps (forward then
inverse), not with a broken inversion, which would show gross misalignment.
This was confirmed visually as well as numerically (Figure
`day4_rectification_gallery.png`).
