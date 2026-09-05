# Day 3 — Synthetic distortion module

Date: 2026-09-05

## Objective (from execution plan)

Build the controlled perspective-distortion tool. Every distortion is
reproducible from a seed and a severity level.

## What was built

- `src/distortion.py`:
  - `distort(image, severity, seed) -> (distorted_image, H)` — the core
    function. Perturbs all 4 image corners independently (a fixed-order,
    single `rng.uniform(-1, 1, size=(4, 2))` draw from a local
    `np.random.RandomState(seed)`, scaled per-axis by the severity's bound),
    builds `H` via `cv2.getPerspectiveTransform`, warps via
    `cv2.warpPerspective`.
  - `load_image_rgb(path)` — reads via `cv2.imread` (BGR) and converts to RGB,
    so this module, rectification (Day 4), and CLIP inference (which expects
    RGB via PIL) all agree on channel order. Documenting this as a deliberate
    project-wide decision made now, not something to reconcile later.
  - `image_id_from_path`, `save_homography` — the "H loggable to disk as
    `.npy`, keyed by image ID and severity" mechanism the plan asks to set up
    now for Day 5's use. Image ID is `"<style>__<filename-stem>"` (e.g.
    `Gothic__catedral_toledo2`) since MonuMAI filenames are not unique across
    style folders — a couple of gallery images turned out to be named things
    like `5d13e2a71d048.jpg`, so folder-name-based disambiguation is not
    optional, it is necessary.
  - `build_distortion_gallery(...)` — picks 6 images via a seeded
    `random.Random(42).sample(...)` (deliberately a separate RNG from the
    per-distortion `np.random.RandomState`, so image *selection* and pixel
    *distortion* are reproducible independently of each other), distorts each
    at all 3 severities, and saves the 6x4 labeled grid.
- `tests/test_distortion.py` — a runnable script (no pytest dependency is
  pinned, so this uses plain `assert`) covering: byte-identical output across
  repeated calls (same image/severity/seed), different seeds actually differ,
  output shape matches input, H is a proper 3x3 matrix, and an invalid
  severity string raises rather than silently doing something odd.

## Severity ranges chosen, and why

`mild = 5%`, `medium = 10%`, `severe = 20%` of image width/height, exactly as
the plan specifies — but the reasoning for defending these specific numbers in
the paper is:

- **Mild (5%)**: modeled as a realistic handheld phone shot — a photographer
  standing roughly in front of a facade, phone not perfectly level, no
  deliberate attempt to angle the shot. At this magnitude the geometry should
  look almost unremarkable; if CLIP's accuracy drops much at this level, that
  itself would be a notable (concerning) finding about CLIP's tilt
  sensitivity.
- **Medium (10%)**: modeled as a shot deliberately taken from an angle —
  e.g. to fit a whole tall facade into frame when standing too close to shoot
  it straight-on, a common real constraint on narrow streets. Visibly skewed,
  but every part of the facade is still fully legible.
- **Severe (20%)**: modeled as an extreme case — the photographer is well off
  to one side, but a human viewer looking at the result could still identify
  the building and, plausibly, its architectural style. This is deliberately
  pitched at "still legible" rather than "unrecognizable," because a severity
  level a human can't parse either would make any CLIP failure uninformative
  (of course it fails if a human would too) — the interesting scientific
  question is whether CLIP degrades faster than a human would at a severity
  a human can still handle.

## What happens visually at each severity (`figures/day3_distortion_gallery.png`)

At **mild**, the warp is subtle: straight architectural lines (window edges,
cornices, arch springlines) show a small, easy-to-miss skew, and the small
black sliver at the image edge where source pixels ran out is barely
noticeable. At **medium**, the skew is unambiguous — verticals lean
consistently in one direction, and a visible black triangular/wedge region
appears along one or two edges where the warp pulled a corner inward past the
original frame. At **severe**, the effect is dramatic: the whole facade reads
as strongly keystoned, a substantial fraction of one or two edges is black
border, and in the most extreme corner draws (this project's random offsets
are independent per corner, so severity is a bound, not a fixed amount — see
below) the facade can appear to lean at what feels like an aggressive
angle while still remaining a single recognizable building. This matches the
intent behind the chosen fractions.

One thing worth naming plainly rather than glossing over: because each corner's
offset is drawn independently within `[-severity_fraction, +severity_fraction]`,
"severe" does not mean every image gets warped by exactly 20% — some draws
land closer to the middle of the range, others near the extremes, and the 4
corners need not move in the same direction or by the same amount. This is
intentional (independent per-corner perturbation is what makes the transform
a true homography rather than an affine warp, per the plan's explicit
requirement), but it does mean "severe" is a *ceiling* on distortion
magnitude, not a constant — two different severe-severity images of the same
building could look meaningfully different from each other. This should be
kept in mind when interpreting per-image results later; it does not affect
the aggregate accuracy numbers the paper actually reports, which average over
many images per severity level.

## Reproducibility — verified, not just claimed

`tests/test_distortion.py` passes in full:

```
test_byte_identical_across_repeated_calls: PASSED
test_different_seeds_differ: PASSED
test_output_shape_matches_input: PASSED
test_homography_is_3x3: PASSED
test_unknown_severity_raises: PASSED

ALL TESTS PASSED
```

Additionally, per the plan's literal success criterion
("`distort(image, "medium", seed=42)` ... same pixels ... on any machine"),
ran `distort` on a real MonuMAI image
(`data/monumai/Gothic/20181212_105032.jpg`) in 3 separate fresh Python
process invocations and compared SHA-256 hashes of both the output pixels and
H:

```
run 1 sha256(pixels)= e107724afe99de54d5605905213c32adf6a64d4c0c0529b53e6bf8159163fa91
run 2 sha256(pixels)= e107724afe99de54d5605905213c32adf6a64d4c0c0529b53e6bf8159163fa91
run 3 sha256(pixels)= e107724afe99de54d5605905213c32adf6a64d4c0c0529b53e6bf8159163fa91
run 1 sha256(H)     = fa14d795968ed2fddaf9fe984937e47c316a3a14a8dbf2948f60e65a517c4139
run 2 sha256(H)     = fa14d795968ed2fddaf9fe984937e47c316a3a14a8dbf2948f60e65a517c4139
run 3 sha256(H)     = fa14d795968ed2fddaf9fe984937e47c316a3a14a8dbf2948f60e65a517c4139
```

Identical across all 3 runs. **Caveat worth being explicit about:** this
guarantees byte-identical reproducibility *within this project's pinned
environment* (`requirements.txt`'s exact `numpy==2.1.3` / `opencv-python
==4.10.0.84`). `np.random.RandomState`'s underlying algorithm (MT19937) and
OpenCV's `getPerspectiveTransform`/`warpPerspective` implementations are
themselves deterministic and stable across platforms for identical library
versions, so this claim is expected to hold across machines too — but it has
only been verified on this one machine, across process restarts, not on a
second physical machine. If cross-machine reproducibility is ever actually
tested (e.g. moving to Colab per the plan's fallback), it should be
re-verified there rather than assumed.

## Homography-saving mechanism (for Day 5)

Exercised now via the gallery build: 18 `.npy` files were saved (6 images x 3
severities) under `results/raw/homographies/<severity>/<image_id>.npy`, e.g.
`results/raw/homographies/medium/Gothic__catedral_toledo2.npy`. Verified one
loads back as a proper `(3, 3)` float64 array. This confirms the mechanism
works; Day 5's full grid run will populate this same directory structure for
all 1,514 images x 3 severities.

## Do-NOT compliance check

- No affine or scipy warp used — `cv2.getPerspectiveTransform` +
  `cv2.warpPerspective` only, confirmed by `test_homography_is_3x3`.
- No rotation, scale, or lens distortion added — the only geometric operation
  is the corner-perturbation-derived homography; `load_image_rgb` only
  changes channel order, not geometry.

## Success criteria check

Met: `distort(image, "medium", seed=42)` verified byte-identical across 3
independent process runs on a real dataset image (see hashes above), and the
full test suite passes.

## Open items carried into Day 4

- Day 4 will need to decide the border-handling policy for
  `rectify_oracle`'s output (black/crop/mean-fill) — Day 3's distortion
  already introduces black borders in the *distorted* image (OpenCV's
  default `warpPerspective` fill), which is a separate, prior decision from
  Day 4's rectification-border policy and is not being revisited here.
- The independent-per-corner "severe is a ceiling, not a constant" behavior
  noted above should be kept in mind if Day 6's failure-gallery analysis ever
  needs to distinguish "a hard severe example" from "an easy one."
