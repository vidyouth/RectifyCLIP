# Day 4 — Oracle rectification module

Date: 2026-09-05

## Objective (from execution plan)

Build the "if we knew the exact distortion, could we undo it?" rectifier and
verify it actually works before Day 5 depends on it.

## What was built

- `src/rectification.py`:
  - `rectify_oracle(distorted_image, H)` — exactly as specified:
    `H_inv = np.linalg.inv(H)`, then `cv2.warpPerspective(distorted_image,
    H_inv, (w, h))`. No border handling inside this function — that is a
    separate, explicit step (see below), so this stays a pure, minimal
    inverse of `distort()`'s warp.
  - `compute_valid_mask(shape, H)` — rather than deriving the "which pixels
    are real vs. border" region analytically from H's geometry, this warps an
    all-255 mask through the *same two-step pipeline* real pixels go through
    (distort()'s H, then rectify_oracle's H_inv) and thresholds the result.
    This automatically captures both (a) border introduced during distortion
    that persists through rectification, and (b) any additional clipping the
    rectification warp itself introduces — without risking a subtly wrong
    closed-form derivation of the quadrilateral geometry.
  - `mean_absolute_error(a, b, mask)` — the sanity-test metric.
  - `largest_rectangle_in_mask`, `apply_border_policy`,
    `save_border_policy_comparison` — the border-policy machinery (see below).
  - `build_rectification_gallery` — extends the Day 3 gallery.

## Sanity test — the key correctness gate (verified with real numbers, not asserted)

Per the plan: `rectify_oracle(distort(I, H), H)` must be pixel-close to `I`
outside the border region, for any image and any H, or it is a bug.

Ran on 3 real MonuMAI images × all 3 severities (`tests/test_rectification.py`,
mean absolute pixel difference on a 0-255 scale, computed only over the valid
mask):

```
data/monumai/Gothic/20181212_105032.jpg            severity=mild   MAE=3.659  valid_fraction=0.958
data/monumai/Gothic/20181212_105032.jpg            severity=medium MAE=3.733  valid_fraction=0.924
data/monumai/Gothic/20181212_105032.jpg            severity=severe MAE=4.042  valid_fraction=0.868
data/monumai/Baroque/20181212_105011.jpg           severity=mild   MAE=3.052  valid_fraction=0.958
data/monumai/Baroque/20181212_105011.jpg           severity=medium MAE=3.067  valid_fraction=0.924
data/monumai/Baroque/20181212_105011.jpg           severity=severe MAE=3.286  valid_fraction=0.868
data/monumai/Hispanic-Muslim/5bae6559d55ef.jpg     severity=mild   MAE=7.476  valid_fraction=0.958
data/monumai/Hispanic-Muslim/5bae6559d55ef.jpg     severity=medium MAE=7.813  valid_fraction=0.923
data/monumai/Hispanic-Muslim/5bae6559d55ef.jpg     severity=severe MAE=8.673  valid_fraction=0.868

test_round_trip_pixel_closeness: PASSED
```

All values are small relative to the 0-255 scale (under 9, generally under 4),
consistent with the blur expected from **two** successive bilinear
interpolations (once during distort's forward warp, once during
rectify_oracle's inverse warp) — not with a broken inversion, which would show
gross misalignment (MAE in the tens-to-hundreds range, or a wildly different
`valid_fraction` than expected). `valid_fraction` also behaves exactly as
expected: it shrinks as severity increases (more corner displacement → more
of the canvas has no real source pixel), from ~0.96 at mild down to ~0.87 at
severe, matching the visual growth of border regions seen in the Day 3
gallery.

**Visual confirmation, not just numbers**: rendered original / distorted
(severe) / rectified / abs-diff-in-valid-region side by side for the
Hispanic-Muslim minaret image (this project's worst-case MAE of the three
test images, 8.673 at severe). The rectified panel is visually
indistinguishable from the original — same vertical alignment, same content,
no residual skew — and the diff panel shows only a fine edge/texture pattern
(brick and lattice-work edges), not any gross misalignment blob. This
confirms the numeric result actually reflects correct geometric restoration
and is not, e.g., a low average that hides a badly-misaligned region.

**Verdict: the sanity check PASSES, both numerically and visually. This is a
working rectifier, not a bug. Day 5 is not blocked.**

### A pattern worth noting (not a bug): Hispanic-Muslim images show consistently higher MAE

Across every test above, and again in the 6-image Day 4 gallery below
(mild severity):

```
Baroque__casa_conde_gabia_granada2            MAE= 4.003
Hispanic-Muslim__medina_azahara24             MAE=10.321
Hispanic-Muslim__5bae6559d55ef                MAE= 7.476
Gothic__catedral_toledo2                      MAE= 2.939
Gothic__5e5a2e8dc057b                         MAE= 2.026
Gothic__5d13e2a71d048                         MAE= 3.657
```

Both Hispanic-Muslim images have noticeably higher round-trip MAE (7.5-10.3)
than every Gothic/Baroque image tested (2.0-4.0), consistently, across
severities. Hypothesis (not confirmed, flagged as a hunch): Hispanic-Muslim
facades in this sample feature fine, repetitive high-frequency texture
(lattice screens, arabesque brickwork) that is more sensitive to the
sub-pixel misalignment introduced by double bilinear interpolation than the
comparatively flatter/smoother stone surfaces in the Gothic/Baroque samples
here — fine periodic texture is exactly where resampling error is most
visible. This is still a small absolute error (well under the 15.0 bug
threshold) and the visual check above confirms no actual misalignment, so
this is noted as an interesting texture-dependent observation for later
discussion, not treated as a problem to fix.

## Border-handling policy — all 3 tested, one chosen with justification

Tested on 5 randomly sampled images (`black`, `mean_fill`, `crop`), medium
severity. Visual inspection confirmed all 3 behave as designed: `black`
leaves stark black wedges where the warp exposed no source pixel;
`mean_fill` replaces those same regions with the flat mean color of the valid
region (blends reasonably with sky/wall-toned backgrounds in these facade
photos); `crop` removes the border entirely by cropping to the largest
axis-aligned rectangle fully inside the valid region, at the cost of
shrinking the image (and, critically, shrinking a *different amount* per
image, since the amount of border depends on that image's own random corner
offsets).

**Trade-offs considered:**

| Policy | Removes black-artifact confound? | Preserves image content/framing? | Compute cost at full-grid (Day 5) scale |
|---|---|---|---|
| `black` | No — leaves a stark artifact absent from the clean condition | Yes, unchanged | Free (default warp output) |
| `crop` | Yes, completely | No — shrinks, by a severity- and image-dependent amount | Expensive (largest-inscribed-rectangle search per image) |
| `mean_fill` | Mostly — replaces black with a flat, much less salient patch | Yes, dimensions unchanged | Cheap (one masked mean + fill) |

**Chosen: `mean_fill`.**

Reasoning, in order of importance:

1. **Avoiding a hidden confound is the deciding factor.** This project's core
   discipline (CLAUDE.md rule 7) is "one variable at a time." `crop` looks
   like the most rigorous fix for the border artifact — it deletes the
   invalid pixels rather than covering them up — but it does so by changing
   each image's effective zoom/field-of-view by an amount that depends on
   that image's own random distortion draw, and that amount *systematically
   grows with severity* (more corner displacement → more border → more must
   be cropped away). That means, under a `crop` policy, "rectified" accuracy
   at severe would be measured on more tightly-cropped (more zoomed-in)
   images than at mild — a second variable riding along with rectification
   and correlated with severity, which is exactly the kind of confound this
   project is designed to avoid. `mean_fill` keeps every image's dimensions
   and field of view identical across all conditions and severities, so the
   only thing that differs between the distorted/rectified/clean conditions
   is the geometric correction itself.
2. **`black` is rejected because the artifact itself is a plausible
   confound.** A hard black rectangle is a strong, structured visual pattern
   with no analog anywhere in the clean condition or in real photography;
   CLIP could plausibly react to "does this image have a black wedge in it"
   as a signal in its own right, independent of the actual architectural
   content being newly available in the correct orientation. `mean_fill`
   still introduces a synthetic patch (real photos don't have flat color
   patches either), but a flat, low-contrast, background-toned patch is far
   less salient than a hard black-edged shape, and is a standard practical
   choice for exactly this reason in the warping/inpainting literature.
3. **Compute cost is a real, secondary consideration**, not the deciding one:
   `crop`'s largest-inscribed-rectangle search is a full
   maximal-rectangle-in-a-binary-matrix computation per image
   (`largest_rectangle_in_mask`, O(rows×cols) with a Python-level double
   loop) — fine for the 5-image comparison test here, but would add
   meaningful, avoidable runtime across Day 5's full grid (1,514 images × 3
   severities). `mean_fill` is a single masked mean plus a fill, negligible
   at any scale.

This decision is made once, here, and will not be revisited across the
remaining days per the plan's explicit "do not keep tuning the border policy
back and forth" instruction. `CHOSEN_BORDER_POLICY = "mean_fill"` is set as a
named constant in `src/rectification.py` for Day 5's runner to import rather
than re-deciding.

## Gallery (`figures/day4_rectification_gallery.png`)

Extends the Day 3 gallery: the same 6 images (same `seed=42`,
same `random.Random(42).sample(...)` selection over the same
`load_monumai` ordering — not a fresh random draw), each shown as original |
mild distorted | mild rectified (with the chosen `mean_fill` policy) | abs
pixel-difference in the valid region only (same metric as the sanity test
above, visualized as a heatmap, clipped to a fixed 0-30 range for
comparability across images). Visually, every rectified panel is very close
to its corresponding original — verticals restored, no visible residual
skew — and the diff panels show only fine edge/texture patterns, consistent
with the numeric MAE table above (including the same Hispanic-Muslim
texture-sensitivity pattern being visible in those two rows' diff panels
being visibly brighter/noisier than the Gothic/Baroque rows).

## Mechanism reused, not duplicated

`compute_valid_mask` and `mean_absolute_error` are written once in
`src/rectification.py` and used by both the standalone sanity test
(`tests/test_rectification.py`) and the gallery builder, so the "is this
pixel real or border" definition and the "how different are two images"
metric are identical everywhere they're used — not two subtly different
implementations that happen to agree on today's test images.

## Success criteria check

Met. `rectify_oracle(distort(I, H), H)` is pixel-close to `I` outside the
border region — verified numerically (MAE 2.0-10.3 across 9 test cases, all
comfortably under a 15.0 bug threshold) and visually (one full-resolution
comparison plus the 6-image gallery). This is a working rectifier, not a bug.
**Day 5 is not blocked and may proceed.**

## Open items carried into Day 5

- Day 5's `experiment_runner.py` should import `CHOSEN_BORDER_POLICY` from
  `src/rectification.py` rather than hardcoding `"mean_fill"` again, so the
  Day 4 decision has exactly one place it lives.
- The Hispanic-Muslim texture-sensitivity observation (higher round-trip MAE)
  is noted here for awareness; it does not block anything and does not need
  action, but may be worth a one-line mention in the paper's limitations if
  it turns out to correlate with any per-class pattern in Day 6's breakdown.
