# RectifyCLIP — paper draft

> Status: skeleton not yet built. Per the execution plan, the full 6-section
> structure (Abstract, Introduction, Related work, Method, Results, Discussion
> & limitations) is a Day 7 task. This file currently holds only the Day 3
> "Distortion protocol" and Day 4 "Rectification protocol" subsections,
> written early because those days explicitly call for them. Do not treat the
> rest of the paper as scoped or started yet.

## Method

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
