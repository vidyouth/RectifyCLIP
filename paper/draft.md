# RectifyCLIP — paper draft

> Status: skeleton not yet built. Per the execution plan, the full 6-section
> structure (Abstract, Introduction, Related work, Method, Results, Discussion
> & limitations) is a Day 7 task. This file currently holds only the Day 3
> "Distortion protocol" subsection, written early because Day 3 explicitly
> calls for it. Do not treat the rest of the paper as scoped or started yet.

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
