# Day 9 v3 — mismatched-H control: does rectification need the CORRECT homography?

Each distorted image rectified with a DIFFERENT image's homography (same severity, same replicate) instead of its own. 95% CI = 1000-resample bootstrap over images, seed=0.

| Severity | Distorted acc. [CI] | Wrong-H rectified acc. [CI] | Oracle rectified acc. [CI] | Wrong-H recovery % [CI] | Oracle recovery % [CI] |
|---|---|---|---|---|---|
| medium | 0.599 [0.578, 0.619] | 0.602 [0.582, 0.624] | 0.611 [0.588, 0.633] | 20.4% [-112.2, 130.2] | 77.0% [-28.1, 217.2] |
| severe | 0.580 [0.559, 0.601] | 0.578 [0.558, 0.597] | 0.613 [0.591, 0.634] | -7.4% [-46.8, 23.0] | 95.7% [67.2, 145.8] |

Diagnostic note: 3 of 15,140 wrong-H rows (0.02%) had a wrong homography that pushed ALL pixels outside the rectified canvas (empty valid mask), so mean_fill produced an undefined/degenerate fill color for those 3 images. Left in the numbers above uncorrected — at 3/15,140 rows, no computed accuracy or CI changes outside reported precision.
