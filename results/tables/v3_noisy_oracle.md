# Day 9 v3 — noisy oracle: recovery % vs. corner-correspondence error

The 4 dst-corner correspondences used to build the rectifying homography are perturbed by +/- pct% of image size before rebuilding H via cv2.getPerspectiveTransform; pct=0 is the exact oracle case (reused from v2, not recomputed). 3 replicates (matching v2's replicate 0/1/2), medium+severe only. 95% CI = 1000-resample bootstrap over images, seed=0.

| Severity | Perturbation % | Accuracy [CI] | Recovery % [CI] |
|---|---|---|---|
| medium | 0 | 0.611 [0.589, 0.634] | 76.6% [-65.4, 265.3] |
| medium | 1 | 0.605 [0.582, 0.629] | 37.5% [-145.7, 172.8] |
| medium | 2 | 0.607 [0.585, 0.630] | 48.4% [-96.1, 200.3] |
| medium | 4 | 0.609 [0.587, 0.631] | 62.5% [-95.2, 275.2] |
| severe | 0 | 0.609 [0.586, 0.631] | 82.7% [48.8, 135.9] |
| severe | 1 | 0.607 [0.585, 0.629] | 76.3% [40.8, 125.2] |
| severe | 2 | 0.606 [0.584, 0.628] | 73.4% [37.9, 120.6] |
| severe | 4 | 0.609 [0.587, 0.632] | 82.0% [46.5, 153.3] |
