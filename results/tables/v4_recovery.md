# Day 10 v4 — recovery % vs. theta

recovery = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) x 100. 95% CI = 1000-resample bootstrap over images, seed=0. Denominator (clean_acc - distorted_acc) flagged 'near-zero' if |denom| < 0.02 — at that point the ratio is not a meaningful number, regardless of how precise it looks.

| Theta (deg) | Denominator (clean-distorted acc.) | Recovery % [CI] | Reliable? |
|---|---|---|---|
| 5 | 0.0299 | 73.5% [36.5, 121.4] | yes |
| 10 | 0.0357 | 42.6% [8.4, 81.3] | yes |
| 15 | 0.0443 | 27.9% [-0.7, 55.1] | yes |
| 25 | 0.0764 | -11.8% [-36.1, 7.6] | yes |
| 35 | 0.1248 | -3.9% [-20.3, 8.5] | yes |
