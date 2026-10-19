# Day 8 v2 — accuracy, balanced accuracy, macro-F1 by severity and condition

Accuracy mean/std is computed over the 5 independent-tilt replicates (1 'replicate' for clean, which is distortion-free); the 95% CI is a bootstrap CI (1000 resamples, resampled at the image level, seed=0) on the pooled-replicate accuracy.

| Severity | Condition | Accuracy (mean +/- std over replicates) | Pooled accuracy [95% CI] | Balanced acc. | Macro-F1 |
|---|---|---|---|---|---|
| mild | clean | 0.614 +/- 0.000 (n_rep=1) | 0.614 [0.590, 0.638] | 0.653 | 0.638 |
| mild | distorted | 0.610 +/- 0.007 (n_rep=5) | 0.610 [0.589, 0.632] | 0.654 | 0.633 |
| mild | rectified | 0.610 +/- 0.003 (n_rep=5) | 0.610 [0.586, 0.633] | 0.650 | 0.633 |
| medium | clean | 0.614 +/- 0.000 (n_rep=1) | 0.614 [0.590, 0.638] | 0.653 | 0.638 |
| medium | distorted | 0.599 +/- 0.006 (n_rep=5) | 0.599 [0.578, 0.619] | 0.646 | 0.620 |
| medium | rectified | 0.611 +/- 0.002 (n_rep=5) | 0.611 [0.588, 0.633] | 0.650 | 0.635 |
| severe | clean | 0.614 +/- 0.000 (n_rep=1) | 0.614 [0.590, 0.638] | 0.653 | 0.638 |
| severe | distorted | 0.580 +/- 0.007 (n_rep=5) | 0.580 [0.559, 0.601] | 0.626 | 0.604 |
| severe | rectified | 0.613 +/- 0.005 (n_rep=5) | 0.613 [0.591, 0.634] | 0.652 | 0.636 |
