# Day 10 v4 — accuracy vs. valid-area fraction (confound check)

All distorted/rectified rows (pooled across theta and replicates) binned into 10 quantile bins of valid_area_fraction (the fraction of the rectified/distorted canvas covered by real, non-border-fill pixels). theta and valid-area-remaining are correlated (larger theta loses more area) but not identical — the random tilt-axis direction also affects how much area a given theta loses — so this asks whether accuracy tracks area loss specifically, pooled across theta.

## distorted (point-biserial correlation of valid_area_fraction with correctness: +0.069)

| Bin (low to high valid-area) | n | Mean valid-area fraction | Accuracy |
|---|---|---|---|
| 1 | 2271 | 0.246 | 0.462 |
| 2 | 2271 | 0.316 | 0.519 |
| 3 | 2271 | 0.421 | 0.539 |
| 4 | 2271 | 0.485 | 0.533 |
| 5 | 2271 | 0.614 | 0.583 |
| 6 | 2271 | 0.666 | 0.561 |
| 7 | 2271 | 0.728 | 0.579 |
| 8 | 2271 | 0.768 | 0.577 |
| 9 | 2271 | 0.854 | 0.606 |
| 10 | 2271 | 0.877 | 0.563 |

## rectified (point-biserial correlation of valid_area_fraction with correctness: +0.090)

| Bin (low to high valid-area) | n | Mean valid-area fraction | Accuracy |
|---|---|---|---|
| 1 | 2271 | 0.246 | 0.469 |
| 2 | 2271 | 0.316 | 0.506 |
| 3 | 2271 | 0.421 | 0.531 |
| 4 | 2271 | 0.485 | 0.520 |
| 5 | 2271 | 0.614 | 0.586 |
| 6 | 2271 | 0.666 | 0.584 |
| 7 | 2271 | 0.728 | 0.585 |
| 8 | 2271 | 0.768 | 0.598 |
| 9 | 2271 | 0.854 | 0.619 |
| 10 | 2271 | 0.877 | 0.594 |

