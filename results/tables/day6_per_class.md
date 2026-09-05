# Day 6 — per-class breakdown: does rectification help all 4 styles equally?

## Per-class accuracy by condition, pooled across all 3 severities

| Style | Clean | Distorted | Rectified | Recovery % (pooled) |
|---|---|---|---|---|
| Hispanic-Muslim | 0.783 | 0.735 | 0.783 | 100.0% |
| Gothic | 0.705 | 0.670 | 0.653 | -51.4% |
| Renaissance | 0.785 | 0.845 | 0.801 | 73.2% |
| Baroque | 0.341 | 0.258 | 0.346 | 105.5% |

## Per-class accuracy by condition, broken down per severity

| Style | Severity | Clean | Distorted | Rectified |
|---|---|---|---|---|
| Hispanic-Muslim | mild | 0.783 | 0.786 | 0.777 |
| Hispanic-Muslim | medium | 0.783 | 0.758 | 0.786 |
| Hispanic-Muslim | severe | 0.783 | 0.661 | 0.786 |
| Gothic | mild | 0.705 | 0.691 | 0.680 |
| Gothic | medium | 0.705 | 0.660 | 0.641 |
| Gothic | severe | 0.705 | 0.660 | 0.638 |
| Renaissance | mild | 0.785 | 0.827 | 0.798 |
| Renaissance | medium | 0.785 | 0.840 | 0.814 |
| Renaissance | severe | 0.785 | 0.869 | 0.792 |
| Baroque | mild | 0.341 | 0.304 | 0.318 |
| Baroque | medium | 0.341 | 0.248 | 0.339 |
| Baroque | severe | 0.341 | 0.223 | 0.380 |
