# Day 13 v7 — Experiment C, step 2: synthetic-data validation of the blind tilt estimator

Pre-registered gate (fixed in src/experiment_v7.py before this ran): PASS requires pitch-dominant signed correlation >= 0.3 AND pitch-dominant MAE < 20 degrees. Result: **FAILED** — steps 3-5 were SKIPPED per the pre-registered instruction to stop after a failed validation.

| Metric | Value |
|---|---|
| Coverage (estimator found >= 4 inlier vertical lines) | 0.940 (4542 image-angle pairs total) |
| Pitch-dominant subset size (axis within 20.0 deg of pure pitch) | 902 |
| Pitch-dominant signed correlation (estimated vs true tilt) | 0.346 |
| Pitch-dominant MAE (degrees) | 30.0 |
| Overall unsigned correlation (all axes, \|estimate\| vs true angle) | 0.138 |
| Overall unsigned MAE (degrees) | 26.8 |

## Per-angle breakdown (pitch-dominant subset only)

| True angle | n (pitch-dominant, ok) | Mean estimated \|tilt\| | MAE (signed) |
|---|---|---|---|
| 5 | 330 | 33.2 | 32.4 |
| 15 | 329 | 38.1 | 32.9 |
| 35 | 243 | 40.0 | 23.0 |
