# Day 10 v4 — attractor calibration diagnostic (NOT a method; CLEAN-only fit, applied post-hoc)

Per-class additive log-prob offsets fit ONLY on clean images (500 iterations, lr=0.5) so the predicted-class distribution on clean images approximately matches the true class distribution. Applied unchanged to distorted/rectified predictions. This is a diagnostic probe of whether per-class bias (e.g. the Renaissance attractor) is a simple additive effect — it is NOT proposed as a correction method for the main experiment, and fitting on clean-only means it cannot see or correct for any DISTORTION-specific shift in the bias.

## Fitted offsets and clean-image calibration check

| Class | True count (clean, n=1514) | Uncalibrated pred. count | Calibrated pred. count | Fitted offset |
|---|---|---|---|---|
| Hispanic-Muslim | 327 | 302 | 326 | -0.053 |
| Gothic | 359 | 294 | 360 | +0.152 |
| Renaissance | 312 | 699 | 312 | -0.716 |
| Baroque | 516 | 219 | 516 | +0.700 |

## Accuracy and recovery %, before vs. after calibration

| Theta (deg) | Condition | Uncalibrated acc. | Calibrated acc. |
|---|---|---|---|
| 5 | distorted | 0.584 | 0.682 |
| 5 | rectified | 0.606 | 0.698 |
| 10 | distorted | 0.579 | 0.684 |
| 10 | rectified | 0.594 | 0.687 |
| 15 | distorted | 0.570 | 0.666 |
| 15 | rectified | 0.582 | 0.679 |
| 25 | distorted | 0.538 | 0.621 |
| 25 | rectified | 0.529 | 0.623 |
| 35 | distorted | 0.489 | 0.560 |
| 35 | rectified | 0.485 | 0.549 |

Calibrated clean accuracy: 0.682 (same for every theta — clean has no theta dependence; shown once here for reference, and reused as the recovery-% denominator's clean term below). Compare to the UNcalibrated clean accuracy of 0.614 in `v4_accuracy.md` — calibration alone raises clean accuracy by ~6.8 points, confirming a real baseline classification bias independent of any distortion.

| Theta (deg) | Calibrated denominator (clean-distorted) | Calibrated recovery % | Reliable? |
|---|---|---|---|
| 5 | -0.0007 | -2400.0% | **NO — denominator near zero, ratio is noise, do not quote this number** |
| 10 | -0.0024 | -127.3% | **NO — denominator near zero, ratio is noise, do not quote this number** |
| 15 | +0.0154 | 82.9% | **NO — denominator near zero, ratio is noise, do not quote this number** |
| 25 | +0.0608 | 2.9% | yes |
| 35 | +0.1213 | -9.4% | yes |

**Reading this honestly**: at theta=5 and theta=10, calibrated distorted accuracy (0.682, 0.684) is nearly IDENTICAL to calibrated clean accuracy (0.682) — once the per-class bias is removed, mild/moderate tilt barely hurts accuracy at all, which is why the recovery-% denominator collapses toward zero and the ratio becomes meaningless (do not quote the -2400% / -127% figures as real effect sizes). This itself is a finding, not a computation artifact: a large share of what LOOKS like accuracy loss from mild/moderate distortion in the uncalibrated numbers is actually a fixed classification bias CLIP already has on clean images, not information lost to the distortion. At theta=25 and theta=35, calibrated distorted accuracy drops well below calibrated clean (0.621, 0.560 vs. 0.682) — a real, calibration-robust accuracy loss that the bias correction does NOT explain away, consistent with genuine geometric information loss dominating at high tilt angles.
