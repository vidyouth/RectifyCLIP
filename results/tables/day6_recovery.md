# Day 6 — recovery percentage per severity

recovery = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) x 100, computed from exact correct-image counts (out of n=1514 per severity).

| Severity | Clean correct | Distorted correct | Rectified correct | Denominator (clean-distorted) | Numerator (rectified-distorted) | Recovery % |
|---|---|---|---|---|---|---|
| mild | 930 | 920 | 911 | 10 | -9 | -90.0% |
| medium | 930 | 875 | 916 | 55 | 41 | 74.5% |
| severe | 930 | 839 | 929 | 91 | 90 | 98.9% |

## Caution on interpreting mild severity's recovery %

At mild severity the denominator (clean minus distorted correct-count) is only 10 images out of 1,514 — distortion barely hurts accuracy at this severity to begin with. Dividing the numerator by such a small denominator produces a recovery percentage (-90.0%) that looks large and precise but is extremely sensitive to a handful of individual image outcomes: flipping the prediction on just 1-2 images would swing this number by roughly 10-20 percentage points. Per Day 5's McNemar test (results/tables/day5_mcnemar.md), the rectified-vs-distorted gap at mild severity is not statistically distinguishable from chance (p=0.52) in the first place. **Reporting mild severity's recovery % as a precise figure in the paper would overstate the precision the data actually supports** — it should be presented as "not meaningful / statistically indistinguishable from zero effect" at mild severity, with medium and severe as the severities where the recovery-% metric is actually informative (both have a much larger, statistically significant denominator and effect).
