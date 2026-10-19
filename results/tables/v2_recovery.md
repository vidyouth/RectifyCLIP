# Day 8 v2 — recovery % by severity, with 95% bootstrap CI

recovery = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) x 100. CI is a 1000-resample bootstrap over images, seed=0 — recovery % is a ratio of two accuracy differences, so its CI can be very wide when the denominator is small; report the CI, not just the point estimate.

| Severity | Recovery % | 95% CI |
|---|---|---|
| mild | -17.2% | [-705.0%, 820.6%] |
| medium | 77.0% | [-28.1%, 217.2%] |
| severe | 95.7% | [67.2%, 145.8%] |
