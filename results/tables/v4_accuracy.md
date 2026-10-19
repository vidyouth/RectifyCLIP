# Day 10 v4 — accuracy, balanced accuracy, agreement-with-clean vs. theta

95% CI = 1000-resample bootstrap over images, seed=0.

| Theta (deg) | Condition | Accuracy [CI] | Balanced acc. | Agreement with clean |
|---|---|---|---|---|
| 0 (clean) | clean | 0.614 [0.590, 0.638] | 0.653 | n/a |
| 5 | distorted | 0.584 [0.563, 0.607] | 0.636 | 0.826 |
| 5 | rectified | 0.606 [0.584, 0.629] | 0.646 | 0.884 |
| 10 | distorted | 0.579 [0.556, 0.602] | 0.626 | 0.818 |
| 10 | rectified | 0.594 [0.572, 0.617] | 0.636 | 0.853 |
| 15 | distorted | 0.570 [0.548, 0.591] | 0.616 | 0.795 |
| 15 | rectified | 0.582 [0.559, 0.604] | 0.625 | 0.818 |
| 25 | distorted | 0.538 [0.518, 0.558] | 0.577 | 0.723 |
| 25 | rectified | 0.529 [0.508, 0.549] | 0.565 | 0.729 |
| 35 | distorted | 0.489 [0.471, 0.509] | 0.515 | 0.616 |
| 35 | rectified | 0.485 [0.466, 0.504] | 0.508 | 0.615 |
