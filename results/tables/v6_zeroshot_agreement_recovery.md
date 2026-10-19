# Day 12 v6 — Experiment A: agreement with clean + recovery %

Recovery % = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) * 100.
Flagged rows have |clean_acc - distorted_acc| < 0.02 — the ratio is unstable/not meaningful; read the raw accuracies instead.

| Model | Angle | Condition | Agreement with clean |
|---|---|---|---|
| CLIP ViT-B/32 | 5 | distorted | 0.834 |
| CLIP ViT-B/32 | 5 | rectified | 0.884 |
| CLIP ViT-B/32 | 15 | distorted | 0.780 |
| CLIP ViT-B/32 | 15 | rectified | 0.806 |
| CLIP ViT-B/32 | 35 | distorted | 0.583 |
| CLIP ViT-B/32 | 35 | rectified | 0.592 |
| CLIP ViT-L/14 | 5 | distorted | 0.854 |
| CLIP ViT-L/14 | 5 | rectified | 0.889 |
| CLIP ViT-L/14 | 15 | distorted | 0.806 |
| CLIP ViT-L/14 | 15 | rectified | 0.836 |
| CLIP ViT-L/14 | 35 | distorted | 0.636 |
| CLIP ViT-L/14 | 35 | rectified | 0.645 |
| SigLIP ViT-B/16 | 5 | distorted | 0.855 |
| SigLIP ViT-B/16 | 5 | rectified | 0.894 |
| SigLIP ViT-B/16 | 15 | distorted | 0.797 |
| SigLIP ViT-B/16 | 15 | rectified | 0.829 |
| SigLIP ViT-B/16 | 35 | distorted | 0.567 |
| SigLIP ViT-B/16 | 35 | rectified | 0.576 |

| Model | Angle | Recovery % | Denominator (clean-distorted) | Flagged |
|---|---|---|---|---|
| CLIP ViT-B/32 | 5 | 23.5 [-450.0, 478.3] | 0.011 | YES |
| CLIP ViT-B/32 | 15 | -40.0 [-800.0, 486.7] | 0.013 | YES |
| CLIP ViT-B/32 | 35 | -26.1 [-55.3, -3.3] | 0.116 | no |
| CLIP ViT-L/14 | 5 | 25.8 [-125.0, 104.8] | 0.020 | no |
| CLIP ViT-L/14 | 15 | 16.7 [-186.7, 138.0] | 0.020 | YES |
| CLIP ViT-L/14 | 35 | 5.4 [-13.9, 21.6] | 0.122 | no |
| SigLIP ViT-B/16 | 5 | 184.2 [-557.5, 886.0] | -0.013 | YES |
| SigLIP ViT-B/16 | 15 | -58.8 [-963.8, 728.2] | 0.011 | YES |
| SigLIP ViT-B/16 | 35 | -15.5 [-30.3, -3.1] | 0.174 | no |
