# Day 13 v7 — Experiment B: tilt probe

## Task 1: 4-class angle classifier (clean / 5 / 15 / 35 degrees), same 5-fold split as v5/v6

| Model | Accuracy (4-class) |
|---|---|
| CLIP ViT-B/32 | 0.945 [0.939, 0.950] |
| CLIP ViT-L/14 | 0.948 [0.942, 0.954] |
| SigLIP ViT-B/16 | 0.933 [0.927, 0.940] |

## Task 2: clean-vs-distorted(35 deg) classifier, applied to rectified(35 deg) features

'Rate called distorted' on held-out rectified features: an ideal rectifier makes this match the held-out CLEAN rate (i.e. the classifier can no longer tell rectified apart from clean).

| Model | Held-out accuracy (clean vs distorted) | Clean rate called 'distorted' | Rectified rate called 'distorted' |
|---|---|---|---|
| CLIP ViT-B/32 | 0.999 [0.998, 1.000] | 0.001 [0.000, 0.003] | 0.995 [0.991, 0.998] |
| CLIP ViT-L/14 | 1.000 [0.999, 1.000] | 0.000 [0.000, 0.000] | 0.993 [0.989, 0.997] |
| SigLIP ViT-B/16 | 0.998 [0.996, 0.999] | 0.003 [0.001, 0.007] | 0.992 [0.987, 0.996] |
