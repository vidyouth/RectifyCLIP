# Day 13 v7 — Experiment A: feature invariance (cosine similarity to clean)

Cosine similarity between each image's CLEAN feature and its distorted/rectified feature (v6 cached features, already L2-normalized, so cosine similarity = dot product). 'Share rectified closer' = fraction of images where rectified is more similar to clean than distorted is.

| Model | Angle | Mean sim: distorted | Mean sim: rectified | Diff (rectified - distorted) | Share rectified closer |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 5 | 0.927 [0.926, 0.929] | 0.958 [0.957, 0.959] | 0.031 [0.029, 0.032] | 0.906 |
| CLIP ViT-B/32 | 15 | 0.865 [0.863, 0.867] | 0.902 [0.900, 0.904] | 0.037 [0.035, 0.039] | 0.859 |
| CLIP ViT-B/32 | 35 | 0.662 [0.657, 0.668] | 0.667 [0.661, 0.673] | 0.005 [0.002, 0.009] | 0.551 |
| CLIP ViT-L/14 | 5 | 0.945 [0.944, 0.946] | 0.960 [0.959, 0.960] | 0.014 [0.013, 0.015] | 0.822 |
| CLIP ViT-L/14 | 15 | 0.890 [0.889, 0.892] | 0.918 [0.916, 0.919] | 0.027 [0.026, 0.028] | 0.888 |
| CLIP ViT-L/14 | 35 | 0.722 [0.717, 0.727] | 0.736 [0.730, 0.741] | 0.014 [0.012, 0.016] | 0.641 |
| SigLIP ViT-B/16 | 5 | 0.954 [0.953, 0.955] | 0.969 [0.968, 0.969] | 0.015 [0.014, 0.015] | 0.882 |
| SigLIP ViT-B/16 | 15 | 0.904 [0.903, 0.906] | 0.930 [0.928, 0.931] | 0.026 [0.024, 0.027] | 0.882 |
| SigLIP ViT-B/16 | 35 | 0.757 [0.753, 0.760] | 0.773 [0.768, 0.777] | 0.016 [0.014, 0.018] | 0.670 |
