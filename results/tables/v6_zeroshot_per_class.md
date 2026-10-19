# Day 12 v6 — Experiment A: per-class accuracy

| Model | Angle | Condition | Hispanic-Muslim | Gothic | Renaissance | Baroque |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | - | clean | 0.758 | 0.735 | 0.718 | 0.401 |
| CLIP ViT-B/32 | 5 | distorted | 0.758 | 0.719 | 0.798 | 0.331 |
| CLIP ViT-B/32 | 5 | rectified | 0.746 | 0.680 | 0.747 | 0.405 |
| CLIP ViT-B/32 | 15 | distorted | 0.679 | 0.741 | 0.744 | 0.393 |
| CLIP ViT-B/32 | 15 | rectified | 0.725 | 0.655 | 0.766 | 0.395 |
| CLIP ViT-B/32 | 35 | distorted | 0.520 | 0.655 | 0.449 | 0.430 |
| CLIP ViT-B/32 | 35 | rectified | 0.621 | 0.451 | 0.526 | 0.372 |
| CLIP ViT-L/14 | - | clean | 0.936 | 0.716 | 0.429 | 0.734 |
| CLIP ViT-L/14 | 5 | distorted | 0.948 | 0.705 | 0.494 | 0.636 |
| CLIP ViT-L/14 | 5 | rectified | 0.939 | 0.719 | 0.535 | 0.622 |
| CLIP ViT-L/14 | 15 | distorted | 0.933 | 0.738 | 0.471 | 0.638 |
| CLIP ViT-L/14 | 15 | rectified | 0.917 | 0.738 | 0.571 | 0.597 |
| CLIP ViT-L/14 | 35 | distorted | 0.780 | 0.772 | 0.353 | 0.483 |
| CLIP ViT-L/14 | 35 | rectified | 0.737 | 0.747 | 0.465 | 0.479 |
| SigLIP ViT-B/16 | - | clean | 0.893 | 0.760 | 0.545 | 0.672 |
| SigLIP ViT-B/16 | 5 | distorted | 0.902 | 0.772 | 0.574 | 0.678 |
| SigLIP ViT-B/16 | 5 | rectified | 0.884 | 0.763 | 0.548 | 0.643 |
| SigLIP ViT-B/16 | 15 | distorted | 0.896 | 0.774 | 0.545 | 0.628 |
| SigLIP ViT-B/16 | 15 | rectified | 0.872 | 0.760 | 0.532 | 0.641 |
| SigLIP ViT-B/16 | 35 | distorted | 0.862 | 0.652 | 0.420 | 0.331 |
| SigLIP ViT-B/16 | 35 | rectified | 0.807 | 0.543 | 0.481 | 0.326 |
