# Day 11 v5 — clean-image model/prompt/calibration/probe comparison

All numbers on the same 1,514 clean MonuMAI images, no distortion. CIs are 95% bootstrap intervals (1000 resamples, image-level).

| Model | Setup | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|---|
| CLIP ViT-B/32 | Zero-shot, single prompt | 0.614 [0.589, 0.638] | 0.653 [0.631, 0.676] | 0.638 [0.613, 0.661] |
| CLIP ViT-B/32 | Zero-shot, 8-prompt ensemble | 0.623 [0.597, 0.645] | 0.653 [0.632, 0.674] | 0.644 [0.621, 0.665] |
| CLIP ViT-B/32 | Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.684 [0.660, 0.707] | 0.665 [0.643, 0.686] | 0.659 [0.634, 0.681] |
| CLIP ViT-B/32 | Linear probe (5-fold CV, frozen features) | 0.861 [0.844, 0.878] | 0.850 [0.832, 0.866] | 0.854 [0.836, 0.870] |
| CLIP ViT-B/16 | Zero-shot, single prompt | 0.694 [0.671, 0.719] | 0.684 [0.662, 0.706] | 0.682 [0.659, 0.705] |
| CLIP ViT-B/16 | Zero-shot, 8-prompt ensemble | 0.693 [0.671, 0.716] | 0.688 [0.667, 0.710] | 0.680 [0.658, 0.703] |
| CLIP ViT-B/16 | Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.690 [0.668, 0.715] | 0.672 [0.652, 0.693] | 0.667 [0.644, 0.689] |
| CLIP ViT-B/16 | Linear probe (5-fold CV, frozen features) | 0.864 [0.847, 0.880] | 0.854 [0.837, 0.871] | 0.858 [0.841, 0.876] |
| CLIP ViT-L/14 | Zero-shot, single prompt | 0.674 [0.651, 0.697] | 0.653 [0.633, 0.673] | 0.639 [0.615, 0.662] |
| CLIP ViT-L/14 | Zero-shot, 8-prompt ensemble | 0.711 [0.688, 0.732] | 0.704 [0.682, 0.724] | 0.697 [0.673, 0.718] |
| CLIP ViT-L/14 | Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.754 [0.733, 0.775] | 0.739 [0.719, 0.759] | 0.739 [0.718, 0.759] |
| CLIP ViT-L/14 | Linear probe (5-fold CV, frozen features) | 0.878 [0.863, 0.894] | 0.869 [0.854, 0.886] | 0.875 [0.860, 0.892] |
| SigLIP ViT-B/16 | Zero-shot, single prompt | 0.694 [0.670, 0.716] | 0.687 [0.663, 0.710] | 0.697 [0.674, 0.717] |
| SigLIP ViT-B/16 | Zero-shot, 8-prompt ensemble | 0.715 [0.693, 0.738] | 0.718 [0.696, 0.740] | 0.716 [0.695, 0.738] |
| SigLIP ViT-B/16 | Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.742 [0.721, 0.763] | 0.717 [0.697, 0.735] | 0.713 [0.691, 0.733] |
| SigLIP ViT-B/16 | Linear probe (5-fold CV, frozen features) | 0.875 [0.859, 0.894] | 0.868 [0.852, 0.885] | 0.873 [0.857, 0.890] |

## Seconds per image (full-dataset encode, 1,514 images)

| Model | seconds/image |
|---|---|
| CLIP ViT-B/32 | 0.063 |
| CLIP ViT-B/16 | 0.195 |
| CLIP ViT-L/14 | 0.870 |
| SigLIP ViT-B/16 | 0.200 |
