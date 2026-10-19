## Summary (read this first)

- 4/4 models completed on CPU; 0 skipped on time budget, 0 errored.
- Best clean-image result: CLIP ViT-L/14's linear probe at 0.878 accuracy (frozen features, 5-fold CV) — see the full table for every model/setup.
- Recommendation for the next distortion experiment: CLIP ViT-L/14 and SigLIP ViT-B/16, plus the linear probe as a non-zero-shot reference point (ranked by clean linear-probe accuracy, a conservative proxy for which backbones' features carry the most style signal).
- Report the direction honestly either way: if the gap between models/setups is small, that itself is the finding (CLIP zero-shot is not far from the probe ceiling, or vice versa) — do not oversell a small gap as a strong justification.
- Time estimates for the reduced distortion grid (angles 5/15/35, 1 replicate, 1,514 images):
  - CLIP ViT-L/14: ~2.6 hours (0.870s/image measured on clean-image encode)
  - SigLIP ViT-B/16: ~0.6 hours (0.200s/image measured on clean-image encode)
  - CLIP ViT-B/32: ~0.2 hours (0.063s/image measured on clean-image encode)

# Day 11 v5 — clean-image model/prompt/calibration/probe comparison

Dataset: 1514 clean MonuMAI images (no distortion/rectification in this script).
Git commit at run time: a487d12a685fcc022118ad819f7da6db2a5ce1f9

## Models & tags checked

Verified against this environment's `open_clip.list_pretrained()` before running:

- `ViT-B-32` / `laion2b_s34b_b79k`: exists
- `ViT-B-16` / `laion2b_s34b_b88k`: exists
- `ViT-L-14` / `laion2b_s32b_b82k`: exists
- `ViT-B-16-SigLIP` / `webli`: exists

## Prompt templates

Single prompt (matches configs/default.yaml): `a photo of {} architecture`

8-template ensemble (fixed before any model was run; never adjusted by per-class accuracy):

- `a photo of {} architecture`
- `a photo of a {} building`
- `an image of {} style architecture`
- `a photograph of a building in the {} architectural style`
- `{} architecture`
- `a building in the {} style`
- `a facade in {} architectural style`
- `this building is an example of {} architecture`

## Per-model results / status

### CLIP ViT-B/32 (`ViT-B-32` / `laion2b_s34b_b79k`) — done

Measured encode speed: 0.063 s/image.

| Setup | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Zero-shot, single prompt | 0.614 [0.589, 0.638] | 0.653 [0.631, 0.676] | 0.638 [0.613, 0.661] |
| Zero-shot, 8-prompt ensemble | 0.623 [0.597, 0.645] | 0.653 [0.632, 0.674] | 0.644 [0.621, 0.665] |
| Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.684 [0.660, 0.707] | 0.665 [0.643, 0.686] | 0.659 [0.634, 0.681] |
| Linear probe (5-fold CV, frozen features) | 0.861 [0.844, 0.878] | 0.850 [0.832, 0.866] | 0.854 [0.836, 0.870] |

### CLIP ViT-B/16 (`ViT-B-16` / `laion2b_s34b_b88k`) — done

Measured encode speed: 0.195 s/image.

| Setup | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Zero-shot, single prompt | 0.694 [0.671, 0.719] | 0.684 [0.662, 0.706] | 0.682 [0.659, 0.705] |
| Zero-shot, 8-prompt ensemble | 0.693 [0.671, 0.716] | 0.688 [0.667, 0.710] | 0.680 [0.658, 0.703] |
| Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.690 [0.668, 0.715] | 0.672 [0.652, 0.693] | 0.667 [0.644, 0.689] |
| Linear probe (5-fold CV, frozen features) | 0.864 [0.847, 0.880] | 0.854 [0.837, 0.871] | 0.858 [0.841, 0.876] |

### CLIP ViT-L/14 (`ViT-L-14` / `laion2b_s32b_b82k`) — done

Measured encode speed: 0.870 s/image.

| Setup | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Zero-shot, single prompt | 0.674 [0.651, 0.697] | 0.653 [0.633, 0.673] | 0.639 [0.615, 0.662] |
| Zero-shot, 8-prompt ensemble | 0.711 [0.688, 0.732] | 0.704 [0.682, 0.724] | 0.697 [0.673, 0.718] |
| Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.754 [0.733, 0.775] | 0.739 [0.719, 0.759] | 0.739 [0.718, 0.759] |
| Linear probe (5-fold CV, frozen features) | 0.878 [0.863, 0.894] | 0.869 [0.854, 0.886] | 0.875 [0.860, 0.892] |

### SigLIP ViT-B/16 (`ViT-B-16-SigLIP` / `webli`) — done

Measured encode speed: 0.200 s/image.

| Setup | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| Zero-shot, single prompt | 0.694 [0.670, 0.716] | 0.687 [0.663, 0.710] | 0.697 [0.674, 0.717] |
| Zero-shot, 8-prompt ensemble | 0.715 [0.693, 0.738] | 0.718 [0.696, 0.740] | 0.716 [0.695, 0.738] |
| Ensemble + calibration (DIAGNOSTIC, uses labels) | 0.742 [0.721, 0.763] | 0.717 [0.697, 0.735] | 0.713 [0.691, 0.733] |
| Linear probe (5-fold CV, frozen features) | 0.875 [0.859, 0.894] | 0.868 [0.852, 0.885] | 0.873 [0.857, 0.890] |

## Notes

- The 'ensemble + calibration' row fits a per-class logit bias using true labels (2-fold cross-fit, so every image's prediction is from a held-out fit) — it is a diagnostic, not a zero-shot number, and is labelled as such everywhere it's reported.
- Conservative CPU time budget: 90 minutes/model, estimated from a 32-image calibration subset before committing to the full encode.
- This script does not touch v1-v4 code, outputs, or the paper draft.

