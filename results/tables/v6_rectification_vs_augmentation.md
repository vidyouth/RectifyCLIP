# Day 12 v6 — Experiment C: rectification vs augmentation

Four combinations per (model, angle), same held-out images: clean-trained probe on distorted/rectified features, and augmentation-trained probe (clean + distorted pooled over all 3 angles, no rectified images in training) on distorted/rectified features.

| Model | Angle | Combination | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 5 | clean-trained probe -> distorted | 0.857 [0.839, 0.874] | 0.845 [0.827, 0.862] | 0.848 [0.830, 0.866] |
| CLIP ViT-B/32 | 5 | clean-trained probe -> rectified | 0.850 [0.833, 0.868] | 0.840 [0.822, 0.858] | 0.842 [0.823, 0.861] |
| CLIP ViT-B/32 | 5 | augmentation-trained probe -> distorted | 0.873 [0.857, 0.890] | 0.866 [0.849, 0.883] | 0.871 [0.853, 0.887] |
| CLIP ViT-B/32 | 5 | augmentation-trained probe -> rectified | 0.871 [0.853, 0.888] | 0.863 [0.846, 0.880] | 0.866 [0.849, 0.883] |
| CLIP ViT-B/32 | 15 | clean-trained probe -> distorted | 0.820 [0.801, 0.838] | 0.809 [0.790, 0.827] | 0.809 [0.788, 0.827] |
| CLIP ViT-B/32 | 15 | clean-trained probe -> rectified | 0.836 [0.817, 0.855] | 0.833 [0.815, 0.851] | 0.831 [0.813, 0.851] |
| CLIP ViT-B/32 | 15 | augmentation-trained probe -> distorted | 0.842 [0.824, 0.860] | 0.831 [0.812, 0.849] | 0.836 [0.816, 0.854] |
| CLIP ViT-B/32 | 15 | augmentation-trained probe -> rectified | 0.853 [0.836, 0.872] | 0.849 [0.831, 0.868] | 0.850 [0.832, 0.870] |
| CLIP ViT-B/32 | 35 | clean-trained probe -> distorted | 0.614 [0.589, 0.639] | 0.616 [0.594, 0.636] | 0.597 [0.571, 0.622] |
| CLIP ViT-B/32 | 35 | clean-trained probe -> rectified | 0.608 [0.583, 0.633] | 0.620 [0.599, 0.643] | 0.598 [0.572, 0.624] |
| CLIP ViT-B/32 | 35 | augmentation-trained probe -> distorted | 0.703 [0.679, 0.726] | 0.692 [0.667, 0.714] | 0.690 [0.665, 0.713] |
| CLIP ViT-B/32 | 35 | augmentation-trained probe -> rectified | 0.694 [0.667, 0.715] | 0.693 [0.668, 0.717] | 0.686 [0.660, 0.709] |
| CLIP ViT-L/14 | 5 | clean-trained probe -> distorted | 0.876 [0.860, 0.892] | 0.867 [0.851, 0.882] | 0.873 [0.856, 0.888] |
| CLIP ViT-L/14 | 5 | clean-trained probe -> rectified | 0.881 [0.865, 0.897] | 0.877 [0.861, 0.893] | 0.880 [0.865, 0.896] |
| CLIP ViT-L/14 | 5 | augmentation-trained probe -> distorted | 0.896 [0.881, 0.910] | 0.890 [0.875, 0.905] | 0.895 [0.880, 0.910] |
| CLIP ViT-L/14 | 5 | augmentation-trained probe -> rectified | 0.898 [0.883, 0.912] | 0.896 [0.881, 0.910] | 0.897 [0.883, 0.912] |
| CLIP ViT-L/14 | 15 | clean-trained probe -> distorted | 0.855 [0.839, 0.873] | 0.848 [0.832, 0.866] | 0.851 [0.833, 0.868] |
| CLIP ViT-L/14 | 15 | clean-trained probe -> rectified | 0.871 [0.855, 0.888] | 0.870 [0.853, 0.887] | 0.870 [0.854, 0.887] |
| CLIP ViT-L/14 | 15 | augmentation-trained probe -> distorted | 0.873 [0.857, 0.890] | 0.863 [0.847, 0.881] | 0.869 [0.853, 0.887] |
| CLIP ViT-L/14 | 15 | augmentation-trained probe -> rectified | 0.882 [0.867, 0.898] | 0.881 [0.866, 0.897] | 0.882 [0.867, 0.898] |
| CLIP ViT-L/14 | 35 | clean-trained probe -> distorted | 0.720 [0.697, 0.742] | 0.730 [0.710, 0.752] | 0.712 [0.688, 0.735] |
| CLIP ViT-L/14 | 35 | clean-trained probe -> rectified | 0.725 [0.703, 0.747] | 0.738 [0.716, 0.758] | 0.721 [0.698, 0.742] |
| CLIP ViT-L/14 | 35 | augmentation-trained probe -> distorted | 0.775 [0.754, 0.795] | 0.762 [0.741, 0.783] | 0.763 [0.740, 0.784] |
| CLIP ViT-L/14 | 35 | augmentation-trained probe -> rectified | 0.756 [0.732, 0.775] | 0.756 [0.733, 0.776] | 0.751 [0.729, 0.771] |
| SigLIP ViT-B/16 | 5 | clean-trained probe -> distorted | 0.875 [0.857, 0.891] | 0.870 [0.854, 0.887] | 0.873 [0.856, 0.889] |
| SigLIP ViT-B/16 | 5 | clean-trained probe -> rectified | 0.879 [0.862, 0.896] | 0.877 [0.860, 0.894] | 0.879 [0.862, 0.895] |
| SigLIP ViT-B/16 | 5 | augmentation-trained probe -> distorted | 0.884 [0.867, 0.900] | 0.879 [0.863, 0.895] | 0.883 [0.866, 0.898] |
| SigLIP ViT-B/16 | 5 | augmentation-trained probe -> rectified | 0.891 [0.876, 0.908] | 0.886 [0.871, 0.903] | 0.890 [0.875, 0.905] |
| SigLIP ViT-B/16 | 15 | clean-trained probe -> distorted | 0.853 [0.836, 0.871] | 0.852 [0.833, 0.869] | 0.851 [0.832, 0.868] |
| SigLIP ViT-B/16 | 15 | clean-trained probe -> rectified | 0.865 [0.849, 0.881] | 0.865 [0.848, 0.881] | 0.863 [0.847, 0.879] |
| SigLIP ViT-B/16 | 15 | augmentation-trained probe -> distorted | 0.867 [0.850, 0.883] | 0.859 [0.842, 0.876] | 0.863 [0.846, 0.880] |
| SigLIP ViT-B/16 | 15 | augmentation-trained probe -> rectified | 0.878 [0.861, 0.894] | 0.871 [0.855, 0.888] | 0.874 [0.857, 0.890] |
| SigLIP ViT-B/16 | 35 | clean-trained probe -> distorted | 0.672 [0.648, 0.695] | 0.683 [0.662, 0.705] | 0.664 [0.641, 0.686] |
| SigLIP ViT-B/16 | 35 | clean-trained probe -> rectified | 0.662 [0.638, 0.687] | 0.669 [0.647, 0.692] | 0.653 [0.629, 0.678] |
| SigLIP ViT-B/16 | 35 | augmentation-trained probe -> distorted | 0.742 [0.720, 0.765] | 0.727 [0.705, 0.750] | 0.725 [0.702, 0.749] |
| SigLIP ViT-B/16 | 35 | augmentation-trained probe -> rectified | 0.732 [0.709, 0.755] | 0.720 [0.697, 0.743] | 0.716 [0.692, 0.739] |

## Pairwise significance (paired bootstrap 95% CI on accuracy difference, same held-out images)

| Model | Angle | Comparison | Difference [95% CI] |
|---|---|---|---|
| CLIP ViT-B/32 | 5 | Rectification effect, clean-trained probe (rectified - distorted) | -0.007 [-0.020, +0.007] (not significant) |
| CLIP ViT-B/32 | 5 | Rectification effect, augmentation-trained probe (rectified - distorted) | -0.003 [-0.015, +0.009] (not significant) |
| CLIP ViT-B/32 | 5 | Augmentation effect on distorted (aug - clean-trained) | +0.017 [+0.007, +0.027] (significant) |
| CLIP ViT-B/32 | 5 | Augmentation effect on rectified (aug - clean-trained) | +0.020 [+0.011, +0.030] (significant) |
| CLIP ViT-B/32 | 15 | Rectification effect, clean-trained probe (rectified - distorted) | +0.015 [-0.001, +0.032] (not significant) |
| CLIP ViT-B/32 | 15 | Rectification effect, augmentation-trained probe (rectified - distorted) | +0.011 [-0.005, +0.028] (not significant) |
| CLIP ViT-B/32 | 15 | Augmentation effect on distorted (aug - clean-trained) | +0.022 [+0.010, +0.034] (significant) |
| CLIP ViT-B/32 | 15 | Augmentation effect on rectified (aug - clean-trained) | +0.018 [+0.007, +0.030] (significant) |
| CLIP ViT-B/32 | 35 | Rectification effect, clean-trained probe (rectified - distorted) | -0.006 [-0.026, +0.017] (not significant) |
| CLIP ViT-B/32 | 35 | Rectification effect, augmentation-trained probe (rectified - distorted) | -0.010 [-0.032, +0.014] (not significant) |
| CLIP ViT-B/32 | 35 | Augmentation effect on distorted (aug - clean-trained) | +0.090 [+0.067, +0.112] (significant) |
| CLIP ViT-B/32 | 35 | Augmentation effect on rectified (aug - clean-trained) | +0.086 [+0.067, +0.106] (significant) |
| CLIP ViT-L/14 | 5 | Rectification effect, clean-trained probe (rectified - distorted) | +0.005 [-0.005, +0.016] (not significant) |
| CLIP ViT-L/14 | 5 | Rectification effect, augmentation-trained probe (rectified - distorted) | +0.002 [-0.008, +0.012] (not significant) |
| CLIP ViT-L/14 | 5 | Augmentation effect on distorted (aug - clean-trained) | +0.020 [+0.011, +0.030] (significant) |
| CLIP ViT-L/14 | 5 | Augmentation effect on rectified (aug - clean-trained) | +0.017 [+0.006, +0.028] (significant) |
| CLIP ViT-L/14 | 15 | Rectification effect, clean-trained probe (rectified - distorted) | +0.016 [+0.005, +0.028] (significant) |
| CLIP ViT-L/14 | 15 | Rectification effect, augmentation-trained probe (rectified - distorted) | +0.009 [-0.003, +0.022] (not significant) |
| CLIP ViT-L/14 | 15 | Augmentation effect on distorted (aug - clean-trained) | +0.017 [+0.006, +0.028] (significant) |
| CLIP ViT-L/14 | 15 | Augmentation effect on rectified (aug - clean-trained) | +0.011 [+0.001, +0.021] (significant) |
| CLIP ViT-L/14 | 35 | Rectification effect, clean-trained probe (rectified - distorted) | +0.005 [-0.013, +0.022] (not significant) |
| CLIP ViT-L/14 | 35 | Rectification effect, augmentation-trained probe (rectified - distorted) | -0.019 [-0.038, +0.000] (not significant) |
| CLIP ViT-L/14 | 35 | Augmentation effect on distorted (aug - clean-trained) | +0.055 [+0.033, +0.075] (significant) |
| CLIP ViT-L/14 | 35 | Augmentation effect on rectified (aug - clean-trained) | +0.030 [+0.015, +0.046] (significant) |
| SigLIP ViT-B/16 | 5 | Rectification effect, clean-trained probe (rectified - distorted) | +0.005 [-0.007, +0.016] (not significant) |
| SigLIP ViT-B/16 | 5 | Rectification effect, augmentation-trained probe (rectified - distorted) | +0.007 [-0.002, +0.017] (not significant) |
| SigLIP ViT-B/16 | 5 | Augmentation effect on distorted (aug - clean-trained) | +0.009 [+0.001, +0.018] (significant) |
| SigLIP ViT-B/16 | 5 | Augmentation effect on rectified (aug - clean-trained) | +0.012 [+0.003, +0.021] (significant) |
| SigLIP ViT-B/16 | 15 | Rectification effect, clean-trained probe (rectified - distorted) | +0.013 [-0.001, +0.026] (not significant) |
| SigLIP ViT-B/16 | 15 | Rectification effect, augmentation-trained probe (rectified - distorted) | +0.011 [-0.001, +0.023] (not significant) |
| SigLIP ViT-B/16 | 15 | Augmentation effect on distorted (aug - clean-trained) | +0.014 [+0.003, +0.024] (significant) |
| SigLIP ViT-B/16 | 15 | Augmentation effect on rectified (aug - clean-trained) | +0.013 [+0.001, +0.022] (significant) |
| SigLIP ViT-B/16 | 35 | Rectification effect, clean-trained probe (rectified - distorted) | -0.010 [-0.029, +0.011] (not significant) |
| SigLIP ViT-B/16 | 35 | Rectification effect, augmentation-trained probe (rectified - distorted) | -0.010 [-0.030, +0.009] (not significant) |
| SigLIP ViT-B/16 | 35 | Augmentation effect on distorted (aug - clean-trained) | +0.070 [+0.052, +0.091] (significant) |
| SigLIP ViT-B/16 | 35 | Augmentation effect on rectified (aug - clean-trained) | +0.070 [+0.051, +0.089] (significant) |
