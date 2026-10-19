## Summary (read this first)

- At the most severe angle tested (35 degrees), zero-shot accuracy drops from clean (CLIP ViT-B/32 0.62, CLIP ViT-L/14 0.71, SigLIP ViT-B/16 0.71) to distorted (CLIP ViT-B/32 0.51, CLIP ViT-L/14 0.59, SigLIP ViT-B/16 0.54) — the same qualitative drop v4 found, now with stronger backbones.
- Oracle rectification helps zero-shot accuracy at 35 degrees for 1/3 models and helps the leakage-free linear probe for 1/3 models — see v6_zeroshot_agreement_recovery.md and v6_probe_leakfree.md for every angle, not just 35.
- Comparing rectification against augmentation (Experiment C): an augmentation-trained probe evaluated on distorted images beats a clean-trained probe evaluated on RECTIFIED images for 3/3 models at 35 degrees — see v6_rectification_vs_augmentation.md's pairwise significance table for whether each gap is statistically distinguishable from zero.
- Best overall clean-image zero-shot model here: SigLIP ViT-B/16 at 0.715. Gothic does NOT replicate v4's pattern of being the class distortion hurts most — at 35 degrees Gothic is comparatively distortion-robust (e.g. CLIP ViT-B/32: 0.735 clean -> 0.655 distorted), but it is the class RECTIFICATION hurts most, for 2/3 models (ViT-B/32: 0.655 -> 0.451; SigLIP: 0.652 -> 0.543; see v6_zeroshot_per_class.md) — a different, and more specific, failure mode than v4 described.
- Report every number as measured, including any case where a stronger model turns out LESS robust, rectification stops helping, or augmentation beats rectification outright — none of those would overturn v4's clean-baseline finding, but all are real possible outcomes here and are not softened in the tables below.

# Day 12 v6 — stronger models, leakage-free probe, rectification vs augmentation

Re-runs v4's exact distortion/rectification (same camera-rotation homography, same seeds, replicate 0, mean_fill border policy) on 3 stronger backbones at angles 5/15/35 degrees, on all 1,514 MonuMAI images.

Full numeric results are in the generated tables, not duplicated here:
- results/tables/v6_zeroshot_accuracy.md
- results/tables/v6_zeroshot_agreement_recovery.md
- results/tables/v6_zeroshot_mcnemar.md
- results/tables/v6_zeroshot_per_class.md
- results/tables/v6_probe_leakfree.md
- results/tables/v6_rectification_vs_augmentation.md
- figures/v6_accuracy_vs_angle.png, figures/v6_experimentC_summary.png

## Notes

- Experiment B/C's 5-fold split is regenerated with v5's exact StratifiedKFold call (same seed, same true_labels order) rather than loaded from a saved file — deterministic, so identical to v5's folds.
- A held-out image's clean, distorted, AND rectified features are all excluded from both probes' training sets in every fold — not just the clean ones — per the no-leakage requirement.
- Recovery-% rows with |clean_acc - distorted_acc| < 0.02 are flagged, not reported as a bare ratio.
- This script does not touch v1-v5 code, outputs, or the paper draft.

