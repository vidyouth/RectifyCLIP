## Summary (read this first)

- Experiment A (feature invariance): at 35 degrees, CLIP/SigLIP image features are still highly similar to their clean counterpart (cosine sim CLIP ViT-B/32 0.66, CLIP ViT-L/14 0.72, SigLIP ViT-B/16 0.76) — the embedding space is far more tilt-robust than zero-shot accuracy alone suggested; rectification's effect on similarity is small either way (see v7_feature_invariance.md's 'share rectified closer' column, rarely far from 0.5).
- Experiment B (tilt probe): a simple logistic probe CAN tell angle apart from frozen features (CLIP ViT-B/32 0.94, CLIP ViT-L/14 0.95, SigLIP ViT-B/16 0.93 4-class accuracy, chance = 0.25) — distortion leaves a detectable signature in feature space even when it barely changes raw accuracy.
- Task 2's rectified-vs-clean confusability test shows whether rectification actually erases that signature or just moves it — see v7_tilt_probe.md: if the rectified rate called 'distorted' stays far from the clean rate, rectification is not restoring clean-like features even where it helps raw accuracy.
- Experiment C's classical blind tilt estimator was pre-registered to be validated on synthetic data BEFORE running on real photos, and FAILED its own pre-registered gate (pitch-dominant correlation 0.35, MAE 30.0 degrees, needed <20) — per the pre-registered instruction, steps 3-5 (real-photo tilt distribution, accuracy-by-tilt-bin, blind rectification) were NOT run. This is reported as a real null result, not softened.
- Honest bottom line: a from-scratch classical vanishing-point tilt estimator on MonuMAI facade photos (decorative facades, occlusion, non-building lines) is not reliable enough to trust on individual images without further engineering (better line filtering, multi-frame or depth cues, or a learned estimator) — this is itself a legitimate finding about the limits of the oracle-rectification framing for real photos.

# Day 13 v7 — feature invariance, tilt probe, and a classical blind tilt estimator

## Assumptions and limitations (pre-registered, stated up front)

- Blind tilt estimator assumes a PURE CAMERA PITCH (rotation about the camera's local x-axis only); real natural tilt can include yaw/roll/oblique components this model cannot represent.
- Intrinsics assumed, not calibrated: focal length = image width, principal point at the exact image center (same assumption v4 used for the synthetic grid).
- Vertical-line detection (LSD, near-vertical filter, length >= 3% of image diagonal) will pick up non-building vertical edges (poles, people, decorative elements) on a subset of images — no semantic filtering was attempted.
- The vanishing-point-to-angle conversion has an inherent 180-degree sign ambiguity for an undirected line; resolved by assuming the true tilt is in (-90, 90] degrees, verified on synthetic ground truth.
- Validation's 'signed' comparison is only meaningful on the pitch-dominant subset (v4's random tilt axis within 20 degrees of pure pitch); non-pitch-dominant (yaw/oblique) synthetic tilts are a genuine model mismatch, reported unsigned only, and expected to validate worse.
- Because the validation gate failed, Experiment C stopped after step 2: no real-photo tilt distribution, no accuracy-by-tilt-bin analysis, and no blind-rectification comparison were produced.
- This script does not touch v1-v6 code, outputs, or the paper draft.

## Outputs

- results/tables/v7_feature_invariance.md, figures/v7_feature_invariance_vs_angle.png (Experiment A)
- results/tables/v7_tilt_probe.md (Experiment B)
- results/tables/v7_validation_synthetic.md, figures/v7_validation_scatter.png (Experiment C, validation only)

