"""Day 12 v6 — feature/zero-shot extraction for the stronger-model distortion re-run.

Re-runs v4's distortion/rectification study (same camera-rotation homography,
same seeds, same mean_fill border policy) with 3 stronger backbones, and
extracts+caches frozen image features for a later leakage-free linear-probe
analysis (src/day12_v6_analysis.py does Experiments B and C; this file does
feature extraction plus Experiment A, the zero-shot comparison).

Continuity with v4 is load-bearing, not decorative: sample_axis_angle() and
build_rotation_homography() are imported UNCHANGED from src/experiment_v4.py
(same SEED_NAMESPACE, same base seed from configs/default.yaml, replicate
fixed at 0), so for a given image and angle this produces the exact same
homography H v4's CSV used — the only thing that changes between v4 and v6
is which model looks at the resulting pixels. v1-v5 code and outputs are
never modified or re-run here; only imported.

Per model, per angle, this is checkpointed (results/raw/v6_progress.json)
after every (model, angle) block, so an interrupted run resumes instead of
recomputing. A clean-feature block is also checkpointed per model (angle-
independent, computed once).

Before committing to the full 1,514-image run, main() runs an internal
20-image smoke test end-to-end (every model, every angle, both zero-shot and
feature caching) using isolated `_smoketest20`-suffixed output paths, prints
a full report (including a projected full-dataset wall-clock time), and only
proceeds automatically to the real run if the smoke test raises no
exceptions — per the instruction to time-estimate and show a smoke test
first, not to wait for a human to approve it.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from clip_zero_shot import encode_image_arrays, load_config
from data_loader import load_monumai
from distortion import image_id_from_path, load_image_rgb
from experiment_v4 import build_rotation_homography, git_commit_hash, sample_axis_angle
from experiment_v5_models import (
    MODELS as V5_MODELS,
    PROMPT_ENSEMBLE_TEMPLATES,
    encode_text_features_ensemble,
    load_model_generic,
    predict_labels,
)
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

ANGLES = [5, 15, 35]
REPLICATE = 0  # v4's replicate 0 only, per the instruction ("1 replicate")

# The 3 models named in the instruction, in the order given. Reused from
# v5's MODELS list (same name/pretrained/slug/label) rather than re-declared,
# so there is exactly one place in the repo that pins these open_clip tags.
MODEL_SLUGS_V6 = ["clip_vit_l14", "siglip_vit_b16", "clip_vit_b32"]
MODELS = [spec for spec in V5_MODELS if spec["slug"] in MODEL_SLUGS_V6]
assert len(MODELS) == 3, f"expected all 3 of {MODEL_SLUGS_V6} in experiment_v5_models.MODELS"

RESULTS_RAW = Path("results/raw")
PROGRESS_PATH = RESULTS_RAW / "v6_progress.json"
PREDICTIONS_CSV_PATH = RESULTS_RAW / "v6_zeroshot_predictions.csv"
RUN_CONFIG_PATH = Path("experiments/configs/day12_v6_run.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_progress(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": {}}


def save_progress(progress, path):
    """Atomic write (temp file + os.replace), matching v4's manifest convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp_path, path)


def append_predictions_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    import csv

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_slug", "angle", "condition", "image_id", "true_label", "predicted_label"])
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def build_distorted_and_rectified(image, image_id, theta, base_seed):
    """One image's distorted + mean_fill-rectified array at camera-rotation
    angle `theta`, using v4's exact seed derivation and homography (replicate
    fixed at 0, per the instruction)."""
    h, w = image.shape[:2]
    axis_phi_deg, _derived_seed = sample_axis_angle(image_id, theta, REPLICATE, base_seed)
    H = build_rotation_homography(h, w, theta, axis_phi_deg)

    distorted = cv2.warpPerspective(image, H, (w, h))
    mask = compute_valid_mask(image.shape, H)
    rectified_raw = rectify_oracle(distorted, H)
    rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
    return distorted, rectified


def zero_shot_rows(model_slug, angle, condition, image_ids, true_labels, predicted_labels):
    return [
        {
            "model_slug": model_slug,
            "angle": angle,
            "condition": condition,
            "image_id": iid,
            "true_label": tl,
            "predicted_label": pl,
        }
        for iid, tl, pl in zip(image_ids, true_labels, predicted_labels)
    ]


def run_clean_block(spec, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels, class_names, device, batch_size, out_suffix, device_name="cpu"):
    """Clean condition: computed once per model (angle-independent)."""
    image_arrays = [load_image_rgb(p) for p in image_paths]
    features = encode_image_arrays(image_arrays, model, preprocess, device, batch_size=batch_size)
    predicted = predict_labels(features, text_features, class_names)

    features_path = RESULTS_RAW / f"v6_features_{spec['slug']}{out_suffix}_clean.npy"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(features_path, features.numpy())

    rows = zero_shot_rows(spec["slug"], 0, "clean", image_ids, true_labels, predicted)
    return rows, features_path


def run_angle_block(spec, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels, class_names, theta, base_seed, device, batch_size, out_suffix):
    """One (model, angle) block: distort + rectify every image at `theta`
    (replicate 0), extract+cache both conditions' features, zero-shot
    predict both."""
    distorted_arrays, rectified_arrays = [], []
    for path, image_id in zip(image_paths, image_ids):
        image = load_image_rgb(path)
        distorted, rectified = build_distorted_and_rectified(image, image_id, theta, base_seed)
        distorted_arrays.append(distorted)
        rectified_arrays.append(rectified)

    distorted_features = encode_image_arrays(distorted_arrays, model, preprocess, device, batch_size=batch_size)
    rectified_features = encode_image_arrays(rectified_arrays, model, preprocess, device, batch_size=batch_size)

    distorted_pred = predict_labels(distorted_features, text_features, class_names)
    rectified_pred = predict_labels(rectified_features, text_features, class_names)

    distorted_path = RESULTS_RAW / f"v6_features_{spec['slug']}{out_suffix}_distorted_{theta}.npy"
    rectified_path = RESULTS_RAW / f"v6_features_{spec['slug']}{out_suffix}_rectified_{theta}.npy"
    distorted_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(distorted_path, distorted_features.numpy())
    np.save(rectified_path, rectified_features.numpy())

    rows = zero_shot_rows(spec["slug"], theta, "distorted", image_ids, true_labels, distorted_pred)
    rows += zero_shot_rows(spec["slug"], theta, "rectified", image_ids, true_labels, rectified_pred)
    return rows


def run_pipeline(config, image_paths, image_ids, true_labels, class_names, progress_path, predictions_csv_path, out_suffix="", device="cpu", batch_size=32):
    """Run clean + all (model, angle) blocks, resumable via progress_path."""
    base_seed = config["seed"]
    progress = load_progress(progress_path)

    for spec in MODELS:
        slug = spec["slug"]
        model_progress = progress["models"].setdefault(slug, {"clean_done": False, "completed_angles": []})
        log(f"=== Model: {spec['label']} ({spec['name']}, {spec['pretrained']}) ===")

        t0 = time.time()
        model, preprocess, tokenizer = load_model_generic(spec["name"], spec["pretrained"], device)
        text_features = encode_text_features_ensemble(model, tokenizer, class_names, PROMPT_ENSEMBLE_TEMPLATES, device)
        log(f"  model+text ready in {time.time()-t0:.1f}s")

        if not model_progress["clean_done"]:
            t0 = time.time()
            rows, _ = run_clean_block(spec, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels, class_names, device, batch_size, out_suffix)
            append_predictions_csv(rows, predictions_csv_path)
            model_progress["clean_done"] = True
            save_progress(progress, progress_path)
            log(f"  Checkpoint: clean block done in {time.time()-t0:.1f}s ({len(image_paths)} images)")
        else:
            log("  Skipping clean block (already done)")

        for theta in ANGLES:
            if theta in model_progress["completed_angles"]:
                log(f"  Skipping angle={theta} (already done)")
                continue
            t0 = time.time()
            rows = run_angle_block(spec, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels, class_names, theta, base_seed, device, batch_size, out_suffix)
            append_predictions_csv(rows, predictions_csv_path)
            model_progress["completed_angles"].append(theta)
            save_progress(progress, progress_path)
            log(f"  Checkpoint: angle={theta} done in {time.time()-t0:.1f}s ({len(image_paths)} images)")

        del model, preprocess, tokenizer, text_features

    log("ALL BLOCKS COMPLETE")


def run_smoke_test(config, class_names, n_per_class=20 // 4, device="cpu"):
    """20-image (5/class) end-to-end smoke test across every model and
    angle, using isolated `_smoketest20`-suffixed paths that are deleted
    before returning (never left mixed in with the real v6 outputs).
    Returns (ok: bool, report: str, projected_hours: dict[slug, float]).
    """
    dataset_root = config["dataset_root"]
    samples = load_monumai(dataset_root)
    counts = {c: 0 for c in class_names}
    smoke_samples = []
    for path, label in samples:
        if counts[label] < n_per_class:
            smoke_samples.append((path, label))
            counts[label] += 1
    image_paths = [p for p, _ in smoke_samples]
    true_labels = [l for _, l in smoke_samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    n_images = len(image_paths)

    smoke_progress_path = RESULTS_RAW / "v6_progress_smoketest20.json"
    smoke_csv_path = RESULTS_RAW / "v6_zeroshot_predictions_smoketest20.csv"
    for p in [smoke_progress_path, smoke_csv_path]:
        if p.exists():
            p.unlink()

    report_lines = [f"Smoke test: {n_images} images ({n_per_class}/class)."]
    projected_hours = {}
    ok = True
    t_all = time.time()
    try:
        for spec in MODELS:
            slug = spec["slug"]
            t0 = time.time()
            model, preprocess, tokenizer = load_model_generic(spec["name"], spec["pretrained"], device)
            text_features = encode_text_features_ensemble(model, tokenizer, class_names, PROMPT_ENSEMBLE_TEMPLATES, device)

            rows, _ = run_clean_block(spec, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels, class_names, device, 32, "_smoketest20")
            for theta in ANGLES:
                rows += run_angle_block(spec, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels, class_names, theta, config["seed"], device, 32, "_smoketest20")
            elapsed = time.time() - t0
            # n_images * (1 clean + 2 conditions * len(ANGLES) angles) forward passes
            n_forward = n_images * (1 + 2 * len(ANGLES))
            seconds_per_image = elapsed / n_forward
            full_n_forward = 1514 * (1 + 2 * len(ANGLES))
            projected_hours[slug] = seconds_per_image * full_n_forward / 3600.0
            report_lines.append(
                f"  {spec['label']}: smoke test OK in {elapsed:.1f}s "
                f"({seconds_per_image:.3f}s/forward-pass) -> projected {projected_hours[slug]:.2f}h for the full 1,514-image run"
            )
            del model, preprocess, tokenizer, text_features
    except Exception as exc:  # noqa: BLE001
        ok = False
        report_lines.append(f"  SMOKE TEST FAILED: {exc!r}")

    report_lines.append(f"Smoke test total wall time: {time.time()-t_all:.1f}s")

    # Clean up smoke-test artifacts regardless of outcome.
    for p in [smoke_progress_path, smoke_csv_path]:
        if p.exists():
            p.unlink()
    for f in RESULTS_RAW.glob("v6_features_*_smoketest20_*.npy"):
        f.unlink()

    return ok, "\n".join(report_lines), projected_hours


def main():
    parser = argparse.ArgumentParser(description="Day 12 v6: feature extraction + zero-shot (Experiment A).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip the automatic 20-image smoke test (not used in the real unattended run).")
    args = parser.parse_args()

    config = load_config(args.config)
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]

    if not args.skip_smoke_test:
        log("Running 20-image smoke test before committing to the full run ...")
        ok, report, projected_hours = run_smoke_test(config, class_names)
        log("Smoke test report:\n" + report)
        if not ok:
            log("ABORTING full run: smoke test failed. See report above and experiments/logs/day12_v6.md.")
            Path("experiments/logs/day12_v6.md").parent.mkdir(parents=True, exist_ok=True)
            with open("experiments/logs/day12_v6.md", "w", encoding="utf-8") as f:
                f.write("# Day 12 v6 — ABORTED\n\nSmoke test failed before the full run started:\n\n```\n" + report + "\n```\n")
            return
        total_projected = sum(projected_hours.values())
        log(f"Smoke test passed. Total projected wall time for all 3 models: {total_projected:.1f}h. Continuing automatically.")

    log(f"Loading dataset from {dataset_root} ...")
    samples = load_monumai(dataset_root)
    image_paths = [p for p, _ in samples]
    true_labels = [l for _, l in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    log(f"{len(samples)} images loaded.")

    RUN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "day": "12_v6",
                "git_commit": git_commit_hash(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seed": config["seed"],
                "n_images": len(samples),
                "angles": ANGLES,
                "replicate": REPLICATE,
                "models": MODELS,
                "prompt_ensemble_templates": PROMPT_ENSEMBLE_TEMPLATES,
                "border_policy": CHOSEN_BORDER_POLICY,
                "note": "homography/seeds reused unmodified from src/experiment_v4.py for continuity with the v4 CSV",
            },
            f,
            indent=2,
        )

    t_start = time.time()
    run_pipeline(config, image_paths, image_ids, true_labels, class_names, PROGRESS_PATH, PREDICTIONS_CSV_PATH)
    log(f"Total wall-clock time: {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
