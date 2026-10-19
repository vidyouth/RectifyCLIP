"""Day 8 v2 — full experimental grid with per-image, per-replicate random tilts.

Day 5's `experiment_runner.py` (kept frozen, unmodified) passed the single
config-level `seed` straight into every call to `distort()`, for every image.
Because `distort()` seeds a local `np.random.RandomState(seed)` and that seed
never varied, every image at a given severity received the exact same corner
offsets (only rescaled by that image's own width/height), so the share of
images with H[2,0]>0 / H[2,1]>0 was 1.0 at every severity — one random tilt
direction cloned across the whole dataset, not an independent tilt per image.
See experiments/logs/day8_v2.md for the full writeup.

This module reruns the grid with the fix: each (image_id, replicate_index)
pair gets its own seed via `distortion.derive_seed`, so tilt direction is
independent per image and per replicate, and 5 replicates per image per
severity let variance/CI across random tilts be estimated rather than
reporting one arbitrary draw as if it were the whole story.

Output locations are new, parallel paths — nothing under results/raw or
results/tables from Day 5/6 is overwritten:
    results/raw/day8_v2_grid.csv
    results/raw/homographies_v2/<severity>/<image_id>__r<replicate>.npy
    experiments/configs/day8_v2_run.json
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from clip_zero_shot import (
    PROMPT_TEMPLATE_SLUGS,
    encode_image_arrays,
    encode_image_features,
    encode_text_features,
    load_config,
    load_model,
    predict_from_features,
)
from data_loader import load_monumai
from distortion import (
    SEVERITY_FRACTIONS,
    derive_seed,
    distort,
    homography_tilt_angle,
    image_id_from_path,
    load_image_rgb,
)
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

SEVERITIES = ["mild", "medium", "severe"]
N_REPLICATES_DEFAULT = 5
ROW_FIELDNAMES = [
    "image_id",
    "true_label",
    "severity",
    "replicate",
    "condition",
    "predicted_label",
    "correct",
    "top1_confidence",
    "prompt_id",
    "base_seed",
    "derived_seed",
    "tilt_angle_deg",
    "tilt_magnitude",
]


def git_commit_hash():
    """Return HEAD's commit hash, or "unknown" if git is unavailable/dirty repo issues."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - reproducibility metadata, must not crash the run
        return "unknown"


def save_homography_v2(H, image_id, severity, replicate, output_dir="results/raw/homographies_v2"):
    """Save H as output_dir/severity/<image_id>__r<replicate>.npy, creating dirs as needed."""
    output_path = Path(output_dir) / severity / f"{image_id}__r{replicate}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, H)
    return output_path


def run_grid_v2(config, device="cpu", image_limit=None, n_replicates=N_REPLICATES_DEFAULT,
                 batch_size=32, verbose=True):
    """Run the image x severity x replicate x condition grid with independent
    per-(image, replicate) tilts.

    Returns (rows, failed_images, timing), same shape as Day 5's run_grid.
    `rows` includes a "replicate" field (0-indexed) and, for distorted/
    rectified rows, the derived seed and the tilt angle/magnitude of that
    replicate's homography, so the direction check can be done straight off
    the output CSV without re-reading every .npy file.
    """
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    prompt_template = config["prompt_template"]
    base_seed = config["seed"]
    prompt_id = PROMPT_TEMPLATE_SLUGS[prompt_template]

    timing = {}
    failed_images = []

    t0 = time.time()
    samples = load_monumai(dataset_root)
    if image_limit is not None:
        samples = samples[:image_limit]
    image_paths = [path for path, _ in samples]
    true_labels = [label for _, label in samples]
    if verbose:
        print(f"Loaded {len(samples)} images from {dataset_root}")

    model, preprocess, tokenizer = load_model(device)
    text_features = encode_text_features(model, tokenizer, class_names, prompt_template, device)
    timing["setup_seconds"] = time.time() - t0

    # --- Clean condition: one CLIP forward pass per image, independent of severity/replicate ---
    t0 = time.time()
    clean_features, clean_ok_paths, clean_failed_paths = encode_image_features(
        image_paths, model, preprocess, device, batch_size=batch_size
    )
    for path in clean_failed_paths:
        failed_images.append({"image_path": path, "phase": "clean_load", "reason": "failed to load/decode"})

    clean_predictions = predict_from_features(clean_features, text_features, class_names)
    ok_index = {path: i for i, path in enumerate(image_paths)}
    clean_true_labels = [true_labels[ok_index[path]] for path in clean_ok_paths]
    clean_image_ids = [image_id_from_path(path, dataset_root) for path in clean_ok_paths]

    clean_by_image_id = {}
    rows = []
    for image_id, true_label, (predicted_label, confidence) in zip(
        clean_image_ids, clean_true_labels, clean_predictions
    ):
        clean_by_image_id[image_id] = (true_label, predicted_label, confidence)
    timing["clean_condition_seconds"] = time.time() - t0
    if verbose:
        print(f"Clean condition encoded for {len(clean_ok_paths)} images "
              f"({len(clean_failed_paths)} failed to load).")

    # --- Distorted + rectified conditions, per severity, per replicate ---
    for severity in SEVERITIES:
        for replicate in range(n_replicates):
            t_rep = time.time()
            distorted_arrays, rectified_arrays = [], []
            rep_image_ids, rep_true_labels = [], []
            rep_derived_seeds, rep_angles, rep_magnitudes = [], [], []

            for path in clean_ok_paths:
                image_id = image_id_from_path(path, dataset_root)
                try:
                    derived_seed = derive_seed(image_id, replicate, base_seed)
                    image = load_image_rgb(path)
                    distorted, H = distort(image, severity, derived_seed)
                    mask = compute_valid_mask(image.shape, H)
                    rectified_raw = rectify_oracle(distorted, H)
                    rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
                    save_homography_v2(H, image_id, severity, replicate)
                    angle_deg, magnitude = homography_tilt_angle(H)
                except Exception as exc:  # noqa: BLE001 - one bad image must not kill a 47k-row run
                    failed_images.append(
                        {
                            "image_path": path,
                            "phase": f"distort_rectify_{severity}_r{replicate}",
                            "reason": str(exc),
                        }
                    )
                    continue

                distorted_arrays.append(distorted)
                rectified_arrays.append(rectified)
                rep_image_ids.append(image_id)
                rep_true_labels.append(true_labels[ok_index[path]])
                rep_derived_seeds.append(derived_seed)
                rep_angles.append(angle_deg)
                rep_magnitudes.append(magnitude)

            distorted_features = encode_image_arrays(
                distorted_arrays, model, preprocess, device, batch_size=batch_size
            )
            rectified_features = encode_image_arrays(
                rectified_arrays, model, preprocess, device, batch_size=batch_size
            )
            distorted_predictions = predict_from_features(distorted_features, text_features, class_names)
            rectified_predictions = predict_from_features(rectified_features, text_features, class_names)

            for image_id, true_label, derived_seed, angle_deg, magnitude, (dist_pred, dist_conf), (rect_pred, rect_conf) in zip(
                rep_image_ids, rep_true_labels, rep_derived_seeds, rep_angles, rep_magnitudes,
                distorted_predictions, rectified_predictions,
            ):
                clean_true, clean_pred, clean_conf = clean_by_image_id[image_id]
                if replicate == 0:
                    # Clean is replicate/distortion-independent; log it once per
                    # (image, severity) under replicate=0 only, to avoid 5x
                    # duplicating an identical row for no informational gain.
                    rows.append(
                        {
                            "image_id": image_id,
                            "true_label": clean_true,
                            "severity": severity,
                            "replicate": 0,
                            "condition": "clean",
                            "predicted_label": clean_pred,
                            "correct": clean_pred == clean_true,
                            "top1_confidence": clean_conf,
                            "prompt_id": prompt_id,
                            "base_seed": base_seed,
                            "derived_seed": "",
                            "tilt_angle_deg": "",
                            "tilt_magnitude": "",
                        }
                    )
                rows.append(
                    {
                        "image_id": image_id,
                        "true_label": true_label,
                        "severity": severity,
                        "replicate": replicate,
                        "condition": "distorted",
                        "predicted_label": dist_pred,
                        "correct": dist_pred == true_label,
                        "top1_confidence": dist_conf,
                        "prompt_id": prompt_id,
                        "base_seed": base_seed,
                        "derived_seed": derived_seed,
                        "tilt_angle_deg": angle_deg,
                        "tilt_magnitude": magnitude,
                    }
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "true_label": true_label,
                        "severity": severity,
                        "replicate": replicate,
                        "condition": "rectified",
                        "predicted_label": rect_pred,
                        "correct": rect_pred == true_label,
                        "top1_confidence": rect_conf,
                        "prompt_id": prompt_id,
                        "base_seed": base_seed,
                        "derived_seed": derived_seed,
                        "tilt_angle_deg": angle_deg,
                        "tilt_magnitude": magnitude,
                    }
                )

            timing[f"severity_{severity}_replicate_{replicate}_seconds"] = time.time() - t_rep
            if verbose:
                print(
                    f"severity={severity} replicate={replicate}: "
                    f"{len(rep_image_ids)} images processed in "
                    f"{timing[f'severity_{severity}_replicate_{replicate}_seconds']:.1f}s"
                )

    return rows, failed_images, timing


def save_rows_csv(rows, output_path):
    """Write the grid's row dicts to CSV using ROW_FIELDNAMES, creating parent dirs as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def save_run_config(config, output_path, n_replicates, extra=None):
    """Save the exact config + provenance (seed, prompt, git hash, timestamp) for this run as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_seed": config["seed"],
        "n_replicates": n_replicates,
        "prompt_template": config["prompt_template"],
        "class_names": config["class_names"],
        "dataset_root": config["dataset_root"],
        "model_name": config["model_name"],
        "pretrained_weights": config["pretrained_weights"],
        "severity_fractions": SEVERITY_FRACTIONS,
        "border_policy": CHOSEN_BORDER_POLICY,
        "seed_derivation": "sha256(f'{base_seed}:{image_id}:{replicate_index}')[:4 bytes]",
        "git_commit_hash": git_commit_hash(),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def direction_check(rows):
    """Verify tilt direction is no longer shared across images.

    Returns {severity: {share_h20_positive, share_h21_positive, n,
    angle_bins_deg}} computed over all distorted rows (every replicate
    pooled) — the same diagnostic the user ran manually on the v1 .npy files
    (share of H[2,0]>0, H[2,1]>0), now computable straight from the v2 CSV's
    tilt_angle_deg/tilt_magnitude columns. A healthy, unbiased fix should give
    shares near 0.5 on both axes.
    """
    result = {}
    for severity in SEVERITIES:
        subset = [r for r in rows if r["severity"] == severity and r["condition"] == "distorted"]
        angles = np.array([float(r["tilt_angle_deg"]) for r in subset])
        h20 = np.cos(np.radians(angles))  # sign(H[2,0]) == sign(cos(angle))
        h21 = np.sin(np.radians(angles))  # sign(H[2,1]) == sign(sin(angle))
        bin_edges = np.arange(0, 361, 45)
        counts, _ = np.histogram(angles, bins=bin_edges)
        result[severity] = {
            "n": len(subset),
            "share_h20_positive": float(np.mean(h20 > 0)),
            "share_h21_positive": float(np.mean(h21 > 0)),
            "angle_bin_edges_deg": bin_edges.tolist(),
            "angle_bin_counts": counts.tolist(),
            "angle_mean_deg": float(np.mean(angles)),
            "angle_circular_std_deg": float(
                np.degrees(np.sqrt(-2 * np.log(np.hypot(np.mean(h20), np.mean(h21)))))
            ) if np.hypot(np.mean(h20), np.mean(h21)) > 0 else None,
        }
    return result


def print_direction_check(check):
    """Print the direction_check() result in a compact, human-readable form."""
    for severity, stats in check.items():
        print(
            f"  {severity}: n={stats['n']} "
            f"share(H20>0)={stats['share_h20_positive']:.3f} "
            f"share(H21>0)={stats['share_h21_positive']:.3f} "
            f"mean_angle={stats['angle_mean_deg']:.1f}deg"
        )
        print(f"    angle histogram (45deg bins, 0-360): {stats['angle_bin_counts']}")


def main():
    """CLI entry point: run the Day 8 v2 grid (or --n-images-limited smoke test)."""
    parser = argparse.ArgumentParser(description="Day 8 v2 experimental grid (fixed tilt direction).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--n-images", type=int, default=None, help="Limit to the first N images (smoke test)."
    )
    parser.add_argument("--n-replicates", type=int, default=N_REPLICATES_DEFAULT)
    parser.add_argument("--output-csv", default="results/raw/day8_v2_grid.csv")
    parser.add_argument("--output-config-json", default="experiments/configs/day8_v2_run.json")
    args = parser.parse_args()

    config = load_config(args.config)
    t_start = time.time()
    rows, failed_images, timing = run_grid_v2(
        config, image_limit=args.n_images, n_replicates=args.n_replicates
    )
    total_seconds = time.time() - t_start

    save_rows_csv(rows, args.output_csv)
    print(f"Saved {args.output_csv} ({len(rows)} rows)")

    save_run_config(
        config,
        args.output_config_json,
        n_replicates=args.n_replicates,
        extra={
            "n_images_requested": args.n_images,
            "n_rows_written": len(rows),
            "n_images_failed": len(failed_images),
            "total_wall_clock_seconds": total_seconds,
        },
    )
    print(f"Saved {args.output_config_json}")

    print(f"\nTotal wall-clock time: {total_seconds:.1f}s")
    print(f"Failed images: {len(failed_images)}")
    for failure in failed_images:
        print(f"  SKIPPED: {failure['image_path']} (phase={failure['phase']}, reason={failure['reason']})")

    print("\nDirection check (expect shares ~0.5 on both axes):")
    check = direction_check(rows)
    print_direction_check(check)


if __name__ == "__main__":
    main()
