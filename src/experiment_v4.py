"""Day 10 v4 — angle-calibrated severity via a pure camera-rotation homography.

v1/v2/v3's distortion model perturbed the 4 image corners by a fraction of
image size, with no direct physical interpretation of "how many degrees did
the camera tilt." v4 replaces that with H = K R K^-1, the exact homography
induced by rotating a pinhole camera in place by a known angle theta about an
axis lying in the image plane, so "severity" now means a literal camera
tilt angle in degrees, not an arbitrary pixel-fraction knob.

**Intrinsics assumption (stated here per the instruction to log it)**: K is
assumed, not calibrated from the MonuMAI images (no calibration data exists
for this dataset) — focal length = 1.0 x image width (a common rule-of-thumb
for an unknown "normal"-ish lens), principal point at the exact image center
((width-1)/2 is NOT used here; (width/2, height/2) is, for simplicity — a
half-pixel difference, immaterial at any image size MonuMAI uses). This is a
real limitation: real facade photos were not taken with this exact K, so
theta should be read as "the angle a camera with these assumed intrinsics
would need to tilt to produce this homography," not a literal real-world
angle of any MonuMAI photo.

**Rotation convention**: axis = (cos(phi), sin(phi), 0) in camera
coordinates — i.e. an axis lying entirely in the image/XY-plane, pointed in
a uniformly random in-plane direction phi in [0, 360) degrees (phi=0 is
rotation about the x-axis, a pure vertical/pitch tilt; phi=90 is rotation
about the y-axis, a pure horizontal/yaw pan; other phi values are oblique).
R is built from the Rodrigues rotation vector (axis * radians(theta)) via
cv2.Rodrigues. H = K @ R @ inv(K), normalized so H[2,2] = 1 (cv2's own
getPerspectiveTransform normalizes this way too — matching that convention
keeps tilt_magnitude comparable across images/conditions).

H maps ORIGINAL image coordinates to DISTORTED (rotated-camera) image
coordinates, exactly matching v1-v3's H convention — rectification.
rectify_oracle(distorted_image, H) (reused unmodified) inverts it the same
way it always has.

Reuses, but never modifies, v1-v3's generic building blocks: distortion.
derive_seed/image_id_from_path/load_image_rgb/homography_tilt_angle,
rectification.rectify_oracle/apply_border_policy/compute_valid_mask/
CHOSEN_BORDER_POLICY, data_loader.load_monumai, and clip_zero_shot.
load_config/load_model/encode_text_features/encode_image_arrays. Per-class
full softmax scoring is implemented fresh in this file (not added to
clip_zero_shot.py) since logging all 4 scores per row is new behavior v1-v3
never needed project-wide.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from clip_zero_shot import encode_image_arrays, encode_text_features, load_config, load_model
from data_loader import load_monumai
from distortion import derive_seed, homography_tilt_angle, image_id_from_path, load_image_rgb
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

THETA_LEVELS = [5, 10, 15, 25, 35]
N_REPLICATES = 3
SEED_NAMESPACE = "v4_camera_rotation"  # folded into derive_seed's key so v4 seeds never collide with v2/v3's


def git_commit_hash():
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --- camera-rotation homography ------------------------------------


def sample_axis_angle(image_id, theta, replicate, base_seed):
    """Uniformly random in-plane tilt-axis direction phi in [0, 360) degrees,
    deterministic given (image_id, theta, replicate, base_seed), in a seed
    namespace independent of v2/v3 (SEED_NAMESPACE + theta folded into the
    string key passed to distortion.derive_seed, which otherwise only ever
    saw plain image_id strings in v1-v3)."""
    key = f"{SEED_NAMESPACE}:theta{theta}:{image_id}"
    seed = derive_seed(key, replicate, base_seed)
    rng = np.random.RandomState(seed)
    phi = float(rng.uniform(0.0, 360.0))
    return phi, seed


def build_rotation_homography(h, w, theta_deg, axis_phi_deg):
    """H = K R K^-1 for a pure camera rotation by theta_deg about an in-plane
    axis at direction axis_phi_deg. See module docstring for the intrinsics
    assumption and axis convention."""
    f = 1.0 * w
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    axis = np.array(
        [np.cos(np.radians(axis_phi_deg)), np.sin(np.radians(axis_phi_deg)), 0.0], dtype=np.float64
    )
    rotvec = axis * np.radians(theta_deg)
    R, _ = cv2.Rodrigues(rotvec)

    H = K @ R @ K_inv
    H = H / H[2, 2]
    return H.astype(np.float64)


def distorted_valid_fraction(shape, H):
    """Fraction of the DISTORTED (not rectified) canvas covered by real
    source pixels — a single forward warp of an all-255 mask through H, not
    the round-tripped mask rectification.compute_valid_mask computes.
    Larger theta pushes more of the frame outside the original image bounds,
    so this is the valid-area-vs-accuracy confound the instruction asks to
    report against."""
    h, w = shape[:2]
    ones = np.full((h, w), 255, dtype=np.uint8)
    warped = cv2.warpPerspective(ones, H, (w, h))
    return float(np.mean(warped > 250))


# --- CLIP full-softmax scoring (not in clip_zero_shot.py — new behavior) ------------------------------------


def predict_full_scores(image_features, text_features, class_names):
    """Return a list of (predicted_label, top1_confidence, [score_per_class
    in class_names order]) — v1-v3's predict_from_features only kept top1;
    v4 needs all 4 scores per row for the attractor-calibration diagnostic."""
    with torch.no_grad():
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
    results = []
    for i in range(probs.shape[0]):
        class_probs = probs[i].tolist()
        best_idx = max(range(len(class_names)), key=lambda j: class_probs[j])
        results.append((class_names[best_idx], class_probs[best_idx], class_probs))
    return results


# --- checkpoint / resume manifest ------------------------------------


def load_manifest(path):
    path = Path(path)
    if not path.exists():
        return {"clean_done": False, "completed_blocks": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest, path):
    """Atomic write: write to a temp file then os.replace, so a restart
    mid-write can never leave a half-written, corrupt manifest behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)


def append_rows_csv(rows, fieldnames, output_path):
    """Append rows to output_path, writing the header only if the file is
    new — this is the per-block checkpoint write."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


# --- main grid ------------------------------------


def build_fieldnames(class_names):
    return (
        ["image_id", "true_label", "theta", "replicate", "condition", "predicted_label", "correct", "top1_confidence"]
        + [f"score_{c}" for c in class_names]
        + ["valid_area_fraction", "axis_angle_deg", "tilt_angle_deg", "tilt_magnitude", "base_seed", "derived_seed"]
    )


def run_clean_block(config, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels, class_names, device, batch_size, log):
    """Clean condition: computed once, independent of theta/replicate. Logged
    with theta=0 (zero camera rotation IS the clean condition) so it serves
    directly as the theta=0 point on every "accuracy vs theta" plot."""
    image_arrays = [load_image_rgb(p) for p in image_paths]
    features = encode_image_arrays(image_arrays, model, preprocess, device, batch_size=batch_size)
    predictions = predict_full_scores(features, text_features, class_names)

    rows = []
    for image_id, true_label, (pred_label, confidence, scores) in zip(image_ids, true_labels, predictions):
        row = {
            "image_id": image_id,
            "true_label": true_label,
            "theta": 0,
            "replicate": 0,
            "condition": "clean",
            "predicted_label": pred_label,
            "correct": pred_label == true_label,
            "top1_confidence": confidence,
            "valid_area_fraction": 1.0,
            "axis_angle_deg": "",
            "tilt_angle_deg": "",
            "tilt_magnitude": "",
            "base_seed": config["seed"],
            "derived_seed": "",
        }
        for class_name, score in zip(class_names, scores):
            row[f"score_{class_name}"] = score
        rows.append(row)
    log(f"clean: {len(rows)} images encoded")
    return rows


def run_theta_replicate_block(
    config, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels,
    class_names, theta, replicate, device, batch_size, log,
):
    """One (theta, replicate) block: distort every image with its own
    independent camera-rotation homography, rectify with the exact inverse,
    classify both conditions, log all 4 softmax scores + valid-area fraction
    + direction-check fields per row."""
    base_seed = config["seed"]
    distorted_arrays, rectified_arrays = [], []
    block_image_ids, block_true_labels = [], []
    block_valid_fracs, block_axis_angles, block_tilt_angles, block_tilt_mags, block_seeds = [], [], [], [], []

    for path, image_id, true_label in zip(image_paths, image_ids, true_labels):
        image = load_image_rgb(path)
        h, w = image.shape[:2]
        axis_phi_deg, derived_seed = sample_axis_angle(image_id, theta, replicate, base_seed)
        H = build_rotation_homography(h, w, theta, axis_phi_deg)

        distorted = cv2.warpPerspective(image, H, (w, h))
        mask = compute_valid_mask(image.shape, H)
        rectified_raw = rectify_oracle(distorted, H)
        rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)

        valid_frac = distorted_valid_fraction(image.shape, H)
        tilt_angle_deg, tilt_magnitude = homography_tilt_angle(H)

        distorted_arrays.append(distorted)
        rectified_arrays.append(rectified)
        block_image_ids.append(image_id)
        block_true_labels.append(true_label)
        block_valid_fracs.append(valid_frac)
        block_axis_angles.append(axis_phi_deg)
        block_tilt_angles.append(tilt_angle_deg)
        block_tilt_mags.append(tilt_magnitude)
        block_seeds.append(derived_seed)

    distorted_features = encode_image_arrays(distorted_arrays, model, preprocess, device, batch_size=batch_size)
    rectified_features = encode_image_arrays(rectified_arrays, model, preprocess, device, batch_size=batch_size)
    distorted_predictions = predict_full_scores(distorted_features, text_features, class_names)
    rectified_predictions = predict_full_scores(rectified_features, text_features, class_names)

    rows = []
    for i, image_id in enumerate(block_image_ids):
        true_label = block_true_labels[i]
        shared = {
            "image_id": image_id,
            "true_label": true_label,
            "theta": theta,
            "replicate": replicate,
            "valid_area_fraction": block_valid_fracs[i],
            "axis_angle_deg": block_axis_angles[i],
            "tilt_angle_deg": block_tilt_angles[i],
            "tilt_magnitude": block_tilt_mags[i],
            "base_seed": base_seed,
            "derived_seed": block_seeds[i],
        }
        for condition, (pred_label, confidence, scores) in [
            ("distorted", distorted_predictions[i]),
            ("rectified", rectified_predictions[i]),
        ]:
            row = dict(shared)
            row["condition"] = condition
            row["predicted_label"] = pred_label
            row["correct"] = pred_label == true_label
            row["top1_confidence"] = confidence
            for class_name, score in zip(class_names, scores):
                row[f"score_{class_name}"] = score
            rows.append(row)

    log(f"theta={theta} replicate={replicate}: {len(block_image_ids)} images")
    return rows


def run_grid_v4(config, output_csv, manifest_path, device="cpu", image_limit=None, batch_size=32, log=print):
    """Resumable grid runner: reads/writes a JSON manifest tracking which
    blocks are already done, and appends to output_csv after every block, so
    a killed-and-restarted process picks up exactly where it left off
    instead of re-running (and re-duplicating rows for) completed blocks."""
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    prompt_template = config["prompt_template"]

    samples = load_monumai(dataset_root)
    if image_limit is not None:
        samples = samples[:image_limit]
    image_paths = [p for p, _ in samples]
    true_labels = [lbl for _, lbl in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    log(f"Loaded {len(samples)} images from {dataset_root}")

    model, preprocess, tokenizer = load_model(device)
    text_features = encode_text_features(model, tokenizer, class_names, prompt_template, device)
    log("Model + text features ready")

    fieldnames = build_fieldnames(class_names)
    manifest = load_manifest(manifest_path)

    if not manifest["clean_done"]:
        rows = run_clean_block(
            config, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels,
            class_names, device, batch_size, log,
        )
        append_rows_csv(rows, fieldnames, output_csv)
        manifest["clean_done"] = True
        save_manifest(manifest, manifest_path)
        log("Checkpoint: clean block done")
    else:
        log("Skipping clean block (already done per manifest)")

    completed = {tuple(b) for b in manifest["completed_blocks"]}
    for theta in THETA_LEVELS:
        for replicate in range(N_REPLICATES):
            if (theta, replicate) in completed:
                log(f"Skipping theta={theta} replicate={replicate} (already done per manifest)")
                continue
            t0 = time.time()
            rows = run_theta_replicate_block(
                config, model, preprocess, tokenizer, text_features, image_paths, image_ids, true_labels,
                class_names, theta, replicate, device, batch_size, log,
            )
            append_rows_csv(rows, fieldnames, output_csv)
            manifest["completed_blocks"].append([theta, replicate])
            save_manifest(manifest, manifest_path)
            log(f"Checkpoint: theta={theta} replicate={replicate} done in {time.time() - t0:.1f}s")

    log("ALL BLOCKS COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Day 10 v4 angle-calibrated grid.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-images", type=int, default=None, help="Limit to first N images (smoke test).")
    parser.add_argument("--output-csv", default="results/raw/day10_v4_grid.csv")
    parser.add_argument("--manifest", default="results/raw/day10_v4_progress.json")
    parser.add_argument("--output-config-json", default="experiments/configs/day10_v4_run.json")
    args = parser.parse_args()

    config = load_config(args.config)

    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    t_start = time.time()
    run_grid_v4(config, args.output_csv, args.manifest, image_limit=args.n_images, log=log)
    total_seconds = time.time() - t_start

    Path(args.output_config_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_seed": config["seed"],
        "theta_levels_deg": THETA_LEVELS,
        "n_replicates": N_REPLICATES,
        "seed_namespace": SEED_NAMESPACE,
        "intrinsics_assumption": {
            "focal_length": "1.0 * image_width",
            "principal_point": "(image_width/2, image_height/2)",
        },
        "class_names": config["class_names"],
        "dataset_root": config["dataset_root"],
        "model_name": config["model_name"],
        "pretrained_weights": config["pretrained_weights"],
        "border_policy": CHOSEN_BORDER_POLICY,
        "git_commit_hash": git_commit_hash(),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_images_requested": args.n_images,
        "total_wall_clock_seconds": total_seconds,
    }
    with open(args.output_config_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"Saved {args.output_config_json}")
    log(f"Total wall-clock time: {total_seconds:.1f}s")


if __name__ == "__main__":
    main()
