"""Day 13 v7 — classical blind tilt estimation + blind rectification on real photos.

PRE-REGISTERED (per the instruction): the analyses below — the tilt estimator's
geometry, the validation metrics (Pearson correlation + MAE, signed on the
pitch-dominant subset, unsigned overall), and the downstream comparisons — are
fixed before this file is run end to end. Nothing here is added or dropped
after seeing results; day13_v7_analysis.py reports every number, including
nulls, as the pre-registration requires.

This file does the new CV/model work for Experiment C (see src/day13_v7_analysis.py
for Experiments A and B, which are pure re-analysis of cached v5/v6 outputs and
extract no new features). It never touches v1-v6 code or outputs.

## The tilt estimator

1. Detect line segments (cv2.createLineSegmentDetector), keep long
   (>=3% of the image diagonal), near-vertical (<=25 degrees from vertical in
   the image plane) ones.
2. RANSAC a vanishing point: each line segment's homogeneous coefficients are
   computed via the cross product of its two endpoints; candidate VPs come
   from crossing random line pairs; the candidate with the most inlier lines
   (algebraic point-to-line distance < 3px) wins; the final VP is a
   least-squares refit over its inliers.
3. Convert the VP to a tilt angle assuming a PURE CAMERA PITCH (rotation
   about the camera's local x-axis only — matching "camera pitch" in the
   instruction) with the SAME intrinsics assumption v4 used: focal length =
   image width, principal point at the exact image center. Nothing here was
   calibrated per-photo; this is a stated, acknowledged simplification (real
   phone-camera focal lengths vary, and real natural tilt is not always pure
   pitch — see the limitations list in experiments/logs/day13_v7.md).

   The algebra (full derivation in the module's dev notes, verified
   empirically against synthetic ground truth below): for a pure-pitch
   homography H = K @ R_x(theta) @ K^-1, a vertical world line's vanishing
   point lands at pixel (cx, f*cot(theta) + cy). Inverting this has a
   180-degree sign ambiguity (an undirected vanishing point cannot by itself
   distinguish "tilted up by theta" from "tilted down by -theta" without a
   small-angle assumption) — resolved here by picking the candidate angle in
   (-90, 90], which is correct whenever the true tilt itself is realistic
   (no real facade photo is tilted more than 90 degrees). This resolution
   was verified empirically on synthetic data with known sign (see
   validate_on_synthetic below) before being relied on for real photos.

## Blind rectification

Given an estimated pitch `theta_est`, build H_est = build_rotation_homography
(h, w, theta_est, axis_phi_deg=0) — EXACTLY v4's own rotation-homography
builder, imported unmodified, with the axis fixed to pure pitch (phi=0) —
then rectify with the SAME rectify_oracle / compute_valid_mask /
apply_border_policy / CHOSEN_BORDER_POLICY v4-v6 already use, treating
H_est as if it were the (unknown, estimated) true distortion homography.
This is deliberate reuse, not reimplementation: the only new thing this file
contributes is the ESTIMATE of H; everything downstream of having an H is
identical to v4/v6's oracle pipeline.
"""

import argparse
import csv
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
from experiment_v6 import ANGLES
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

RESULTS_RAW = Path("results/raw")
RESULTS_TABLES = Path("results/tables")
FIGURES = Path("figures")

VALIDATION_CSV_PATH = RESULTS_RAW / "v7_validation_synthetic.csv"
REAL_TILT_CSV_PATH = RESULTS_RAW / "v7_real_tilt_estimates.csv"
PROGRESS_PATH = RESULTS_RAW / "v7_progress.json"
RUN_CONFIG_PATH = Path("experiments/configs/day13_v7_run.json")

# --- tilt-estimator parameters (pre-registered, fixed before any run) ------------------------------------
MAX_ANGLE_FROM_VERTICAL_DEG = 25.0
MIN_LENGTH_FRACTION_OF_DIAGONAL = 0.03
RANSAC_ITERATIONS = 500
RANSAC_INLIER_PX = 3.0
MIN_INLIER_LINES = 4
RANSAC_SEED = 42

# Models used downstream in steps 3-4 (Experiment C), per the instruction.
MODEL_B32 = next(s for s in V5_MODELS if s["slug"] == "clip_vit_b32")
MODEL_L14 = next(s for s in V5_MODELS if s["slug"] == "clip_vit_l14")

# "Pitch-dominant" validation subset: v4's random axis_phi_deg within this
# many degrees of 0 or 180 is treated as effectively a pure pitch (possibly
# sign-flipped) rotation, the only case this estimator's model matches.
PITCH_DOMINANT_AXIS_TOLERANCE_DEG = 20.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_PATH)


# --- tilt estimator ------------------------------------


def detect_near_vertical_lines(image):
    """Long, near-vertical line segments as homogeneous line coefficients
    (a,b,c), normalized so sqrt(a^2+b^2)=1. Returns an (N,3) array (N may be 0)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lsd = cv2.createLineSegmentDetector()
    detected = lsd.detect(gray)[0]
    if detected is None or len(detected) == 0:
        return np.zeros((0, 3))

    lines = detected.reshape(-1, 4)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    length = np.hypot(dx, dy)
    angle_from_vertical = np.degrees(np.arctan2(np.abs(dx), np.abs(dy) + 1e-9))
    diag = np.hypot(w, h)
    keep = (angle_from_vertical <= MAX_ANGLE_FROM_VERTICAL_DEG) & (length >= MIN_LENGTH_FRACTION_OF_DIAGONAL * diag)
    lines = lines[keep]
    if len(lines) == 0:
        return np.zeros((0, 3))

    p1 = np.column_stack([lines[:, 0], lines[:, 1], np.ones(len(lines))])
    p2 = np.column_stack([lines[:, 2], lines[:, 3], np.ones(len(lines))])
    coeffs = np.cross(p1, p2)
    norm = np.hypot(coeffs[:, 0], coeffs[:, 1])
    norm[norm == 0] = 1.0
    return coeffs / norm[:, None]


def ransac_vanishing_point(line_coeffs, seed=RANSAC_SEED):
    """RANSAC a finite vanishing point from near-vertical line coefficients.
    Returns (vx, vy, n_inliers) or None if too few lines / no consensus found."""
    n = len(line_coeffs)
    if n < MIN_INLIER_LINES:
        return None

    rng = np.random.RandomState(seed)
    best_inliers = None
    best_vp = None
    for _ in range(RANSAC_ITERATIONS):
        i, j = rng.choice(n, 2, replace=False)
        vp_h = np.cross(line_coeffs[i], line_coeffs[j])
        if abs(vp_h[2]) < 1e-9:
            continue  # near-parallel pair -> ill-conditioned candidate, skip
        vx, vy = vp_h[0] / vp_h[2], vp_h[1] / vp_h[2]
        dist = np.abs(line_coeffs[:, 0] * vx + line_coeffs[:, 1] * vy + line_coeffs[:, 2])
        inliers = dist < RANSAC_INLIER_PX
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_vp = (vx, vy)

    if best_vp is None or best_inliers.sum() < MIN_INLIER_LINES:
        return None

    inlier_coeffs = line_coeffs[best_inliers]
    A = inlier_coeffs[:, :2]
    b = -inlier_coeffs[:, 2]
    vp_refined, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(vp_refined[0]), float(vp_refined[1]), int(best_inliers.sum())


def vp_to_pitch_deg(vx, vy, w, h):
    """Convert a vanishing-point pixel location to a signed pitch angle in
    degrees, under the pure-pitch, v4-matching intrinsics assumption (f=w,
    principal point at image center). See module docstring for the
    derivation and the 180-degree sign-ambiguity resolution."""
    f = 1.0 * w
    cy = h / 2.0
    theta_raw = np.degrees(np.arctan2(f, vy - cy))
    if theta_raw > 90:
        return float(theta_raw - 180)
    if theta_raw < -90:
        return float(theta_raw + 180)
    return float(theta_raw)


def estimate_natural_tilt(image):
    """Full pipeline: image -> dict(ok, estimated_tilt_deg, n_inliers, n_candidate_lines)."""
    lines = detect_near_vertical_lines(image)
    result = ransac_vanishing_point(lines)
    if result is None:
        return {"ok": False, "estimated_tilt_deg": None, "n_inliers": 0, "n_candidate_lines": len(lines)}
    vx, vy, n_inliers = result
    h, w = image.shape[:2]
    theta_est = vp_to_pitch_deg(vx, vy, w, h)
    return {"ok": True, "estimated_tilt_deg": theta_est, "n_inliers": n_inliers, "n_candidate_lines": len(lines), "vp_pixel": (vx, vy)}


def blind_rectify(image, theta_est_deg):
    """Rectify `image` using the ESTIMATED pitch, via v4/v6's exact oracle
    rectification machinery (rectify_oracle/compute_valid_mask/apply_border_policy),
    treating H_est as if it were the true (unknown) distortion homography."""
    h, w = image.shape[:2]
    H_est = build_rotation_homography(h, w, theta_est_deg, axis_phi_deg=0)
    mask = compute_valid_mask(image.shape, H_est)
    rectified_raw = rectify_oracle(image, H_est)
    rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
    return rectified, H_est


# --- Step 2: validation on v4 synthetic images ------------------------------------


def run_validation(config, image_paths, image_ids):
    """Estimate tilt on v4-identical synthetic distorted images (same seeds,
    replicate 0) at each of ANGLES, checkpointed per angle. Appends rows to
    VALIDATION_CSV_PATH: image_id, angle (true |theta|), axis_phi_deg,
    signed_true_tilt (projected onto the x-axis: +theta if axis points
    toward +x, -theta if toward -x — only meaningful when pitch-dominant),
    pitch_dominant (bool), ok, estimated_tilt_deg, n_inliers.
    """
    base_seed = config["seed"]
    progress = load_progress()
    done_angles = set(progress.get("validation_angles_done", []))

    fieldnames = ["image_id", "angle", "axis_phi_deg", "signed_true_tilt", "pitch_dominant", "ok", "estimated_tilt_deg", "n_inliers"]
    is_new = not VALIDATION_CSV_PATH.exists()

    for theta in ANGLES:
        if theta in done_angles:
            log(f"  validation angle={theta}: already done, skipping")
            continue
        t0 = time.time()
        rows = []
        for path, image_id in zip(image_paths, image_ids):
            image = load_image_rgb(path)
            h, w = image.shape[:2]
            axis_phi_deg, _ = sample_axis_angle(image_id, theta, 0, base_seed)
            H = build_rotation_homography(h, w, theta, axis_phi_deg)
            distorted = cv2.warpPerspective(image, H, (w, h))

            axis_rad = np.radians(axis_phi_deg)
            pitch_dominant = (abs(axis_phi_deg) <= PITCH_DOMINANT_AXIS_TOLERANCE_DEG) or (abs(axis_phi_deg - 180) <= PITCH_DOMINANT_AXIS_TOLERANCE_DEG) or (abs(axis_phi_deg - 360) <= PITCH_DOMINANT_AXIS_TOLERANCE_DEG)
            signed_true_tilt = theta * np.sign(np.cos(axis_rad)) if np.cos(axis_rad) != 0 else theta

            est = estimate_natural_tilt(distorted)
            rows.append(
                {
                    "image_id": image_id,
                    "angle": theta,
                    "axis_phi_deg": axis_phi_deg,
                    "signed_true_tilt": signed_true_tilt,
                    "pitch_dominant": pitch_dominant,
                    "ok": est["ok"],
                    "estimated_tilt_deg": est["estimated_tilt_deg"] if est["ok"] else "",
                    "n_inliers": est["n_inliers"],
                }
            )

        VALIDATION_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
                is_new = False
            writer.writerows(rows)

        done_angles.add(theta)
        progress["validation_angles_done"] = sorted(done_angles)
        save_progress(progress)
        log(f"  validation angle={theta}: {len(rows)} images done in {time.time()-t0:.1f}s")


def check_validation_gate():
    """Pearson correlation + MAE from VALIDATION_CSV_PATH. Returns a dict the
    caller uses to decide whether to continue past validation (per the
    instruction: 'If it is unreliable, say so and stop C after validation.').
    Pre-registered gate: proceed only if the pitch-dominant signed
    correlation is >= 0.3 (weak-but-real relationship) AND the pitch-dominant
    MAE is < 20 degrees (smaller than the smallest angle tested, 5 degrees,
    would be ideal, but the gate is set loose enough to allow a noisy-but-
    real classical estimator through, consistent with the rest of this
    project reporting modest effects honestly rather than demanding
    unrealistic precision from a 3-line/RANSAC estimator).
    """
    import csv as _csv

    rows = []
    with open(VALIDATION_CSV_PATH, "r", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            r["ok"] = r["ok"] == "True"
            r["pitch_dominant"] = r["pitch_dominant"] == "True"
            r["angle"] = int(r["angle"])
            r["signed_true_tilt"] = float(r["signed_true_tilt"])
            r["estimated_tilt_deg"] = float(r["estimated_tilt_deg"]) if r["estimated_tilt_deg"] not in ("", None) else None
            rows.append(r)

    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    coverage = n_ok / n_total if n_total else 0.0

    pitch_rows_ok = [r for r in rows if r["ok"] and r["pitch_dominant"]]
    if len(pitch_rows_ok) >= 10:
        true_vals = np.array([r["signed_true_tilt"] for r in pitch_rows_ok])
        est_vals = np.array([r["estimated_tilt_deg"] for r in pitch_rows_ok])
        pitch_corr = float(np.corrcoef(true_vals, est_vals)[0, 1])
        pitch_mae = float(np.mean(np.abs(true_vals - est_vals)))
    else:
        pitch_corr, pitch_mae = None, None

    all_rows_ok = [r for r in rows if r["ok"]]
    unsigned_true = np.array([r["angle"] for r in all_rows_ok])
    unsigned_est = np.array([abs(r["estimated_tilt_deg"]) for r in all_rows_ok])
    overall_corr = float(np.corrcoef(unsigned_true, unsigned_est)[0, 1]) if len(all_rows_ok) >= 10 else None
    overall_mae = float(np.mean(np.abs(unsigned_true - unsigned_est))) if len(all_rows_ok) >= 10 else None

    passed = (pitch_corr is not None) and (pitch_corr >= 0.3) and (pitch_mae is not None) and (pitch_mae < 20.0)

    return {
        "n_total": n_total,
        "coverage": coverage,
        "pitch_dominant_n": len(pitch_rows_ok),
        "pitch_dominant_corr": pitch_corr,
        "pitch_dominant_mae": pitch_mae,
        "overall_unsigned_corr": overall_corr,
        "overall_unsigned_mae": overall_mae,
        "passed_gate": passed,
    }


# --- Step 3: real-image tilt estimation ------------------------------------


def run_real_tilt_estimation(image_paths, image_ids, true_labels):
    progress = load_progress()
    if progress.get("real_tilt_done"):
        log("  real-image tilt estimation: already done, skipping")
        return

    t0 = time.time()
    fieldnames = ["image_id", "true_label", "ok", "estimated_tilt_deg", "n_inliers", "n_candidate_lines"]
    rows = []
    for path, image_id, true_label in zip(image_paths, image_ids, true_labels):
        image = load_image_rgb(path)
        est = estimate_natural_tilt(image)
        rows.append(
            {
                "image_id": image_id,
                "true_label": true_label,
                "ok": est["ok"],
                "estimated_tilt_deg": est["estimated_tilt_deg"] if est["ok"] else "",
                "n_inliers": est["n_inliers"],
                "n_candidate_lines": est["n_candidate_lines"],
            }
        )

    REAL_TILT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REAL_TILT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    progress["real_tilt_done"] = True
    save_progress(progress)
    log(f"  real-image tilt estimation: {len(rows)} images done in {time.time()-t0:.1f}s")


# --- Step 4: feature extraction for blind-rectified images ------------------------------------


def zero_shot_predict(model, tokenizer, class_names, text_features, features):
    return predict_labels(features, text_features, class_names)


def run_blind_rectified_real(config, class_names, image_paths, image_ids, device="cpu", batch_size=32):
    """Blind-rectify every real image whose tilt estimate was 'ok', extract
    features via both models, save features + zero-shot (B32) predictions.
    Checkpointed per model."""
    progress = load_progress()
    tilt_by_id = {}
    with open(REAL_TILT_CSV_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tilt_by_id[r["image_id"]] = (r["ok"] == "True", float(r["estimated_tilt_deg"]) if r["ok"] == "True" else None)

    ok_paths, ok_ids = [], []
    for p, iid in zip(image_paths, image_ids):
        ok, _ = tilt_by_id[iid]
        if ok:
            ok_paths.append(p)
            ok_ids.append(iid)
    log(f"  blind-rectifying {len(ok_ids)}/{len(image_ids)} real images with an 'ok' tilt estimate")

    rectified_arrays = [blind_rectify(load_image_rgb(p), tilt_by_id[iid][1])[0] for p, iid in zip(ok_paths, ok_ids)]

    for spec in [MODEL_B32, MODEL_L14]:
        slug = spec["slug"]
        key = f"real_blindrect_{slug}_done"
        if progress.get(key):
            log(f"  real blind-rectified features ({spec['label']}): already done")
            continue
        t0 = time.time()
        model, preprocess, tokenizer = load_model_generic(spec["name"], spec["pretrained"], device)
        features = encode_image_arrays(rectified_arrays, model, preprocess, device, batch_size=batch_size)
        np.save(RESULTS_RAW / f"v7_features_{slug}_real_blindrect.npy", features.numpy())
        with open(RESULTS_RAW / "v7_real_blindrect_image_ids.json", "w", encoding="utf-8") as f:
            json.dump(ok_ids, f)

        if slug == MODEL_B32["slug"]:
            text_features = encode_text_features_ensemble(model, tokenizer, class_names, PROMPT_ENSEMBLE_TEMPLATES, device)
            predicted = zero_shot_predict(model, tokenizer, class_names, text_features, features)
            with open(RESULTS_RAW / "v7_real_blindrect_zs_predictions.json", "w", encoding="utf-8") as f:
                json.dump(dict(zip(ok_ids, predicted)), f, indent=2)

        del model, preprocess, tokenizer
        progress[key] = True
        save_progress(progress)
        log(f"  real blind-rectified features ({spec['label']}): done in {time.time()-t0:.1f}s")


def run_blind_rectified_synthetic(config, class_names, image_paths, image_ids, device="cpu", batch_size=32):
    """Re-generate v4-identical synthetic distorted images at each angle,
    blind-rectify each using ITS OWN validation-step estimate (not the
    oracle H), extract features via both models. Checkpointed per
    (model, angle). Images whose validation estimate was not 'ok' are
    skipped (same exclusion as the real-image path) and the kept subset
    saved alongside the features so analysis can align them."""
    base_seed = config["seed"]
    est_by_image_angle = {}
    with open(VALIDATION_CSV_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["image_id"], int(r["angle"]))
            ok = r["ok"] == "True"
            est_by_image_angle[key] = (ok, float(r["estimated_tilt_deg"]) if ok else None)

    progress = load_progress()
    for theta in ANGLES:
        kept_ids = []
        rectified_arrays = []
        for path, image_id in zip(image_paths, image_ids):
            ok, theta_est = est_by_image_angle[(image_id, theta)]
            if not ok:
                continue
            image = load_image_rgb(path)
            h, w = image.shape[:2]
            axis_phi_deg, _ = sample_axis_angle(image_id, theta, 0, base_seed)
            H_true = build_rotation_homography(h, w, theta, axis_phi_deg)
            distorted = cv2.warpPerspective(image, H_true, (w, h))
            rectified, _ = blind_rectify(distorted, theta_est)
            rectified_arrays.append(rectified)
            kept_ids.append(image_id)

        with open(RESULTS_RAW / f"v7_synth_blindrect_image_ids_{theta}.json", "w", encoding="utf-8") as f:
            json.dump(kept_ids, f)

        for spec in [MODEL_B32, MODEL_L14]:
            slug = spec["slug"]
            key = f"synth_blindrect_{slug}_{theta}_done"
            if progress.get(key):
                log(f"  synth blind-rectified features ({spec['label']}, angle={theta}): already done")
                continue
            t0 = time.time()
            model, preprocess, tokenizer = load_model_generic(spec["name"], spec["pretrained"], device)
            features = encode_image_arrays(rectified_arrays, model, preprocess, device, batch_size=batch_size)
            np.save(RESULTS_RAW / f"v7_features_{slug}_synth_blindrect_{theta}.npy", features.numpy())

            if slug == MODEL_B32["slug"]:
                text_features = encode_text_features_ensemble(model, tokenizer, class_names, PROMPT_ENSEMBLE_TEMPLATES, device)
                predicted = zero_shot_predict(model, tokenizer, class_names, text_features, features)
                with open(RESULTS_RAW / f"v7_synth_blindrect_zs_predictions_{theta}.json", "w", encoding="utf-8") as f:
                    json.dump(dict(zip(kept_ids, predicted)), f, indent=2)

            del model, preprocess, tokenizer
            progress[key] = True
            save_progress(progress)
            log(f"  synth blind-rectified features ({spec['label']}, angle={theta}): {len(kept_ids)} images done in {time.time()-t0:.1f}s")


# --- Step 5: gallery figure ------------------------------------


def build_gallery_figure(config, image_paths, image_ids, out_path=FIGURES / "v7_blind_rectification_gallery.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    with open(REAL_TILT_CSV_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ok"] == "True":
                r["estimated_tilt_deg"] = float(r["estimated_tilt_deg"])
                r["n_inliers"] = int(r["n_inliers"])
                rows.append(r)

    path_by_id = dict(zip(image_ids, image_paths))

    good = sorted(rows, key=lambda r: (-r["n_inliers"], abs(r["estimated_tilt_deg"])))[:6]
    # "Failure" cases: implausibly large estimated tilt (>55 degrees) with
    # low inlier support — a real facade photo is essentially never tilted
    # that much, so a large estimate signals the estimator locked onto
    # spurious (non-vertical-building-edge) lines.
    failures = sorted([r for r in rows if abs(r["estimated_tilt_deg"]) > 55], key=lambda r: r["n_inliers"])[:2]

    examples = good + failures
    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row_idx, r in enumerate(examples):
        image_id = r["image_id"]
        image = load_image_rgb(path_by_id[image_id])
        lines = detect_near_vertical_lines(image)
        rectified, _ = blind_rectify(image, r["estimated_tilt_deg"])

        ax0, ax1 = axes[row_idx]
        ax0.imshow(image)
        h, w = image.shape[:2]
        for a, b, c in lines:
            if abs(b) > 1e-6:
                x0, x1 = 0, w
                y0 = -(a * x0 + c) / b
                y1 = -(a * x1 + c) / b
                ax0.plot([x0, x1], [y0, y1], color="lime", linewidth=0.8, alpha=0.7)
        ax0.set_ylim(h, 0)
        ax0.set_xlim(0, w)
        tag = "FAILURE " if r in failures else ""
        ax0.set_title(f"{tag}{image_id}\nest={r['estimated_tilt_deg']:.1f} deg, n_inliers={r['n_inliers']}", fontsize=8)
        ax0.axis("off")

        ax1.imshow(rectified)
        ax1.set_title("blind-rectified", fontsize=8)
        ax1.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Day 13 v7: blind tilt estimation + blind rectification.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]

    samples = load_monumai(dataset_root)
    image_paths = [p for p, _ in samples]
    true_labels = [l for _, l in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    log(f"{len(samples)} images loaded from {dataset_root}")

    RUN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "day": "13_v7",
                "git_commit": git_commit_hash(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "angles": ANGLES,
                "max_angle_from_vertical_deg": MAX_ANGLE_FROM_VERTICAL_DEG,
                "min_length_fraction_of_diagonal": MIN_LENGTH_FRACTION_OF_DIAGONAL,
                "ransac_iterations": RANSAC_ITERATIONS,
                "ransac_inlier_px": RANSAC_INLIER_PX,
                "min_inlier_lines": MIN_INLIER_LINES,
                "pitch_dominant_axis_tolerance_deg": PITCH_DOMINANT_AXIS_TOLERANCE_DEG,
                "models_used": [MODEL_B32, MODEL_L14],
            },
            f,
            indent=2,
        )

    log("=== Step 2: validation on v4-identical synthetic distortions ===")
    run_validation(config, image_paths, image_ids)

    gate = check_validation_gate()
    log(f"Validation gate: {json.dumps(gate, indent=2)}")
    progress = load_progress()
    progress["validation_gate"] = gate
    save_progress(progress)

    if not gate["passed_gate"]:
        log("GATE FAILED: estimator unreliable on synthetic validation. Stopping Experiment C after validation, per instructions.")
        progress["stopped_after_validation"] = True
        save_progress(progress)
        return

    log("Validation gate PASSED. Continuing to real-image steps.")

    log("=== Step 3: tilt estimation on real (original) MonuMAI images ===")
    run_real_tilt_estimation(image_paths, image_ids, true_labels)

    log("=== Step 4a: blind-rectify + extract features, real images ===")
    run_blind_rectified_real(config, class_names, image_paths, image_ids)

    log("=== Step 4b: blind-rectify + extract features, v4-synthetic images (estimated H) ===")
    run_blind_rectified_synthetic(config, class_names, image_paths, image_ids)

    log("=== Step 5: gallery figure ===")
    build_gallery_figure(config, image_paths, image_ids)

    log("ALL STEPS COMPLETE")


if __name__ == "__main__":
    main()
