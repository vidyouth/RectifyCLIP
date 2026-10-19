"""Day 9 v3 — two control/diagnostic experiments, reusing v2's exact pipeline and seeds.

Does NOT import or modify src/experiment_runner_v2.py or any v1/v2 output —
reuses the same lower-level building blocks v2 used (distortion.derive_seed,
distortion.distort, rectification.rectify_oracle/apply_border_policy/
compute_valid_mask, clip_zero_shot's model loading) with the SAME base_seed
and the SAME derive_seed(image_id, replicate, base_seed) formula, so every
v3 "distorted" image is pixel-identical to its v2 counterpart and v3 rows
pair 1:1 with v2 rows by (image_id, severity, replicate).

Experiment 1 — mismatched-H control
------------------------------------
Question: does rectification recover accuracy because it undoes THIS image's
specific distortion, or would rectifying with basically any plausible-looking
homography do about as well (e.g. because warpPerspective's resampling/
smoothing alone helps CLIP, independent of whether the geometry is actually
corrected)? Each distorted image is rectified using a DIFFERENT image's
homography (same severity, same replicate) instead of its own. The pairing
is a deterministic cyclic shift per (severity, replicate) group — guaranteed
to never pair an image with itself, fully reproducible from base_seed.

Experiment 3 — noisy oracle
------------------------------------
Question: how much does rectification accuracy degrade as knowledge of the
true homography becomes imperfect (as it would be for real, blind
rectification via line/vanishing-point detection, which is out of scope per
CLAUDE.md but whose error sensitivity this diagnostic can still probe)? The
4 (src_corner, dst_corner) correspondences `distort()` used to build the true
H are reused, but the dst_corner half (representing noisy localization of
where the distorted corners actually are) is perturbed by 0/1/2/4% of image
size before rebuilding the rectifying homography via
cv2.getPerspectiveTransform. 0% is the exact oracle case and is NOT
recomputed — those rows are pulled directly from the existing v2 CSV
(read-only) since they are pixel-identical.

Experiment 2 (attractor calibration) is NOT implemented here: it requires the
full 4-class softmax per prediction, which v2's CSV does not store (only
top1_confidence) — see experiments/logs/day9_v3.md for why this is reported,
not worked around by rerunning v2.
"""

import argparse
import csv
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

from clip_zero_shot import (
    encode_image_arrays,
    encode_text_features,
    load_config,
    load_model,
    predict_from_features,
)
from data_loader import load_monumai
from distortion import derive_seed, distort, image_id_from_path, load_image_rgb
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

SEVERITIES_V3 = ["medium", "severe"]  # mild excluded per the user's spec for experiments 1 and 3


def stable_string_seed(key_string, base_seed):
    """Same sha256-based scheme as distortion.derive_seed, generalized to an
    arbitrary string key (derive_seed is restricted to (image_id,
    replicate_index) pairs) — used here to seed the mismatched-H partner
    permutation and the per-(image, pct) corner-noise draws independently."""
    key = f"{base_seed}:{key_string}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big")


# --- Experiment 1: mismatched-H control ------------------------------------


def build_wrong_partner_map(image_ids, severity, replicate, base_seed):
    """Deterministic derangement: pairs each image_id with a DIFFERENT
    image_id, via a random (but fixed, reproducible) cyclic shift of the
    image order, so no image is ever paired with itself and every image at
    this (severity, replicate) gets exactly one partner."""
    n = len(image_ids)
    seed = stable_string_seed(f"wrongH_partner:{severity}", base_seed)
    # replicate folded into the shift itself (not just the seed) so the
    # mapping actually differs per replicate, not just the RNG internals.
    rng = np.random.RandomState(seed + replicate)
    shift = 1 + int(rng.randint(0, n - 1))
    return {image_ids[i]: image_ids[(i + shift) % n] for i in range(n)}


def run_mismatched_h(config, device="cpu", image_limit=None, n_replicates=5, batch_size=32, verbose=True):
    """For each image, severity (medium/severe), replicate: rectify the
    distorted image with a DIFFERENT image's homography instead of its own.

    Returns (rows, timing). rows have condition="wrong_h_rectified".
    "distorted" and "rectified" (oracle) accuracy for comparison are NOT
    recomputed here — they are pixel-identical to v2's and should be read
    from results/raw/day8_v2_grid.csv directly by the analysis script.
    """
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    prompt_template = config["prompt_template"]
    base_seed = config["seed"]

    timing = {}
    t0 = time.time()
    samples = load_monumai(dataset_root)
    if image_limit is not None:
        samples = samples[:image_limit]
    image_paths = [path for path, _ in samples]
    true_labels = [label for _, label in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    true_label_by_id = dict(zip(image_ids, true_labels))

    model, preprocess, tokenizer = load_model(device)
    text_features = encode_text_features(model, tokenizer, class_names, prompt_template, device)
    timing["setup_seconds"] = time.time() - t0
    if verbose:
        print(f"Loaded {len(samples)} images, model ready ({timing['setup_seconds']:.1f}s)")

    rows = []
    for severity in SEVERITIES_V3:
        for replicate in range(n_replicates):
            t_rep = time.time()
            partner_map = build_wrong_partner_map(image_ids, severity, replicate, base_seed)

            # Pass A: distort every image with ITS OWN v2-identical seed, caching
            # (distorted_array, H, shape) by image_id so Pass B can look up any
            # image's H as another image's "wrong" rectifying homography without
            # re-running distort() a second time per lookup.
            distorted_by_id = {}
            H_by_id = {}
            for path, image_id in zip(image_paths, image_ids):
                own_seed = derive_seed(image_id, replicate, base_seed)
                image = load_image_rgb(path)
                distorted, H = distort(image, severity, own_seed)
                distorted_by_id[image_id] = distorted
                H_by_id[image_id] = H

            # Pass B: rectify each image's distorted array with its PARTNER's H.
            rectified_arrays = []
            rep_image_ids = []
            for image_id in image_ids:
                partner_id = partner_map[image_id]
                wrong_H = H_by_id[partner_id]
                distorted = distorted_by_id[image_id]
                mask = compute_valid_mask(distorted.shape, wrong_H)
                rectified_raw = rectify_oracle(distorted, wrong_H)
                rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
                rectified_arrays.append(rectified)
                rep_image_ids.append(image_id)

            rectified_features = encode_image_arrays(
                rectified_arrays, model, preprocess, device, batch_size=batch_size
            )
            predictions = predict_from_features(rectified_features, text_features, class_names)

            for image_id, (pred_label, confidence) in zip(rep_image_ids, predictions):
                true_label = true_label_by_id[image_id]
                rows.append(
                    {
                        "image_id": image_id,
                        "true_label": true_label,
                        "severity": severity,
                        "replicate": replicate,
                        "condition": "wrong_h_rectified",
                        "partner_image_id": partner_map[image_id],
                        "predicted_label": pred_label,
                        "correct": pred_label == true_label,
                        "top1_confidence": confidence,
                        "base_seed": base_seed,
                    }
                )

            timing[f"{severity}_r{replicate}_seconds"] = time.time() - t_rep
            if verbose:
                print(f"  severity={severity} replicate={replicate}: {len(rep_image_ids)} images in {timing[f'{severity}_r{replicate}_seconds']:.1f}s")

    return rows, timing


# --- Experiment 3: noisy oracle ------------------------------------


PERTURBATION_PCTS = [0, 1, 2, 4]
NOISY_REPLICATES = [0, 1, 2]  # "3 reps" per the spec; reuses v2's replicate 0/1/2 derive_seed values


def perturb_dst_corners(true_dst_corners, h, w, pct, seed):
    """Perturb the 4 distorted-corner correspondences by +/- pct% of image
    size (symmetric, per-corner-independent), representing imperfect
    localization of where the distorted corners actually are — the
    noisy-oracle analogue of distortion.distort()'s own corner-offset draw.
    pct=0 returns true_dst_corners unchanged (no RNG call), matching the
    exact-oracle case exactly.
    """
    if pct == 0:
        return true_dst_corners.copy()
    rng = np.random.RandomState(seed)
    max_dx = (pct / 100.0) * w
    max_dy = (pct / 100.0) * h
    unit_offsets = rng.uniform(-1.0, 1.0, size=(4, 2))
    noise = unit_offsets * np.array([max_dx, max_dy])
    return (true_dst_corners + noise).astype(np.float32)


def run_noisy_oracle(config, device="cpu", image_limit=None, pcts=None, batch_size=32, verbose=True):
    """For each image, severity (medium/severe), replicate in {0,1,2}, and
    each non-zero perturbation level in `pcts`: rebuild the rectifying
    homography from corner correspondences perturbed by that % of image
    size, rectify, and classify. pct=0 is intentionally excluded here — it
    is read directly from v2's existing "rectified" rows by the analysis
    script, since it is pixel-identical to the exact oracle case.

    Returns (rows, timing). rows have condition="noisy_rectified" and a
    "perturbation_pct" field.
    """
    if pcts is None:
        pcts = [p for p in PERTURBATION_PCTS if p != 0]

    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    prompt_template = config["prompt_template"]
    base_seed = config["seed"]

    timing = {}
    t0 = time.time()
    samples = load_monumai(dataset_root)
    if image_limit is not None:
        samples = samples[:image_limit]
    image_paths = [path for path, _ in samples]
    true_labels = [label for _, label in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]

    model, preprocess, tokenizer = load_model(device)
    text_features = encode_text_features(model, tokenizer, class_names, prompt_template, device)
    timing["setup_seconds"] = time.time() - t0
    if verbose:
        print(f"Loaded {len(samples)} images, model ready ({timing['setup_seconds']:.1f}s)")

    rows = []
    for severity in SEVERITIES_V3:
        for replicate in NOISY_REPLICATES:
            t_rep = time.time()

            # Pass A: distort every image once (v2-identical seed/pixels),
            # keeping the true H's own src/dst corners so each pct level's
            # noisy H can be rebuilt from the SAME true correspondences.
            cache = []  # list of (image_id, true_label, distorted, src_corners, true_dst_corners, h, w)
            for path, image_id, true_label in zip(image_paths, image_ids, true_labels):
                own_seed = derive_seed(image_id, replicate, base_seed)
                image = load_image_rgb(path)
                h, w = image.shape[:2]
                distorted, H = distort(image, severity, own_seed)
                # Recover the exact corner correspondences distort() used:
                # src_corners are fixed (the 4 image corners); dst_corners are
                # found by applying H to src_corners (round-trips exactly,
                # since H was built FROM these exact correspondences).
                src_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                src_h = cv2.convertPointsToHomogeneous(src_corners).reshape(-1, 3).T
                dst_h = H @ src_h
                dst_corners = (dst_h[:2] / dst_h[2]).T.astype(np.float32)
                cache.append((image_id, true_label, distorted, src_corners, dst_corners, h, w))

            for pct in pcts:
                rectified_arrays = []
                rep_image_ids, rep_true_labels = [], []
                for image_id, true_label, distorted, src_corners, true_dst_corners, h, w in cache:
                    noise_seed = stable_string_seed(f"noisyH:{image_id}:{severity}:{pct}", base_seed) + replicate
                    noisy_dst_corners = perturb_dst_corners(true_dst_corners, h, w, pct, noise_seed)
                    H_noisy = cv2.getPerspectiveTransform(src_corners, noisy_dst_corners)
                    mask = compute_valid_mask(distorted.shape, H_noisy)
                    rectified_raw = rectify_oracle(distorted, H_noisy)
                    rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
                    rectified_arrays.append(rectified)
                    rep_image_ids.append(image_id)
                    rep_true_labels.append(true_label)

                rectified_features = encode_image_arrays(
                    rectified_arrays, model, preprocess, device, batch_size=batch_size
                )
                predictions = predict_from_features(rectified_features, text_features, class_names)

                for image_id, true_label, (pred_label, confidence) in zip(
                    rep_image_ids, rep_true_labels, predictions
                ):
                    rows.append(
                        {
                            "image_id": image_id,
                            "true_label": true_label,
                            "severity": severity,
                            "replicate": replicate,
                            "perturbation_pct": pct,
                            "condition": "noisy_rectified",
                            "predicted_label": pred_label,
                            "correct": pred_label == true_label,
                            "top1_confidence": confidence,
                            "base_seed": base_seed,
                        }
                    )

            timing[f"{severity}_r{replicate}_seconds"] = time.time() - t_rep
            if verbose:
                print(f"  severity={severity} replicate={replicate}: {len(pcts)} pct levels in {timing[f'{severity}_r{replicate}_seconds']:.1f}s")

    return rows, timing


# --- shared CSV I/O ------------------------------------


def save_rows_csv(rows, fieldnames, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


MISMATCHED_H_FIELDNAMES = [
    "image_id", "true_label", "severity", "replicate", "condition",
    "partner_image_id", "predicted_label", "correct", "top1_confidence", "base_seed",
]
NOISY_ORACLE_FIELDNAMES = [
    "image_id", "true_label", "severity", "replicate", "perturbation_pct", "condition",
    "predicted_label", "correct", "top1_confidence", "base_seed",
]


def main():
    parser = argparse.ArgumentParser(description="Day 9 v3 experiments.")
    parser.add_argument("--experiment", required=True, choices=["mismatched_h", "noisy_oracle"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-images", type=int, default=None, help="Limit to first N images (smoke test).")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    t_start = time.time()

    if args.experiment == "mismatched_h":
        rows, timing = run_mismatched_h(config, image_limit=args.n_images)
        output_csv = args.output_csv or "results/raw/v3_mismatched_h.csv"
        save_rows_csv(rows, MISMATCHED_H_FIELDNAMES, output_csv)
    else:
        rows, timing = run_noisy_oracle(config, image_limit=args.n_images)
        output_csv = args.output_csv or "results/raw/v3_noisy_oracle.csv"
        save_rows_csv(rows, NOISY_ORACLE_FIELDNAMES, output_csv)

    total_seconds = time.time() - t_start
    print(f"\nSaved {output_csv} ({len(rows)} rows)")
    print(f"Total wall-clock time: {total_seconds:.1f}s")


if __name__ == "__main__":
    main()
