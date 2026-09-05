"""Day 5 full experimental grid: image x severity x condition.

For every image, for each of the 3 severities, produces the distorted and
rectified (oracle, mean_fill border policy) versions, classifies all 3
conditions (clean, distorted, rectified) with frozen zero-shot CLIP, and logs
one row per (image, severity, condition) — including 3 separate "clean" rows
per image (one per severity), exactly as the plan specifies, so a clean vs.
distorted vs. rectified comparison can be made independently per severity
without a join.

Efficiency note (does not change any logged value): the "clean" condition is
identical across all 3 severities for a given image, so its CLIP forward pass
is computed once per image, not 3 times — the resulting prediction is simply
copied into all 3 severity rows for that image and condition. This mirrors
Day 2's "encode once, reuse" approach and is purely a compute optimization.

Config is read entirely from configs/default.yaml (model, prompt, class
names, seed, dataset root) — no new config fields are introduced here.
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

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
    distort,
    image_id_from_path,
    load_image_rgb,
    save_homography,
)
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

SEVERITIES = ["mild", "medium", "severe"]
ROW_FIELDNAMES = [
    "image_id",
    "true_label",
    "severity",
    "condition",
    "predicted_label",
    "correct",
    "top1_confidence",
    "prompt_id",
    "seed",
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


def run_grid(config, device="cpu", image_limit=None, batch_size=32, verbose=True):
    """Run the full (or a truncated, for image_limit) image x severity x condition grid.

    Returns (rows, failed_images, timing) where timing is a dict of wall-clock
    seconds for the major phases, and failed_images is a list of
    {"image_path": ..., "phase": ..., "reason": ...} dicts for anything
    skipped rather than crashing the whole run.
    """
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    prompt_template = config["prompt_template"]
    seed = config["seed"]
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

    # --- Clean condition: one CLIP forward pass per image, reused across all 3 severities ---
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

    # --- Distorted + rectified conditions, per severity ---
    ok_paths_set_index = {path: i for i, path in enumerate(clean_ok_paths)}

    for severity in SEVERITIES:
        t_severity = time.time()
        distorted_arrays = []
        rectified_arrays = []
        severity_image_ids = []
        severity_true_labels = []

        for path in clean_ok_paths:
            image_id = image_id_from_path(path, dataset_root)
            try:
                image = load_image_rgb(path)
                distorted, H = distort(image, severity, seed)
                mask = compute_valid_mask(image.shape, H)
                rectified_raw = rectify_oracle(distorted, H)
                rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
                save_homography(H, image_id, severity)
            except Exception as exc:  # noqa: BLE001 - one bad image must not kill a 13k-row run
                failed_images.append(
                    {"image_path": path, "phase": f"distort_rectify_{severity}", "reason": str(exc)}
                )
                continue

            distorted_arrays.append(distorted)
            rectified_arrays.append(rectified)
            severity_image_ids.append(image_id)
            severity_true_labels.append(true_labels[ok_index[path]])

        distorted_features = encode_image_arrays(
            distorted_arrays, model, preprocess, device, batch_size=batch_size
        )
        rectified_features = encode_image_arrays(
            rectified_arrays, model, preprocess, device, batch_size=batch_size
        )
        distorted_predictions = predict_from_features(distorted_features, text_features, class_names)
        rectified_predictions = predict_from_features(rectified_features, text_features, class_names)

        for image_id, true_label, (dist_pred, dist_conf), (rect_pred, rect_conf) in zip(
            severity_image_ids, severity_true_labels, distorted_predictions, rectified_predictions
        ):
            clean_true, clean_pred, clean_conf = clean_by_image_id[image_id]
            rows.append(
                {
                    "image_id": image_id,
                    "true_label": clean_true,
                    "severity": severity,
                    "condition": "clean",
                    "predicted_label": clean_pred,
                    "correct": clean_pred == clean_true,
                    "top1_confidence": clean_conf,
                    "prompt_id": prompt_id,
                    "seed": seed,
                }
            )
            rows.append(
                {
                    "image_id": image_id,
                    "true_label": true_label,
                    "severity": severity,
                    "condition": "distorted",
                    "predicted_label": dist_pred,
                    "correct": dist_pred == true_label,
                    "top1_confidence": dist_conf,
                    "prompt_id": prompt_id,
                    "seed": seed,
                }
            )
            rows.append(
                {
                    "image_id": image_id,
                    "true_label": true_label,
                    "severity": severity,
                    "condition": "rectified",
                    "predicted_label": rect_pred,
                    "correct": rect_pred == true_label,
                    "top1_confidence": rect_conf,
                    "prompt_id": prompt_id,
                    "seed": seed,
                }
            )

        timing[f"severity_{severity}_seconds"] = time.time() - t_severity
        if verbose:
            print(
                f"severity={severity}: {len(severity_image_ids)} images processed "
                f"in {timing[f'severity_{severity}_seconds']:.1f}s"
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


def save_run_config(config, output_path, extra=None):
    """Save the exact config + provenance (seed, prompt, git hash, timestamp) for this run as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": config["seed"],
        "prompt_template": config["prompt_template"],
        "class_names": config["class_names"],
        "dataset_root": config["dataset_root"],
        "model_name": config["model_name"],
        "pretrained_weights": config["pretrained_weights"],
        "severity_fractions": SEVERITY_FRACTIONS,
        "border_policy": CHOSEN_BORDER_POLICY,
        "git_commit_hash": git_commit_hash(),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_headline_table(rows):
    """Return {severity: {condition: accuracy}} for the 3 conditions."""
    headline = {}
    for severity in SEVERITIES:
        headline[severity] = {}
        for condition in ["clean", "distorted", "rectified"]:
            subset = [r for r in rows if r["severity"] == severity and r["condition"] == condition]
            correct = sum(1 for r in subset if r["correct"])
            headline[severity][condition] = correct / len(subset) if subset else None
    return headline


def save_headline_table(headline, output_path):
    """Write the clean/distorted/rectified accuracy-by-severity Markdown table, with the
    clean >= rectified >= distorted sanity check spelled out per row."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Day 5 headline results — accuracy by severity and condition", ""]
    lines.append("| Severity | Clean | Distorted | Rectified | clean >= rectified >= distorted? |")
    lines.append("|---|---|---|---|---|")
    for severity in SEVERITIES:
        acc = headline[severity]
        clean, distorted, rectified = acc["clean"], acc["distorted"], acc["rectified"]
        holds = "yes" if (clean >= rectified >= distorted) else "NO"
        lines.append(
            f"| {severity} | {clean:.3f} | {distorted:.3f} | {rectified:.3f} | {holds} |"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    """CLI entry point: run the full (or --n-images-limited, for a smoke test) Day 5 grid."""
    parser = argparse.ArgumentParser(description="Day 5 full experimental grid.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--n-images", type=int, default=None, help="Limit to the first N images (smoke test)."
    )
    parser.add_argument("--output-csv", default="results/raw/day5_full_grid.csv")
    parser.add_argument("--output-config-json", default="experiments/configs/day5_run.json")
    parser.add_argument("--output-headline", default="results/tables/day5_headline.md")
    args = parser.parse_args()

    config = load_config(args.config)
    t_start = time.time()
    rows, failed_images, timing = run_grid(config, image_limit=args.n_images)
    total_seconds = time.time() - t_start

    save_rows_csv(rows, args.output_csv)
    print(f"Saved {args.output_csv} ({len(rows)} rows)")

    save_run_config(
        config,
        args.output_config_json,
        extra={
            "n_images_requested": args.n_images,
            "n_rows_written": len(rows),
            "n_images_failed": len(failed_images),
            "total_wall_clock_seconds": total_seconds,
        },
    )
    print(f"Saved {args.output_config_json}")

    headline = compute_headline_table(rows)
    save_headline_table(headline, args.output_headline)
    print(f"Saved {args.output_headline}")

    print(f"\nTotal wall-clock time: {total_seconds:.1f}s")
    print(f"Failed images: {len(failed_images)}")
    for failure in failed_images:
        print(f"  SKIPPED: {failure['image_path']} (phase={failure['phase']}, reason={failure['reason']})")

    for severity, acc in headline.items():
        print(
            f"{severity}: clean={acc['clean']:.3f} distorted={acc['distorted']:.3f} "
            f"rectified={acc['rectified']:.3f}"
        )


if __name__ == "__main__":
    main()
