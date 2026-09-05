"""Day 6 — analysis, metrics, plots, ablations, failure gallery.

Turns results/raw/day5_full_grid.csv into every remaining figure/table the
paper needs. Per the plan, nothing new gets computed after today except the
one explicitly-listed prompt-sensitivity sweep (medium severity, 3 prompts)
and the confidence metric, both of which need a fresh CLIP forward pass
because Day 5's CSV only stored the top1 (predicted-class) confidence, not
the full per-class softmax vector needed to know "confidence assigned to the
TRUE class" when a prediction was wrong. Recovery %, the severity curve, and
the per-class breakdown are all computed directly from Day 5's existing CSV
— no rerun, exactly reusing the frozen Day 5 source of truth.

This module does NOT modify src/experiment_runner.py (Day 5's runner stays
frozen as the artifact that produced day5_full_grid.csv) — it is a separate,
read-mostly analysis layer, reusing distortion/rectification/CLIP functions
without altering them.
"""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from clip_zero_shot import (
    PROMPT_TEMPLATE_SLUGS,
    PROMPT_TEMPLATES_PILOT,
    encode_image_arrays,
    encode_image_features,
    encode_text_features,
    load_config,
    load_model,
)
from data_loader import load_monumai
from distortion import distort, image_id_from_path, load_image_rgb
from metrics import per_class_accuracy, top1_accuracy
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

SEVERITIES = ["mild", "medium", "severe"]
CONDITIONS = ["clean", "distorted", "rectified"]

DAY5_CSV = "results/raw/day5_full_grid.csv"


def load_day5_rows():
    with open(DAY5_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["correct"] = row["correct"] == "True"
    return rows


# --- Task 1: recovery percentage ------------------------------------------------


def compute_recovery(rows):
    """Return {severity: {clean, distorted, rectified, recovery_pct, denominator, notes}}.

    recovery = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) * 100,
    computed from exact integer correct-counts (not the already-rounded
    accuracy floats) to avoid compounding rounding error in a ratio of two
    small differences.
    """
    result = {}
    for severity in SEVERITIES:
        counts = {}
        n = None
        for condition in CONDITIONS:
            subset = [r for r in rows if r["severity"] == severity and r["condition"] == condition]
            n = len(subset)
            counts[condition] = sum(1 for r in subset if r["correct"])

        denom_count = counts["clean"] - counts["distorted"]
        numer_count = counts["rectified"] - counts["distorted"]
        recovery_pct = (numer_count / denom_count * 100) if denom_count != 0 else None

        result[severity] = {
            "n": n,
            "clean_acc": counts["clean"] / n,
            "distorted_acc": counts["distorted"] / n,
            "rectified_acc": counts["rectified"] / n,
            "clean_correct": counts["clean"],
            "distorted_correct": counts["distorted"],
            "rectified_correct": counts["rectified"],
            "denom_count": denom_count,
            "numer_count": numer_count,
            "recovery_pct": recovery_pct,
        }
    return result


def save_recovery_table(recovery, output_path):
    lines = ["# Day 6 — recovery percentage per severity", ""]
    lines.append(
        "recovery = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) x 100, "
        "computed from exact correct-image counts (out of n=1514 per severity)."
    )
    lines.append("")
    lines.append(
        "| Severity | Clean correct | Distorted correct | Rectified correct | "
        "Denominator (clean-distorted) | Numerator (rectified-distorted) | Recovery % |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for severity in SEVERITIES:
        r = recovery[severity]
        recovery_str = f"{r['recovery_pct']:.1f}%" if r["recovery_pct"] is not None else "undefined (denominator=0)"
        lines.append(
            f"| {severity} | {r['clean_correct']} | {r['distorted_correct']} | {r['rectified_correct']} | "
            f"{r['denom_count']} | {r['numer_count']} | {recovery_str} |"
        )
    lines.append("")
    lines.append("## Caution on interpreting mild severity's recovery %")
    lines.append("")
    lines.append(
        "At mild severity the denominator (clean minus distorted correct-count) is only "
        f"{recovery['mild']['denom_count']} images out of 1,514 — distortion barely hurts "
        "accuracy at this severity to begin with. Dividing the numerator by such a small "
        "denominator produces a recovery percentage "
        f"({recovery['mild']['recovery_pct']:.1f}%) that looks large and precise but is "
        "extremely sensitive to a handful of individual image outcomes: flipping the "
        "prediction on just 1-2 images would swing this number by roughly 10-20 percentage "
        "points. Per Day 5's McNemar test (results/tables/day5_mcnemar.md), the "
        "rectified-vs-distorted gap at mild severity is not statistically distinguishable "
        "from chance (p=0.52) in the first place. **Reporting mild severity's recovery % as "
        "a precise figure in the paper would overstate the precision the data actually "
        "supports** — it should be presented as \"not meaningful / statistically "
        "indistinguishable from zero effect\" at mild severity, with medium and severe as "
        "the severities where the recovery-% metric is actually informative (both have a "
        "much larger, statistically significant denominator and effect)."
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Task 2: severity curve ------------------------------------------------


def save_severity_curve(recovery, output_path):
    import matplotlib.pyplot as plt

    x_labels = ["clean (0)", "mild", "medium", "severe"]
    x = list(range(len(x_labels)))

    clean_y = [recovery["mild"]["clean_acc"]] * len(x_labels)  # clean is identical at every severity
    distorted_y = [recovery["mild"]["clean_acc"]] + [recovery[s]["distorted_acc"] for s in SEVERITIES]
    rectified_y = [recovery["mild"]["clean_acc"]] + [recovery[s]["rectified_acc"] for s in SEVERITIES]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, clean_y, marker="o", label="clean", color="tab:green")
    ax.plot(x, distorted_y, marker="o", label="distorted", color="tab:red")
    ax.plot(x, rectified_y, marker="o", label="rectified", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xlabel("Distortion severity")
    ax.set_title("Day 6 — accuracy vs. severity (n=1514 per point)")
    ax.set_ylim(0.5, 0.65)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# --- Task 6: per-class breakdown ------------------------------------------------


def save_per_class_table(rows, class_names, output_path):
    lines = ["# Day 6 — per-class breakdown: does rectification help all 4 styles equally?", ""]

    lines.append("## Per-class accuracy by condition, pooled across all 3 severities")
    lines.append("")
    lines.append("| Style | Clean | Distorted | Rectified | Recovery % (pooled) |")
    lines.append("|---|---|---|---|---|")
    for class_name in class_names:
        class_rows = [r for r in rows if r["true_label"] == class_name]
        accs = {}
        counts = {}
        n = None
        for condition in CONDITIONS:
            subset = [r for r in class_rows if r["condition"] == condition]
            n = len(subset)
            counts[condition] = sum(1 for r in subset if r["correct"])
            accs[condition] = counts[condition] / n if n else None
        denom = counts["clean"] - counts["distorted"]
        numer = counts["rectified"] - counts["distorted"]
        recovery_str = f"{numer / denom * 100:.1f}%" if denom != 0 else "undefined"
        lines.append(
            f"| {class_name} | {accs['clean']:.3f} | {accs['distorted']:.3f} | "
            f"{accs['rectified']:.3f} | {recovery_str} |"
        )

    lines.append("")
    lines.append("## Per-class accuracy by condition, broken down per severity")
    lines.append("")
    lines.append("| Style | Severity | Clean | Distorted | Rectified |")
    lines.append("|---|---|---|---|---|")
    for class_name in class_names:
        for severity in SEVERITIES:
            subset_by_cond = {}
            for condition in CONDITIONS:
                subset = [
                    r for r in rows
                    if r["true_label"] == class_name and r["severity"] == severity and r["condition"] == condition
                ]
                subset_by_cond[condition] = sum(1 for r in subset if r["correct"]) / len(subset) if subset else None
            lines.append(
                f"| {class_name} | {severity} | {subset_by_cond['clean']:.3f} | "
                f"{subset_by_cond['distorted']:.3f} | {subset_by_cond['rectified']:.3f} |"
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- Task 5: failure gallery ------------------------------------------------


def find_hurt_and_helped_candidates(rows):
    """Return (hurt, helped): lists of {severity, image_id, true_label, distorted_pred, rectified_pred}.

    hurt = distorted correct, rectified wrong (rectification broke a
    previously-correct prediction). helped = the reverse. Iterates in a
    fixed sort order (severity, image_id) so any seeded sample drawn from
    these lists is reproducible regardless of CSV row order.
    """
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["severity"], row["image_id"])][row["condition"]] = row

    hurt, helped = [], []
    for key in sorted(by_key):
        severity, image_id = key
        conds = by_key[key]
        d, r = conds["distorted"], conds["rectified"]
        record = {
            "severity": severity,
            "image_id": image_id,
            "true_label": d["true_label"],
            "distorted_pred": d["predicted_label"],
            "rectified_pred": r["predicted_label"],
        }
        if d["correct"] and not r["correct"]:
            hurt.append(record)
        elif not d["correct"] and r["correct"]:
            helped.append(record)
    return hurt, helped


def class_breakdown(records, class_names):
    counts = defaultdict(int)
    for record in records:
        counts[record["true_label"]] += 1
    return {name: counts.get(name, 0) for name in class_names}


def build_failure_gallery(
    hurt_candidates,
    dataset_root="data/monumai",
    output_path="figures/day6_failure_gallery.png",
    n_images=8,
    seed=42,
):
    """8 side-by-sides (distorted, rectified) where rectification broke a
    previously-correct prediction. Sampled uniformly at random (not
    stratified by class) from the full candidate pool, per the plan's "do not
    cherry-pick" rule — whatever class mix results is itself part of the
    finding (see experiments/logs/day6.md for the full per-class breakdown).
    """
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    chosen = rng.sample(hurt_candidates, n_images)

    samples = {image_id_from_path(p, dataset_root): (p, label) for p, label in load_monumai(dataset_root)}

    fig, axes = plt.subplots(n_images, 2, figsize=(7, 3.4 * n_images))
    for row, record in enumerate(chosen):
        image_path, _ = samples[record["image_id"]]
        image = load_image_rgb(image_path)
        distorted, H = distort(image, record["severity"], 42)
        mask = compute_valid_mask(image.shape, H)
        rectified_raw = rectify_oracle(distorted, H)
        rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)

        axes[row, 0].imshow(distorted)
        axes[row, 0].set_title(
            f"distorted — pred: {record['distorted_pred']} (correct)", fontsize=8
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(
            f"{record['true_label']}\n({record['image_id']}, {record['severity']})", fontsize=7
        )

        axes[row, 1].imshow(rectified)
        axes[row, 1].set_title(
            f"rectified — pred: {record['rectified_pred']} (WRONG)", fontsize=8
        )
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

    fig.suptitle("Day 6 — 8 cases where rectification broke a correct prediction (seed=42 sample)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
    return chosen


# --- Tasks 3 & 4: confidence + prompt-sensitivity (require a fresh forward pass) ------


def run_confidence_and_prompt_sweep(config, device="cpu", image_limit=None, batch_size=32):
    """One combined rerun covering both Day 6 tasks that need fresh CLIP scores.

    Day 5's CSV only stored top1 (predicted-class) confidence, not the full
    per-class softmax vector, so "confidence on the TRUE class" (needed even
    when the prediction was wrong) cannot be recovered from it — this
    requires one fresh forward pass. Rather than running that twice (once for
    the confidence metric, once for the prompt-sensitivity sweep the plan
    explicitly asks for), both are computed from a single pass: image
    features (the expensive part) are encoded once per image per
    condition/severity, exactly reproducing Day 5's pixels (same seed=42,
    same distort/rectify/border-policy code); the default prompt is
    evaluated at all 3 severities (needed for the confidence metric, and to
    reproduce Day 5's own accuracy numbers as an internal consistency check);
    the 2 non-default prompts are evaluated ONLY at medium severity (all
    that's needed for the prompt-sensitivity table, and all the plan asks
    for) — reusing the same cached image features, so this costs only a few
    extra cheap matmuls, not additional CLIP image encoding.
    """
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    default_prompt = config["prompt_template"]
    seed = config["seed"]

    samples = load_monumai(dataset_root)
    if image_limit is not None:
        samples = samples[:image_limit]
    image_paths = [path for path, _ in samples]
    true_labels = [label for _, label in samples]

    model, preprocess, tokenizer = load_model(device)

    text_features_by_prompt = {
        prompt: encode_text_features(model, tokenizer, class_names, prompt, device)
        for prompt in PROMPT_TEMPLATES_PILOT
    }

    clean_features, clean_ok_paths, clean_failed = encode_image_features(
        image_paths, model, preprocess, device, batch_size=batch_size
    )
    ok_index = {path: i for i, path in enumerate(image_paths)}
    clean_true_labels = [true_labels[ok_index[p]] for p in clean_ok_paths]
    clean_image_ids = [image_id_from_path(p, dataset_root) for p in clean_ok_paths]

    rows = []

    def classify_and_emit(features, true_labels_local, image_ids_local, severity, condition, prompts_to_use):
        for prompt in prompts_to_use:
            text_features = text_features_by_prompt[prompt]
            with __import__("torch").no_grad():
                logits = 100.0 * features @ text_features.T
                probs = logits.softmax(dim=-1)
            prompt_id = PROMPT_TEMPLATE_SLUGS[prompt]
            for i, image_id in enumerate(image_ids_local):
                class_probs = probs[i].tolist()
                true_label = true_labels_local[i]
                true_idx = class_names.index(true_label)
                best_idx = max(range(len(class_names)), key=lambda j: class_probs[j])
                predicted_label = class_names[best_idx]
                rows.append(
                    {
                        "image_id": image_id,
                        "true_label": true_label,
                        "severity": severity,
                        "condition": condition,
                        "prompt_id": prompt_id,
                        "predicted_label": predicted_label,
                        "correct": predicted_label == true_label,
                        "top1_confidence": class_probs[best_idx],
                        "true_class_confidence": class_probs[true_idx],
                    }
                )

    for severity in SEVERITIES:
        distorted_arrays, rectified_arrays = [], []
        severity_image_ids, severity_true_labels = [], []
        for path in clean_ok_paths:
            image_id = image_id_from_path(path, dataset_root)
            image = load_image_rgb(path)
            distorted, H = distort(image, severity, seed)
            mask = compute_valid_mask(image.shape, H)
            rectified_raw = rectify_oracle(distorted, H)
            rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
            distorted_arrays.append(distorted)
            rectified_arrays.append(rectified)
            severity_image_ids.append(image_id)
            severity_true_labels.append(true_labels[ok_index[path]])

        distorted_features = encode_image_arrays(distorted_arrays, model, preprocess, device, batch_size)
        rectified_features = encode_image_arrays(rectified_arrays, model, preprocess, device, batch_size)

        prompts_here = PROMPT_TEMPLATES_PILOT if severity == "medium" else [default_prompt]

        classify_and_emit(clean_features, clean_true_labels, clean_image_ids, severity, "clean", prompts_here)
        classify_and_emit(
            distorted_features, severity_true_labels, severity_image_ids, severity, "distorted", prompts_here
        )
        classify_and_emit(
            rectified_features, severity_true_labels, severity_image_ids, severity, "rectified", prompts_here
        )
        print(f"severity={severity}: done ({len(prompts_here)} prompt(s))")

    return rows


def save_scores_csv(rows, output_path):
    fieldnames = [
        "image_id", "true_label", "severity", "condition", "prompt_id",
        "predicted_label", "correct", "top1_confidence", "true_class_confidence",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confidence_table_and_plot(scores_rows, default_prompt_id, table_path, plot_path):
    import matplotlib.pyplot as plt

    default_rows = [r for r in scores_rows if r["prompt_id"] == default_prompt_id]

    lines = ["# Day 6 — mean CLIP confidence on the TRUE class, per condition per severity", ""]
    lines.append("| Severity | Clean | Distorted | Rectified |")
    lines.append("|---|---|---|---|")
    means = {}
    for severity in SEVERITIES:
        means[severity] = {}
        for condition in CONDITIONS:
            subset = [r for r in default_rows if r["severity"] == severity and r["condition"] == condition]
            means[severity][condition] = sum(r["true_class_confidence"] for r in subset) / len(subset)
        lines.append(
            f"| {severity} | {means[severity]['clean']:.4f} | {means[severity]['distorted']:.4f} | "
            f"{means[severity]['rectified']:.4f} |"
        )
    Path(table_path).parent.mkdir(parents=True, exist_ok=True)
    Path(table_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    x_labels = ["clean (0)", "mild", "medium", "severe"]
    x = list(range(len(x_labels)))
    clean_y = [means["mild"]["clean"]] * len(x_labels)
    distorted_y = [means["mild"]["clean"]] + [means[s]["distorted"] for s in SEVERITIES]
    rectified_y = [means["mild"]["clean"]] + [means[s]["rectified"] for s in SEVERITIES]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, clean_y, marker="o", label="clean", color="tab:green")
    ax.plot(x, distorted_y, marker="o", label="distorted", color="tab:red")
    ax.plot(x, rectified_y, marker="o", label="rectified", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean CLIP confidence on the TRUE class")
    ax.set_xlabel("Distortion severity")
    ax.set_title("Day 6 — mean confidence on correct class vs. severity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")
    return means


def save_prompt_sensitivity_table(scores_rows, output_path):
    medium_rows = [r for r in scores_rows if r["severity"] == "medium"]
    prompt_ids = sorted(set(r["prompt_id"] for r in medium_rows))

    lines = ["# Day 6 — prompt-sensitivity ablation (medium severity only, 3 prompts)", ""]
    lines.append(
        "Goal: check whether the DIRECTION of the finding (rectification recovers some "
        "accuracy lost to distortion) holds regardless of prompt choice — not that the "
        "exact numbers match Day 5's default-prompt run."
    )
    lines.append("")
    lines.append("| Prompt | Clean | Distorted | Rectified | clean >= rectified >= distorted? |")
    lines.append("|---|---|---|---|---|")
    for prompt_id in prompt_ids:
        accs = {}
        for condition in CONDITIONS:
            subset = [r for r in medium_rows if r["prompt_id"] == prompt_id and r["condition"] == condition]
            accs[condition] = sum(1 for r in subset if r["correct"]) / len(subset)
        holds = "yes" if accs["clean"] >= accs["rectified"] >= accs["distorted"] else "NO"
        lines.append(
            f"| `{prompt_id}` | {accs['clean']:.3f} | {accs['distorted']:.3f} | {accs['rectified']:.3f} | {holds} |"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
