"""Day 12 v6 analysis — Experiments A (zero-shot), B (leakage-free probe), C (rectification vs augmentation).

Reads src/experiment_v6.py's cached outputs (results/raw/v6_zeroshot_predictions.csv
and results/raw/v6_features_<slug>_{clean,distorted_<angle>,rectified_<angle>}.npy)
and produces every v6 table/figure plus experiments/logs/day12_v6.md. Does not
touch v1-v5 code or outputs, and does not re-extract any feature — this file
is pure analysis over what experiment_v6.py already cached.

Experiment B / C's 5-fold split is NOT re-derived from v5's saved fold
assignment (v5 didn't save one) — it is regenerated with the identical
sklearn call v5 used (StratifiedKFold(n_splits=5, shuffle=True,
random_state=LINEAR_PROBE_SEED) on the same true_labels, in the same
load_monumai order), which is deterministic and therefore identical to v5's
own folds by construction, not by coincidence.

Experiment C's "clean-trained probe" predictions are computed once per fold
(inside run_probes_for_model) and reused as Experiment B's own leakage-free
probe result, rather than being fit twice — Experiment B is a subset of what
Experiment C already computes.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from data_loader import load_monumai
from distortion import image_id_from_path
from experiment_v5_models import (
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    LINEAR_PROBE_SEED,
    compute_metrics_with_ci,
)
from experiment_v6 import ANGLES, MODELS, RESULTS_RAW
from metrics import (
    balanced_accuracy,
    macro_f1,
    paired_mcnemar,
    per_class_accuracy,
    prediction_agreement,
    recovery_percentage,
    top1_accuracy,
)

RESULTS_TABLES = Path("results/tables")
FIGURES = Path("figures")
LOG_MD_PATH = Path("experiments/logs/day12_v6.md")
ZEROSHOT_CSV_PATH = RESULTS_RAW / "v6_zeroshot_predictions.csv"

RECOVERY_DENOM_FLAG_THRESHOLD = 0.02  # clean_acc - distorted_acc smaller than this -> flag, don't trust the ratio


def load_dataset(config):
    dataset_root = config["dataset_root"]
    samples = load_monumai(dataset_root)
    image_paths = [p for p, _ in samples]
    true_labels = [l for _, l in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    return image_ids, true_labels


def load_zeroshot_rows(path=ZEROSHOT_CSV_PATH):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["angle"] = int(row["angle"])
            row["correct"] = row["predicted_label"] == row["true_label"]
            rows.append(row)
    return rows


def rows_for(rows, model_slug=None, angle=None, condition=None):
    out = rows
    if model_slug is not None:
        out = [r for r in out if r["model_slug"] == model_slug]
    if angle is not None:
        out = [r for r in out if r["angle"] == angle]
    if condition is not None:
        out = [r for r in out if r["condition"] == condition]
    return out


def recovery_with_flag(rows, threshold=RECOVERY_DENOM_FLAG_THRESHOLD):
    """recovery_percentage + bootstrap CI, with the denominator (clean_acc -
    distorted_acc) checked and flagged if too close to zero to trust the
    resulting ratio, per the instruction to flag rather than report it."""
    from metrics import bootstrap_ci

    clean_acc = top1_accuracy([r for r in rows if r["condition"] == "clean"])
    distorted_acc = top1_accuracy([r for r in rows if r["condition"] == "distorted"])
    denom = clean_acc - distorted_acc
    point, lo, hi = bootstrap_ci(rows, recovery_percentage, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")
    return {"point": point, "lo": lo, "hi": hi, "denom": denom, "flagged": abs(denom) < threshold}


# --- Experiment B / C: leakage-free probes ------------------------------------


def make_folds(true_labels, n_splits=5, seed=LINEAR_PROBE_SEED):
    """Identical call to v5's StratifiedKFold — see module docstring."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = np.array(true_labels)
    return list(skf.split(np.zeros(len(y)), y)), y


def load_features(slug):
    clean = np.load(RESULTS_RAW / f"v6_features_{slug}_clean.npy")
    distorted = {a: np.load(RESULTS_RAW / f"v6_features_{slug}_distorted_{a}.npy") for a in ANGLES}
    rectified = {a: np.load(RESULTS_RAW / f"v6_features_{slug}_rectified_{a}.npy") for a in ANGLES}
    return clean, distorted, rectified


def run_probes_for_model(clean, distorted, rectified, y, folds, seed=LINEAR_PROBE_SEED):
    """Fit, per fold, (a) a probe on that fold's training CLEAN features only
    (Experiment B / C's "clean-trained" probe) and (b) a probe on that
    fold's training clean + distorted features pooled over all angles
    (Experiment C's "augmentation-trained" probe). Predict both on the held-
    out fold's clean/distorted/rectified features. A held-out image's
    features never enter either training set in any form (clean,
    distorted, or rectified) — both probes are trained on train_idx only.

    Returns a dict of condition_key -> list of predicted labels, aligned
    with the original image order (every entry filled exactly once, by
    whichever fold held that image out).
    """
    n = len(y)
    pred = {"clean": [None] * n}
    for a in ANGLES:
        pred[f"clean_on_distorted_{a}"] = [None] * n
        pred[f"clean_on_rectified_{a}"] = [None] * n
        pred[f"aug_on_distorted_{a}"] = [None] * n
        pred[f"aug_on_rectified_{a}"] = [None] * n

    for train_idx, test_idx in folds:
        clf_clean = LogisticRegression(max_iter=2000, random_state=seed).fit(clean[train_idx], y[train_idx])

        X_aug = np.concatenate([clean[train_idx]] + [distorted[a][train_idx] for a in ANGLES], axis=0)
        y_aug = np.concatenate([y[train_idx]] * (1 + len(ANGLES)))
        clf_aug = LogisticRegression(max_iter=2000, random_state=seed).fit(X_aug, y_aug)

        clean_pred = clf_clean.predict(clean[test_idx])
        for i, idx in enumerate(test_idx):
            pred["clean"][idx] = clean_pred[i]

        for a in ANGLES:
            dp_clean = clf_clean.predict(distorted[a][test_idx])
            rp_clean = clf_clean.predict(rectified[a][test_idx])
            dp_aug = clf_aug.predict(distorted[a][test_idx])
            rp_aug = clf_aug.predict(rectified[a][test_idx])
            for i, idx in enumerate(test_idx):
                pred[f"clean_on_distorted_{a}"][idx] = dp_clean[i]
                pred[f"clean_on_rectified_{a}"][idx] = rp_clean[i]
                pred[f"aug_on_distorted_{a}"][idx] = dp_aug[i]
                pred[f"aug_on_rectified_{a}"][idx] = rp_aug[i]

    return pred


def rows_from_pred(pred_list, image_ids, true_labels):
    return [{"image_id": iid, "true_label": tl, "predicted_label": pl} for iid, tl, pl in zip(image_ids, true_labels, pred_list)]


def paired_bootstrap_diff(rows_a, rows_b, n_resamples=BOOTSTRAP_N_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Bootstrap 95% CI on (accuracy_a - accuracy_b), resampled at the image
    level, for two prediction sets covering the SAME held-out images (a
    paired comparison) — e.g. the same probe evaluated on distorted vs
    rectified features of the same images."""
    acc_a = {r["image_id"]: (r["predicted_label"] == r["true_label"]) for r in rows_a}
    acc_b = {r["image_id"]: (r["predicted_label"] == r["true_label"]) for r in rows_b}
    image_ids = np.array(sorted(acc_a.keys()))
    assert set(acc_a) == set(acc_b), "paired_bootstrap_diff requires both sides to cover the same image set"

    point = float(np.mean([acc_a[i] for i in image_ids]) - np.mean([acc_b[i] for i in image_ids]))

    rng = np.random.RandomState(seed)
    n = len(image_ids)
    diffs = np.empty(n_resamples)
    for k in range(n_resamples):
        sample = rng.choice(image_ids, size=n, replace=True)
        diffs[k] = np.mean([acc_a[i] for i in sample]) - np.mean([acc_b[i] for i in sample])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt_ci(point_lo_hi):
    point, lo, hi = point_lo_hi
    if point is None:
        return "n/a"
    if lo is None or hi is None:
        return f"{point:.3f} [n/a]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def fmt_diff_ci(point, lo, hi):
    sig = "significant" if (lo > 0) or (hi < 0) else "not significant"
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}] ({sig})"


# --- Experiment A: zero-shot tables ------------------------------------


def write_zeroshot_accuracy_table(zs_rows, class_names, out_path=RESULTS_TABLES / "v6_zeroshot_accuracy.md"):
    lines = [
        "# Day 12 v6 — Experiment A: zero-shot (8-prompt ensemble) accuracy",
        "",
        "| Model | Angle | Condition | Accuracy | Balanced accuracy | Macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        clean_rows = rows_for(zs_rows, slug, condition="clean")
        m = compute_metrics_with_ci(clean_rows, class_names)
        lines.append(f"| {spec['label']} | - | clean | {fmt_ci(m['accuracy'])} | {fmt_ci(m['balanced_accuracy'])} | {fmt_ci(m['macro_f1'])} |")
        for angle in ANGLES:
            for condition in ["distorted", "rectified"]:
                sub = rows_for(zs_rows, slug, angle, condition)
                m = compute_metrics_with_ci(sub, class_names)
                lines.append(f"| {spec['label']} | {angle} | {condition} | {fmt_ci(m['accuracy'])} | {fmt_ci(m['balanced_accuracy'])} | {fmt_ci(m['macro_f1'])} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_zeroshot_agreement_recovery_table(zs_rows, out_path=RESULTS_TABLES / "v6_zeroshot_agreement_recovery.md"):
    lines = [
        "# Day 12 v6 — Experiment A: agreement with clean + recovery %",
        "",
        "Recovery % = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) * 100.",
        "Flagged rows have |clean_acc - distorted_acc| < " + f"{RECOVERY_DENOM_FLAG_THRESHOLD:.2f}" + " — the ratio is unstable/not meaningful; read the raw accuracies instead.",
        "",
        "| Model | Angle | Condition | Agreement with clean |",
        "|---|---|---|---|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        clean_rows = rows_for(zs_rows, slug, condition="clean")
        clean_by_id = {r["image_id"]: r["predicted_label"] for r in clean_rows}
        for angle in ANGLES:
            for condition in ["distorted", "rectified"]:
                sub = rows_for(zs_rows, slug, angle, condition)
                agree = prediction_agreement(sub, clean_by_id)
                lines.append(f"| {spec['label']} | {angle} | {condition} | {agree:.3f} |")

    lines += ["", "| Model | Angle | Recovery % | Denominator (clean-distorted) | Flagged |", "|---|---|---|---|---|"]
    for spec in MODELS:
        slug = spec["slug"]
        for angle in ANGLES:
            combined = rows_for(zs_rows, slug, condition="clean") + rows_for(zs_rows, slug, angle, "distorted") + rows_for(zs_rows, slug, angle, "rectified")
            r = recovery_with_flag(combined)
            recov_str = f"{r['point']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]" if r["point"] is not None and r["lo"] is not None else "n/a (denom=0)"
            lines.append(f"| {spec['label']} | {angle} | {recov_str} | {r['denom']:.3f} | {'YES' if r['flagged'] else 'no'} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_zeroshot_mcnemar_table(zs_rows, out_path=RESULTS_TABLES / "v6_zeroshot_mcnemar.md"):
    lines = [
        "# Day 12 v6 — Experiment A: McNemar's test, distorted vs rectified",
        "",
        "| Model | Angle | b (distorted-correct, rectified-wrong) | c (distorted-wrong, rectified-correct) | n discordant | p-value |",
        "|---|---|---|---|---|---|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        for angle in ANGLES:
            distorted_rows = rows_for(zs_rows, slug, angle, "distorted")
            rectified_rows = rows_for(zs_rows, slug, angle, "rectified")
            result = paired_mcnemar(distorted_rows, rectified_rows, key_field="image_id")
            lines.append(f"| {spec['label']} | {angle} | {result['b']} | {result['c']} | {result['n_discordant']} | {result['p_value']:.4f} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_zeroshot_per_class_table(zs_rows, class_names, out_path=RESULTS_TABLES / "v6_zeroshot_per_class.md"):
    lines = [
        "# Day 12 v6 — Experiment A: per-class accuracy",
        "",
        "| Model | Angle | Condition | " + " | ".join(class_names) + " |",
        "|---" * (3 + len(class_names)) + "|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        clean_rows = rows_for(zs_rows, slug, condition="clean")
        pc = per_class_accuracy(clean_rows, class_names)
        lines.append(f"| {spec['label']} | - | clean | " + " | ".join(f"{pc[c]:.3f}" if pc[c] is not None else "n/a" for c in class_names) + " |")
        for angle in ANGLES:
            for condition in ["distorted", "rectified"]:
                sub = rows_for(zs_rows, slug, angle, condition)
                pc = per_class_accuracy(sub, class_names)
                lines.append(f"| {spec['label']} | {angle} | {condition} | " + " | ".join(f"{pc[c]:.3f}" if pc[c] is not None else "n/a" for c in class_names) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Experiment B table ------------------------------------


def write_probe_table(probe_results, class_names, out_path=RESULTS_TABLES / "v6_probe_leakfree.md"):
    lines = [
        "# Day 12 v6 — Experiment B: leakage-free linear probe (clean-trained)",
        "",
        "5-fold stratified CV, same split as v5; each fold trains on that fold's "
        "training images' CLEAN features only, then predicts held-out clean/distorted/rectified features.",
        "",
        "| Model | Angle | Condition | Accuracy | Balanced accuracy | Macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        rows = probe_results[slug]["rows"]
        m = compute_metrics_with_ci(rows["clean"], class_names)
        lines.append(f"| {spec['label']} | - | clean | {fmt_ci(m['accuracy'])} | {fmt_ci(m['balanced_accuracy'])} | {fmt_ci(m['macro_f1'])} |")
        for angle in ANGLES:
            for condition, key in [("distorted", f"clean_on_distorted_{angle}"), ("rectified", f"clean_on_rectified_{angle}")]:
                m = compute_metrics_with_ci(rows[key], class_names)
                lines.append(f"| {spec['label']} | {angle} | {condition} | {fmt_ci(m['accuracy'])} | {fmt_ci(m['balanced_accuracy'])} | {fmt_ci(m['macro_f1'])} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Experiment C table ------------------------------------


def write_rectification_vs_augmentation_table(probe_results, class_names, out_path=RESULTS_TABLES / "v6_rectification_vs_augmentation.md"):
    lines = [
        "# Day 12 v6 — Experiment C: rectification vs augmentation",
        "",
        "Four combinations per (model, angle), same held-out images: clean-trained probe "
        "on distorted/rectified features, and augmentation-trained probe (clean + distorted "
        "pooled over all 3 angles, no rectified images in training) on distorted/rectified features.",
        "",
        "| Model | Angle | Combination | Accuracy | Balanced accuracy | Macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    combo_labels = [
        ("clean_on_distorted", "clean-trained probe -> distorted"),
        ("clean_on_rectified", "clean-trained probe -> rectified"),
        ("aug_on_distorted", "augmentation-trained probe -> distorted"),
        ("aug_on_rectified", "augmentation-trained probe -> rectified"),
    ]
    for spec in MODELS:
        slug = spec["slug"]
        rows = probe_results[slug]["rows"]
        for angle in ANGLES:
            for combo_key, combo_label in combo_labels:
                key = f"{combo_key}_{angle}"
                m = compute_metrics_with_ci(rows[key], class_names)
                lines.append(f"| {spec['label']} | {angle} | {combo_label} | {fmt_ci(m['accuracy'])} | {fmt_ci(m['balanced_accuracy'])} | {fmt_ci(m['macro_f1'])} |")

    lines += [
        "",
        "## Pairwise significance (paired bootstrap 95% CI on accuracy difference, same held-out images)",
        "",
        "| Model | Angle | Comparison | Difference [95% CI] |",
        "|---|---|---|---|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        rows = probe_results[slug]["rows"]
        for angle in ANGLES:
            comparisons = [
                ("Rectification effect, clean-trained probe (rectified - distorted)", f"clean_on_rectified_{angle}", f"clean_on_distorted_{angle}"),
                ("Rectification effect, augmentation-trained probe (rectified - distorted)", f"aug_on_rectified_{angle}", f"aug_on_distorted_{angle}"),
                ("Augmentation effect on distorted (aug - clean-trained)", f"aug_on_distorted_{angle}", f"clean_on_distorted_{angle}"),
                ("Augmentation effect on rectified (aug - clean-trained)", f"aug_on_rectified_{angle}", f"clean_on_rectified_{angle}"),
            ]
            for label, key_a, key_b in comparisons:
                point, lo, hi = paired_bootstrap_diff(rows[key_a], rows[key_b])
                lines.append(f"| {spec['label']} | {angle} | {label} | {fmt_diff_ci(point, lo, hi)} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- figures ------------------------------------


def write_accuracy_vs_angle_figure(zs_rows, probe_results, class_names, out_path=FIGURES / "v6_accuracy_vs_angle.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.5 * len(MODELS), 4.5), sharey=True)
    for ax, spec in zip(axes, MODELS):
        slug = spec["slug"]
        x = [0] + ANGLES

        def series(get_rows_fn):
            points, los, his = [], [], []
            for angle in x:
                rows = get_rows_fn(angle)
                m = compute_metrics_with_ci(rows, class_names)
                p, lo, hi = m["accuracy"]
                points.append(p if p is not None else np.nan)
                los.append((p - lo) if (p is not None and lo is not None) else 0.0)
                his.append((hi - p) if (p is not None and hi is not None) else 0.0)
            return points, [los, his]

        clean_rows = rows_for(zs_rows, slug, condition="clean")

        zs_distorted = series(lambda a: clean_rows if a == 0 else rows_for(zs_rows, slug, a, "distorted"))
        zs_rectified = series(lambda a: clean_rows if a == 0 else rows_for(zs_rows, slug, a, "rectified"))
        probe_distorted = series(lambda a: probe_results[slug]["rows"]["clean"] if a == 0 else probe_results[slug]["rows"][f"clean_on_distorted_{a}"])
        probe_rectified = series(lambda a: probe_results[slug]["rows"]["clean"] if a == 0 else probe_results[slug]["rows"][f"clean_on_rectified_{a}"])

        ax.errorbar(x, zs_distorted[0], yerr=zs_distorted[1], marker="o", label="zero-shot: distorted", capsize=3)
        ax.errorbar(x, zs_rectified[0], yerr=zs_rectified[1], marker="s", label="zero-shot: rectified", capsize=3)
        ax.errorbar(x, probe_distorted[0], yerr=probe_distorted[1], marker="^", linestyle="--", label="probe: distorted", capsize=3)
        ax.errorbar(x, probe_rectified[0], yerr=probe_rectified[1], marker="v", linestyle="--", label="probe: rectified", capsize=3)

        ax.set_title(spec["label"])
        ax.set_xlabel("Camera tilt angle (degrees); 0 = clean")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Day 12 v6 — accuracy vs angle, zero-shot vs leakage-free probe (95% bootstrap CI)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_experiment_c_summary_figure(probe_results, class_names, out_path=FIGURES / "v6_experimentC_summary.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combo_keys = ["clean_on_distorted", "clean_on_rectified", "aug_on_distorted", "aug_on_rectified"]
    combo_labels = ["clean probe\n-> distorted", "clean probe\n-> rectified", "aug probe\n-> distorted", "aug probe\n-> rectified"]

    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4.5), sharey=True)
    width = 0.8 / len(combo_keys)
    x = np.arange(len(ANGLES))

    for ax, spec in zip(axes, MODELS):
        slug = spec["slug"]
        rows = probe_results[slug]["rows"]
        for i, (combo_key, combo_label) in enumerate(zip(combo_keys, combo_labels)):
            values, errs_lo, errs_hi = [], [], []
            for angle in ANGLES:
                m = compute_metrics_with_ci(rows[f"{combo_key}_{angle}"], class_names)
                p, lo, hi = m["accuracy"]
                values.append(p if p is not None else 0.0)
                errs_lo.append((p - lo) if (p is not None and lo is not None) else 0.0)
                errs_hi.append((hi - p) if (p is not None and hi is not None) else 0.0)
            offset = (i - (len(combo_keys) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=combo_label, yerr=[errs_lo, errs_hi], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ANGLES])
        ax.set_xlabel("Camera tilt angle (degrees)")
        ax.set_title(spec["label"])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Day 12 v6 — Experiment C: rectification vs augmentation (95% bootstrap CI)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- log ------------------------------------


def write_log(zs_rows, probe_results, class_names):
    lines = []

    # --- Summary (5 lines) ---
    summary_facts = []
    for spec in MODELS:
        slug = spec["slug"]
        rows = probe_results[slug]["rows"]
        zs_clean_acc = top1_accuracy(rows_for(zs_rows, slug, condition="clean"))
        zs_35_distorted = top1_accuracy(rows_for(zs_rows, slug, 35, "distorted"))
        zs_35_rectified = top1_accuracy(rows_for(zs_rows, slug, 35, "rectified"))
        probe_35_distorted = top1_accuracy(rows["clean_on_distorted_35"])
        probe_35_rectified = top1_accuracy(rows["clean_on_rectified_35"])
        aug_35_distorted = top1_accuracy(rows["aug_on_distorted_35"])
        summary_facts.append(
            {
                "label": spec["label"],
                "zs_clean": zs_clean_acc,
                "zs_35_distorted": zs_35_distorted,
                "zs_35_rectified": zs_35_rectified,
                "probe_35_distorted": probe_35_distorted,
                "probe_35_rectified": probe_35_rectified,
                "aug_35_distorted": aug_35_distorted,
            }
        )

    best_model = max(summary_facts, key=lambda f: f["zs_clean"])
    rect_helps_zs = sum(1 for f in summary_facts if f["zs_35_rectified"] > f["zs_35_distorted"])
    rect_helps_probe = sum(1 for f in summary_facts if f["probe_35_rectified"] > f["probe_35_distorted"])
    aug_beats_rect_probe = sum(1 for f in summary_facts if f["aug_35_distorted"] > f["probe_35_rectified"])

    clean_str = ", ".join(f"{fct['label']} {fct['zs_clean']:.2f}" for fct in summary_facts)
    distorted35_str = ", ".join(f"{fct['label']} {fct['zs_35_distorted']:.2f}" for fct in summary_facts)

    lines.append("## Summary (read this first)")
    lines.append("")
    lines.append(
        f"- At the most severe angle tested (35 degrees), zero-shot accuracy drops from clean "
        f"({clean_str}) to distorted ({distorted35_str}) — the same qualitative drop v4 found, now with stronger backbones."
    )
    lines.append(
        f"- Oracle rectification helps zero-shot accuracy at 35 degrees for {rect_helps_zs}/{len(summary_facts)} models "
        f"and helps the leakage-free linear probe for {rect_helps_probe}/{len(summary_facts)} models — see v6_zeroshot_agreement_recovery.md "
        "and v6_probe_leakfree.md for every angle, not just 35."
    )
    lines.append(
        f"- Comparing rectification against augmentation (Experiment C): an augmentation-trained probe evaluated on distorted images "
        f"beats a clean-trained probe evaluated on RECTIFIED images for {aug_beats_rect_probe}/{len(summary_facts)} models at 35 degrees "
        "— see v6_rectification_vs_augmentation.md's pairwise significance table for whether each gap is statistically distinguishable from zero."
    )
    lines.append(
        f"- Best overall clean-image zero-shot model here: {best_model['label']} at {best_model['zs_clean']:.3f} — "
        "see v6_zeroshot_per_class.md for whether Gothic specifically is still the weakest/most-affected class, as v4 found."
    )
    lines.append(
        "- Report every number as measured, including any case where a stronger model turns out LESS robust, "
        "rectification stops helping, or augmentation beats rectification outright — none of those would overturn v4's "
        "clean-baseline finding, but all are real possible outcomes here and are not softened in the tables below."
    )
    lines.append("")

    lines.append("# Day 12 v6 — stronger models, leakage-free probe, rectification vs augmentation")
    lines.append("")
    lines.append(
        "Re-runs v4's exact distortion/rectification (same camera-rotation homography, same seeds, "
        "replicate 0, mean_fill border policy) on 3 stronger backbones at angles 5/15/35 degrees, on all 1,514 MonuMAI images."
    )
    lines.append("")
    lines.append("Full numeric results are in the generated tables, not duplicated here:")
    lines.append("- results/tables/v6_zeroshot_accuracy.md")
    lines.append("- results/tables/v6_zeroshot_agreement_recovery.md")
    lines.append("- results/tables/v6_zeroshot_mcnemar.md")
    lines.append("- results/tables/v6_zeroshot_per_class.md")
    lines.append("- results/tables/v6_probe_leakfree.md")
    lines.append("- results/tables/v6_rectification_vs_augmentation.md")
    lines.append("- figures/v6_accuracy_vs_angle.png, figures/v6_experimentC_summary.png")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Experiment B/C's 5-fold split is regenerated with v5's exact StratifiedKFold call (same seed, "
        "same true_labels order) rather than loaded from a saved file — deterministic, so identical to v5's folds."
    )
    lines.append(
        "- A held-out image's clean, distorted, AND rectified features are all excluded from both probes' "
        "training sets in every fold — not just the clean ones — per the no-leakage requirement."
    )
    lines.append(f"- Recovery-% rows with |clean_acc - distorted_acc| < {RECOVERY_DENOM_FLAG_THRESHOLD} are flagged, not reported as a bare ratio.")
    lines.append("- This script does not touch v1-v5 code, outputs, or the paper draft.")
    lines.append("")

    LOG_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    from clip_zero_shot import load_config

    config = load_config("configs/default.yaml")
    class_names = config["class_names"]
    image_ids, true_labels = load_dataset(config)

    zs_rows = load_zeroshot_rows()

    folds, y = make_folds(true_labels)

    probe_results = {}
    for spec in MODELS:
        slug = spec["slug"]
        print(f"Running probes for {spec['label']} ...", flush=True)
        clean, distorted, rectified = load_features(slug)
        assert clean.shape[0] == len(true_labels), f"{slug}: feature count mismatch with dataset"
        pred = run_probes_for_model(clean, distorted, rectified, y, folds)
        rows = {key: rows_from_pred(pred_list, image_ids, true_labels) for key, pred_list in pred.items()}
        probe_results[slug] = {"pred": pred, "rows": rows}

    print("Writing tables ...", flush=True)
    write_zeroshot_accuracy_table(zs_rows, class_names)
    write_zeroshot_agreement_recovery_table(zs_rows)
    write_zeroshot_mcnemar_table(zs_rows)
    write_zeroshot_per_class_table(zs_rows, class_names)
    write_probe_table(probe_results, class_names)
    write_rectification_vs_augmentation_table(probe_results, class_names)

    print("Writing figures ...", flush=True)
    write_accuracy_vs_angle_figure(zs_rows, probe_results, class_names)
    write_experiment_c_summary_figure(probe_results, class_names)

    print("Writing log ...", flush=True)
    write_log(zs_rows, probe_results, class_names)

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
