"""Day 8 v2 — analysis of the fixed-direction, 5-replicate experimental grid.

Turns results/raw/day8_v2_grid.csv (produced by src/experiment_runner_v2.py)
into the v2 tables/figures and re-tests whether Day 6's two headline
per-class findings (Gothic's negative pooled recovery, the "Renaissance
attractor" effect) survive once tilt direction is independent per image
instead of one shared draw per severity.

Does not modify src/experiment_runner_v2.py, src/day6_analysis.py, or any
Day 5/6 output — purely a read-only analysis layer over the new v2 CSV,
writing only to results/tables/v2_*.md and figures/v2_*.png.
"""

import argparse
import csv
from pathlib import Path

from clip_zero_shot import load_config
from metrics import (
    balanced_accuracy,
    bootstrap_ci,
    macro_f1,
    paired_mcnemar,
    prediction_agreement,
    recovery_percentage,
    top1_accuracy,
)

SEVERITIES = ["mild", "medium", "severe"]
CONDITIONS = ["clean", "distorted", "rectified"]
V2_CSV = "results/raw/day8_v2_grid.csv"
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


def load_v2_rows(csv_path=V2_CSV):
    """Load results/raw/day8_v2_grid.csv, coercing "correct" to bool and
    "replicate" to int. "clean" rows are logged once per (image, severity)
    at replicate=0 only (see experiment_runner_v2.run_grid_v2's docstring)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["correct"] = row["correct"] == "True"
        row["replicate"] = int(row["replicate"])
    return rows


def rows_for(rows, severity=None, condition=None):
    """Filter helper: rows matching the given severity/condition (either may be None = no filter)."""
    out = rows
    if severity is not None:
        out = [r for r in out if r["severity"] == severity]
    if condition is not None:
        out = [r for r in out if r["condition"] == condition]
    return out


# --- Accuracy with replicate mean/std + bootstrap CI ------------------------------------


def accuracy_summary(rows, class_names):
    """Return {severity: {condition: {accuracy_mean, accuracy_std, ci_low, ci_high,
    balanced_accuracy, macro_f1}}}.

    accuracy_mean/std are computed directly over the 5 per-replicate accuracy
    values (clean has only 1 "replicate" since it is distortion-independent,
    so its std is reported as 0.0/n=1, not undefined). ci_low/ci_high are the
    95% bootstrap CI (1000 resamples over images) on the pooled-replicate
    accuracy, per metrics.bootstrap_ci.
    """
    import numpy as np

    summary = {}
    for severity in SEVERITIES:
        summary[severity] = {}
        for condition in CONDITIONS:
            subset = rows_for(rows, severity, condition)
            n_replicates = len(set(r["replicate"] for r in subset))
            per_rep_acc = [
                top1_accuracy([r for r in subset if r["replicate"] == rep])
                for rep in sorted(set(r["replicate"] for r in subset))
            ]
            point, lo, hi = bootstrap_ci(
                subset, lambda rs: top1_accuracy(rs), n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED
            )
            summary[severity][condition] = {
                "n_replicates": n_replicates,
                "accuracy_mean": float(np.mean(per_rep_acc)),
                "accuracy_std": float(np.std(per_rep_acc)),
                "accuracy_pooled": point,
                "ci_low": lo,
                "ci_high": hi,
                "balanced_accuracy": balanced_accuracy(subset, class_names),
                "macro_f1": macro_f1(subset, class_names),
            }
    return summary


def save_accuracy_table(summary, output_path):
    lines = ["# Day 8 v2 — accuracy, balanced accuracy, macro-F1 by severity and condition", ""]
    lines.append(
        "Accuracy mean/std is computed over the 5 independent-tilt replicates "
        "(1 'replicate' for clean, which is distortion-free); the 95% CI is a "
        f"bootstrap CI ({N_RESAMPLES} resamples, resampled at the image level, "
        "seed=0) on the pooled-replicate accuracy."
    )
    lines.append("")
    lines.append(
        "| Severity | Condition | Accuracy (mean +/- std over replicates) | "
        "Pooled accuracy [95% CI] | Balanced acc. | Macro-F1 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for severity in SEVERITIES:
        for condition in CONDITIONS:
            s = summary[severity][condition]
            ci_str = f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if s["ci_low"] is not None else "n/a"
            lines.append(
                f"| {severity} | {condition} | {s['accuracy_mean']:.3f} +/- {s['accuracy_std']:.3f} "
                f"(n_rep={s['n_replicates']}) | {s['accuracy_pooled']:.3f} {ci_str} | "
                f"{s['balanced_accuracy']:.3f} | {s['macro_f1']:.3f} |"
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


def save_severity_curve(summary, output_path):
    """Plot accuracy vs. severity with bootstrap-CI error bars, one line per condition."""
    import matplotlib.pyplot as plt

    x_labels = ["clean (0)", "mild", "medium", "severe"]
    x = list(range(len(x_labels)))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"clean": "tab:green", "distorted": "tab:red", "rectified": "tab:blue"}
    for condition in CONDITIONS:
        clean_acc = summary["mild"]["clean"]["accuracy_pooled"]
        clean_err = (
            clean_acc - summary["mild"]["clean"]["ci_low"],
            summary["mild"]["clean"]["ci_high"] - clean_acc,
        )
        ys = [clean_acc] + [summary[s][condition]["accuracy_pooled"] for s in SEVERITIES]
        err_lo = [clean_err[0]] + [
            summary[s][condition]["accuracy_pooled"] - summary[s][condition]["ci_low"] for s in SEVERITIES
        ]
        err_hi = [clean_err[1]] + [
            summary[s][condition]["ci_high"] - summary[s][condition]["accuracy_pooled"] for s in SEVERITIES
        ]
        ax.errorbar(
            x, ys, yerr=[err_lo, err_hi], marker="o", label=condition, color=colors[condition], capsize=3
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xlabel("Distortion severity")
    ax.set_title("Day 8 v2 — accuracy vs. severity (5 independent-tilt replicates,\n95% bootstrap CI, n=1514 images per point)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# --- Recovery % with bootstrap CI ------------------------------------


def recovery_summary(rows):
    """Return {severity: {recovery_pct, ci_low, ci_high}}."""
    summary = {}
    for severity in SEVERITIES:
        subset = rows_for(rows, severity)
        point, lo, hi = bootstrap_ci(subset, recovery_percentage, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED)
        summary[severity] = {"recovery_pct": point, "ci_low": lo, "ci_high": hi}
    return summary


def save_recovery_table(summary, output_path):
    lines = ["# Day 8 v2 — recovery % by severity, with 95% bootstrap CI", ""]
    lines.append(
        "recovery = (acc_rectified - acc_distorted) / (acc_clean - acc_distorted) x 100. "
        f"CI is a {N_RESAMPLES}-resample bootstrap over images, seed=0 — recovery % is a "
        "ratio of two accuracy differences, so its CI can be very wide when the "
        "denominator is small; report the CI, not just the point estimate."
    )
    lines.append("")
    lines.append("| Severity | Recovery % | 95% CI |")
    lines.append("|---|---|---|")
    for severity in SEVERITIES:
        s = summary[severity]
        rec_str = f"{s['recovery_pct']:.1f}%" if s["recovery_pct"] is not None else "undefined"
        ci_str = f"[{s['ci_low']:.1f}%, {s['ci_high']:.1f}%]" if s["ci_low"] is not None else "n/a"
        lines.append(f"| {severity} | {rec_str} | {ci_str} |")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


# --- Prediction agreement with clean ------------------------------------


def agreement_summary(rows):
    """Return {severity: {condition: agreement_fraction}} for distorted/rectified."""
    summary = {}
    for severity in SEVERITIES:
        clean_rows = rows_for(rows, severity, "clean")
        clean_by_image_id = {r["image_id"]: r["predicted_label"] for r in clean_rows}
        summary[severity] = {}
        for condition in ["distorted", "rectified"]:
            subset = rows_for(rows, severity, condition)
            summary[severity][condition] = prediction_agreement(subset, clean_by_image_id)
    return summary


def save_agreement_table(summary, output_path):
    lines = ["# Day 8 v2 — prediction agreement with clean, by severity", ""]
    lines.append(
        "Fraction of (image, replicate) predictions unchanged from that image's clean prediction."
    )
    lines.append("")
    lines.append("| Severity | Distorted agreement | Rectified agreement |")
    lines.append("|---|---|---|")
    for severity in SEVERITIES:
        s = summary[severity]
        lines.append(f"| {severity} | {s['distorted']:.3f} | {s['rectified']:.3f} |")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


# --- Per-class recovery with CI (pooled across severities, matching Day 6's table) ------


def per_class_recovery_summary(rows, class_names):
    """Return {class_name: {recovery_pct, ci_low, ci_high, clean_acc, distorted_acc, rectified_acc}},
    pooled across all 3 severities x 5 replicates, matching Day 6's pooled per-class table."""
    summary = {}
    for class_name in class_names:
        class_rows = [r for r in rows if r["true_label"] == class_name]
        point, lo, hi = bootstrap_ci(
            class_rows, recovery_percentage, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        summary[class_name] = {
            "recovery_pct": point,
            "ci_low": lo,
            "ci_high": hi,
            "clean_acc": top1_accuracy([r for r in class_rows if r["condition"] == "clean"]),
            "distorted_acc": top1_accuracy([r for r in class_rows if r["condition"] == "distorted"]),
            "rectified_acc": top1_accuracy([r for r in class_rows if r["condition"] == "rectified"]),
        }
    return summary


def save_per_class_recovery_table(summary, class_names, output_path):
    lines = ["# Day 8 v2 — per-class recovery %, pooled across severities, with 95% bootstrap CI", ""]
    lines.append("| Style | Clean acc. | Distorted acc. | Rectified acc. | Recovery % | 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    for class_name in class_names:
        s = summary[class_name]
        rec_str = f"{s['recovery_pct']:.1f}%" if s["recovery_pct"] is not None else "undefined"
        ci_str = f"[{s['ci_low']:.1f}%, {s['ci_high']:.1f}%]" if s["ci_low"] is not None else "n/a"
        lines.append(
            f"| {class_name} | {s['clean_acc']:.3f} | {s['distorted_acc']:.3f} | "
            f"{s['rectified_acc']:.3f} | {rec_str} | {ci_str} |"
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


def save_per_class_recovery_figure(summary, class_names, output_path, zoom_range=(-150, 250)):
    """Bar chart of per-class recovery % with bootstrap-CI error bars.

    Two panels: the left shows the full CI range honestly (Gothic's CI is
    orders of magnitude wider than the other 3 classes', since its clean and
    distorted accuracy are nearly identical — a near-zero denominator, not a
    huge effect), which would otherwise make Renaissance/Baroque/Hispanic-
    Muslim's bars invisible on a single shared axis. The right panel zooms to
    `zoom_range` so those 3 classes' CIs are actually readable; any bar/CI
    extending past the zoom range is marked, not silently clipped.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # recovery_pct (and its CI) is None when a class's clean and distorted
    # accuracy are exactly equal (zero denominator) — plot those bars as 0
    # with an "undefined" annotation rather than letting None reach
    # matplotlib's bar height arithmetic.
    values = [summary[c]["recovery_pct"] if summary[c]["recovery_pct"] is not None else 0.0 for c in class_names]
    err_lo = [
        (summary[c]["recovery_pct"] - summary[c]["ci_low"])
        if summary[c]["recovery_pct"] is not None and summary[c]["ci_low"] is not None
        else 0
        for c in class_names
    ]
    err_hi = [
        (summary[c]["ci_high"] - summary[c]["recovery_pct"])
        if summary[c]["recovery_pct"] is not None and summary[c]["ci_high"] is not None
        else 0
        for c in class_names
    ]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(class_names))

    for ax, title, ylim in [
        (ax_full, "full range (honest, Gothic's CI dominates)", None),
        (ax_zoom, f"zoomed to {zoom_range} (CIs past this range marked ^/v)", zoom_range),
    ]:
        bars = ax.bar(x, values, yerr=[err_lo, err_hi], capsize=4, color="tab:blue", alpha=0.8)
        for i, class_name in enumerate(class_names):
            if summary[class_name]["recovery_pct"] is None:
                bars[i].set_color("lightgray")
                ax.text(i, 2, "undefined", ha="center", va="bottom", fontsize=8, rotation=90)
            elif ylim is not None:
                ci_hi = summary[class_name]["ci_high"]
                ci_lo = summary[class_name]["ci_low"]
                if ci_hi is not None and ci_hi > ylim[1]:
                    ax.annotate("^", (i, ylim[1] - 10), ha="center", fontsize=14, color="black")
                if ci_lo is not None and ci_lo < ylim[0]:
                    ax.annotate("v", (i, ylim[0] + 10), ha="center", fontsize=14, color="black")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_ylabel("Recovery % (95% bootstrap CI)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if ylim is not None:
            ax.set_ylim(ylim)

    fig.suptitle("Day 8 v2 — per-class recovery %, pooled across severities")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# --- McNemar per severity, pooled replicates ------------------------------------


def mcnemar_summary(rows):
    """McNemar (distorted vs rectified) per severity, pooling all 5 replicates as
    paired samples keyed by (image_id, replicate) — NOT by image_id alone, which
    would silently collide replicates and drop data (see metrics.paired_mcnemar)."""
    summary = {}
    for severity in SEVERITIES:
        dist_rows = rows_for(rows, severity, "distorted")
        rect_rows = rows_for(rows, severity, "rectified")
        summary[severity] = paired_mcnemar(dist_rows, rect_rows, key_field=("image_id", "replicate"))
    return summary


def save_mcnemar_table(summary, output_path):
    lines = ["# Day 8 v2 — McNemar exact test, distorted vs. rectified, per severity (pooled replicates)", ""]
    lines.append(
        "Paired on (image_id, replicate) — each of the 5 independent-tilt "
        "replicates per image counted separately, not collapsed."
    )
    lines.append("")
    lines.append("| Severity | n discordant | b (dist correct, rect wrong) | c (dist wrong, rect correct) | p-value |")
    lines.append("|---|---|---|---|---|")
    for severity in SEVERITIES:
        s = summary[severity]
        lines.append(f"| {severity} | {s['n_discordant']} | {s['b']} | {s['c']} | {s['p_value']:.4f} |")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Day 8 v2 analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--csv", default=V2_CSV)
    args = parser.parse_args()

    config = load_config(args.config)
    class_names = config["class_names"]
    rows = load_v2_rows(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    acc_summary = accuracy_summary(rows, class_names)
    save_accuracy_table(acc_summary, "results/tables/v2_accuracy.md")
    save_severity_curve(acc_summary, "figures/v2_severity_curve.png")

    rec_summary = recovery_summary(rows)
    save_recovery_table(rec_summary, "results/tables/v2_recovery.md")

    agree_summary = agreement_summary(rows)
    save_agreement_table(agree_summary, "results/tables/v2_agreement.md")

    per_class_summary = per_class_recovery_summary(rows, class_names)
    save_per_class_recovery_table(per_class_summary, class_names, "results/tables/v2_per_class_recovery.md")
    save_per_class_recovery_figure(per_class_summary, class_names, "figures/v2_per_class_recovery.png")

    mc_summary = mcnemar_summary(rows)
    save_mcnemar_table(mc_summary, "results/tables/v2_mcnemar.md")

    print("\n--- Headline re-test ---")
    for severity in SEVERITIES:
        s = acc_summary[severity]
        print(
            f"{severity}: clean={s['clean']['accuracy_pooled']:.3f} "
            f"distorted={s['distorted']['accuracy_pooled']:.3f} "
            f"rectified={s['rectified']['accuracy_pooled']:.3f}"
        )
    print("\nPer-class recovery % (pooled, 95% CI):")
    for class_name in class_names:
        s = per_class_summary[class_name]
        rec_str = f"{s['recovery_pct']:.1f}%" if s["recovery_pct"] is not None else "undefined"
        ci_str = f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]" if s["ci_low"] is not None else "n/a"
        print(f"  {class_name}: {rec_str} {ci_str}")


if __name__ == "__main__":
    main()
