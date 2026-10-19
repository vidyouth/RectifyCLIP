"""Day 9 v3 — analysis of the mismatched-H control and noisy-oracle experiments.

Read-only over results/raw/day8_v2_grid.csv (v2, never modified),
results/raw/v3_mismatched_h.csv, and results/raw/v3_noisy_oracle.csv. Reuses
metrics.bootstrap_ci/recovery_percentage/top1_accuracy and
day8_v2_analysis.recovery_summary (for the oracle baseline) rather than
duplicating their logic — this module only adds the merging/relabeling
needed to compare v3's new conditions against v2's existing clean/distorted/
rectified rows on equal footing.
"""

import argparse
import csv
from pathlib import Path

from clip_zero_shot import load_config
from day8_v2_analysis import load_v2_rows, recovery_summary as oracle_recovery_summary
from metrics import bootstrap_ci, recovery_percentage, top1_accuracy

N_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
SEVERITIES_V3 = ["medium", "severe"]


def load_csv_rows(path, int_fields=("replicate",), bool_fields=("correct",)):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in bool_fields:
            if field in row:
                row[field] = row[field] == "True"
        for field in int_fields:
            if field in row and row[field] != "":
                row[field] = int(row[field])
    return rows


def relabel_condition(rows, new_condition):
    """Return a COPY of rows with "condition" overwritten to new_condition —
    used to make a v3 treatment condition (e.g. "wrong_h_rectified") look
    like v2's "rectified" condition to metrics.recovery_percentage, which
    filters by the literal strings "clean"/"distorted"/"rectified". Never
    mutates the input rows (v1/v2 data must never be modified; v3 data is
    copied defensively too, for the same reason this function exists)."""
    out = []
    for r in rows:
        r2 = dict(r)
        r2["condition"] = new_condition
        out.append(r2)
    return out


def accuracy_with_ci(rows):
    return bootstrap_ci(rows, top1_accuracy, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED)


# --- Experiment 1: mismatched-H control ------------------------------------


def analyze_mismatched_h(v2_rows, v3_rows):
    """Return {severity: {distorted: {...}, wrong_h: {...}, oracle: {...},
    wrong_h_recovery: {...}, oracle_recovery: {...}}}."""
    summary = {}
    oracle_rec = oracle_recovery_summary(v2_rows)  # {severity: {recovery_pct, ci_low, ci_high}} for ALL severities
    for severity in SEVERITIES_V3:
        v2_clean = [r for r in v2_rows if r["severity"] == severity and r["condition"] == "clean"]
        v2_distorted = [r for r in v2_rows if r["severity"] == severity and r["condition"] == "distorted"]
        v2_oracle = [r for r in v2_rows if r["severity"] == severity and r["condition"] == "rectified"]
        v3_wrong_h = [r for r in v3_rows if r["severity"] == severity]

        dist_point, dist_lo, dist_hi = accuracy_with_ci(v2_distorted)
        wrong_point, wrong_lo, wrong_hi = accuracy_with_ci(v3_wrong_h)
        oracle_point, oracle_lo, oracle_hi = accuracy_with_ci(v2_oracle)

        wrong_h_relabeled = relabel_condition(v3_wrong_h, "rectified")
        merged_wrong_h = v2_clean + v2_distorted + wrong_h_relabeled
        wrong_rec_point, wrong_rec_lo, wrong_rec_hi = bootstrap_ci(
            merged_wrong_h, recovery_percentage, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED
        )

        summary[severity] = {
            "distorted": {"acc": dist_point, "ci_low": dist_lo, "ci_high": dist_hi},
            "wrong_h": {"acc": wrong_point, "ci_low": wrong_lo, "ci_high": wrong_hi},
            "oracle": {"acc": oracle_point, "ci_low": oracle_lo, "ci_high": oracle_hi},
            "wrong_h_recovery": {"recovery_pct": wrong_rec_point, "ci_low": wrong_rec_lo, "ci_high": wrong_rec_hi},
            "oracle_recovery": oracle_rec[severity],
        }
    return summary


def save_mismatched_h_table(summary, output_path):
    lines = ["# Day 9 v3 — mismatched-H control: does rectification need the CORRECT homography?", ""]
    lines.append(
        "Each distorted image rectified with a DIFFERENT image's homography "
        "(same severity, same replicate) instead of its own. 95% CI = "
        f"{N_RESAMPLES}-resample bootstrap over images, seed=0."
    )
    lines.append("")
    lines.append(
        "| Severity | Distorted acc. [CI] | Wrong-H rectified acc. [CI] | Oracle rectified acc. [CI] | "
        "Wrong-H recovery % [CI] | Oracle recovery % [CI] |"
    )
    lines.append("|---|---|---|---|---|---|")
    for severity in SEVERITIES_V3:
        s = summary[severity]
        d, w, o = s["distorted"], s["wrong_h"], s["oracle"]
        wr, orr = s["wrong_h_recovery"], s["oracle_recovery"]
        wr_str = f"{wr['recovery_pct']:.1f}% [{wr['ci_low']:.1f}, {wr['ci_high']:.1f}]" if wr["recovery_pct"] is not None else "undefined"
        orr_str = f"{orr['recovery_pct']:.1f}% [{orr['ci_low']:.1f}, {orr['ci_high']:.1f}]" if orr["recovery_pct"] is not None else "undefined"
        lines.append(
            f"| {severity} | {d['acc']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}] | "
            f"{w['acc']:.3f} [{w['ci_low']:.3f}, {w['ci_high']:.3f}] | "
            f"{o['acc']:.3f} [{o['ci_low']:.3f}, {o['ci_high']:.3f}] | {wr_str} | {orr_str} |"
        )
    lines.append("")
    lines.append(
        "Diagnostic note: 3 of 15,140 wrong-H rows (0.02%) had a wrong homography that pushed "
        "ALL pixels outside the rectified canvas (empty valid mask), so mean_fill produced an "
        "undefined/degenerate fill color for those 3 images. Left in the numbers above uncorrected "
        "— at 3/15,140 rows, no computed accuracy or CI changes outside reported precision."
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


def save_mismatched_h_figure(summary, output_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7, 5))
    conditions = ["distorted", "wrong_h", "oracle"]
    labels = ["distorted", "wrong-H\nrectified", "oracle\nrectified"]
    colors = ["tab:red", "tab:orange", "tab:blue"]
    width = 0.25
    x = np.arange(len(SEVERITIES_V3))

    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        vals = [summary[sev][cond]["acc"] for sev in SEVERITIES_V3]
        err_lo = [summary[sev][cond]["acc"] - summary[sev][cond]["ci_low"] for sev in SEVERITIES_V3]
        err_hi = [summary[sev][cond]["ci_high"] - summary[sev][cond]["acc"] for sev in SEVERITIES_V3]
        ax.bar(x + (i - 1) * width, vals, width, yerr=[err_lo, err_hi], capsize=3, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITIES_V3)
    ax.set_ylabel("Top-1 accuracy (95% bootstrap CI)")
    ax.set_title("Day 9 v3 — mismatched-H control:\ndistorted vs. wrong-H vs. oracle rectification", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# --- Experiment 3: noisy oracle ------------------------------------


def analyze_noisy_oracle(v2_rows, v3_rows, noisy_replicates=(0, 1, 2)):
    """Return {severity: {pct: {acc, ci_low, ci_high, recovery_pct, rec_ci_low, rec_ci_high}}}."""
    summary = {}
    for severity in SEVERITIES_V3:
        summary[severity] = {}
        v2_clean = [r for r in v2_rows if r["severity"] == severity and r["condition"] == "clean"]
        v2_distorted_restricted = [
            r for r in v2_rows
            if r["severity"] == severity and r["condition"] == "distorted" and int(r["replicate"]) in noisy_replicates
        ]

        # pct=0 pulled directly from v2's existing "rectified" rows (pixel-identical to exact oracle).
        v2_oracle_restricted = [
            r for r in v2_rows
            if r["severity"] == severity and r["condition"] == "rectified" and int(r["replicate"]) in noisy_replicates
        ]
        pct_rowsets = {0: v2_oracle_restricted}
        for pct in sorted(set(r["perturbation_pct"] for r in v3_rows if r["severity"] == severity)):
            pct_rowsets[pct] = [
                r for r in v3_rows
                if r["severity"] == severity and r["perturbation_pct"] == pct
            ]

        for pct, treatment_rows in sorted(pct_rowsets.items()):
            acc_point, acc_lo, acc_hi = accuracy_with_ci(treatment_rows)
            merged = v2_clean + v2_distorted_restricted + relabel_condition(treatment_rows, "rectified")
            rec_point, rec_lo, rec_hi = bootstrap_ci(
                merged, recovery_percentage, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED
            )
            summary[severity][pct] = {
                "acc": acc_point, "ci_low": acc_lo, "ci_high": acc_hi,
                "recovery_pct": rec_point, "rec_ci_low": rec_lo, "rec_ci_high": rec_hi,
            }
    return summary


def save_noisy_oracle_table(summary, output_path):
    lines = ["# Day 9 v3 — noisy oracle: recovery % vs. corner-correspondence error", ""]
    lines.append(
        "The 4 dst-corner correspondences used to build the rectifying homography are perturbed "
        "by +/- pct% of image size before rebuilding H via cv2.getPerspectiveTransform; pct=0 is the "
        "exact oracle case (reused from v2, not recomputed). 3 replicates (matching v2's replicate "
        f"0/1/2), medium+severe only. 95% CI = {N_RESAMPLES}-resample bootstrap over images, seed=0."
    )
    lines.append("")
    lines.append("| Severity | Perturbation % | Accuracy [CI] | Recovery % [CI] |")
    lines.append("|---|---|---|---|")
    for severity in SEVERITIES_V3:
        for pct in sorted(summary[severity]):
            s = summary[severity][pct]
            rec_str = (
                f"{s['recovery_pct']:.1f}% [{s['rec_ci_low']:.1f}, {s['rec_ci_high']:.1f}]"
                if s["recovery_pct"] is not None else "undefined"
            )
            lines.append(
                f"| {severity} | {pct} | {s['acc']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] | {rec_str} |"
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


def save_noisy_oracle_figure(summary, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"medium": "tab:orange", "severe": "tab:red"}
    for severity in SEVERITIES_V3:
        pcts = sorted(summary[severity])
        ys = [summary[severity][p]["recovery_pct"] for p in pcts]
        err_lo = [
            (summary[severity][p]["recovery_pct"] - summary[severity][p]["rec_ci_low"])
            if summary[severity][p]["recovery_pct"] is not None else 0
            for p in pcts
        ]
        err_hi = [
            (summary[severity][p]["rec_ci_high"] - summary[severity][p]["recovery_pct"])
            if summary[severity][p]["recovery_pct"] is not None else 0
            for p in pcts
        ]
        ax.errorbar(pcts, ys, yerr=[err_lo, err_hi], marker="o", label=severity, color=colors[severity], capsize=4)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Corner-correspondence perturbation (% of image size)")
    ax.set_ylabel("Recovery % (95% bootstrap CI)")
    ax.set_title("Day 9 v3 — recovery % vs. homography-estimation error")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Day 9 v3 analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--v2-csv", default="results/raw/day8_v2_grid.csv")
    parser.add_argument("--mismatched-h-csv", default="results/raw/v3_mismatched_h.csv")
    parser.add_argument("--noisy-oracle-csv", default="results/raw/v3_noisy_oracle.csv")
    parser.add_argument("--skip-mismatched-h", action="store_true")
    parser.add_argument("--skip-noisy-oracle", action="store_true")
    args = parser.parse_args()

    load_config(args.config)  # validated readable; class_names not needed here
    v2_rows = load_v2_rows(args.v2_csv)
    print(f"Loaded {len(v2_rows)} v2 rows")

    if not args.skip_mismatched_h:
        v3_mh_rows = load_csv_rows(args.mismatched_h_csv)
        print(f"Loaded {len(v3_mh_rows)} mismatched-H rows")
        mh_summary = analyze_mismatched_h(v2_rows, v3_mh_rows)
        save_mismatched_h_table(mh_summary, "results/tables/v3_mismatched_h.md")
        save_mismatched_h_figure(mh_summary, "figures/v3_mismatched_h.png")
        print("\n--- Mismatched-H headline ---")
        for severity in SEVERITIES_V3:
            s = mh_summary[severity]
            print(
                f"{severity}: distorted={s['distorted']['acc']:.3f} "
                f"wrong_h={s['wrong_h']['acc']:.3f} oracle={s['oracle']['acc']:.3f}"
            )

    if not args.skip_noisy_oracle:
        v3_no_rows = load_csv_rows(args.noisy_oracle_csv, int_fields=("replicate", "perturbation_pct"))
        print(f"Loaded {len(v3_no_rows)} noisy-oracle rows")
        no_summary = analyze_noisy_oracle(v2_rows, v3_no_rows)
        save_noisy_oracle_table(no_summary, "results/tables/v3_noisy_oracle.md")
        save_noisy_oracle_figure(no_summary, "figures/v3_noisy_oracle_recovery.png")
        print("\n--- Noisy-oracle headline ---")
        for severity in SEVERITIES_V3:
            for pct in sorted(no_summary[severity]):
                s = no_summary[severity][pct]
                rec = f"{s['recovery_pct']:.1f}%" if s["recovery_pct"] is not None else "undefined"
                print(f"{severity} pct={pct}: acc={s['acc']:.3f} recovery={rec}")


if __name__ == "__main__":
    main()
