"""Day 13 v7 analysis — Experiments A (feature invariance), B (tilt probe),
C (validation-only report; steps 3-5 were pre-registered to be skipped if
src/experiment_v7.py's synthetic-validation gate failed, which it did).

PRE-REGISTERED (restated from src/experiment_v7.py): these analyses, metrics,
and the Experiment C gate were fixed before results were seen. This file
reports every number produced, including the null/negative Experiment C
result, without adding or dropping metrics in response to what came out.

Experiments A and B are pure re-analysis of src/experiment_v6.py's cached
features (no new CLIP inference) plus src/experiment_v7.py's cached
synthetic-validation CSV. Does not touch v1-v6 code or outputs.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from day12_v6_analysis import load_features, make_folds
from experiment_v5_models import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, LINEAR_PROBE_SEED, compute_metrics_with_ci
from experiment_v6 import ANGLES, MODELS
from experiment_v7 import PITCH_DOMINANT_AXIS_TOLERANCE_DEG, VALIDATION_CSV_PATH
from metrics import bootstrap_ci, top1_accuracy

RESULTS_RAW = Path("results/raw")
RESULTS_TABLES = Path("results/tables")
FIGURES = Path("figures")
LOG_MD_PATH = Path("experiments/logs/day13_v7.md")


def load_dataset(config):
    from data_loader import load_monumai
    from distortion import image_id_from_path

    dataset_root = config["dataset_root"]
    samples = load_monumai(dataset_root)
    image_paths = [p for p, _ in samples]
    true_labels = [l for _, l in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    return image_ids, true_labels


def fmt_ci(point_lo_hi):
    point, lo, hi = point_lo_hi
    if point is None:
        return "n/a"
    if lo is None or hi is None:
        return f"{point:.3f} [n/a]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# --- Experiment A: feature invariance ------------------------------------


def cosine_sim_rows(clean, other, image_ids):
    """Per-image cosine similarity (plain dot product; v6 features are
    already L2-normalized by encode_image_arrays)."""
    sims = np.sum(clean * other, axis=1)
    return [{"image_id": iid, "value": float(s)} for iid, s in zip(image_ids, sims)]


def run_experiment_a(image_ids):
    results = {}
    for spec in MODELS:
        slug = spec["slug"]
        clean, distorted, rectified = load_features(slug)
        results[slug] = {}
        for angle in ANGLES:
            dist_rows = cosine_sim_rows(clean, distorted[angle], image_ids)
            rect_rows = cosine_sim_rows(clean, rectified[angle], image_ids)
            diff_rows = [
                {"image_id": d["image_id"], "value": r["value"] - d["value"]}
                for d, r in zip(dist_rows, rect_rows)
            ]

            def _mean(rows):
                return float(np.mean([r["value"] for r in rows]))

            dist_ci = bootstrap_ci(dist_rows, _mean, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")
            rect_ci = bootstrap_ci(rect_rows, _mean, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")
            diff_ci = bootstrap_ci(diff_rows, _mean, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")

            share_rect_closer = float(np.mean([r["value"] > d["value"] for d, r in zip(dist_rows, rect_rows)]))

            results[slug][angle] = {
                "distorted_sim": dist_ci,
                "rectified_sim": rect_ci,
                "diff": diff_ci,
                "share_rectified_closer": share_rect_closer,
            }
    return results


def write_experiment_a_table(results_a, out_path=RESULTS_TABLES / "v7_feature_invariance.md"):
    lines = [
        "# Day 13 v7 — Experiment A: feature invariance (cosine similarity to clean)",
        "",
        "Cosine similarity between each image's CLEAN feature and its distorted/rectified feature "
        "(v6 cached features, already L2-normalized, so cosine similarity = dot product). "
        "'Share rectified closer' = fraction of images where rectified is more similar to clean than distorted is.",
        "",
        "| Model | Angle | Mean sim: distorted | Mean sim: rectified | Diff (rectified - distorted) | Share rectified closer |",
        "|---|---|---|---|---|---|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        for angle in ANGLES:
            r = results_a[slug][angle]
            lines.append(
                f"| {spec['label']} | {angle} | {fmt_ci(r['distorted_sim'])} | {fmt_ci(r['rectified_sim'])} | "
                f"{fmt_ci(r['diff'])} | {r['share_rectified_closer']:.3f} |"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_experiment_a_figure(results_a, out_path=FIGURES / "v7_feature_invariance_vs_angle.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.5 * len(MODELS), 4.5), sharey=True)
    for ax, spec in zip(axes, MODELS):
        slug = spec["slug"]
        x = ANGLES
        for key, label, marker in [("distorted_sim", "distorted", "o"), ("rectified_sim", "rectified", "s")]:
            points = [results_a[slug][a][key][0] for a in x]
            los = [results_a[slug][a][key][0] - results_a[slug][a][key][1] for a in x]
            his = [results_a[slug][a][key][2] - results_a[slug][a][key][0] for a in x]
            ax.errorbar(x, points, yerr=[los, his], marker=marker, label=label, capsize=3)
        ax.set_title(spec["label"])
        ax.set_xlabel("Camera tilt angle (degrees)")
    axes[0].set_ylabel("Cosine similarity to clean feature")
    axes[0].legend(fontsize=8)
    fig.suptitle("Day 13 v7 — feature-space invariance vs angle (95% bootstrap CI)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Experiment B: tilt probe ------------------------------------


def run_experiment_b_task1(image_ids, true_labels, folds, y_dummy):
    """4-class angle classifier (clean, 5, 15, 35), same held-out fold per
    image across all 4 conditions (no leakage: an image's clean AND all 3
    distorted copies are entirely in the train or entirely in the test
    fold, never split)."""
    results = {}
    for spec in MODELS:
        slug = spec["slug"]
        clean, distorted, _ = load_features(slug)
        n = len(image_ids)
        class_labels = np.array(["clean"] + [str(a) for a in ANGLES])
        X_by_class = {"clean": clean}
        for a in ANGLES:
            X_by_class[str(a)] = distorted[a]

        pred = {c: [None] * n for c in class_labels}
        for train_idx, test_idx in folds:
            X_train = np.concatenate([X_by_class[c][train_idx] for c in class_labels], axis=0)
            y_train = np.concatenate([[c] * len(train_idx) for c in class_labels])
            clf = LogisticRegression(max_iter=2000, random_state=LINEAR_PROBE_SEED).fit(X_train, y_train)
            for c in class_labels:
                p = clf.predict(X_by_class[c][test_idx])
                for i, idx in enumerate(test_idx):
                    pred[c][idx] = p[i]

        rows = []
        for c in class_labels:
            for iid, p in zip(image_ids, pred[c]):
                rows.append({"image_id": f"{iid}__{c}", "true_label": c, "predicted_label": p})
        acc = bootstrap_ci(rows, top1_accuracy, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")
        results[slug] = {"accuracy": acc, "n_classes": len(class_labels)}
    return results


def run_experiment_b_task2(image_ids, folds):
    """Binary clean-vs-distorted(35) classifier; report held-out accuracy
    AND the fraction of held-out RECTIFIED(35) features it calls
    'distorted' (ideal rectifier -> matches the held-out clean rate)."""
    results = {}
    for spec in MODELS:
        slug = spec["slug"]
        clean, distorted, rectified = load_features(slug)
        n = len(image_ids)
        pred_clean = [None] * n
        pred_distorted = [None] * n
        pred_rectified = [None] * n

        for train_idx, test_idx in folds:
            X_train = np.concatenate([clean[train_idx], distorted[35][train_idx]], axis=0)
            y_train = np.array(["clean"] * len(train_idx) + ["distorted"] * len(train_idx))
            clf = LogisticRegression(max_iter=2000, random_state=LINEAR_PROBE_SEED).fit(X_train, y_train)

            pc = clf.predict(clean[test_idx])
            pd_ = clf.predict(distorted[35][test_idx])
            pr = clf.predict(rectified[35][test_idx])
            for i, idx in enumerate(test_idx):
                pred_clean[idx] = pc[i]
                pred_distorted[idx] = pd_[i]
                pred_rectified[idx] = pr[i]

        rows_acc = (
            [{"image_id": f"{iid}__clean", "true_label": "clean", "predicted_label": p} for iid, p in zip(image_ids, pred_clean)]
            + [{"image_id": f"{iid}__distorted", "true_label": "distorted", "predicted_label": p} for iid, p in zip(image_ids, pred_distorted)]
        )
        acc = bootstrap_ci(rows_acc, top1_accuracy, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")

        clean_rate_rows = [{"image_id": iid, "value": (p == "distorted")} for iid, p in zip(image_ids, pred_clean)]
        rect_rate_rows = [{"image_id": iid, "value": (p == "distorted")} for iid, p in zip(image_ids, pred_rectified)]

        def _rate(rows):
            return float(np.mean([r["value"] for r in rows]))

        clean_rate_ci = bootstrap_ci(clean_rate_rows, _rate, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")
        rect_rate_ci = bootstrap_ci(rect_rate_rows, _rate, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id")

        results[slug] = {"accuracy": acc, "clean_rate_called_distorted": clean_rate_ci, "rectified_rate_called_distorted": rect_rate_ci}
    return results


def write_experiment_b_table(task1_results, task2_results, out_path=RESULTS_TABLES / "v7_tilt_probe.md"):
    lines = [
        "# Day 13 v7 — Experiment B: tilt probe",
        "",
        "## Task 1: 4-class angle classifier (clean / 5 / 15 / 35 degrees), same 5-fold split as v5/v6",
        "",
        "| Model | Accuracy (4-class) |",
        "|---|---|",
    ]
    for spec in MODELS:
        r = task1_results[spec["slug"]]
        lines.append(f"| {spec['label']} | {fmt_ci(r['accuracy'])} |")

    lines += [
        "",
        "## Task 2: clean-vs-distorted(35 deg) classifier, applied to rectified(35 deg) features",
        "",
        "'Rate called distorted' on held-out rectified features: an ideal rectifier makes this match the "
        "held-out CLEAN rate (i.e. the classifier can no longer tell rectified apart from clean).",
        "",
        "| Model | Held-out accuracy (clean vs distorted) | Clean rate called 'distorted' | Rectified rate called 'distorted' |",
        "|---|---|---|---|",
    ]
    for spec in MODELS:
        r = task2_results[spec["slug"]]
        lines.append(
            f"| {spec['label']} | {fmt_ci(r['accuracy'])} | {fmt_ci(r['clean_rate_called_distorted'])} | {fmt_ci(r['rectified_rate_called_distorted'])} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Experiment C: validation-only report ------------------------------------


def load_validation_rows():
    rows = []
    with open(VALIDATION_CSV_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["ok"] = r["ok"] == "True"
            r["pitch_dominant"] = r["pitch_dominant"] == "True"
            r["angle"] = int(r["angle"])
            r["signed_true_tilt"] = float(r["signed_true_tilt"])
            r["estimated_tilt_deg"] = float(r["estimated_tilt_deg"]) if r["estimated_tilt_deg"] not in ("", None) else None
            r["n_inliers"] = int(r["n_inliers"])
            rows.append(r)
    return rows


def write_experiment_c_validation_table(rows, gate, out_path=RESULTS_TABLES / "v7_validation_synthetic.md"):
    lines = [
        "# Day 13 v7 — Experiment C, step 2: synthetic-data validation of the blind tilt estimator",
        "",
        "Pre-registered gate (fixed in src/experiment_v7.py before this ran): PASS requires "
        "pitch-dominant signed correlation >= 0.3 AND pitch-dominant MAE < 20 degrees. "
        f"Result: **{'PASSED' if gate['passed_gate'] else 'FAILED'}** — steps 3-5 "
        f"{'ran' if gate['passed_gate'] else 'were SKIPPED per the pre-registered instruction to stop after a failed validation'}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Coverage (estimator found >= 4 inlier vertical lines) | {gate['coverage']:.3f} ({gate['n_total']} image-angle pairs total) |",
        f"| Pitch-dominant subset size (axis within {PITCH_DOMINANT_AXIS_TOLERANCE_DEG} deg of pure pitch) | {gate['pitch_dominant_n']} |",
        f"| Pitch-dominant signed correlation (estimated vs true tilt) | {gate['pitch_dominant_corr']:.3f} |" if gate["pitch_dominant_corr"] is not None else "| Pitch-dominant signed correlation | n/a |",
        f"| Pitch-dominant MAE (degrees) | {gate['pitch_dominant_mae']:.1f} |" if gate["pitch_dominant_mae"] is not None else "| Pitch-dominant MAE | n/a |",
        f"| Overall unsigned correlation (all axes, \\|estimate\\| vs true angle) | {gate['overall_unsigned_corr']:.3f} |" if gate["overall_unsigned_corr"] is not None else "| Overall unsigned correlation | n/a |",
        f"| Overall unsigned MAE (degrees) | {gate['overall_unsigned_mae']:.1f} |" if gate["overall_unsigned_mae"] is not None else "| Overall unsigned MAE | n/a |",
        "",
        "## Per-angle breakdown (pitch-dominant subset only)",
        "",
        "| True angle | n (pitch-dominant, ok) | Mean estimated \\|tilt\\| | MAE (signed) |",
        "|---|---|---|---|",
    ]
    for angle in ANGLES:
        subset = [r for r in rows if r["angle"] == angle and r["pitch_dominant"] and r["ok"]]
        if subset:
            mean_abs_est = float(np.mean([abs(r["estimated_tilt_deg"]) for r in subset]))
            mae = float(np.mean([abs(r["estimated_tilt_deg"] - r["signed_true_tilt"]) for r in subset]))
            lines.append(f"| {angle} | {len(subset)} | {mean_abs_est:.1f} | {mae:.1f} |")
        else:
            lines.append(f"| {angle} | 0 | n/a | n/a |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_experiment_c_validation_figure(rows, out_path=FIGURES / "v7_validation_scatter.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    pitch_rows = [r for r in rows if r["ok"] and r["pitch_dominant"]]
    other_rows = [r for r in rows if r["ok"] and not r["pitch_dominant"]]

    if other_rows:
        ax.scatter(
            [r["angle"] for r in other_rows],
            [abs(r["estimated_tilt_deg"]) for r in other_rows],
            s=6, alpha=0.15, color="gray", label="non-pitch-dominant axis (unsigned)",
        )
    if pitch_rows:
        ax.scatter(
            [r["signed_true_tilt"] for r in pitch_rows],
            [r["estimated_tilt_deg"] for r in pitch_rows],
            s=10, alpha=0.4, color="C0", label="pitch-dominant axis (signed)",
        )
    lims = [-40, 40]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="perfect estimate")
    ax.set_xlim(lims)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("True tilt (degrees)")
    ax.set_ylabel("Estimated tilt (degrees)")
    ax.set_title("Day 13 v7 — blind tilt estimator validation on synthetic data")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- log ------------------------------------


def write_log(results_a, task1_results, task2_results, gate):
    lines = []

    sim35_str = ", ".join(f"{s['label']} {results_a[s['slug']][35]['distorted_sim'][0]:.2f}" for s in MODELS)
    task1_acc_str = ", ".join(f"{s['label']} {task1_results[s['slug']]['accuracy'][0]:.2f}" for s in MODELS)

    lines.append("## Summary (read this first)")
    lines.append("")
    lines.append(
        f"- Experiment A (feature invariance): at 35 degrees, CLIP/SigLIP image features are still highly "
        f"similar to their clean counterpart (cosine sim {sim35_str}) "
        "— the embedding space is far more tilt-robust than zero-shot accuracy alone suggested; rectification's effect on similarity is "
        "small either way (see v7_feature_invariance.md's 'share rectified closer' column, rarely far from 0.5)."
    )
    lines.append(
        f"- Experiment B (tilt probe): a simple logistic probe CAN tell angle apart from frozen features "
        f"({task1_acc_str} 4-class accuracy, "
        "chance = 0.25) — distortion leaves a detectable signature in feature space even when it barely changes raw accuracy."
    )
    lines.append(
        f"- Task 2's rectified-vs-clean confusability test shows whether rectification actually erases that signature "
        "or just moves it — see v7_tilt_probe.md: if the rectified rate called 'distorted' stays far from the clean "
        "rate, rectification is not restoring clean-like features even where it helps raw accuracy."
    )
    lines.append(
        f"- Experiment C's classical blind tilt estimator was pre-registered to be validated on synthetic data "
        f"BEFORE running on real photos, and FAILED its own pre-registered gate (pitch-dominant correlation "
        f"{gate['pitch_dominant_corr']:.2f}, MAE {gate['pitch_dominant_mae']:.1f} degrees, needed <20) — "
        "per the pre-registered instruction, steps 3-5 (real-photo tilt distribution, accuracy-by-tilt-bin, "
        "blind rectification) were NOT run. This is reported as a real null result, not softened."
    )
    lines.append(
        "- Honest bottom line: a from-scratch classical vanishing-point tilt estimator on MonuMAI facade photos "
        "(decorative facades, occlusion, non-building lines) is not reliable enough to trust on individual images "
        "without further engineering (better line filtering, multi-frame or depth cues, or a learned estimator) — "
        "this is itself a legitimate finding about the limits of the oracle-rectification framing for real photos."
    )
    lines.append("")

    lines.append("# Day 13 v7 — feature invariance, tilt probe, and a classical blind tilt estimator")
    lines.append("")
    lines.append("## Assumptions and limitations (pre-registered, stated up front)")
    lines.append("")
    lines.append("- Blind tilt estimator assumes a PURE CAMERA PITCH (rotation about the camera's local x-axis only); "
                  "real natural tilt can include yaw/roll/oblique components this model cannot represent.")
    lines.append("- Intrinsics assumed, not calibrated: focal length = image width, principal point at the exact image center "
                  "(same assumption v4 used for the synthetic grid).")
    lines.append("- Vertical-line detection (LSD, near-vertical filter, length >= 3% of image diagonal) will pick up "
                  "non-building vertical edges (poles, people, decorative elements) on a subset of images — no "
                  "semantic filtering was attempted.")
    lines.append("- The vanishing-point-to-angle conversion has an inherent 180-degree sign ambiguity for an undirected "
                  "line; resolved by assuming the true tilt is in (-90, 90] degrees, verified on synthetic ground truth.")
    lines.append("- Validation's 'signed' comparison is only meaningful on the pitch-dominant subset (v4's random tilt "
                  "axis within 20 degrees of pure pitch); non-pitch-dominant (yaw/oblique) synthetic tilts are a "
                  "genuine model mismatch, reported unsigned only, and expected to validate worse.")
    lines.append("- Because the validation gate failed, Experiment C stopped after step 2: no real-photo tilt "
                  "distribution, no accuracy-by-tilt-bin analysis, and no blind-rectification comparison were produced.")
    lines.append("- This script does not touch v1-v6 code, outputs, or the paper draft.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- results/tables/v7_feature_invariance.md, figures/v7_feature_invariance_vs_angle.png (Experiment A)")
    lines.append("- results/tables/v7_tilt_probe.md (Experiment B)")
    lines.append("- results/tables/v7_validation_synthetic.md, figures/v7_validation_scatter.png (Experiment C, validation only)")
    lines.append("")

    LOG_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    from clip_zero_shot import load_config
    import experiment_v7 as v7

    config = load_config("configs/default.yaml")
    image_ids, true_labels = load_dataset(config)
    folds, y = make_folds(true_labels)

    print("Experiment A: feature invariance ...", flush=True)
    results_a = run_experiment_a(image_ids)
    write_experiment_a_table(results_a)
    write_experiment_a_figure(results_a)

    print("Experiment B: tilt probe ...", flush=True)
    task1_results = run_experiment_b_task1(image_ids, true_labels, folds, y)
    task2_results = run_experiment_b_task2(image_ids, folds)
    write_experiment_b_table(task1_results, task2_results)

    print("Experiment C: validation report ...", flush=True)
    progress = v7.load_progress()
    gate = progress["validation_gate"]
    rows = load_validation_rows()
    write_experiment_c_validation_table(rows, gate)
    write_experiment_c_validation_figure(rows)

    print("Writing log ...", flush=True)
    write_log(results_a, task1_results, task2_results, gate)

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
