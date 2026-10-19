"""Day 11 v5 — stronger zero-shot setups and a linear-probe baseline, clean images only.

Motivation: v1-v4 all measured ViT-B/32 zero-shot CLIP under perspective
distortion and found it mediocre even on clean images. Before extending the
distortion experiment, this answers "why use CLIP at all if it performs
poorly?" — by checking, on the 1,514 clean MonuMAI images only (no
distortion/rectification here), whether a bigger CLIP backbone, a SigLIP
backbone, prompt ensembling, a label-supervised calibration diagnostic, or a
frozen-feature linear probe does meaningfully better than the v1-v4 baseline.
This never touches or recomputes any v1-v4 file.

Models evaluated: see MODELS below. Pretrained tags are verified against
this environment's open_clip.list_pretrained() at run start (not assumed) —
see check_and_log_pretrained_tags() and the "Models & tags checked" section
written to experiments/logs/day11_v5.md.

Per-model flow, each step independently cached/resumable (see run_model()):
  1. time-estimate on a small fixed calibration subset; skip the model
     (logged, not silently) if the extrapolated full-dataset encode time
     exceeds TIME_BUDGET_SECONDS — a conservative cutoff chosen because this
     runs unattended across 4 models plus a 5-fold probe on CPU only.
  2. encode all images once -> results/raw/v5_features_<slug>.npy (frozen
     features reused by every later step, per the instruction).
  3. zero-shot, single prompt (PROMPT_SINGLE, same template as
     configs/default.yaml's existing baseline).
  4. zero-shot, 8-template ensemble (PROMPT_ENSEMBLE_TEMPLATES, averaged
     L2-normalized text features) — templates fixed below BEFORE any model
     was run, never adjusted by per-class accuracy.
  5. ensemble + calibration diagnostic: per-class additive logit bias fit by
     2-fold cross-fitting (fit on one stratified half, evaluate on the other,
     swap, so every image gets a held-out-fit prediction). This uses true
     labels to fit the bias, so it is NOT a zero-shot number — it is reported
     and labelled explicitly as a diagnostic, never as a zero-shot result.
  6. linear probe: 5-fold stratified cross-validated logistic regression on
     the same frozen features (out-of-fold predictions only, so every image's
     prediction comes from a fold that never saw it during training).

All four setups are scored with top-1 accuracy, balanced accuracy, and
macro-F1, each with a bootstrap 95% CI resampled at the image level (1000
resamples, fixed seed) via metrics.bootstrap_ci.

A note on cross-model logit scale: predicted-label argmax is invariant to
multiplying all 4 class logits by the same positive constant, so every model
here (including SigLIP, whose own training uses a learned logit_scale/bias
this script does not reuse) is scored with the same fixed "100 * cosine
similarity" convention clip_zero_shot.py already uses elsewhere in this
project. This only matters for the calibration diagnostic's bias fit, which
is itself invariant to that same global rescaling (it is absorbed into the
fitted bias).
"""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from clip_zero_shot import encode_image_features, encode_text_features, load_config
from data_loader import load_monumai
from distortion import image_id_from_path
from metrics import balanced_accuracy, bootstrap_ci, macro_f1, top1_accuracy

MODELS = [
    {
        "name": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "slug": "clip_vit_b32",
        "label": "CLIP ViT-B/32",
    },
    {
        "name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "slug": "clip_vit_b16",
        "label": "CLIP ViT-B/16",
    },
    {
        "name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "slug": "clip_vit_l14",
        "label": "CLIP ViT-L/14",
    },
    {
        "name": "ViT-B-16-SigLIP",
        "pretrained": "webli",
        "slug": "siglip_vit_b16",
        "label": "SigLIP ViT-B/16",
    },
]

# Conservative cutoff: this runs unattended, overnight, across 4 models plus
# a 5-fold linear probe on CPU only. A model whose calibration estimate
# implies more than this is skipped rather than risking the rest of the
# night on one model. See experiments/logs/day11_v5.md for the reasoning
# restated next to whichever model(s) this affects.
TIME_BUDGET_SECONDS = 90 * 60

# Number of images used for the pre-commit time estimate (same images every
# run: the dataset's first N in load_monumai's deterministic order).
CALIBRATION_PROBE_N = 32

BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
CALIBRATION_SPLIT_SEED = 42
LINEAR_PROBE_SEED = 42
LINEAR_PROBE_N_FOLDS = 5

# Matches configs/default.yaml's existing prompt_template exactly, so this is
# a genuine "does ensembling/bigger-model help over the existing baseline"
# comparison, not a new single prompt.
PROMPT_SINGLE = "a photo of {} architecture"

# Fixed in advance, before any model was run or any accuracy was looked at.
# Do NOT edit this list after seeing results — that would make it a
# per-class-tuned ensemble instead of a fixed one.
PROMPT_ENSEMBLE_TEMPLATES = [
    "a photo of {} architecture",
    "a photo of a {} building",
    "an image of {} style architecture",
    "a photograph of a building in the {} architectural style",
    "{} architecture",
    "a building in the {} style",
    "a facade in {} architectural style",
    "this building is an example of {} architecture",
]

RESULTS_RAW = Path("results/raw")
RESULTS_TABLES = Path("results/tables")
FIGURES = Path("figures")
PROGRESS_PATH = RESULTS_RAW / "v5_progress.json"
RUN_CONFIG_PATH = Path("experiments/configs/day11_v5_run.json")
LOG_MD_PATH = Path("experiments/logs/day11_v5.md")


def ts():
    """Short timestamp prefix for stdout log lines (this process's own log, not experiments/logs)."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def git_commit_hash():
    """Current git commit hash, or 'unknown' if unavailable (same pattern as src/experiment_v4.py)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_progress(path):
    """Load the resumability progress file, or an empty skeleton if absent."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"pretrained_tags_checked": False, "models": {}}


def save_progress(progress, path):
    """Persist the progress dict, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def cached_json(path, compute_fn):
    """Return the JSON at `path` if present, else call compute_fn(), save it, and return it.

    The resumability primitive for every per-model, per-setup prediction
    output: if the process is interrupted and restarted, every step whose
    output file already exists on disk is loaded instead of recomputed.
    """
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    result = compute_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def check_and_log_pretrained_tags():
    """Verify every (name, pretrained) pair in MODELS actually exists in this
    open_clip install's own list_pretrained() registry, rather than assuming
    the plan's intended tags are still valid. Returns a list of per-model
    check dicts for logging; does not raise on a missing tag (that model is
    skipped later, in run()), so one bad tag doesn't crash the whole run.
    """
    available = set(open_clip.list_pretrained())
    checks = []
    for spec in MODELS:
        exists = (spec["name"], spec["pretrained"]) in available
        checks.append({"name": spec["name"], "pretrained": spec["pretrained"], "exists": exists})
    return checks


def subsample_per_class(samples, n_per_class, class_names):
    """First n_per_class images of each class, in load_monumai's own
    deterministic order — used only by --smoke-test, never by the real run."""
    out = []
    counts = {c: 0 for c in class_names}
    for path, label in samples:
        if counts[label] < n_per_class:
            out.append((path, label))
            counts[label] += 1
    return out


def load_model_generic(name, pretrained, device="cpu"):
    """Load any open_clip model/pretrained pair (unlike clip_zero_shot.load_model,
    which hardcodes ViT-B/32 — this file evaluates several backbones)."""
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(name)
    return model, preprocess, tokenizer


def encode_text_features_ensemble(model, tokenizer, class_names, templates, device="cpu"):
    """Average L2-normalized per-template text features across `templates`,
    then renormalize — the standard CLIP prompt-ensembling recipe."""
    per_template = []
    for template in templates:
        feats = encode_text_features(model, tokenizer, class_names, template, device)
        per_template.append(feats)
    stacked = torch.stack(per_template, dim=0)
    mean_feats = stacked.mean(dim=0)
    mean_feats = mean_feats / mean_feats.norm(dim=-1, keepdim=True)
    return mean_feats


def predict_labels(image_features, text_features, class_names, bias=None):
    """Argmax predicted label per image from image/text features, optionally
    with a per-class additive logit bias (used by the calibration diagnostic;
    None for the plain zero-shot setups)."""
    with torch.no_grad():
        logits = 100.0 * image_features @ text_features.T
        if bias is not None:
            logits = logits + torch.as_tensor(bias, dtype=logits.dtype)
        best_idx = logits.argmax(dim=-1).tolist()
    return [class_names[i] for i in best_idx]


def fit_bias_offsets(logits, true_idx, n_classes):
    """Fit a per-class additive logit bias b (b[0] fixed at 0 for
    identifiability — softmax is invariant to a uniform shift of all
    classes) that maximizes log-likelihood of true_idx under
    softmax(logits + b), via BFGS. Deterministic: this is a small convex
    problem solved from a fixed zero start, so no seed is needed for the
    optimization itself (the randomness is only in which half of the data
    gets passed in, upstream)."""

    def neg_log_lik(b_free):
        b = np.concatenate([[0.0], b_free])
        shifted = logits + b
        shifted = shifted - shifted.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(true_idx)), true_idx].mean()

    x0 = np.zeros(n_classes - 1)
    res = minimize(neg_log_lik, x0, method="BFGS")
    return np.concatenate([[0.0], res.x])


def cross_fit_calibration(image_features, text_features, class_names, true_labels, seed=CALIBRATION_SPLIT_SEED):
    """2-fold cross-fit calibration diagnostic: split images into two
    stratified halves (fixed seed), fit a per-class bias on each half, and
    predict each half using the bias fit on the OTHER half. Every image gets
    exactly one held-out-fit prediction. This uses true labels to fit the
    bias, so the result is a diagnostic, not a zero-shot number — callers
    must label it as such.
    """
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    true_idx = np.array([class_to_idx[label] for label in true_labels])
    n = len(true_labels)

    with torch.no_grad():
        logits = (100.0 * image_features @ text_features.T).cpu().numpy()

    half_a, half_b = train_test_split(
        np.arange(n), test_size=0.5, random_state=seed, stratify=true_idx
    )

    bias_from_a = fit_bias_offsets(logits[half_a], true_idx[half_a], len(class_names))
    bias_from_b = fit_bias_offsets(logits[half_b], true_idx[half_b], len(class_names))

    predicted = [None] * n
    for i in half_b:
        predicted[i] = class_names[int(np.argmax(logits[i] + bias_from_a))]
    for i in half_a:
        predicted[i] = class_names[int(np.argmax(logits[i] + bias_from_b))]

    diagnostics = {"bias_from_half_a": bias_from_a.tolist(), "bias_from_half_b": bias_from_b.tolist()}
    return predicted, diagnostics


def run_linear_probe(features_np, true_labels, seed=LINEAR_PROBE_SEED, n_folds=LINEAR_PROBE_N_FOLDS):
    """5-fold stratified cross-validated logistic regression on frozen
    features. Returns out-of-fold predicted labels for every image — each
    image's prediction comes from a fold whose training set never saw it."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    y = np.array(true_labels)
    oof_pred = cross_val_predict(clf, features_np, y, cv=skf)
    return oof_pred.tolist()


def estimate_seconds_per_image(model, preprocess, device, calibration_paths):
    """Time a forward pass over `calibration_paths` (same images every run)
    and return seconds/image, used to decide (before committing to the full
    1,514-image encode) whether a model fits the time budget."""
    t0 = time.time()
    encode_image_features(calibration_paths, model, preprocess, device, batch_size=len(calibration_paths))
    elapsed = time.time() - t0
    return elapsed / len(calibration_paths)


def compute_metrics_with_ci(rows, class_names):
    """accuracy / balanced accuracy / macro-F1, each as (point, ci_low, ci_high)
    from a 1000-resample image-level bootstrap, fixed seed."""

    def _acc(r):
        return top1_accuracy(r)

    def _bal(r):
        return balanced_accuracy(r, class_names)

    def _f1(r):
        return macro_f1(r, class_names)

    return {
        "accuracy": bootstrap_ci(rows, _acc, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id"),
        "balanced_accuracy": bootstrap_ci(rows, _bal, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id"),
        "macro_f1": bootstrap_ci(rows, _f1, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, image_id_field="image_id"),
    }


def run_model(spec, image_paths, image_ids, true_labels, class_names, out_prefix, device="cpu"):
    """Run the full per-model pipeline (steps 2-6 of the module docstring).

    Returns a dict: {"status": "done", "seconds_per_image": float,
    "setups": {setup_name: {"metrics": ..., }}} or
    {"status": "skipped"/"error", "reason": str}.
    Every expensive step is cached to disk under `out_prefix` first, so a
    second run resumes instead of recomputing.
    """
    slug = spec["slug"]
    print(f"[{ts()}]   loading model {spec['name']} / {spec['pretrained']} ...", flush=True)
    model, preprocess, tokenizer = load_model_generic(spec["name"], spec["pretrained"], device)

    calibration_paths = image_paths[:CALIBRATION_PROBE_N]
    sec_per_img_estimate = estimate_seconds_per_image(model, preprocess, device, calibration_paths)
    est_total = sec_per_img_estimate * len(image_paths)
    print(
        f"[{ts()}]   time estimate: {sec_per_img_estimate:.3f}s/image "
        f"-> {est_total/60:.1f} min for {len(image_paths)} images (budget {TIME_BUDGET_SECONDS/60:.0f} min)",
        flush=True,
    )
    if est_total > TIME_BUDGET_SECONDS:
        reason = (
            f"estimated {est_total/60:.1f} min for {len(image_paths)} images exceeds the "
            f"{TIME_BUDGET_SECONDS/60:.0f}-min conservative CPU budget (estimate from a "
            f"{CALIBRATION_PROBE_N}-image calibration subset)"
        )
        return {"status": "skipped", "reason": reason, "estimated_seconds_per_image": sec_per_img_estimate}

    features_path = RESULTS_RAW / f"v5_features_{out_prefix}.npy"
    if features_path.exists():
        features_np = np.load(features_path)
        if features_np.shape[0] != len(image_paths):
            raise RuntimeError(
                f"{features_path} has {features_np.shape[0]} rows but dataset now has "
                f"{len(image_paths)} images — delete the stale cache to recompute."
            )
        image_features = torch.tensor(features_np, dtype=torch.float32)
        actual_sec_per_img = None
        print(f"[{ts()}]   reusing cached features from {features_path}", flush=True)
    else:
        print(f"[{ts()}]   encoding all {len(image_paths)} images ...", flush=True)
        t0 = time.time()
        image_features, ok_paths, failed_paths = encode_image_features(image_paths, model, preprocess, device)
        elapsed = time.time() - t0
        if failed_paths:
            raise RuntimeError(
                f"{len(failed_paths)} image(s) failed to load for {slug}: {failed_paths} — "
                "MonuMAI images should all load; investigate rather than silently dropping rows "
                "(image_ids/true_labels are not re-aligned to ok_paths here)."
            )
        actual_sec_per_img = elapsed / len(ok_paths)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(features_path, image_features.numpy())
        print(f"[{ts()}]   encoded in {elapsed/60:.1f} min ({actual_sec_per_img:.3f}s/image); saved {features_path}", flush=True)

    sec_per_img = actual_sec_per_img if actual_sec_per_img is not None else sec_per_img_estimate

    def rows_from_pred_map(pred_map):
        return [
            {"image_id": iid, "true_label": tl, "predicted_label": pred_map[iid]}
            for iid, tl in zip(image_ids, true_labels)
        ]

    setups = {}

    # --- zero-shot, single prompt ---
    print(f"[{ts()}]   zero-shot single prompt ...", flush=True)
    single_path = RESULTS_RAW / f"v5_{out_prefix}_zs_single.json"

    def compute_single():
        text_features = encode_text_features(model, tokenizer, class_names, PROMPT_SINGLE, device)
        preds = predict_labels(image_features, text_features, class_names)
        return dict(zip(image_ids, preds))

    pred_map = cached_json(single_path, compute_single)
    setups["zero_shot_single"] = {"metrics": compute_metrics_with_ci(rows_from_pred_map(pred_map), class_names)}

    # --- zero-shot, 8-template ensemble ---
    print(f"[{ts()}]   zero-shot ensemble ({len(PROMPT_ENSEMBLE_TEMPLATES)} templates) ...", flush=True)
    ensemble_path = RESULTS_RAW / f"v5_{out_prefix}_zs_ensemble.json"

    def compute_ensemble():
        text_features = encode_text_features_ensemble(model, tokenizer, class_names, PROMPT_ENSEMBLE_TEMPLATES, device)
        preds = predict_labels(image_features, text_features, class_names)
        return dict(zip(image_ids, preds))

    ensemble_pred_map = cached_json(ensemble_path, compute_ensemble)
    setups["zero_shot_ensemble"] = {
        "metrics": compute_metrics_with_ci(rows_from_pred_map(ensemble_pred_map), class_names)
    }

    # --- ensemble + calibration diagnostic (uses labels; not zero-shot) ---
    print(f"[{ts()}]   ensemble + calibration diagnostic (uses labels; DIAGNOSTIC only) ...", flush=True)
    calibrated_path = RESULTS_RAW / f"v5_{out_prefix}_ensemble_calibrated.json"

    def compute_calibrated():
        text_features = encode_text_features_ensemble(model, tokenizer, class_names, PROMPT_ENSEMBLE_TEMPLATES, device)
        preds, diagnostics = cross_fit_calibration(image_features, text_features, class_names, true_labels)
        return {"predictions": dict(zip(image_ids, preds)), "diagnostics": diagnostics}

    calibrated_result = cached_json(calibrated_path, compute_calibrated)
    setups["ensemble_calibrated_diagnostic"] = {
        "metrics": compute_metrics_with_ci(rows_from_pred_map(calibrated_result["predictions"]), class_names)
    }

    # --- linear probe ---
    print(f"[{ts()}]   linear probe ({LINEAR_PROBE_N_FOLDS}-fold stratified CV) ...", flush=True)
    probe_path = RESULTS_RAW / f"v5_{out_prefix}_linear_probe.json"

    def compute_probe():
        preds = run_linear_probe(image_features.numpy(), true_labels)
        return dict(zip(image_ids, preds))

    probe_pred_map = cached_json(probe_path, compute_probe)
    setups["linear_probe"] = {"metrics": compute_metrics_with_ci(rows_from_pred_map(probe_pred_map), class_names)}

    return {"status": "done", "seconds_per_image": sec_per_img, "setups": setups}


def write_run_config(class_names, n_images, seed):
    """experiments/configs/day11_v5_run.json — exact config for reproducibility,
    per CLAUDE.md working rule 1."""
    RUN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "day": "11_v5",
        "git_commit": git_commit_hash(),
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "n_images": n_images,
        "class_names": class_names,
        "models": MODELS,
        "prompt_single": PROMPT_SINGLE,
        "prompt_ensemble_templates": PROMPT_ENSEMBLE_TEMPLATES,
        "calibration_split_seed": CALIBRATION_SPLIT_SEED,
        "linear_probe_seed": LINEAR_PROBE_SEED,
        "linear_probe_n_folds": LINEAR_PROBE_N_FOLDS,
        "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
    }
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


SETUP_ORDER = [
    ("zero_shot_single", "Zero-shot, single prompt"),
    ("zero_shot_ensemble", "Zero-shot, 8-prompt ensemble"),
    ("ensemble_calibrated_diagnostic", "Ensemble + calibration (DIAGNOSTIC, uses labels)"),
    ("linear_probe", "Linear probe (5-fold CV, frozen features)"),
]


def fmt_ci(point_lo_hi):
    """'0.812 [0.791, 0.833]' from a (point, lo, hi) bootstrap_ci tuple, or 'n/a'."""
    point, lo, hi = point_lo_hi
    if point is None:
        return "n/a"
    if lo is None or hi is None:
        return f"{point:.3f} [n/a]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def write_table(progress, out_path=RESULTS_TABLES / "v5_clean_models.md"):
    """results/tables/v5_clean_models.md — rows = model x setup, columns =
    accuracy / balanced accuracy / macro-F1 (each with 95% bootstrap CI)."""
    lines = [
        "# Day 11 v5 — clean-image model/prompt/calibration/probe comparison",
        "",
        "All numbers on the same 1,514 clean MonuMAI images, no distortion. "
        "CIs are 95% bootstrap intervals (1000 resamples, image-level).",
        "",
        "| Model | Setup | Accuracy | Balanced accuracy | Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for spec in MODELS:
        slug = spec["slug"]
        model_entry = progress["models"].get(slug, {})
        if model_entry.get("status") == "skipped":
            lines.append(f"| {spec['label']} | *(skipped)* | {model_entry.get('reason', '')} | | |")
            continue
        if model_entry.get("status") != "done":
            lines.append(f"| {spec['label']} | *(not completed: {model_entry.get('status', 'pending')})* | | | |")
            continue
        for setup_key, setup_label in SETUP_ORDER:
            setup = model_entry["setups"].get(setup_key)
            if setup is None:
                continue
            m = setup["metrics"]
            lines.append(
                f"| {spec['label']} | {setup_label} | {fmt_ci(m['accuracy'])} | "
                f"{fmt_ci(m['balanced_accuracy'])} | {fmt_ci(m['macro_f1'])} |"
            )

    lines.append("")
    lines.append("## Seconds per image (full-dataset encode, 1,514 images)")
    lines.append("")
    lines.append("| Model | seconds/image |")
    lines.append("|---|---|")
    for spec in MODELS:
        model_entry = progress["models"].get(spec["slug"], {})
        if model_entry.get("status") == "done":
            lines.append(f"| {spec['label']} | {model_entry['seconds_per_image']:.3f} |")
        elif model_entry.get("status") == "skipped":
            lines.append(f"| {spec['label']} | skipped (est. {model_entry.get('estimated_seconds_per_image', float('nan')):.3f}) |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_figure(progress, out_path=FIGURES / "v5_clean_models.png"):
    """figures/v5_clean_models.png — grouped bar chart, accuracy by model x setup."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done_specs = [s for s in MODELS if progress["models"].get(s["slug"], {}).get("status") == "done"]
    if not done_specs:
        print(f"[{ts()}] no completed models — skipping figure", flush=True)
        return

    setup_keys = [k for k, _ in SETUP_ORDER]
    n_setups = len(setup_keys)
    n_models = len(done_specs)
    x = np.arange(n_models)
    width = 0.8 / n_setups

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (setup_key, setup_label) in enumerate(SETUP_ORDER):
        values = []
        errs_lo = []
        errs_hi = []
        for spec in done_specs:
            m = progress["models"][spec["slug"]]["setups"][setup_key]["metrics"]["accuracy"]
            point, lo, hi = m
            values.append(point if point is not None else 0.0)
            errs_lo.append((point - lo) if (point is not None and lo is not None) else 0.0)
            errs_hi.append((hi - point) if (point is not None and hi is not None) else 0.0)
        offset = (i - (n_setups - 1) / 2) * width
        ax.bar(
            x + offset, values, width, label=setup_label,
            yerr=[errs_lo, errs_hi], capsize=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in done_specs], rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Day 11 v5 — clean-image accuracy by model x setup (95% bootstrap CI)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def project_reduced_grid_hours(seconds_per_image, n_images=1514, n_angles=3, n_replicates=1):
    """Rough wall-clock estimate for the reduced distortion grid named in the
    instructions (angles 5/15/35, 1 replicate, all 1,514 images): the clean
    condition is computed once per image regardless of angle, while
    distorted and rectified are each computed once per (image, angle,
    replicate) — i.e. n_images * (1 + 2 * n_angles * n_replicates) forward
    passes total. This is a projection, not a measurement — no distortion or
    rectification code runs in this script."""
    n_forward_passes = n_images * (1 + 2 * n_angles * n_replicates)
    return seconds_per_image * n_forward_passes / 3600.0


def write_log(progress, class_names, n_images, pretrained_checks):
    """Regenerate experiments/logs/day11_v5.md in full from `progress` every
    time it's called (same spirit as 'plots are always regenerable from
    CSVs' — this log is always regenerable from progress.json), so a run
    interrupted partway still leaves an accurate log, and resuming never
    produces duplicated sections."""
    lines = []

    all_statuses = [progress["models"].get(s["slug"], {}).get("status") for s in MODELS]
    run_complete = all(status in ("done", "skipped", "error") for status in all_statuses) and None not in all_statuses

    if run_complete:
        done = [s for s in MODELS if progress["models"][s["slug"]]["status"] == "done"]
        skipped = [s for s in MODELS if progress["models"][s["slug"]]["status"] == "skipped"]
        errored = [s for s in MODELS if progress["models"][s["slug"]]["status"] == "error"]

        # Rank completed models by linear-probe accuracy (the best-case
        # upper bound each backbone's frozen features can reach) to pick a
        # conservative recommendation.
        ranked = sorted(
            done,
            key=lambda s: progress["models"][s["slug"]]["setups"]["linear_probe"]["metrics"]["accuracy"][0] or 0.0,
            reverse=True,
        )
        top2 = ranked[:2]

        lines.append("## Summary (read this first)")
        lines.append("")
        if top2:
            names = " and ".join(s["label"] for s in top2)
            lines.append(
                f"- {len(done)}/{len(MODELS)} models completed on CPU; "
                f"{len(skipped)} skipped on time budget, {len(errored)} errored."
            )
            best = ranked[0]
            best_acc = progress["models"][best["slug"]]["setups"]["linear_probe"]["metrics"]["accuracy"][0]
            lines.append(
                f"- Best clean-image result: {best['label']}'s linear probe at "
                f"{best_acc:.3f} accuracy (frozen features, 5-fold CV) — see the "
                "full table for every model/setup."
            )
            lines.append(
                f"- Recommendation for the next distortion experiment: {names}, plus the linear "
                "probe as a non-zero-shot reference point (ranked by clean linear-probe accuracy, "
                "a conservative proxy for which backbones' features carry the most style signal)."
            )
            lines.append(
                "- Report the direction honestly either way: if the gap between models/setups is "
                "small, that itself is the finding (CLIP zero-shot is not far from the probe "
                "ceiling, or vice versa) — do not oversell a small gap as a strong justification."
            )
            lines.append("- Time estimates for the reduced distortion grid (angles 5/15/35, 1 replicate, 1,514 images):")
            for spec in top2 + ([ranked[-1]] if len(ranked) > 2 and ranked[-1] not in top2 else []):
                sec = progress["models"][spec["slug"]]["seconds_per_image"]
                hours = project_reduced_grid_hours(sec)
                lines.append(f"  - {spec['label']}: ~{hours:.1f} hours ({sec:.3f}s/image measured on clean-image encode)")
        else:
            lines.append("- No model completed within the time budget. See per-model skip reasons below.")
        lines.append("")

    lines.append("# Day 11 v5 — clean-image model/prompt/calibration/probe comparison")
    lines.append("")
    lines.append(f"Dataset: {n_images} clean MonuMAI images (no distortion/rectification in this script).")
    lines.append(f"Git commit at run time: {git_commit_hash()}")
    lines.append("")

    lines.append("## Models & tags checked")
    lines.append("")
    lines.append("Verified against this environment's `open_clip.list_pretrained()` before running:")
    lines.append("")
    for check in pretrained_checks:
        status = "exists" if check["exists"] else "DOES NOT EXIST — this model will fail/be skipped"
        lines.append(f"- `{check['name']}` / `{check['pretrained']}`: {status}")
    lines.append("")

    lines.append("## Prompt templates")
    lines.append("")
    lines.append(f"Single prompt (matches configs/default.yaml): `{PROMPT_SINGLE}`")
    lines.append("")
    lines.append("8-template ensemble (fixed before any model was run; never adjusted by per-class accuracy):")
    lines.append("")
    for t in PROMPT_ENSEMBLE_TEMPLATES:
        lines.append(f"- `{t}`")
    lines.append("")

    lines.append("## Per-model results / status")
    lines.append("")
    for spec in MODELS:
        entry = progress["models"].get(spec["slug"], {})
        status = entry.get("status", "pending")
        lines.append(f"### {spec['label']} (`{spec['name']}` / `{spec['pretrained']}`) — {status}")
        lines.append("")
        if status == "skipped":
            lines.append(f"Skipped: {entry.get('reason')}")
        elif status == "error":
            lines.append(f"Errored: {entry.get('reason')}")
        elif status == "done":
            lines.append(f"Measured encode speed: {entry['seconds_per_image']:.3f} s/image.")
            lines.append("")
            lines.append("| Setup | Accuracy | Balanced accuracy | Macro-F1 |")
            lines.append("|---|---|---|---|")
            for setup_key, setup_label in SETUP_ORDER:
                m = entry["setups"][setup_key]["metrics"]
                lines.append(
                    f"| {setup_label} | {fmt_ci(m['accuracy'])} | {fmt_ci(m['balanced_accuracy'])} | {fmt_ci(m['macro_f1'])} |"
                )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The 'ensemble + calibration' row fits a per-class logit bias using true labels "
        "(2-fold cross-fit, so every image's prediction is from a held-out fit) — it is a "
        "diagnostic, not a zero-shot number, and is labelled as such everywhere it's reported."
    )
    lines.append(
        f"- Conservative CPU time budget: {TIME_BUDGET_SECONDS/60:.0f} minutes/model, estimated from "
        f"a {CALIBRATION_PROBE_N}-image calibration subset before committing to the full encode."
    )
    lines.append("- This script does not touch v1-v4 code, outputs, or the paper draft.")
    lines.append("")

    LOG_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Day 11 v5: clean-image model/prompt/calibration/probe comparison.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=0,
        help="If >0, use only this many images per class, writing to isolated _smoketest-suffixed "
        "paths (never the real v5 outputs/progress file) — for a fast correctness check only.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    seed = config["seed"]

    print(f"[{ts()}] Loading dataset from {dataset_root} ...", flush=True)
    samples = load_monumai(dataset_root)

    suffix = ""
    if args.smoke_test:
        samples = subsample_per_class(samples, args.smoke_test, class_names)
        suffix = "_smoketest"
        global PROGRESS_PATH, LOG_MD_PATH, RUN_CONFIG_PATH
        PROGRESS_PATH = RESULTS_RAW / "v5_progress_smoketest.json"
        LOG_MD_PATH = Path("experiments/logs/day11_v5_smoketest.md")
        RUN_CONFIG_PATH = Path("experiments/configs/day11_v5_smoketest_run.json")

    image_paths = [p for p, _ in samples]
    true_labels = [label for _, label in samples]
    image_ids = [image_id_from_path(p, dataset_root) for p in image_paths]
    print(f"[{ts()}] {len(samples)} images loaded.", flush=True)

    write_run_config(class_names, len(samples), seed)

    progress = load_progress(PROGRESS_PATH)
    if not progress.get("pretrained_tags_checked"):
        pretrained_checks = check_and_log_pretrained_tags()
        progress["pretrained_tags_checked"] = True
        progress["pretrained_checks"] = pretrained_checks
        save_progress(progress, PROGRESS_PATH)
    else:
        pretrained_checks = progress["pretrained_checks"]

    for check in pretrained_checks:
        tag = "OK" if check["exists"] else "MISSING"
        print(f"[{ts()}] pretrained tag check: {check['name']} / {check['pretrained']} -> {tag}", flush=True)

    for spec in MODELS:
        slug = spec["slug"]
        existing = progress["models"].get(slug, {})
        if existing.get("status") in ("done", "skipped"):
            print(f"[{ts()}] === {spec['label']}: already {existing['status']} (resumed) ===", flush=True)
            continue

        print(f"[{ts()}] === Model: {spec['label']} ({spec['name']}, {spec['pretrained']}) ===", flush=True)
        check = next((c for c in pretrained_checks if c["name"] == spec["name"] and c["pretrained"] == spec["pretrained"]), None)
        if check is not None and not check["exists"]:
            result = {"status": "skipped", "reason": f"pretrained tag {spec['pretrained']!r} not found in this open_clip install"}
        else:
            out_prefix = slug + suffix
            try:
                result = run_model(spec, image_paths, image_ids, true_labels, class_names, out_prefix)
            except Exception as exc:  # noqa: BLE001 - keep the run going for the remaining models
                print(f"[{ts()}]   ERROR: {exc}", flush=True)
                result = {"status": "error", "reason": str(exc)}

        progress["models"][slug] = result
        save_progress(progress, PROGRESS_PATH)
        write_log(progress, class_names, len(samples), pretrained_checks)
        print(f"[{ts()}] === {spec['label']}: {result['status']} ===", flush=True)

    if not args.smoke_test:
        write_table(progress)
        write_figure(progress)
        print(f"[{ts()}] Saved {RESULTS_TABLES / 'v5_clean_models.md'} and {FIGURES / 'v5_clean_models.png'}", flush=True)

    print(f"[{ts()}] DONE.", flush=True)


if __name__ == "__main__":
    main()
