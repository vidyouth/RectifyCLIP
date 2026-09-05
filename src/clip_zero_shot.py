"""Zero-shot architectural-style classification with a frozen CLIP model.

Day 1 scope: load open_clip ViT-B/32 (laion2b_s34b_b79k pretrained weights),
encode a fixed prompt per class, and classify one image given on the command
line (`--image`). This proved the pipeline works end to end before anything
else was built.

Day 2 scope: batch inference over the whole MonuMAI dataset (`--batch`),
reading model/prompt/class-name/path settings from configs/default.yaml
instead of the hardcoded constants Day 1 used, logging every prediction to a
CSV, and running the same batch over the 3 prompt-template pilot options
without picking a winner (that is Day 6's ablation).

Assumption carried over from Day 1 (documented again here since it still
applies): image preprocessing is entirely open_clip's own default transform
for ViT-B/32 — no separate resize/crop step is added, per the Day 2 "do not
touch preprocessing beyond CLIP's own defaults" rule. `image_size` in
configs/default.yaml is informational only.
"""

import argparse
import csv
from pathlib import Path

import open_clip
import torch
import yaml
from PIL import Image

from data_loader import load_monumai
from metrics import confusion_matrix_counts, per_class_accuracy, top1_accuracy

# --- Day 1 hardcoded defaults, kept only for the single-image --image CLI ---
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
CLASS_NAMES = ["Hispanic-Muslim", "Gothic", "Renaissance", "Baroque"]
PROMPT_TEMPLATE = "a photo of {} architecture"

# Day 2 prompt-template pilot: 3 templates, evaluated but not ranked/chosen
# here. Order matches the plan doc; the config file's own prompt_template
# (used for the "clean baseline" CSV) is one of these three.
PROMPT_TEMPLATES_PILOT = [
    "{}",
    "a photo of {} architecture",
    "a facade in {} architectural style",
]

# Explicit, human-readable slugs for output filenames — auto-slugifying the
# template text breaks for "{}" (every character is non-alphanumeric, so it
# collapses to an empty string). Keyed by the literal template string above.
PROMPT_TEMPLATE_SLUGS = {
    "{}": "bare_class_name",
    "a photo of {} architecture": "a_photo_of_architecture",
    "a facade in {} architectural style": "a_facade_in_architectural_style",
}


def load_config(config_path="configs/default.yaml"):
    """Load the YAML config (model, prompt, class names, paths, seed)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(device: str = "cpu"):
    """Load the frozen CLIP model, its preprocessing transform, and tokenizer.

    Returns (model, preprocess, tokenizer). The model is put in eval mode since
    it is used purely for zero-shot inference — no gradients, no fine-tuning.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer


def classify_image(image_path: str, model, preprocess, tokenizer, device: str = "cpu"):
    """Classify one image against CLASS_NAMES using frozen zero-shot CLIP.

    Day 1 single-image path. Returns (predicted_class, scores) where scores is
    a list of (class_name, softmax_probability) pairs in CLASS_NAMES order.
    """
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    prompts = [PROMPT_TEMPLATE.format(name) for name in CLASS_NAMES]
    text = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1).squeeze(0).tolist()

    scores = list(zip(CLASS_NAMES, probs))
    predicted_class = max(scores, key=lambda pair: pair[1])[0]
    return predicted_class, scores


def encode_text_features(model, tokenizer, class_names, prompt_template, device="cpu"):
    """Encode one prompt per class, return an L2-normalized (C, D) tensor."""
    prompts = [prompt_template.format(name) for name in class_names]
    text = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def encode_image_features(image_paths, model, preprocess, device="cpu", batch_size=32):
    """Encode every image once, return an L2-normalized (N, D) tensor.

    Image features do not depend on the prompt template, so they are computed
    once and reused across all 3 pilot templates rather than re-encoding
    ~1,514 images 3 times over — a deliberate implementation choice for CPU
    throughput, documented in experiments/logs/day2.md. It does not change any
    result: the classification math is identical to encoding fresh per
    template, only the redundant image forward passes are skipped.

    Images that fail to load/decode are skipped with a warning printed to
    stdout; the caller receives features only for the images that succeeded,
    paired with a matching list of the paths that succeeded (same order).
    """
    features = []
    ok_paths = []
    failed_paths = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        batch_ok_paths = []
        for path in batch_paths:
            try:
                tensors.append(preprocess(Image.open(path).convert("RGB")))
                batch_ok_paths.append(path)
            except Exception as exc:  # noqa: BLE001 - log and skip, don't crash a 1.5k-image run
                print(f"WARNING: failed to load {path}: {exc}")
                failed_paths.append(path)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features)
        ok_paths.extend(batch_ok_paths)

        done = start + len(batch_paths)
        print(f"  encoded {done}/{len(image_paths)} images", end="\r")

    print()
    return torch.cat(features, dim=0), ok_paths, failed_paths


def classify_from_features(
    image_features, text_features, image_paths, true_labels, class_names, prompt_template
):
    """Score every image against every class's text feature.

    Returns a list of row dicts: image_path, true_label, predicted_label,
    correct, prompt_template, and one score_<class_name> column per class
    (all 4 softmax scores — this is a 4-way task, so "top-4" is all of them).
    """
    with torch.no_grad():
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)

    rows = []
    for i, image_path in enumerate(image_paths):
        class_probs = probs[i].tolist()
        predicted_label = class_names[max(range(len(class_names)), key=lambda j: class_probs[j])]
        row = {
            "image_path": image_path,
            "true_label": true_labels[i],
            "predicted_label": predicted_label,
            "correct": predicted_label == true_labels[i],
            "prompt_template": prompt_template,
        }
        for class_name, prob in zip(class_names, class_probs):
            row[f"score_{class_name}"] = prob
        rows.append(row)
    return rows


def save_predictions_csv(rows, class_names, output_path):
    """Write prediction rows to CSV, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image_path", "true_label", "predicted_label", "correct", "prompt_template"]
    fieldnames += [f"score_{name}" for name in class_names]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(matrix, class_names, output_path, title="Confusion matrix"):
    """Save a 4x4 confusion-matrix heatmap (counts) to output_path."""
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    max_val = matrix.max() if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_batch(config, device="cpu"):
    """Run the Day 2 clean-baseline batch inference plus the prompt pilot.

    Returns a dict with everything the caller (main()) needs to write
    deliverables: predictions for the config's default prompt_template,
    predictions for all 3 pilot templates, and the list of any images that
    failed to load.
    """
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]

    print(f"Loading dataset from {dataset_root} ...")
    samples = load_monumai(dataset_root)
    image_paths = [path for path, _ in samples]
    true_labels = [label for _, label in samples]
    print(f"Loaded {len(samples)} images.")

    print("Loading CLIP model ...")
    model, preprocess, tokenizer = load_model(device)

    print("Encoding all images once (reused across every prompt template) ...")
    image_features, ok_paths, failed_paths = encode_image_features(
        image_paths, model, preprocess, device
    )
    # Re-align true_labels with any images that failed to load.
    ok_index = {path: i for i, path in enumerate(image_paths)}
    ok_true_labels = [true_labels[ok_index[path]] for path in ok_paths]

    pilot_rows_by_template = {}
    for template in PROMPT_TEMPLATES_PILOT:
        print(f"Classifying with prompt template: {template!r}")
        text_features = encode_text_features(model, tokenizer, class_names, template, device)
        rows = classify_from_features(
            image_features, text_features, ok_paths, ok_true_labels, class_names, template
        )
        pilot_rows_by_template[template] = rows

    default_template = config["prompt_template"]
    clean_baseline_rows = pilot_rows_by_template[default_template]

    return {
        "clean_baseline_rows": clean_baseline_rows,
        "pilot_rows_by_template": pilot_rows_by_template,
        "failed_paths": failed_paths,
        "class_names": class_names,
    }


def write_baseline_summary(pilot_rows_by_template, class_names, default_template, output_path):
    """Write results/tables/day2_baseline_summary.md: accuracy overall + per class + per prompt."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Day 2 clean baseline — accuracy summary", ""]
    lines.append(f"Default (clean baseline) prompt template: `{default_template}`")
    lines.append("")
    lines.append("## Overall + per-class accuracy, per prompt template")
    lines.append("")
    lines.append("| Prompt template | Overall accuracy | " + " | ".join(class_names) + " |")
    lines.append("|---" * (2 + len(class_names)) + "|")

    for template, rows in pilot_rows_by_template.items():
        overall = top1_accuracy(rows)
        per_class = per_class_accuracy(rows, class_names)
        per_class_str = " | ".join(
            f"{per_class[name]:.3f}" if per_class[name] is not None else "n/a"
            for name in class_names
        )
        marker = " (clean baseline)" if template == default_template else ""
        lines.append(f"| `{template}`{marker} | {overall:.3f} | {per_class_str} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot classify facade image(s) with frozen CLIP."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Path to a single input image (Day 1 check).")
    mode.add_argument(
        "--batch", action="store_true", help="Run full-dataset batch inference (Day 2)."
    )
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config (batch mode)."
    )
    args = parser.parse_args()

    device = "cpu"

    if args.image:
        model, preprocess, tokenizer = load_model(device)
        predicted_class, scores = classify_image(args.image, model, preprocess, tokenizer, device)
        print(f"Predicted class: {predicted_class}")
        print("Softmax scores:")
        for name, prob in scores:
            print(f"  {name:16s} {prob:.4f}")
        return

    config = load_config(args.config)
    class_names = config["class_names"]
    default_template = config["prompt_template"]

    result = run_batch(config, device)

    save_predictions_csv(
        result["clean_baseline_rows"], class_names, "results/raw/day2_clean_baseline.csv"
    )
    print("Saved results/raw/day2_clean_baseline.csv")

    for template, rows in result["pilot_rows_by_template"].items():
        if template == default_template:
            continue
        slug = PROMPT_TEMPLATE_SLUGS[template]
        save_predictions_csv(rows, class_names, f"results/raw/day2_prompt_pilot_{slug}.csv")
        print(f"Saved results/raw/day2_prompt_pilot_{slug}.csv")

    matrix = confusion_matrix_counts(result["clean_baseline_rows"], class_names)
    plot_confusion_matrix(
        matrix,
        class_names,
        "figures/day2_confusion_clean.png",
        title=f"Day 2 clean baseline confusion matrix\n(prompt: {default_template!r})",
    )
    print("Saved figures/day2_confusion_clean.png")

    write_baseline_summary(
        result["pilot_rows_by_template"],
        class_names,
        default_template,
        "results/tables/day2_baseline_summary.md",
    )
    print("Saved results/tables/day2_baseline_summary.md")

    overall_accuracy = top1_accuracy(result["clean_baseline_rows"])
    print(f"\nClean baseline accuracy (prompt={default_template!r}): {overall_accuracy:.4f}")
    if result["failed_paths"]:
        print(f"WARNING: {len(result['failed_paths'])} images failed to load: {result['failed_paths']}")


if __name__ == "__main__":
    main()
