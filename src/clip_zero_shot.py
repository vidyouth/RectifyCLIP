"""Zero-shot architectural-style classification with a frozen CLIP model.

Day 1 scope only: load open_clip ViT-B/32 (laion2b_s34b_b79k pretrained weights),
encode a fixed prompt for each of the 4 MonuMAI style classes, and classify one
image given on the command line. This is the "smallest possible end-to-end
pipeline" check from the Day 1 plan — prove image-in / label-out works before
building batch inference, prompt ablation, or config-driven runs (Day 2).

Assumption (documented per Day 1 log): class names and the prompt template are
hardcoded here rather than read from configs/default.yaml. The execution plan's
Day 2 section is what introduces default.yaml with model_name, pretrained_weights,
class_names, prompt_template, image_size, seed, dataset_root — writing that file
today would be doing Day 2's task ahead of schedule (scope discipline in
CLAUDE.md). The single prompt template used today ("a photo of {} architecture")
is a reasonable default, not a tuned choice; Day 2 pilots 3 templates and Day 6
runs the full prompt-sensitivity ablation.
"""

import argparse

import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Exact spelling matches CLAUDE.md and the execution plan — CLIP is
# prompt-sensitive, so this spelling is treated as fixed, not a free choice.
CLASS_NAMES = ["Hispanic-Muslim", "Gothic", "Renaissance", "Baroque"]

PROMPT_TEMPLATE = "a photo of {} architecture"


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

    Returns (predicted_class, scores) where scores is a list of
    (class_name, softmax_probability) pairs in CLASS_NAMES order.
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


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot classify one facade image with frozen CLIP."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    args = parser.parse_args()

    device = "cpu"
    model, preprocess, tokenizer = load_model(device)
    predicted_class, scores = classify_image(
        args.image, model, preprocess, tokenizer, device
    )

    print(f"Predicted class: {predicted_class}")
    print("Softmax scores:")
    for name, prob in scores:
        print(f"  {name:16s} {prob:.4f}")


if __name__ == "__main__":
    main()
