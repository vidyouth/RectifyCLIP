"""Loader for the MonuMAI dataset directory layout.

MonuMAI (as distributed at https://github.com/ari-dasci/OD-MonuMAI, in its
MonuMAI_dataset/ subfolder) is laid out as one subfolder per architectural
style, with images directly inside that folder and a nested xml/ subfolder
holding Pascal VOC object-detection annotations for that style:

    data/monumai/
    ├── Baroque/
    │   ├── xml/          (Pascal VOC XML annotations — not used here)
    │   └── *.jpg
    ├── Gothic/
    │   ├── xml/
    │   └── *.jpg
    ├── Hispanic-Muslim/
    │   ├── xml/
    │   └── *.jpg
    └── Renaissance/
        ├── xml/
        └── *.jpg

This project only does whole-image zero-shot classification, so the xml/
annotations (object-level bounding boxes for architectural elements) are
intentionally ignored — the folder name is the only label this project needs.
"""

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Style folder names double as the class labels. Fixed order matters wherever
# label order is compared to CLASS_NAMES elsewhere (e.g. clip_zero_shot.py) —
# keep this list in sync with that one.
STYLE_FOLDERS = ["Hispanic-Muslim", "Gothic", "Renaissance", "Baroque"]


def load_monumai(root):
    """Return a list of (image_path, label_string) tuples for the MonuMAI dataset.

    `root` is the path to the directory containing one subfolder per style
    (e.g. "data/monumai"). Each style subfolder's own nested "xml" directory is
    skipped; every other file directly inside a style subfolder whose
    extension is in IMAGE_EXTENSIONS is collected, labelled with that style's
    folder name.

    Images are returned in sorted order within each style, and styles are
    walked in STYLE_FOLDERS order, so the result is deterministic given the
    same files on disk.
    """
    root = Path(root)
    samples = []

    for style in STYLE_FOLDERS:
        style_dir = root / style
        if not style_dir.is_dir():
            raise FileNotFoundError(f"Expected style folder not found: {style_dir}")

        image_paths = sorted(
            p
            for p in style_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            samples.append((str(image_path), style))

    return samples


if __name__ == "__main__":
    # Quick manual check: python src/data_loader.py [root]
    import sys

    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "data/monumai"
    data = load_monumai(dataset_root)
    print(f"Loaded {len(data)} (image_path, label) pairs from {dataset_root}")
    counts = {}
    for _, label in data:
        counts[label] = counts.get(label, 0) + 1
    for style in STYLE_FOLDERS:
        print(f"  {style:16s} {counts.get(style, 0)}")
