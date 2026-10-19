"""Controlled synthetic perspective (projective) distortion.

Core function `distort(image, severity, seed)` takes the 4 corners of an
image, perturbs each corner independently by a random offset bounded by the
severity level, builds the 3x3 homography that maps the original corners to
the perturbed ones via `cv2.getPerspectiveTransform`, and warps the image with
`cv2.warpPerspective`. The homography H is returned so Day 4's oracle
rectification can invert it exactly.

This is deliberately a single variable: only a projective warp is applied.
No rotation, scale, or lens distortion is mixed in (that would conflate
multiple distinct causes of "looks wrong" into the one experimental
variable this project isolates), and no affine warp is used in place of a
homography — independently perturbing all 4 corners (rather than picking 3
free points and deriving the 4th, as an affine transform would) is what makes
this a true projective transform with 8 degrees of freedom, matching what an
off-axis camera tilt actually produces geometrically, not an approximation
of it.
"""

import argparse
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

from data_loader import load_monumai

# Corner-offset bound as a fraction of image width/height, per severity.
# Chosen so mild ~ a realistic handheld phone-shot tilt, medium ~ a
# deliberately angled shot to fit a full facade in frame, and severe ~ an
# extreme tilt at the edge of where a human viewer could still recognise the
# building. See experiments/logs/day3.md for the full justification.
SEVERITY_FRACTIONS = {"mild": 0.05, "medium": 0.10, "severe": 0.20}


def distort(image, severity, seed):
    """Apply a random projective (perspective) warp to `image`.

    Args:
        image: HxWxC (or HxW) numpy array. Channel order is irrelevant here —
            this is a pure geometric warp.
        severity: one of "mild", "medium", "severe" (keys of
            SEVERITY_FRACTIONS).
        seed: integer seed. The same (image, severity, seed) triple always
            produces byte-identical output — see tests/test_distortion.py.

    Returns:
        (distorted_image, H): the warped image (same shape as input, with
        black border-fill in any newly-exposed corner regions — OpenCV's
        default `cv2.warpPerspective` border handling, not a special choice
        for this module) and the 3x3 homography H used to produce it.
    """
    if severity not in SEVERITY_FRACTIONS:
        raise ValueError(
            f"Unknown severity {severity!r}; expected one of {list(SEVERITY_FRACTIONS)}"
        )

    h, w = image.shape[:2]
    fraction = SEVERITY_FRACTIONS[severity]
    max_dx = fraction * w
    max_dy = fraction * h

    # A local, seeded RandomState (not the global numpy random state) so this
    # function's output depends only on its own arguments, never on call
    # order or other code's random draws elsewhere in the process.
    rng = np.random.RandomState(seed)
    # One offset (dx, dy) per corner, drawn in a single call so the exact
    # sequence of random numbers consumed is fixed and documented here:
    # 8 uniform draws in [-1, 1], reshaped to (4 corners, 2 axes), then
    # scaled per-axis by the severity bound.
    unit_offsets = rng.uniform(-1.0, 1.0, size=(4, 2))
    offsets = unit_offsets * np.array([max_dx, max_dy])

    src_corners = np.float32(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    )
    dst_corners = (src_corners + offsets).astype(np.float32)

    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    distorted = cv2.warpPerspective(image, H, (w, h))
    return distorted, H


def derive_seed(image_id, replicate_index, base_seed):
    """Derive a per-(image_id, replicate_index) seed from a stable hash.

    Day 5's grid passed the single config-level `base_seed` straight into
    every call to `distort()`, so every image at a given severity got the
    exact same `np.random.RandomState(seed)` draw — one shared random tilt
    cloned across the whole dataset, not an independent tilt per image (see
    experiments/logs/day8_v2.md). This function fixes that: each
    (image_id, replicate_index) pair gets its own seed, so each image/
    replicate gets its own independent draw, while staying fully
    reproducible (same inputs -> same seed -> same output, every run, any
    machine).

    Uses hashlib.sha256 rather than Python's built-in hash() because hash()
    is randomly salted per-process for strings (PYTHONHASHSEED) unless
    explicitly disabled — it would not reproduce across runs/machines.
    Returns an int in [0, 2**32 - 1], the valid range for
    np.random.RandomState.
    """
    key = f"{base_seed}:{image_id}:{replicate_index}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def homography_tilt_angle(H):
    """Return (angle_deg, magnitude) summarizing a homography's tilt direction.

    H[2,0] and H[2,1] are the two "perspective" entries of a 3x3 homography
    — together they encode the direction and strength of the vanishing-point
    tilt that makes the warp projective rather than affine (both are exactly
    0 for a pure affine transform). angle_deg = atan2(H[2,1], H[2,0]) in
    degrees (0-360) gives the tilt's direction in that 2D plane; magnitude is
    its Euclidean norm. Used by the Day 8 v2 direction-check to verify tilts
    now point in roughly uniform random directions rather than all sharing
    one sign (see experiments/logs/day8_v2.md).
    """
    angle_deg = np.degrees(np.arctan2(H[2, 1], H[2, 0])) % 360
    magnitude = float(np.hypot(H[2, 0], H[2, 1]))
    return float(angle_deg), magnitude


def load_image_rgb(path):
    """Load an image from disk as an RGB numpy array (H, W, 3).

    OpenCV's cv2.imread returns BGR; converting to RGB here keeps every
    module in this project (distortion, rectification, and eventually CLIP
    inference, which expects RGB via PIL) working in one consistent channel
    order, rather than tracking BGR-vs-RGB per function.

    Deliberately reads via np.fromfile + cv2.imdecode rather than
    cv2.imread(path) directly: cv2.imread opens the path through a narrow
    (non-Unicode) file API on Windows, so it silently fails (returns None) on
    paths containing non-ASCII characters. Several MonuMAI filenames contain
    Spanish accented characters (e.g. "iglesia_nuestra_señora_...jpg"), which
    PIL (used for the "clean" condition elsewhere in this project) reads
    without issue — confirming these are valid image files, not corrupt data,
    and that cv2.imread's failure was a Windows-path-encoding bug in this
    code, not a dataset problem. Discovered during the Day 5 full grid run;
    see experiments/logs/day5.md.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Could not read image (empty or missing file): {path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_id_from_path(path, dataset_root):
    """Derive a unique, filesystem-safe image ID from a dataset image path.

    MonuMAI filenames (e.g. "20181212_105032.jpg") are not unique across the
    4 style folders, so the ID is "<style>__<filename-without-extension>",
    e.g. "Gothic__20181212_105032". This is the key used to name saved
    homographies (and, in Day 5, to key CSV rows) so every image has one
    stable, unambiguous identifier project-wide.
    """
    path = Path(path)
    dataset_root = Path(dataset_root)
    relative = path.relative_to(dataset_root)
    style = relative.parts[0]
    stem = Path(relative.parts[-1]).stem
    return f"{style}__{stem}"


def save_homography(H, image_id, severity, output_dir="results/raw/homographies"):
    """Save H as output_dir/severity/image_id.npy, creating dirs as needed.

    This is the mechanism Day 5's full grid will use to log every distorted
    image's homography to disk, keyed by image ID and severity, sitting
    alongside (not inside) the per-image prediction CSV. Set up and exercised
    now (via the Day 3 gallery) even though the full grid run is Day 5.
    """
    output_path = Path(output_dir) / severity / f"{image_id}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, H)
    return output_path


def build_distortion_gallery(
    dataset_root="data/monumai",
    output_path="figures/day3_distortion_gallery.png",
    n_images=6,
    seed=42,
    homography_dir="results/raw/homographies",
):
    """Build the 6x4 (original + mild/medium/severe) qualitative gallery.

    Image selection uses `seed` (via Python's own random.Random, kept separate
    from the per-distortion numpy RandomState) so which 6 images get chosen is
    itself reproducible. Each severity's homography is also saved via
    save_homography, both to exercise that mechanism now and so the exact
    warp behind every gallery panel is inspectable/reproducible later.
    """
    import matplotlib.pyplot as plt

    samples = load_monumai(dataset_root)
    rng = random.Random(seed)
    chosen = rng.sample(range(len(samples)), n_images)

    severities = ["mild", "medium", "severe"]
    n_cols = 1 + len(severities)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(3.2 * n_cols, 3.2 * n_images))

    for row, idx in enumerate(chosen):
        image_path, label = samples[idx]
        image_id = image_id_from_path(image_path, dataset_root)
        original = load_image_rgb(image_path)

        axes[row, 0].imshow(original)
        axes[row, 0].set_ylabel(f"{label}\n({image_id})", fontsize=8)
        if row == 0:
            axes[row, 0].set_title("original")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        for col, severity in enumerate(severities, start=1):
            distorted, H = distort(original, severity, seed)
            save_homography(H, image_id, severity, homography_dir)

            axes[row, col].imshow(distorted)
            if row == 0:
                axes[row, col].set_title(severity)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Day 3 — synthetic perspective distortion gallery (seed=42)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main():
    """CLI entry point: --gallery builds the Day 3 qualitative distortion gallery."""
    parser = argparse.ArgumentParser(description="Synthetic perspective distortion.")
    parser.add_argument(
        "--gallery", action="store_true", help="Build the Day 3 qualitative gallery."
    )
    parser.add_argument("--dataset-root", default="data/monumai")
    parser.add_argument("--output", default="figures/day3_distortion_gallery.png")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.gallery:
        build_distortion_gallery(
            dataset_root=args.dataset_root,
            output_path=args.output,
            n_images=args.n_images,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
