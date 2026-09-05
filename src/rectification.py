"""Oracle perspective rectification: invert a known distortion homography.

Core function `rectify_oracle(distorted_image, H)` inverts the exact
homography `H` that `distort()` (src/distortion.py) used to produce
`distorted_image`, and warps the image back. This is intentionally "oracle"
rectification — it assumes perfect knowledge of H, which a real deployment
would not have. That is not a shortcut or a cheat: the scientific question
this project asks is "if perspective could be perfectly undone, would CLIP's
accuracy recover?", which is exactly what oracle rectification tests. Blind
rectification (line/vanishing-point detection, no known H) is explicitly out
of scope until Day 7's stretch goal.

This module also provides the machinery to decide and apply a border-handling
policy (black / crop / mean-fill) for the regions of a rectified image that
have no corresponding real source pixel — see `experiments/logs/day4.md` for
the written justification of which policy was chosen and why.
"""

import random
from pathlib import Path

import cv2
import numpy as np

from data_loader import load_monumai
from distortion import distort, image_id_from_path, load_image_rgb


def rectify_oracle(distorted_image, H):
    """Invert a known distortion homography H and warp the image back.

    Implementation exactly as specified by the plan: H_inv = inverse(H), then
    cv2.warpPerspective(distorted_image, H_inv, (w, h)). Uses OpenCV's default
    bilinear interpolation and black (zero-value) border fill — no
    border-handling policy is applied here; that is a separate, explicit step
    (see `apply_border_policy`) so this function stays a pure, minimal inverse
    of `distort()`'s warp.
    """
    h, w = distorted_image.shape[:2]
    H_inv = np.linalg.inv(H)
    return cv2.warpPerspective(distorted_image, H_inv, (w, h))


def compute_valid_mask(shape, H):
    """Return a boolean mask of which rectified-image pixels are "real".

    A pixel is only "real" (as opposed to border) if it survives BOTH warps:
    it must have come from inside the original image during distort(), AND
    the rectification warp itself must sample it from inside the distorted
    image's own canvas. Rather than deriving this analytically from H's
    geometry, this mimics the actual pipeline empirically: warp an
    all-255 mask through distort()'s H, then through rectify_oracle's H_inv,
    exactly mirroring what happens to real pixel data. This automatically
    accounts for any clipping or interpolation nuances in the real pipeline,
    rather than risking a subtly-wrong closed-form derivation.

    A pixel is only counted valid if the round-tripped mask value is above
    250/255 — just below full white — rather than any positive value, so
    that partially-black antialiased edge pixels (a blend of real content and
    border, produced by bilinear interpolation right at the border) are
    excluded from "valid" and don't contaminate the sanity-test error metric
    or the mean-fill color computation.
    """
    h, w = shape[:2]
    ones = np.full((h, w), 255, dtype=np.uint8)
    distorted_mask = cv2.warpPerspective(ones, H, (w, h))
    rectified_mask = cv2.warpPerspective(distorted_mask, np.linalg.inv(H), (w, h))
    return rectified_mask > 250


def mean_absolute_error(image_a, image_b, mask):
    """Mean absolute per-pixel, per-channel difference, restricted to mask.

    Used by the Day 4 sanity test to check rectify_oracle(distort(I, H), H)
    is pixel-close to I *outside* the border region. Returns None if mask has
    no True pixels (would otherwise divide by zero).
    """
    if not np.any(mask):
        return None
    diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
    return float(diff[mask].mean())


def largest_rectangle_in_mask(mask):
    """Largest axis-aligned all-valid rectangle inside a boolean mask.

    Classic "maximal rectangle in a binary matrix" algorithm (histogram +
    monotonic stack per row), O(rows*cols). Returns (top, bottom, left,
    right), all inclusive, describing the crop bounds for the "crop to
    central rectangle" border-handling policy.
    """
    rows, cols = mask.shape
    heights = [0] * cols
    best_area = 0
    best_rect = (0, 0, 0, 0)

    for r in range(rows):
        for c in range(cols):
            heights[c] = heights[c] + 1 if mask[r, c] else 0

        stack = []
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            while stack and heights[stack[-1]] >= h:
                top_idx = stack.pop()
                height = heights[top_idx]
                left = stack[-1] + 1 if stack else 0
                right = c - 1
                area = height * (right - left + 1)
                if area > best_area:
                    best_area = area
                    best_rect = (r - height + 1, r, left, right)
            stack.append(c)

    return best_rect


# Border-handling policy chosen for the main experiment (Day 5 onward), after
# testing all 3 options on 5 sample images. See experiments/logs/day4.md for
# the full written justification. Summary: "black" introduces a stark
# artifact absent from the clean condition; "crop" removes the artifact but
# shrinks each image by a different, severity-dependent amount, confounding
# rectification with an implicit zoom change (violates "one variable at a
# time" — CLAUDE.md rule 7); "mean_fill" avoids both problems at negligible
# compute cost.
CHOSEN_BORDER_POLICY = "mean_fill"


def apply_border_policy(rectified_image, mask, policy):
    """Apply one of the 3 candidate border-handling policies.

    policy: "black" (no-op — rectify_oracle's own zero-fill is left as is),
    "mean_fill" (invalid pixels replaced with the mean color of the valid
    region, image dimensions unchanged), or "crop" (cropped to the largest
    all-valid axis-aligned rectangle — image dimensions DO change, generally
    shrinking, and shrink by a different amount per image depending on how
    much of that image's own random distortion pushed content outside the
    frame).
    """
    if policy == "black":
        return rectified_image

    if policy == "mean_fill":
        output = rectified_image.copy()
        mean_color = rectified_image[mask].mean(axis=0)
        output[~mask] = mean_color
        return output

    if policy == "crop":
        top, bottom, left, right = largest_rectangle_in_mask(mask)
        return rectified_image[top : bottom + 1, left : right + 1]

    raise ValueError(f"Unknown border policy: {policy!r}")


def save_border_policy_comparison(image, distorted, H, output_dir):
    """Save one image's rectification under all 3 border policies, for inspection.

    Used by the Day 4 border-policy test on 5 sample images — not part of the
    main pipeline.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rectified = rectify_oracle(distorted, H)
    mask = compute_valid_mask(image.shape, H)

    policies = ["black", "mean_fill", "crop"]
    fig, axes = plt.subplots(1, 1 + len(policies), figsize=(4 * (1 + len(policies)), 4))
    axes[0].imshow(distorted)
    axes[0].set_title("distorted (input)")
    axes[0].axis("off")
    for ax, policy in zip(axes[1:], policies):
        result = apply_border_policy(rectified, mask, policy)
        ax.imshow(result)
        ax.set_title(policy)
        ax.axis("off")
    fig.tight_layout()
    return fig


def build_rectification_gallery(
    dataset_root="data/monumai",
    output_path="figures/day4_rectification_gallery.png",
    n_images=6,
    seed=42,
    severity="mild",
    border_policy=CHOSEN_BORDER_POLICY,
):
    """Extend the Day 3 gallery: original | mild distorted | mild rectified | diff.

    Uses the exact same image selection as Day 3's
    build_distortion_gallery (same seed, same random.Random(seed).sample call
    over the same load_monumai ordering), so this shows rectification results
    for the identical 6 images Day 3 showed distortion for — a continuation of
    that gallery, not a fresh sample. The "diff" column shows mean absolute
    per-channel difference between the original and rectified image,
    restricted to the valid (non-border) mask — matching the Day 4 sanity
    test's own metric, not a separate ad hoc visualization.
    """
    import matplotlib.pyplot as plt

    samples = load_monumai(dataset_root)
    rng = random.Random(seed)
    chosen = rng.sample(range(len(samples)), n_images)

    fig, axes = plt.subplots(n_images, 4, figsize=(3.2 * 4, 3.2 * n_images))

    for row, idx in enumerate(chosen):
        image_path, label = samples[idx]
        image_id = image_id_from_path(image_path, dataset_root)
        image = load_image_rgb(image_path)
        distorted, H = distort(image, severity, seed)
        rectified_raw = rectify_oracle(distorted, H)
        mask = compute_valid_mask(image.shape, H)
        rectified = apply_border_policy(rectified_raw, mask, border_policy)

        diff = np.abs(image.astype(np.int16) - rectified_raw.astype(np.int16)).mean(axis=2)
        diff_display = np.zeros_like(diff)
        diff_display[mask] = diff[mask]

        panels = [image, distorted, rectified, diff_display]
        titles = ["original", f"{severity} distorted", f"{severity} rectified ({border_policy})", "abs diff (valid region)"]

        for col, panel in enumerate(panels):
            ax = axes[row, col]
            if col == 3:
                im = ax.imshow(panel, cmap="inferno", vmin=0, vmax=30)
            else:
                ax.imshow(panel)
            if row == 0:
                ax.set_title(titles[col], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{label}\n({image_id})", fontsize=8)

    fig.suptitle(f"Day 4 — oracle rectification gallery (severity={severity}, seed={seed})")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    build_rectification_gallery()
