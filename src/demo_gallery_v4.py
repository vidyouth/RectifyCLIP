"""Day 10 v4 qualitative gallery — one figure illustrating the v4 patterns.

Writes a single image, figures/v4_example_gallery.png, built entirely from
existing v4 outputs and code: predictions/confidences/correctness come from
reading results/raw/day10_v4_grid.csv (never recomputed — this script runs
no new CLIP inference and no new experiment, per instruction), and pixel
content is reproduced by calling the exact same distortion/rectification
building blocks v4 used (experiment_v4.sample_axis_angle/
build_rotation_homography, rectification.rectify_oracle/
apply_border_policy/compute_valid_mask/CHOSEN_BORDER_POLICY), with the same
seeds, so the images shown are pixel-identical to whatever produced the
logged CSV rows. Nothing here reimplements distortion, rectification, or
CLIP-scoring logic — it only composes existing functions.

Layout:
  Part A — a 6-row x 7-column grid (original | distorted/rectified at 5, 15,
  35 degrees), one real MonuMAI image per row, each row chosen (with a fixed
  seed, among all images satisfying that row's criterion) to illustrate a
  specific pattern from the v4 analysis — see SELECTION CRITERIA below and
  results/tables/v4_example_gallery_ids.md for exactly which image and why.

  Part B — one "direction strip": a single image distorted at a FIXED 15
  degree tilt but with 6 different EXPLICIT tilt-axis directions (0, 60,
  120, 180, 240, 300 degrees), each followed by its oracle-rectified
  version, to visually demonstrate that the tilt direction is random, not
  one-sided (the property verified numerically in v4_direction_check.md).
  These 6 explicit-angle images were never part of the v4 grid (v4 only
  ever sampled one RANDOM direction per image/replicate) — they are
  generated here, from the same homography-building code, purely for this
  illustration, not logged as a new "result."
"""

import csv
import random
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from clip_zero_shot import load_config
from data_loader import load_monumai
from distortion import image_id_from_path, load_image_rgb
from experiment_v4 import build_rotation_homography, sample_axis_angle
from day10_v4_analysis import load_v4_rows
from rectification import CHOSEN_BORDER_POLICY, apply_border_policy, compute_valid_mask, rectify_oracle

GALLERY_SEED = 42
REPLICATE_SHOWN = 0  # Part A reads/reproduces replicate=0 throughout, so captions match the CSV exactly
THETAS_SHOWN = [5, 15, 35]
DIRECTION_STRIP_THETA = 15
DIRECTION_STRIP_ANGLES = [0, 60, 120, 180, 240, 300]

# (criterion label, short on-figure description, full description for the IDs table), in display order.
ROW_SPECS = [
    ("row1_hispanic_muslim_fixed", "rectification fixes a wrong prediction (15deg)",
     "Hispanic-Muslim: rectification fixes a wrong prediction (15deg)"),
    ("row2_baroque_fixed", "rectification fixes a wrong prediction (15deg)",
     "Baroque: rectification fixes a wrong prediction (15deg)"),
    ("row3_gothic_broken", "rectification breaks a correct prediction (15deg)",
     "Gothic: rectification breaks a correct prediction (15deg)"),
    ("row4_gothic_broken_2", "rectification breaks a correct prediction (15deg), 2nd example",
     "Gothic: rectification breaks a correct prediction (15deg) -- second example"),
    ("row5_easy_case", "easy case: correct in all 7 columns",
     "Easy case: correct in all 7 columns"),
    ("row6_hard_case", "hard case: both fail at 35deg",
     "Hard case: both distorted and rectified fail at 35deg"),
]


def distort_and_rectify(image, theta_deg, axis_phi_deg):
    """New helper (lives only here — experiment_v4.py is not modified):
    produce (distorted, rectified) for an EXPLICIT axis angle, composed
    entirely from existing building blocks. build_rotation_homography
    already accepts an explicit axis_phi_deg (it is what sample_axis_angle
    feeds it internally for the real grid); the rest of the chain
    (warpPerspective, compute_valid_mask, rectify_oracle,
    apply_border_policy with CHOSEN_BORDER_POLICY) is exactly
    experiment_v4.run_theta_replicate_block's own per-image sequence,
    reused here rather than duplicated with any variation.
    """
    h, w = image.shape[:2]
    H = build_rotation_homography(h, w, theta_deg, axis_phi_deg)
    distorted = cv2.warpPerspective(image, H, (w, h))
    mask = compute_valid_mask(image.shape, H)
    rectified_raw = rectify_oracle(distorted, H)
    rectified = apply_border_policy(rectified_raw, mask, CHOSEN_BORDER_POLICY)
    return distorted, rectified


def build_path_lookup(dataset_root):
    """{image_id: image_path} for every MonuMAI image, via the same
    image_id_from_path v4's own CSV keys were built with."""
    samples = load_monumai(dataset_root)
    return {image_id_from_path(path, dataset_root): path for path, _ in samples}


def build_row_lookup(rows):
    """{(image_id, theta, replicate, condition): row} for O(1) lookup."""
    return {(r["image_id"], r["theta"], r["replicate"], r["condition"]): r for r in rows}


def cell_info(by_key, image_id, theta, condition, replicate=REPLICATE_SHOWN):
    r = by_key.get((image_id, theta, replicate, condition))
    if r is None:
        return None
    return {
        "predicted_label": r["predicted_label"],
        "correct": r["correct"],
        "confidence": float(r["top1_confidence"]),
    }


def build_image_summaries(rows):
    """{image_id: {true_label, clean, distorted_5, rectified_5, distorted_15,
    rectified_15, distorted_35, rectified_35}} — one dict per image, pulling
    only replicate=0 rows (so Part A's captions and pixels always agree)."""
    by_key = build_row_lookup(rows)
    true_label_by_id = {r["image_id"]: r["true_label"] for r in rows}
    summaries = {}
    for image_id in true_label_by_id:
        s = {"true_label": true_label_by_id[image_id], "clean": cell_info(by_key, image_id, 0, "clean")}
        for theta in THETAS_SHOWN:
            s[f"distorted_{theta}"] = cell_info(by_key, image_id, theta, "distorted")
            s[f"rectified_{theta}"] = cell_info(by_key, image_id, theta, "rectified")
        summaries[image_id] = s
    return summaries


def select_gallery_rows(summaries):
    """Pick the 6 Part-A rows per SELECTION CRITERIA, and 1 Part-B strip
    image, deterministically from a fixed-seed random.Random — see each
    criterion's lambda for the exact rule. Candidates are sorted by
    image_id before the seeded choice, so the pick only depends on
    GALLERY_SEED and the CSV's content, never on dict/set iteration order.
    """
    rng = random.Random(GALLERY_SEED)

    def candidates(predicate):
        return sorted(image_id for image_id, s in summaries.items() if predicate(s))

    def all_correct_15(s):
        return (
            s["clean"]["correct"]
            and s["distorted_5"]["correct"] and s["rectified_5"]["correct"]
            and s["distorted_15"]["correct"] and s["rectified_15"]["correct"]
            and s["distorted_35"]["correct"] and s["rectified_35"]["correct"]
        )

    c1 = candidates(lambda s: s["true_label"] == "Hispanic-Muslim" and not s["distorted_15"]["correct"] and s["rectified_15"]["correct"])
    c2 = candidates(lambda s: s["true_label"] == "Baroque" and not s["distorted_15"]["correct"] and s["rectified_15"]["correct"])
    c3 = candidates(lambda s: s["true_label"] == "Gothic" and s["distorted_15"]["correct"] and not s["rectified_15"]["correct"])
    c5 = candidates(all_correct_15)
    c6 = candidates(lambda s: not s["distorted_35"]["correct"] and not s["rectified_35"]["correct"])

    for name, cands in [("row1 (Hispanic-Muslim fixed)", c1), ("row2 (Baroque fixed)", c2),
                         ("row3/4 (Gothic broken)", c3), ("row5 (easy case)", c5), ("row6 (hard case)", c6)]:
        if not cands:
            raise RuntimeError(f"No candidates found for {name} — check results/raw/day10_v4_grid.csv")

    row1 = rng.choice(c1)
    row2 = rng.choice(c2)
    row3 = rng.choice(c3)
    c4 = [iid for iid in c3 if iid != row3]
    if not c4:
        raise RuntimeError("Only one Gothic 'rectification breaks a correct prediction' candidate exists — need a second, distinct one for row4")
    row4 = rng.choice(c4)
    row5 = rng.choice(c5)
    row6 = rng.choice(c6)

    strip_image_id = rng.choice(sorted(summaries.keys()))

    chosen = [row1, row2, row3, row4, row5, row6]
    return chosen, strip_image_id


def predicted_text(cell):
    label = cell["predicted_label"]
    conf = cell["confidence"]
    color = "green" if cell["correct"] else "red"
    return f"{label}\n{conf:.1%}", color


def render_cell(ax, image, caption=None, caption_color="black", row_label=None, col_title=None):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if col_title:
        ax.set_title(col_title, fontsize=13)
    if row_label:
        ax.set_ylabel(row_label, fontsize=9, rotation=0, ha="right", va="center", labelpad=10)
    if caption:
        ax.text(
            0.5, -0.06, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=caption_color, fontweight="bold",
        )


def build_gallery(config, csv_path="results/raw/day10_v4_grid.csv", output_path="figures/v4_example_gallery.png"):
    class_names = config["class_names"]
    dataset_root = config["dataset_root"]
    base_seed = config["seed"]

    rows = load_v4_rows(csv_path)
    print(f"Loaded {len(rows)} v4 rows")
    summaries = build_image_summaries(rows)
    path_lookup = build_path_lookup(dataset_root)

    chosen_ids, strip_image_id = select_gallery_rows(summaries)
    print("Selected Part A rows:", chosen_ids)
    print("Selected Part B direction-strip image:", strip_image_id)

    n_rows_a = len(chosen_ids)
    n_rows_b = 2
    n_title_rows = 1
    total_rows = n_rows_a + n_title_rows + n_rows_b
    col_titles = ["original"] + [
        f"{cond} {theta}deg" for theta in THETAS_SHOWN for cond in ["distorted", "rectified"]
    ]
    n_cols_a = len(col_titles)  # 1 + 3*2 = 7

    row_h = 3.3  # inches per image row
    title_h = 0.5  # inches for the Part-B title row
    top_reserve = 2.0  # inches reserved at the top for suptitle + subtitle + a gap before the grid
    fig_width = 3.2 * n_cols_a
    fig_height = row_h * n_rows_a + title_h + row_h * n_rows_b + top_reserve

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)
    fig.patch.set_facecolor("white")
    # Explicit GridSpec margins (not tight_layout, which fought with the
    # manually-placed title text and produced unpredictable blank space and
    # overlaps) — top/bottom/left/right are fractions of the figure, held
    # fixed regardless of content, so row labels have a guaranteed left
    # margin and the title band never overlaps the image rows below it.
    top_margin = 1.0 - (top_reserve - 0.3) / fig_height  # leaves a clear gap below the subtitle
    gs = fig.add_gridspec(
        total_rows, n_cols_a,
        height_ratios=[row_h] * n_rows_a + [title_h] + [row_h] * n_rows_b,
        left=0.10, right=0.985, top=top_margin, bottom=0.01,
        hspace=0.55, wspace=0.08,
    )

    fig.suptitle(
        "RectifyCLIP — Day 10 v4 qualitative gallery (camera-rotation distortion)",
        fontsize=18, y=1.0 - (0.5 / fig_height),
    )
    fig.text(
        0.5, 1.0 - (1.1 / fig_height),
        "Examples selected to illustrate each pattern; see tables for overall rates.",
        ha="center", fontsize=12, style="italic",
    )

    # --- Part A ---
    for row_idx, (image_id, (_, reason, _full_reason)) in enumerate(zip(chosen_ids, ROW_SPECS)):
        path = path_lookup[image_id]
        original = load_image_rgb(path)
        s = summaries[image_id]
        true_label = s["true_label"]

        col = 0
        ax = fig.add_subplot(gs[row_idx, col])
        caption, color = predicted_text(s["clean"])
        wrapped_reason = "\n".join(textwrap.wrap(reason, width=16))
        row_label = f"TRUE:\n{true_label}\n{wrapped_reason}"
        render_cell(
            ax, original, caption=caption, caption_color=color,
            row_label=row_label, col_title=col_titles[0] if row_idx == 0 else None,
        )

        col_i = 1
        for theta in THETAS_SHOWN:
            phi, _ = sample_axis_angle(image_id, theta, REPLICATE_SHOWN, base_seed)
            distorted, rectified = distort_and_rectify(original, theta, phi)

            ax_d = fig.add_subplot(gs[row_idx, col_i])
            cap_d, color_d = predicted_text(s[f"distorted_{theta}"])
            render_cell(ax_d, distorted, caption=cap_d, caption_color=color_d,
                        col_title=col_titles[col_i] if row_idx == 0 else None)
            col_i += 1

            ax_r = fig.add_subplot(gs[row_idx, col_i])
            cap_r, color_r = predicted_text(s[f"rectified_{theta}"])
            render_cell(ax_r, rectified, caption=cap_r, caption_color=color_r,
                        col_title=col_titles[col_i] if row_idx == 0 else None)
            col_i += 1

    # --- Part B: direction strip ---
    strip_path = path_lookup[strip_image_id]
    strip_original = load_image_rgb(strip_path)
    strip_true_label = summaries[strip_image_id]["true_label"]

    part_b_title_row = n_rows_a  # dedicated GridSpec row, not a hand-positioned fig.text —
    ax_title = fig.add_subplot(gs[part_b_title_row, :])  # guarantees no overlap with the rows above/below it
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        f"Tilt direction is random, not one-sided  (theta={DIRECTION_STRIP_THETA}deg, "
        f"image: {strip_image_id}, true style: {strip_true_label})",
        ha="center", va="center", fontsize=14, fontweight="bold", transform=ax_title.transAxes,
    )

    part_b_row = n_rows_a + n_title_rows
    n_cols_b = len(DIRECTION_STRIP_ANGLES)
    col_span = n_cols_a / n_cols_b
    for i, angle in enumerate(DIRECTION_STRIP_ANGLES):
        distorted, rectified = distort_and_rectify(strip_original, DIRECTION_STRIP_THETA, angle)
        start_col = int(round(i * col_span))
        end_col = int(round((i + 1) * col_span))

        ax_d = fig.add_subplot(gs[part_b_row, start_col:end_col])
        render_cell(ax_d, distorted, col_title=f"phi={angle}deg", row_label="distorted" if i == 0 else None)

        ax_r = fig.add_subplot(gs[part_b_row + 1, start_col:end_col])
        render_cell(ax_r, rectified, row_label="rectified" if i == 0 else None)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")

    return chosen_ids, strip_image_id, summaries


def save_ids_table(chosen_ids, strip_image_id, summaries, output_path="results/tables/v4_example_gallery_ids.md"):
    lines = ["# Day 10 v4 — example gallery image selection", ""]
    lines.append(f"Selected with a fixed seed (GALLERY_SEED={GALLERY_SEED}) from `results/raw/day10_v4_grid.csv`.")
    lines.append("")
    lines.append("## Part A rows")
    lines.append("")
    lines.append("| Row | image_id | True style | Selection criterion |")
    lines.append("|---|---|---|---|")
    for i, (image_id, (_, _short_reason, reason)) in enumerate(zip(chosen_ids, ROW_SPECS), start=1):
        lines.append(f"| {i} | {image_id} | {summaries[image_id]['true_label']} | {reason} |")
    lines.append("")
    lines.append("## Part B direction strip")
    lines.append("")
    lines.append(f"Image: `{strip_image_id}` (true style: {summaries[strip_image_id]['true_label']})")
    lines.append(f"Theta: {DIRECTION_STRIP_THETA} degrees. Explicit axis angles (not sampled, not in the v4 CSV): {DIRECTION_STRIP_ANGLES}.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {output_path}")


def main():
    config = load_config("configs/default.yaml")
    chosen_ids, strip_image_id, summaries = build_gallery(config)
    save_ids_table(chosen_ids, strip_image_id, summaries)


if __name__ == "__main__":
    main()
