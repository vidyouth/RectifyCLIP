# CLAUDE.md

This file gives Claude Code context for working in this repository.

## Project overview

RectifyCLIP is a solo research project (course: BCSE417L Machine Vision, VIT) testing
whether classical OpenCV perspective rectification recovers the zero-shot architectural-style
classification accuracy that a frozen CLIP model loses under synthetic perspective distortion
of building facade images. Target output is a short research paper.

This is a controlled empirical study, not a new-method paper — the novelty is isolating one
variable (perspective distortion, with/without rectification) on a frozen foundation model.
Keep that framing in mind; do not oversell results as a novel technique.

The authoritative plan for what to build, in what order, is `RectifyCLIP_Execution_Plan.md`
in the repo root. Always read the relevant day's section there before starting work — this
file describes how to work, that file describes what to build and when.

## Tech stack

- Python 3.13, virtual environment at `.venv`
- PyTorch + `open_clip_torch` (ViT-B/32, `laion2b_s34b_b79k` pretrained weights) for
  zero-shot CLIP classification
- OpenCV (`opencv-python`) for perspective distortion and rectification (homography-based)
- pandas / numpy for results tables, matplotlib for figures, scikit-learn for metrics,
  pyyaml for configs
- Dataset: MonuMAI, 1,514 images, 4 architectural styles (Hispanic-Muslim, Gothic,
  Renaissance, Baroque), stored in `data/monumai/` (gitignored — never commit images).
  1,514 is the official release's full image count, confirmed against the source
  repo's own README (see `experiments/logs/day1.md`) — an earlier "1,092" figure here
  was stale, apparently from a different paper's filtered subset, not an official
  MonuMAI train/test split; since this project runs CLIP zero-shot with no
  fine-tuning, there is no split to honor and all 1,514 images are used as the
  evaluation set.
- Platform: Windows + VS Code + PowerShell locally; Colab T4 as fallback only if CPU is too slow

## Repository structure
rectify-clip/
├── configs/ default.yaml — model, prompts, seeds, paths
├── data/monumai/ dataset, gitignored
├── src/
│ ├── data_loader.py load_monumai(root) -> [(image_path, label), ...]
│ ├── clip_zero_shot.py CLIP loading + zero-shot classification
│ ├── distortion.py distort(image, severity, seed) -> (image, H)
│ ├── rectification.py rectify_oracle(image, H) -> image
│ ├── experiment_runner.py full grid: image × severity × condition
│ └── metrics.py accuracy, confidence, recovery %, etc.
├── experiments/
│ ├── logs/ dayN.md — what was done, decisions, numbers
│ └── configs/ per-run JSON configs (seed, prompt, git hash, etc.)
├── results/
│ ├── raw/ per-image predictions, CSV (gitignored, large)
│ └── tables/ aggregated markdown tables
├── figures/ plots, galleries — always regenerable from results/
├── paper/draft.md the paper itself
└── notebooks/ exploration only — never the source of truth

## Working rules

1. **Reproducibility first.** Fix random seeds everywhere. Every experiment's exact config
   (seed, prompt template, severity ranges, git commit hash) gets saved as JSON in
   `experiments/configs/`.
2. **Log every experiment.** Each day of work gets an entry in `experiments/logs/dayN.md`
   — what was built, key numbers, decisions made and why, anything surprising.
3. **Never overwrite results.** Version or timestamp instead of clobbering a previous run's output.
4. **Commit at the end of each unit of work**, with a message naming the day/deliverable
   (e.g. "Day 3: distortion module + gallery").
5. **Rectification is oracle-based** for the main experiment (inverse of the known
   homography) — this is intentional, not a shortcut. Do not build blind rectification
   (line/vanishing-point detection) unless the plan explicitly calls for it as a Day 7
   stretch goal.
6. **Every claim in the paper must trace to a number in `results/` or an image in
   `figures/`.** No hand-wavy claims.
7. **One variable at a time.** Distortion is projective/perspective only — no rotation,
   scale, or lens distortion mixed in unless the plan says otherwise.
8. **Plots are always regenerable from CSVs in `results/`** — never hand-edited in an
   image editor.

## Coding conventions

- One module docstring per file in `src/`, docstrings on every function.
- Functions in `src/` should be pure/testable where possible (e.g. `distort()` and
  `rectify_oracle()` are deterministic given a seed — no hidden state).
- Config values (model name, prompts, paths, seeds) belong in `configs/default.yaml`,
  not hardcoded in scripts.
- Prefer explicit CLI args (`--image`, `--severity`, etc.) over interactive prompts for
  anything meant to run unattended or in a batch/grid.

## Scope discipline

- Do not implement a later day's tasks ahead of schedule, even if it seems efficient —
  the plan sequences things deliberately (e.g. don't write distortion code before the
  clean baseline is frozen).
- Do not restructure the repo layout after the experimental grid (Day 5 in the plan) is
  complete, without a clear reason stated up front.
- If a task in the plan seems to conflict with something already built, stop and flag it
  rather than silently resolving it.
- Null/negative results are valid and should be reported honestly, not massaged.