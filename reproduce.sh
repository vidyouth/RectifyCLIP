#!/usr/bin/env bash
# Reproduce the full RectifyCLIP experiment end to end, in the exact order
# the 7-day plan built it.
#
# Prerequisites (not automated here — one-time, environment-specific setup;
# see README.md "Setup" and "Dataset" sections):
#   1. .venv created and `pip install -r requirements.txt` run.
#   2. MonuMAI downloaded into data/monumai/ (see README.md for the exact
#      clone/copy commands).
#
# Total runtime on local CPU (no GPU/Colab used anywhere in this project):
# roughly 35-40 minutes, dominated by Day 5's full grid (~18 min) and Day 6's
# confidence/prompt-sensitivity rerun (~20 min). Everything else (Day 2's
# baseline, Day 3/4's galleries, Day 6's CSV-only analysis) is under a minute
# each. Every step is deterministic (seed=42 fixed throughout) — rerunning
# this script reproduces byte-identical CSVs and numerically identical
# figures every time.
#
# Usage: bash reproduce.sh   (from the repo root)

set -e
cd "$(dirname "$0")"

PYTHON=".venv/Scripts/python.exe"   # Windows venv layout used throughout this project;
                                      # use .venv/bin/python on Linux/Mac instead.

echo "== Day 2: clean-baseline batch inference + prompt pilot =="
"$PYTHON" src/clip_zero_shot.py --batch

echo "== Day 3: distortion module qualitative gallery =="
"$PYTHON" src/distortion.py --gallery

echo "== Day 4: oracle rectification gallery =="
"$PYTHON" src/rectification.py

echo "== Day 5: full experimental grid (image x severity x condition, ~18 min) =="
"$PYTHON" src/experiment_runner.py

echo "== Day 6: analysis, recovery %, severity curve, per-class, failure gallery,"
echo "          confidence + prompt-sensitivity rerun (~20 min) =="
"$PYTHON" - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from clip_zero_shot import load_config, PROMPT_TEMPLATE_SLUGS
from day6_analysis import (
    load_day5_rows, compute_recovery, save_recovery_table, save_severity_curve,
    save_per_class_table, find_hurt_and_helped_candidates, build_failure_gallery,
    run_confidence_and_prompt_sweep, save_scores_csv, save_confidence_table_and_plot,
    save_prompt_sensitivity_table,
)

config = load_config("configs/default.yaml")
rows = load_day5_rows()

recovery = compute_recovery(rows)
save_recovery_table(recovery, "results/tables/day6_recovery.md")
save_severity_curve(recovery, "figures/day6_severity_curve.png")
save_per_class_table(rows, config["class_names"], "results/tables/day6_per_class.md")

hurt, helped = find_hurt_and_helped_candidates(rows)
build_failure_gallery(hurt, n_images=8, seed=42)

scores_rows = run_confidence_and_prompt_sweep(config)
save_scores_csv(scores_rows, "results/raw/day6_scores.csv")
default_prompt_id = PROMPT_TEMPLATE_SLUGS[config["prompt_template"]]
save_confidence_table_and_plot(
    scores_rows, default_prompt_id,
    "results/tables/day6_confidence.md", "figures/day6_confidence.png",
)
save_prompt_sensitivity_table(scores_rows, "results/tables/day6_prompt_sensitivity.md")
PYEOF

echo ""
echo "Done. Headline results: results/tables/day5_headline.md, day6_recovery.md,"
echo "day6_per_class.md. Figures: figures/day6_severity_curve.png and friends."
