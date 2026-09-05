# Day 7 — Paper draft, README, repo polish

Date: 2026-09-05

## Objective (from execution plan)

Package the week's work so it can be sent to a professor and pushed to
GitHub as a portfolio piece. Leave a clear handoff for the paper-writing
week. No new experiments today — everything here synthesizes Days 1-6.

## What shipped

- **`paper/draft.md`**: all 6 sections filled in (Abstract, Introduction,
  Related work, Method, Results, Discussion & limitations, plus a Future
  Work section) — the existing Day 3/4 "Distortion protocol" and
  "Rectification protocol" subsections were left completely unchanged and
  referenced from the new Method section rather than duplicated. Every
  number in the Results section traces to a specific file under `results/`
  or `figures/`; nothing is hand-typed from memory without a source. Per
  the plan's own caution and the user's explicit instruction, every
  citation detail not independently re-verified is flagged in-line as
  unverified rather than presented as confirmed (BATCLIP's exact venue,
  MonuMAI's exact venue/year, and the GLNet/OpenFACADES citations
  specifically) — these need a real bibliography check before submission.
- **`README.md`**: rewritten as a portfolio-grade README — pitch,
  architecture diagram (unchanged mermaid block), an honest headline-results
  table (including the mild-severity statistical caveat and the Gothic
  negative-recovery finding, not just the favorable numbers), the
  Renaissance-attractor finding summarized with a pointer to the full
  analysis, dataset citation/download instructions (unchanged from Day 1),
  setup + reproduce instructions pointing at `reproduce.sh`, the repository
  structure, and a license section.
- **`LICENSE`**: added — an MIT license file did not exist despite Day 1's
  task list calling for "public, MIT licence"; this was a real gap from
  Week 1, not something deferred deliberately. Added now since the new
  README explicitly claims an MIT license and that claim needs to be true.
- **`reproduce.sh`**: a single documented script that runs the exact command
  sequence used to produce every result this week, in order (Day 2's batch
  baseline → Day 3's gallery → Day 4's gallery → Day 5's full grid → Day 6's
  full analysis pipeline, inline via a heredoc since Day 6's outputs were
  originally produced via ad hoc Python snippets rather than a single
  script). Documented prerequisites (venv + dataset) and expected runtime
  (~35-40 minutes on CPU) rather than silently assuming they're obvious.
  Syntax-checked (`bash -n`) but **not executed end-to-end** today — it
  would take ~40 minutes and would only reproduce results already verified
  correct throughout the week; each of its steps was run and verified
  individually on its own day.
- **Repo hygiene — docstrings**: audited every function in every `src/*.py`
  file via `ast.get_docstring`; found 13 functions across
  `clip_zero_shot.py`, `day6_analysis.py`, `experiment_runner.py`, and
  `metrics.py` missing a docstring (mostly small CLI `main()` entry points
  and simple save/write helpers written quickly during Days 5-6's
  time-pressured background runs) and added one-line docstrings to all of
  them. Every module already had a module-level docstring from the day it
  was written. Re-ran `tests/test_distortion.py` and
  `tests/test_rectification.py` after these edits — both still pass with
  identical output, confirming the docstring-only changes introduced no
  regressions.
- **Repo hygiene — `requirements.txt`**: confirmed still pinned exactly as
  Day 1 left it, with one real gap found and fixed: Day 5 added
  `src/metrics.py:paired_mcnemar`, which imports `scipy.stats` directly —
  `scipy` was installed (as scikit-learn's transitive dependency) but never
  explicitly pinned in `requirements.txt`, even though this project's own
  code now imports it directly. Added `scipy==1.18.1` (the exact version
  already installed and used all week) as an explicit pin. This is a small
  but real reproducibility gap that would have bitten a stranger installing
  fresh from `requirements.txt` alone if scikit-learn's own dependency
  resolution ever pulled a different scipy version.
- **`experiments/logs/day7.md`** (this file).

## What was deliberately NOT done today (per the plan's explicit scope)

- **Blind rectification was not implemented.** Per the plan's Day 7 scope,
  it is described only as future work (paper Section 6) — no
  `cv2.HoughLinesP`/vanishing-point code was written. Nothing from Days 1-6
  slipped, so there was no catch-up need for the buffer time either.
- **No new experiments were run.** All paper numbers come from Day 5's
  frozen `day5_full_grid.csv` and Day 6's `day6_scores.csv` — no new CLIP
  forward passes happened today.
- **Git tagging and pushing were not done** — per explicit instruction, the
  user handles all git operations (add/commit/tag/push) after reviewing
  today's changes themselves.
- **The off-GitHub backup (Google Drive zip) was not done** — this is a
  manual action on the user's own Google Drive account, not something this
  session has access to or should attempt; flagging it here as an
  outstanding task for the user, not silently skipped without mention.

## Honest next step

In priority order, matching the paper's own Future Work section:

1. **Blind rectification** (line/vanishing-point-based, no ground-truth
   homography) — the natural next experiment, since it tests whether this
   week's oracle-rectification benefit survives contact with a realistic
   deployment that doesn't know the true distortion.
2. **Investigate the Renaissance-attractor mechanism directly** — this
   week's most novel and least-understood finding. A targeted follow-up
   (e.g., varying only the Renaissance class name/prompt while holding the
   image set fixed, or an embedding-space analysis) could distinguish a
   prompt-wording artifact from a pretraining-data frequency prior from a
   genuine cross-style visual-similarity effect.
3. **A second dataset** (e.g. WikiChurches) to check whether the per-class
   unevenness and the attractor bias are specific to MonuMAI's particular
   4-style distribution or a more general property of how CLIP handles
   architectural-style classification under geometric perturbation.

## Success criterion check

Per the plan: "a stranger could clone the repo, run one command, and
reproduce the headline result... `paper/draft.md` is complete enough that
turning it into a formatted conference submission is now a writing task, not
a research task." `reproduce.sh` provides that one command (with documented
prerequisites); `paper/draft.md` has all 6 sections filled in with every
claim traced to a `results/`/`figures/` artifact, and the only remaining
work flagged for a human before submission is bibliographic verification of
a handful of citation details this project never had independent means to
confirm — a writing/verification task, not a research task.

## Files touched today

- `paper/draft.md` — filled in (Method/Results/Discussion sections and
  Abstract/Intro/Related-work/Future-work added; existing Day 3/4
  subsections untouched).
- `README.md` — rewritten, portfolio-grade.
- `LICENSE` — added (MIT).
- `reproduce.sh` — added.
- `requirements.txt` — added missing explicit `scipy==1.18.1` pin.
- `src/clip_zero_shot.py`, `src/day6_analysis.py`,
  `src/experiment_runner.py`, `src/metrics.py` — added 13 missing function
  docstrings; no logic changes (tests re-verified, identical output).
- `experiments/logs/day7.md` — this file.
