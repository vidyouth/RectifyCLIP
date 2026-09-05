RectifyCLIP — 7-Day Execution Plan

Course: BCSE417L Machine Vision, VIT Type: Solo research project, target output = short research paper Owner: Vidyuth (23BAI0173)

One-line pitch

RectifyCLIP tests if classical image rectification can undo the accuracy CLIP loses when a building photo is taken at an angle.

Research question

Does classical (OpenCV-based) perspective rectification recover the zero-shot architectural-style classification accuracy that a frozen CLIP model loses under controlled synthetic perspective distortion of building facade images?

Why this is a research problem

Frozen vision-language models like CLIP are increasingly used "as-is" on built-environment tasks (e.g. Tarkhan et al., Scientific Reports, 2025, on facade material mapping). Real facade photos are almost never straight-on, and Tarkhan et al. explicitly flag viewing-angle distortion as a possible source of error but never test or fix it. Standard CLIP robustness benchmarks (ImageNet-C style) test blur, noise, JPEG, and weather, but not true projective/perspective tilt. RectifyCLIP fills that specific gap with a clean, controlled experiment: same model, same task, same images — only distortion severity and the rectify/don't-rectify choice change.

Novelty (honest framing)

This is not a new-method paper. It is a controlled empirical study that isolates one variable (perspective distortion, with and without classical rectification) on a frozen foundation model. The novelty is the isolation, not the technique. Faculty and reviewers should see it framed this way — do not oversell.

Architecture diagram
Clean facade image
Synthetic distortionmild, medium, severe
Frozen CLIPclean baseline
Distorted image
Frozen CLIPno rectification
Rectify OpenCVinvert the homography
Frozen CLIPafter rectification
Compare resultsaccuracy and confidence
Working principles
Keep everything reproducible. Fix random seeds. Save every config as JSON alongside every result.
Log every experiment. Every run writes to experiments/logs/YYYY-MM-DD_run.md.
Never overwrite results. Timestamp or version everything.
Commit at end of every day with a message that names the day and the deliverable.
Rectification is "oracle" (inverse of the known homography) for the main experiment. This is intentional — it isolates the CLIP question and does not cheat because the framing is "if we could perfectly rectify, would CLIP recover?" Blind rectification is a stretch goal, not a requirement.
Every claim in the final paper must be backed by a number in results/ or an image in figures/.
Repository structure
rectify-clip/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   └── monumai/                 (gitignored)
├── src/
│   ├── data_loader.py
│   ├── clip_zero_shot.py
│   ├── distortion.py
│   ├── rectification.py
│   ├── experiment_runner.py
│   └── metrics.py
├── experiments/
│   ├── logs/                    (daily run notes)
│   └── configs/                 (per-run JSON configs)
├── results/
│   ├── raw/                     (per-image predictions, CSV)
│   └── tables/                  (aggregated tables)
├── figures/                     (plots and failure gallery)
├── paper/
│   └── draft.md
└── notebooks/                   (exploration only, not source of truth)
Environment
Windows + VS Code + PowerShell locally
Python 3.10+, PyTorch (CPU is fine — CLIP zero-shot on ~1000 images is quick)
Colab T4 only if the full grid feels slow locally
Key libraries: open_clip_torch, torch, torchvision, opencv-python, numpy, pandas, matplotlib, scikit-learn, pyyaml
Dataset

MonuMAI — 1,092 images, 4 architectural styles (Hispanic-Muslim, Gothic, Renaissance, Baroque). Download from the official GitHub release. Expert-labelled, small enough to run everything on CPU, no domain expertise needed to read.

Day 1 — Repo, environment, dataset, first CLIP forward pass
Objective

Get the smallest possible end-to-end pipeline running: load 1 image, run CLIP, print a prediction. Prove the plumbing works before building anything.

Tasks
Create the GitHub repo rectify-clip (public, MIT licence).
Set up virtualenv, requirements.txt, .gitignore (ignore data/, results/raw/, __pycache__, .venv).
Write README.md skeleton: title, one-line pitch, research question, architecture diagram (paste the mermaid block from this document).
Download MonuMAI into data/monumai/. Do not commit the images.
Write src/data_loader.py — one function load_monumai(root) that returns a list of (image_path, label_string) tuples.
Write src/clip_zero_shot.py — load open_clip ViT-B/32, encode 4 class-name prompts, classify one image, print predicted class + confidence.
Deliverables
Repo initialised, first commit pushed.
python src/clip_zero_shot.py --image data/monumai/<something>.jpg prints a class prediction and 4 softmax scores.
Documentation to save
experiments/logs/day1.md: what you set up, exact package versions (pip freeze), Python version, OS, which CLIP model & pretrained weights you chose and why (ViT-B/32 with laion2b_s34b_b79k is a solid default — cheap, well-known, easy to defend in the paper).
README.md — dataset citation and download instructions.
Success criteria

End-to-end: image path in, class label out. Nothing more. If this works, everything else is just structured expansion.

Do NOT
Do not start writing the distortion code today.
Do not tune prompts today.
Do not run on all images today.
Day 2 — Zero-shot CLIP baseline on clean data
Objective

Establish and freeze the "clean image" ceiling. Every later result gets measured against this number.

Tasks
Write configs/default.yaml with fields: model_name, pretrained_weights, class_names, prompt_template, image_size, seed, dataset_root.
Extend src/clip_zero_shot.py: batch inference over the full MonuMAI test split, log every prediction (image path, true label, predicted label, top-4 softmax) to results/raw/day2_clean_baseline.csv.
Compute overall top-1 accuracy, per-class accuracy, and a 4×4 confusion matrix. Save the confusion matrix plot to figures/day2_confusion_clean.png.
Prompt-template pilot: try 3 templates (e.g. "{}", "a photo of {} architecture", "a facade in {} architectural style"). Log each. Do NOT pick a winner yet — this becomes the prompt-sensitivity ablation on Day 6.
Commit and tag v0.1-baseline.
Deliverables
results/raw/day2_clean_baseline.csv
results/tables/day2_baseline_summary.md (accuracy overall + per class + per prompt)
figures/day2_confusion_clean.png
Documentation to save
experiments/logs/day2.md: exact prompts tried, exact class names used (spelling matters — CLIP is prompt-sensitive), baseline numbers, and one sentence on which style CLIP is best/worst at and any early hunch why.
Success criteria

You have one number ("clean baseline accuracy = X%") that you would defend in the paper. If this number is very low (e.g. <35% on a 4-way task, where random is 25%), stop and fix prompts before proceeding — the rest of the experiment is meaningless without a working baseline.

Do NOT
Do not tune prompts to death. Pick something reasonable and move on.
Do not touch the model weights or image preprocessing beyond CLIP's own defaults.
Day 3 — Synthetic distortion module
Objective

Build the controlled perspective-distortion tool. Every distortion is reproducible from a seed and a severity level.

Tasks
Write src/distortion.py. Core function: distort(image, severity, seed) → (distorted_image, homography_H).
Distortion recipe: take the 4 image corners; for each corner, pick a random offset from a bounded range that scales with severity (e.g. mild = up to 5% of image size, medium = 10%, severe = 20%). Use cv2.getPerspectiveTransform to build H, then cv2.warpPerspective to warp the pixels.
Return H so it can be inverted tomorrow.
Unit-test: for a fixed seed and severity, the function must return byte-identical output every call.
Build a small qualitative gallery: pick 6 random MonuMAI images, distort each at all 3 severities, save a 6×4 grid (original + 3 severities) to figures/day3_distortion_gallery.png. This is going straight into the paper.
Commit.
Deliverables
src/distortion.py with tests.
figures/day3_distortion_gallery.png.
Every distorted image's H matrix loggable to disk (as .npy next to the CSV, keyed by image ID and severity).
Documentation to save
experiments/logs/day3.md: exact severity ranges chosen, why those ranges (justify: mild = realistic phone shot, severe = extreme case where a human might still recognise the building), one paragraph on what happens visually at each severity.
In the paper draft: add a short "Distortion protocol" subsection describing this exactly.
Success criteria

Anyone can rerun distort(image, "medium", seed=42) on the same image on any machine and get the same pixels out.

Do NOT
Do not use a random affine or a scipy affine warp. Use a proper 3×3 homography via OpenCV — that is what a real camera tilt produces, and it is what a reviewer will expect.
Do not add rotation, scale, or lens distortion. Only projective warp. One variable at a time.
Day 4 — Oracle rectification module
Objective

Build the "if we knew the exact distortion, could we undo it?" rectifier. This is one function and one guarantee.

Tasks
Write src/rectification.py. Core function: rectify_oracle(distorted_image, H) → rectified_image.
Implementation: compute H_inv = np.linalg.inv(H), then cv2.warpPerspective(distorted_image, H_inv, (w, h)).
Sanity test: for any image I and any H, rectify_oracle(distort(I, H), H) should be pixel-close to I (ignoring interpolation and black-border artefacts). Quantify with mean absolute pixel difference on the non-border region.
Save the small artefact analysis: rectified images will have black triangular borders where pixels came from outside the original frame. Decide the policy now — you have three options: (a) leave the borders black; (b) crop to the central rectangle; (c) fill with mean colour. Test all three on 5 sample images and pick one with a written justification. This will come up in the paper and in questions.
Extend the Day 3 gallery: 6 images × 4 columns (original, mild distorted, mild rectified, side-by-side diff). Save to figures/day4_rectification_gallery.png.
Commit.
Deliverables
src/rectification.py with the sanity test.
figures/day4_rectification_gallery.png.
A short note in experiments/logs/day4.md documenting the border policy decision.
Documentation to save
experiments/logs/day4.md: chosen border-handling policy and why. Mean pixel error of oracle rectification (should be near-zero except for borders).
In the paper draft: "Rectification protocol" subsection.
Success criteria

rectify_oracle(distort(I, H), H) returns something visually indistinguishable from I except for the border region. If not, there is a bug — do not proceed to Day 5.

Do NOT
Do not attempt blind rectification (line detection, vanishing points) today. That is Day 7 stretch or paper future-work.
Do not fine-tune the border-handling policy across the week. Pick once, document, move on.
Day 5 — Full experimental grid
Objective

Run the actual experiment. Every (image, severity, condition) triple gets one row in one big table.

Tasks
Write src/experiment_runner.py. It loops: for each image → for each of 3 severities → produce distorted + rectified versions → run CLIP on {clean, distorted, rectified} → log a row.
Row schema: image_id, true_label, severity, condition ("clean"|"distorted"|"rectified"), predicted_label, correct (bool), top1_confidence, prompt_id, seed. Note: the "clean" condition is the same for every severity — that's fine, keep it for a clean comparison.
Total row count = 1092 images × 3 severities × 3 conditions ≈ 9,828 CLIP forward passes. On CPU with ViT-B/32 this is roughly 30–60 minutes. If it drags, move this single script to a Colab T4 notebook that clones your repo and runs the same code.
Save output to results/raw/day5_full_grid.csv. Also save the exact config used to experiments/configs/day5_run.json (seed, prompt, severity ranges, git commit hash).
Compute headline numbers: for each severity, accuracy of clean, distorted, rectified. Save to results/tables/day5_headline.md.
Commit and tag v0.5-grid-complete.
Deliverables
results/raw/day5_full_grid.csv (the source of truth for everything else in the paper).
results/tables/day5_headline.md.
Reproducible config JSON.
Documentation to save
experiments/logs/day5.md: wall-clock time, hardware used (local CPU vs Colab T4), any images that failed to load (there will be a couple — handle gracefully and note them).
Success criteria

The headline table exists and passes the sanity check: clean ≥ rectified ≥ distorted at every severity, OR you have a clear reason why not (e.g. rectification artefacts hurt at low severity). Either outcome is a real finding.

Do NOT
Do not start writing the paper yet.
Do not add more experimental conditions today. If Day 6 shows the design is missing something, you can rerun then — the runner is one function.
Day 6 — Analysis, metrics, plots, ablations, failure gallery
Objective

Turn the CSV into every figure and number the paper needs. Nothing new gets computed after today.

Tasks
Compute the recovery percentage per severity: recovery = (acc_rectified − acc_distorted) / (acc_clean − acc_distorted) × 100. This is the paper's headline metric.
Plot the severity curve: x-axis = severity (0/mild/medium/severe), 3 lines (clean, distorted, rectified). Save to figures/day6_severity_curve.png.
Compute mean CLIP confidence on the correct class per condition per severity. Save to results/tables/day6_confidence.md and plot to figures/day6_confidence.png.
Prompt-sensitivity ablation: rerun the Day 5 grid on just the medium severity, once per prompt template (3 prompts). Save to results/tables/day6_prompt_sensitivity.md. The point is to show the direction of the finding is robust to prompt choice, not that the exact numbers are.
Failure gallery: find 8 images where rectification hurt (correct-before, wrong-after). Save side-by-sides (distorted, rectified, CLIP predictions on each) to figures/day6_failure_gallery.png. Write one paragraph on the common pattern (probably border artefacts or over-warp).
Per-class breakdown: does rectification help all 4 styles equally? Save results/tables/day6_per_class.md.
Commit.
Deliverables
All figures under figures/day6_*.
All tables under results/tables/day6_*.
One-sentence answer to the research question, based on the numbers.
Documentation to save
experiments/logs/day6.md: the one-sentence answer, plus the 3–5 most surprising numbers.
Success criteria

You could hand the results/ and figures/ folders to someone else and they could write the paper. Do not proceed to Day 7 until this is true.

Do NOT
Do not run new experiments today (except the small prompt-ablation sweep listed above).
Do not cherry-pick figures. If rectification does not help, show that clearly — a well-reported null result is publishable.
Day 7 — Paper draft skeleton, README, repo polish, buffer
Objective

Package the work so it can be sent to a professor and pushed to GitHub as a portfolio piece. Leave a clear handoff for the paper-writing week.

Tasks
Fill paper/draft.md with the standard 6-section structure:
Abstract (150 words, written last).
Introduction — motivate with the Tarkhan et al. gap.
Related work — CLIP robustness (BATCLIP), zero-shot facade parsing (Tarkhan et al.), classical rectification for downstream models (GLNet, OpenFACADES).
Method — distortion protocol, oracle rectification, experimental grid, metrics.
Results — the severity curve, the recovery-% table, prompt sensitivity, per-class, failure gallery.
Discussion & limitations — oracle vs blind, single dataset, single model, single task; future work = blind rectification.
Write proper README.md: pitch, diagram, how to reproduce (single command ideally), citation, licence, results table.
Repo hygiene: docstrings on every function in src/, one-paragraph module docstring at the top of each file, requirements.txt pinned, Makefile or shell script for the "reproduce everything" flow.
Tag v1.0-paper-ready and push.
Buffer time: if anything from Days 1–6 slipped, catch it up here. If nothing slipped, spend it on the paper draft or on the blind-rectification stretch goal (line detection with cv2.HoughLinesP, vanishing-point estimation, homography-from-lines).
Backup the whole repo somewhere off GitHub (Google Drive zip) so a bad commit cannot lose the work.
Deliverables
paper/draft.md with all 6 sections, at least skeletal.
Portfolio-grade README.md.
Repo tagged v1.0-paper-ready.
Documentation to save
experiments/logs/day7.md: what shipped, what was cut, what the honest next step is (probably: blind rectification, then a second dataset like WikiChurches).
Success criteria

A stranger could clone the repo, run one command, and reproduce the headline result. The paper/draft.md is complete enough that turning it into a formatted conference submission is now a writing task, not a research task.

Do NOT
Do not add new experiments unless a specific reviewer would ask for them.
Do not spend more than 2 hours today on repo aesthetics. Function over polish.
Cross-cutting checklist (all 7 days)
 Every day ends with a git commit and a git push.
 Every experiment logs its config to experiments/configs/.
 Every plot is regenerable from CSVs — no manual edits in Photoshop.
 Random seeds are fixed everywhere.
 experiments/logs/dayN.md exists for every day, even if short.
 Nothing gets renamed or restructured after Day 5 without a very good reason.
Risks and honest fallbacks
CLIP baseline is too low (~random). Very unlikely on MonuMAI, but if it happens on Day 2, spend one extra day on prompt engineering and stronger class descriptions. Do not proceed with a broken baseline.
Oracle rectification does not visually restore the image. This is a bug in your distortion or inversion code, not a scientific finding. Fix on Day 4 before Day 5.
Rectification does not help CLIP at all (null result). This is still publishable — reframe the paper as "CLIP is more perspective-invariant than expected on facade classification." Do not fake positive results.
Full grid takes too long on CPU. Move Day 5 to Colab T4, or subsample MonuMAI to 500 images while keeping class balance. Report subsample size honestly.
You slip a day. Day 7 is the buffer. If you slip more, cut the blind-rectification stretch, cut the per-class breakdown, but never cut the prompt-sensitivity ablation — that is what makes a reviewer trust the headline finding.
Publication venue targets

Realistic: a Scopus-indexed IEEE India conference, a Springer LNNS/LNCS proceedings, or an arXiv preprint. Not realistic and not the goal: CVPR/ICCV.

Reference works to cite
Tarkhan et al., Mapping facade materials utilizing zero-shot segmentation…, Scientific Reports 15, 5492 (2025). DOI 10.1038/s41598-025-86307-1. — the gap that motivates this paper.
Radford et al., Learning Transferable Visual Models from Natural Language Supervision, ICML 2021. — the CLIP paper.
Sanyal et al., BATCLIP: Bimodal Online Test-Time Adaptation for CLIP, arXiv:2412.02837, 2024. — CLIP corruption robustness prior work.
Hartley & Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., 2004. — homography reference.
Ilyas Sikora et al., MonuMAI dataset paper — dataset reference.

Always double-check every citation before submission — Claude may hallucinate exact titles or authors.