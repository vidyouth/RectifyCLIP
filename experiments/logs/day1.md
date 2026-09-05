# Day 1 — Repo, environment, dataset, first CLIP forward pass

Date: 2026-09-05

## Objective (from execution plan)

Get the smallest possible end-to-end pipeline running: load 1 image, run CLIP,
print a prediction. Prove the plumbing works before building anything else.

## Environment

- OS: Windows 11 Home Single Language (build 10.0.26200)
- Python: 3.13.3, virtualenv at `.venv`
- Editor/shell: VS Code + PowerShell (per CLAUDE.md conventions), commands run
  via a Git Bash-compatible tool
- Hardware: local CPU only, no CUDA (see PyTorch choice below)

## Package installation

`requirements.txt` was written with **exact pinned versions**, CPU-only PyTorch,
no CUDA — per explicit instruction. Full `pip freeze` after install:

```
anyio==4.15.0
certifi==2026.7.22
click==8.5.0
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
filelock==3.32.5
fonttools==4.64.0
fsspec==2026.7.0
ftfy==6.3.1
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.30.0
idna==3.19
Jinja2==3.1.6
joblib==1.6.0
kiwisolver==1.5.1
MarkupSafe==3.0.3
matplotlib==3.9.2
mpmath==1.3.0
networkx==3.6.1
numpy==2.1.3
open_clip_torch==2.29.0
opencv-python==4.10.0.84
packaging==26.3
pandas==2.2.3
pillow==10.4.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.3.post1
PyYAML==6.0.2
regex==2026.9.3
safetensors==0.8.0
scikit-learn==1.5.2
scipy==1.18.1
setuptools==84.0.0
six==1.17.0
sympy==1.13.1
threadpoolctl==3.6.0
timm==1.0.29
torch==2.6.0+cpu
torchvision==0.21.0+cpu
tqdm==4.70.0
typing_extensions==4.16.0
tzdata==2026.3
wcwidth==0.8.3
```

`requirements.txt` pins the direct dependencies (`torch`, `torchvision`,
`open_clip_torch`, `opencv-python`, `numpy`, `pandas`, `matplotlib`,
`scikit-learn`, `pyyaml`, `Pillow`) plus a `--extra-index-url
https://download.pytorch.org/whl/cpu` line so `pip install -r requirements.txt`
resolves CPU-only torch/torchvision wheels with no separate install step.
Transitive dependencies (`safetensors`, `huggingface_hub`, `ftfy`, `timm`, etc.,
pulled in by `open_clip_torch`) are recorded above via `pip freeze` for exact
reproducibility, but are not hand-pinned in `requirements.txt` itself.

### Dead end: numpy==1.26.4 failed to install

First attempt pinned `numpy==1.26.4`. This has no prebuilt wheel for CPython
3.13 (`cp313`) on PyPI — pip fell back to building from source via Meson, which
failed because no C/C++ compiler is present on this machine (no MSVC, no
vswhere.exe, no gcc/clang). Fix: bumped to `numpy==2.1.3`, the first 2.x release
with `cp313` Windows wheels. Confirmed compatible with the rest of the pinned
stack (torch 2.6.0 supports numpy 2.x; opencv-python 4.10.0.84 and
scikit-learn 1.5.2 both installed cleanly against it).

**Lesson for future days:** when pinning exact versions for a package that
ships compiled wheels, check that the pinned version has a wheel for the
target Python version before assuming any old known-good version will work —
"known good together" from memory can be stale with respect to a specific,
recently-released Python (3.13 here).

### Verification

```
torch 2.6.0+cpu cuda_available= False
torchvision 0.21.0+cpu
open_clip 2.29.0
opencv 4.10.0
numpy 2.1.3
```

`torch.cuda.is_available()` is `False` as expected/required — confirms the CPU-only
build, no CUDA present.

## Model choice: CLIP ViT-B/32, `laion2b_s34b_b79k` pretrained weights

Chosen per CLAUDE.md's explicit default. Reasoning (for the paper's method
section):
- **Cheap**: ViT-B/32 is the smallest common CLIP vision backbone in `open_clip`,
  making ~9,800 forward passes (the eventual Day 5 grid) tractable on CPU in
  well under an hour.
- **Well-known**: ViT-B/32 is the architecture used in the original CLIP paper
  (Radford et al., 2021), so results are easy to contextualize against prior
  CLIP robustness literature (e.g. BATCLIP).
- **`laion2b_s34b_b79k`**: an open, widely-used, reproducible pretraining
  checkpoint (LAION-2B, 34B samples seen, batch size 79K) distributed via
  `open_clip`/HuggingFace Hub — no proprietary or closed weights, so the
  experiment is fully re-runnable by a reviewer with no special access.

No alternative backbones were evaluated on Day 1 — that would be tuning ahead
of schedule. This is a default choice made once and fixed for the whole project
per the "one variable at a time" working rule.

## Dataset: MonuMAI

Source: https://github.com/ari-dasci/OD-MonuMAI (official repo), dataset
contents live in `MonuMAI_dataset/`.

Procedure: cloned the repo into a temp directory outside the project
(`/tmp/OD-MonuMAI`), copied `MonuMAI_dataset/*` into `data/monumai/` preserving
the four style subfolders and their nested `xml/` annotation folders, then
deleted the temp clone. Confirmed `data/monumai/` remains gitignored
(`git check-ignore` matches every file under it via the `data/` rule in
`.gitignore`; `git status --porcelain data/` is empty).

Per-style image counts on disk after download:

| Style | Images | XML annotations |
|---|---|---|
| Baroque | 516 | 516 |
| Gothic | 359 | 359 |
| Hispanic-Muslim | 327 | 327 |
| Renaissance | 312 | 312 |
| **Total** | **1,514** | 1,514 |

These match the source repo's own README counts exactly.

### Flagged discrepancy: 1,514 vs. CLAUDE.md's "1,092 images"

CLAUDE.md states the dataset is "1,092 images, 4 architectural styles." The
actual download totals **1,514** images. This is a real discrepancy, not
something silently resolved:

- The 1,514 figure matches the official MonuMAI GitHub repo's own README
  exactly, so there is no download error.
- Possible explanations not yet confirmed: (a) 1,092 in CLAUDE.md was an
  approximate/remembered figure and should simply be corrected to 1,514; or
  (b) the plan's Day 2 phrase "batch inference over the full MonuMAI **test
  split**" implies the intended experimental set is a defined train/test split
  smaller than all 1,514 images, and 1,092 might refer to such a split (e.g. if
  the original MonuMAI paper's classification benchmark used a subset). No
  train/test split file was found in the downloaded repo during this check —
  this needs confirming before Day 2 commits to "the full test split."
- **Decision for now**: treating 1,514 as the correct total-images-on-disk
  figure for Day 1 (this is what `data_loader.py` reports and what was
  verified against the source README). Whether Day 2's experiment runs on all
  1,514 or a specific subset is an open question flagged for the start of Day 2,
  not resolved here.

## `src/data_loader.py`

`load_monumai(root)` walks the four style subfolders (`Hispanic-Muslim`,
`Gothic`, `Renaissance`, `Baroque`) in that fixed order, collects every file
directly inside each folder whose extension is `.jpg`/`.jpeg`/`.png`, and
returns `(image_path, label_string)` tuples. The nested `xml/` subfolder in
each style folder (Pascal VOC object-detection annotations) is intentionally
skipped — this project only needs whole-image classification labels, not the
object-level bounding boxes MonuMAI also ships for a detection task.

Verified against real data:

```
Loaded 1514 (image_path, label) pairs from data/monumai
  Hispanic-Muslim  327
  Gothic           359
  Renaissance      312
  Baroque          516
```

Matches the table above exactly.

## `src/clip_zero_shot.py`

Loads `ViT-B-32` / `laion2b_s34b_b79k` via `open_clip`, encodes one hardcoded
prompt per class (`"a photo of {} architecture"`), and classifies one image
given via `--image`.

**Assumption documented here (not just in code comments):** class names and the
prompt template are hardcoded in this script rather than read from
`configs/default.yaml`. CLAUDE.md's coding conventions say config values
"belong in `configs/default.yaml`, not hardcoded in scripts" — but the
execution plan's Day 2 section is explicitly what creates `default.yaml` with
its full field list (`model_name`, `pretrained_weights`, `class_names`,
`prompt_template`, `image_size`, `seed`, `dataset_root`). Writing that file
today, ahead of the plan's own schedule for it, would violate the "do not
implement a later day's tasks ahead of schedule" scope rule. Resolved this in
favor of scope discipline for Day 1 only: hardcode the minimum needed to prove
the pipeline works, then move the same values into `configs/default.yaml` on
Day 2 as the plan specifies.

The prompt template `"a photo of {} architecture"` is a reasonable, untuned
default — not a chosen winner. Day 2 explicitly pilots 3 templates without
picking one; Day 6 runs the real prompt-sensitivity ablation. This single
template is only meant to prove the forward pass works end to end.

Class name spelling (`Hispanic-Muslim`, `Gothic`, `Renaissance`, `Baroque`) is
taken directly from the MonuMAI folder names on disk, so it is guaranteed to
match `data_loader.py`'s labels exactly — no separate spelling decision was
needed here.

### First real run

```
$ python src/clip_zero_shot.py --image data/monumai/Gothic/20181212_105032.jpg
Predicted class: Gothic
Softmax scores:
  Hispanic-Muslim  0.0046
  Gothic           0.7968
  Renaissance      0.1917
  Baroque          0.0069
```

Correct prediction (true label: Gothic, from the folder it was drawn from),
high confidence (79.7%). This is a single anecdotal image, not a baseline
number — Day 2 is where a real accuracy figure gets computed over the whole
dataset/split. Two benign warnings appeared on this run and are not errors:
an "unauthenticated requests to the HF Hub" rate-limit notice (weights
download still succeeded), and a Windows-specific note that `huggingface_hub`'s
symlink cache is disabled without Developer Mode/admin (falls back to full
copies — functionally fine, just slightly more disk use for cached weights).

## Success criteria check

End-to-end works: image path in, class label out, non-trivial confidence
scores across all 4 classes. Per the plan, "if this works, everything else is
just structured expansion" — Day 1's objective is met.

## Open items carried into Day 2

1. Confirm whether MonuMAI defines an official train/test split (relevant to
   Day 2's "batch inference over the full MonuMAI test split" wording) and
   reconcile the 1,514-vs-1,092 discrepancy noted above before computing the
   frozen baseline number.
2. Move hardcoded class names/prompt template from `clip_zero_shot.py` into
   `configs/default.yaml` as Day 2 specifies.
