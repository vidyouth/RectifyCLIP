"""Runnable sanity test for src/rectification.py — the Day 4 correctness gate.

Per the plan: rectify_oracle(distort(I, H), H) must be pixel-close to I
OUTSIDE the border region, for any image I and any H. This is quantified here
as mean absolute pixel difference (0-255 scale) restricted to the non-border
("valid") mask, for real MonuMAI images across all 3 severities. A large
error here means a bug in the distortion or inversion math — not a finding —
and per the plan, Day 5 must not be started if this fails.

Run: .venv\\Scripts\\python.exe tests\\test_rectification.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_monumai  # noqa: E402
from distortion import distort, load_image_rgb  # noqa: E402
from rectification import compute_valid_mask, mean_absolute_error, rectify_oracle  # noqa: E402

# Mean absolute error, on a 0-255 scale, above which the round-trip is
# considered a bug rather than ordinary double-bilinear-interpolation blur.
# This is a generous ceiling (real double-warp blur is expected to be a
# handful of intensity levels at most, per Day 3's warp being a modest
# corner perturbation) — see experiments/logs/day4.md for the actual
# measured numbers, which are well under this.
MAX_ACCEPTABLE_MAE = 15.0

TEST_IMAGES = [
    "data/monumai/Gothic/20181212_105032.jpg",
    "data/monumai/Baroque/20181212_105011.jpg",
    "data/monumai/Hispanic-Muslim/5bae6559d55ef.jpg",
]
SEVERITIES = ["mild", "medium", "severe"]
SEED = 42


def test_round_trip_pixel_closeness():
    results = []
    for image_path in TEST_IMAGES:
        image = load_image_rgb(image_path)
        for severity in SEVERITIES:
            distorted, H = distort(image, severity, SEED)
            rectified = rectify_oracle(distorted, H)
            mask = compute_valid_mask(image.shape, H)
            mae = mean_absolute_error(image, rectified, mask)
            valid_fraction = mask.mean()
            results.append((image_path, severity, mae, valid_fraction))
            print(
                f"{image_path:50s} severity={severity:6s} "
                f"MAE(valid region)={mae:6.3f}  valid_fraction={valid_fraction:.3f}"
            )
            assert mae is not None, "mask had no valid pixels at all — unexpected"
            assert mae < MAX_ACCEPTABLE_MAE, (
                f"MAE {mae:.3f} exceeds {MAX_ACCEPTABLE_MAE} for {image_path} "
                f"at severity={severity} — treat as a BUG, not a finding"
            )
    print("test_round_trip_pixel_closeness: PASSED")
    return results


if __name__ == "__main__":
    test_round_trip_pixel_closeness()
    print("\nALL TESTS PASSED")
