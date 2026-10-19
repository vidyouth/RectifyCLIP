# Day 10 v4 — example gallery image selection

Selected with a fixed seed (GALLERY_SEED=42) from `results/raw/day10_v4_grid.csv`.

## Part A rows

| Row | image_id | True style | Selection criterion |
|---|---|---|---|
| 1 | Hispanic-Muslim__5c700b74d7c96 | Hispanic-Muslim | Hispanic-Muslim: rectification fixes a wrong prediction (15deg) |
| 2 | Baroque__5b8a5bebf2b05 | Baroque | Baroque: rectification fixes a wrong prediction (15deg) |
| 3 | Gothic__5cadb9254e5bb | Gothic | Gothic: rectification breaks a correct prediction (15deg) |
| 4 | Gothic__5c699dfd45a1a | Gothic | Gothic: rectification breaks a correct prediction (15deg) -- second example |
| 5 | Gothic__catedral_leon13 | Gothic | Easy case: correct in all 7 columns |
| 6 | Baroque__5cf7e81e4e982 | Baroque | Hard case: both distorted and rectified fail at 35deg |

## Part B direction strip

Image: `Renaissance__san_pedro_cordoba7` (true style: Renaissance)
Theta: 15 degrees. Explicit axis angles (not sampled, not in the v4 CSV): [0, 60, 120, 180, 240, 300].
