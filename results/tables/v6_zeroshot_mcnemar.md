# Day 12 v6 — Experiment A: McNemar's test, distorted vs rectified

| Model | Angle | b (distorted-correct, rectified-wrong) | c (distorted-wrong, rectified-correct) | n discordant | p-value |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 5 | 101 | 105 | 206 | 0.8345 |
| CLIP ViT-B/32 | 15 | 134 | 126 | 260 | 0.6643 |
| CLIP ViT-B/32 | 35 | 226 | 180 | 406 | 0.0254 |
| CLIP ViT-L/14 | 5 | 73 | 81 | 154 | 0.5728 |
| CLIP ViT-L/14 | 15 | 88 | 93 | 181 | 0.7663 |
| CLIP ViT-L/14 | 35 | 141 | 151 | 292 | 0.5985 |
| SigLIP ViT-B/16 | 5 | 103 | 68 | 171 | 0.0091 |
| SigLIP ViT-B/16 | 15 | 112 | 102 | 214 | 0.5385 |
| SigLIP ViT-B/16 | 35 | 157 | 116 | 273 | 0.0153 |
