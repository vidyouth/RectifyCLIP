# Day 6 — prompt-sensitivity ablation (medium severity only, 3 prompts)

Goal: check whether the DIRECTION of the finding (rectification recovers some accuracy lost to distortion) holds regardless of prompt choice — not that the exact numbers match Day 5's default-prompt run.

| Prompt | Clean | Distorted | Rectified | clean >= rectified >= distorted? |
|---|---|---|---|---|
| `a_facade_in_architectural_style` | 0.595 | 0.585 | 0.596 | NO |
| `a_photo_of_architecture` | 0.614 | 0.578 | 0.605 | yes |
| `bare_class_name` | 0.368 | 0.341 | 0.368 | yes |
