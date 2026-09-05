# Day 5 — McNemar paired significance test (distorted vs rectified)

Tests whether the accuracy gap between distorted and rectified at each severity
is distinguishable from chance, using only the discordant pairs (images where the
two conditions disagree), since both are evaluated on the identical 1,514 images.

| Severity | b (distorted-only correct) | c (rectified-only correct) | n discordant | p-value | significant at p<0.05? |
|---|---|---|---|---|---|
| mild | 81 | 72 | 153 | 0.5179 | no |
| medium | 81 | 122 | 203 | 0.0049 | yes |
| severe | 96 | 186 | 282 | 0.0000 | yes |
