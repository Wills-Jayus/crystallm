# Grouped Split Leakage Audit

Created: 2026-06-19T02:46:40Z

Split policy: deterministic SHA256 bucket on a prototype-aware group key. The key includes SG, anonymized reduced formula, element count, Wyckoff multiplicity and letter sequence, free-symbol pattern, row count, crystal system, and a prototype fingerprint. Identical keys are never split across train/dev.

| split | records | groups | rows>=7 | datasets |
|---|---:|---:|---:|---|
| train_core | 41337 | 12947 | 9027 | {"mp20": 21184, "mpts52": 20153} |
| dev_model | 6566 | 1675 | 1109 | {"mp20": 3193, "mpts52": 3373} |
| dev_gate | 4724 | 1578 | 1033 | {"mp20": 2252, "mpts52": 2472} |
| fold_a | 5388 | 1568 | 1083 | {"mp20": 2602, "mpts52": 2786} |
| fold_b | 5902 | 1685 | 1059 | {"mp20": 2843, "mpts52": 3059} |

Train/dev group overlap count: 0.
Grouped fold/train overlap count: 0.

Status: pass for exact prototype-key leakage. Similarity beyond the explicit key is not used to move samples back into train; future stricter clustering must update the manifest before any model comparison.
