# Canonical Representation Report

Created: 2026-06-19T02:46:40Z

Representation: `opentry5_symmetry_canonical_v1_lossless_structured_symcif_v4`.

The dataset preserves the structured SymCIF-v4 fields already extracted with symmetry-aware Wyckoff rows, lattice parameters, free-parameter metadata, source coordinates, and a frozen copy of the source CIF under `data/canonical_train` or `data/canonical_dev`. This first representation is lossless by design; destructive origin/axis equivalence reduction is deferred until it can be proven by round-trip audit.

Key fields: formula, GT-SG, crystal system, canonical skeleton key, canonical W/A key, row order, multiplicity, Wyckoff letter, free-symbol mask, free params, lattice, source coordinate, and canonical CIF path.

| split | canonical records | CIF copies |
|---|---:|---:|
| train_core | 41337 | 41332 |
| dev_model | 6566 | 6566 |
| dev_gate | 4724 | 4724 |
| fold_a | 5388 | 5388 |
| fold_b | 5902 | 5902 |

Round-trip audit is executed by `scripts/audit_canonical_roundtrip.py` and writes `eval/canonical_roundtrip_audit.json`.

## Round-Trip Audit

Audit time: 2026-06-19T02:46:51Z
Records checked: 2047
Records considered: 2048
Skipped missing/invalid CIF: 1
Success rate: 1.0000
Parse failures: 0
Composition mismatches: 0
StructureMatcher failures: 0
Pass: True

This audit reads canonical CIF copies and source CIFs only from train/dev data; no test files are accessed.
