# Canonical Evaluation Protocol

Created: 2026-06-19T02:47:58Z

Scope: opentry_5 train/dev only. Full-test sample-level labels, CIFs, StructureMatcher results, and tuned seeds remain forbidden. Historical full-test aggregate metrics are background only.

Frozen samples and order:

- `data/grouped_split_train_core.jsonl`
- `data/grouped_split_dev_model.jsonl`
- `data/grouped_split_dev_gate.jsonl`
- `data/grouped_split_fold_a.jsonl`
- `data/grouped_split_fold_b.jsonl`

The sample order is the JSONL order written in `manifests/canonical_sample_manifest.json`. The split hash is `{'dev_gate': '7c03864f56d58807c08be9b7b003b5c07296203cb5c4384fefa732a16d95656a', 'dev_model': '20f7f77f2225c8d91fdcd4498883c29a40891f4b048433855eaecafd167e6527', 'fold_a': 'a57c441544b4beb0e4e579642b9a361efa05dc4e5e88b9e2ccf02df44444b03e', 'fold_b': '7d2db348ae447212308d36765f1a34be4dfab736c0ed061ad20b61f3b16a00c2', 'train_core': '7c2fe905ec1ed8d9d0ae6123a433c656f369818d7788e140dcc37dacbc71725c'}`.

Metrics:

- match@1/5/20/50: native candidate order by `generation_index`; invalid candidates occupy their original slot.
- RMSE@1/5/20: best positive candidate RMSD among the first K slots; no positive means NA for that sample.
- rows>=7: `row_count >= 7` from canonical metadata.
- composition exact: reduced composition equality after CIF parse.
- SG/Wyckoff legal: generated rows are legal for GT-SG and parseable.
- positive-any: any positive within the fixed first K slots.

StructureMatcher:

- ltol=0.2
- stol=0.3
- angle_tol=5.0
- primitive_cell=true
- scale=true
- attempt_supercell=false

No-ranking rule: candidates may only be ordered by fixed generation_index and seed. Log-probability, confidence, energy, collision count, validity, self-score, learned threshold, oracle label, invalid deletion, or any post-generation sorting is forbidden.
