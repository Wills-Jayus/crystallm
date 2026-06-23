# No-Ranking Audit

Created: 2026-06-19T02:47:58Z
Updated: 2026-06-19T14:45:00Z

Pass for generated opentry_5 Phase 0-4 data artifacts:

- No test sample-level labels or CIFs are read.
- No energy checkpoint, scorer, ranker, selector, threshold, oracle label, collision ordering, self-score, or learned quality sort is used to order candidates.
- OOF W/A prediction is a training/inference condition-gap dataset, not candidate selection.
- Hard-negative rows preserve native train candidate rank only as provenance for corruption training; they are not inserted into inference output.
- All future generated candidates must remain ordered by generation_index and seed.

## E8009-E8038 Fixed-Order Joint Smoke

- Scripts: `scripts/train_eval_fixed_order_joint_smoke.py`, `scripts/train_eval_fixed_order_geom_sampler_smoke.py`.
- `gen_index=0` is deterministic greedy decode.
- `gen_index=1..19` are single-path decodes using the frozen seed list in `configs/model_d_joint_generation.yaml` when `k=20`.
- The decoder uses SG/formula exact-cover masks as hard validity constraints; it does not build a candidate pool and then sort/select.
- No `score`, `generation_score`, energy, logprob, selector, ranker, or oracle label is written or used for candidate order.
- Invalid or failed decodes are retained at their original `gen_index` and count as failures.
- E8009 through E8038 reports record `candidate_order: generation_index_then_fixed_seed` and `no_ranking: true`.
- E8013/E8014 changed only the training curriculum toward rows>=7 examples; candidate generation and ordering were unchanged from E8009-E8012.
- E8015/E8016 changed only loss weights on top of the E8013/E8014 curriculum; candidate generation and ordering were unchanged.
- E8017-E8020 changed the geometry objective to heteroscedastic free-param/lattice NLL and used fixed-seed geometry sampling. The generated rows record `geometry_sampler: heteroscedastic_nll_fixed_seed`; no likelihood, scale, score, or evaluator metric is used to reorder candidates.
- E8019/E8020 changed training scale, loss balance, and fixed temperature/sample-scale settings only; candidate order remained generation_index.
- E8021 added only training-time representative row-pair auxiliary losses; no row-pair distance, collision flag, likelihood, or evaluator metric is used to order or filter inference candidates.
- E8022 added only training-time expanded atom-cloud pair-distance and separation losses; no expanded distance, collision flag, likelihood, or evaluator metric is used to order or filter inference candidates.
- E8023 added only training-time cartesian atom-cloud pair-distance and separation losses; no cartesian distance, collision flag, likelihood, or evaluator metric is used to order or filter inference candidates.
- E8024 combined higher rows>=7 training curriculum with training-time expanded and cartesian atom-cloud auxiliary losses; no expanded/cartesian distance, collision flag, likelihood, or evaluator metric is used to order or filter inference candidates.
- E8025 changed only sampling temperature/scale and training-time cartesian auxiliary weights on the same fixed-order path; no cartesian distance, collision flag, likelihood, or evaluator metric is used to order or filter inference candidates.
- E8026 changed only decoder conditioning to include the previous representative coordinate during training and generation; candidate order remained generation_index and no geometry metric is used for sorting or filtering.
- E8027 added only training-time target-local/nearest-neighbor cartesian auxiliary losses; no local distance, collision flag, likelihood, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8028 repeated the E8027 training/inference protocol on fold_a; candidate order remained generation_index plus fixed seed and no fold_a metric is used to sort or filter candidates.
- E8029 changed only the training-time local separation term to active-only violations on fold_b; no local distance, collision flag, likelihood, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8030 changed only training-time symbolic loss weights and active-local-separation strength on fold_b; no symbolic probability, local distance, collision flag, likelihood, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8031 changed only training-time local pair-distance weight and removed active separation on fold_b; no local distance, collision flag, likelihood, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8032 changed only coordinate/lattice sampling scale while loading the E8031 train-loss-selected checkpoint in eval-only mode; no validity, distance, collision flag, likelihood, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8033 added only a training-time target-relative cartesian separation loss on fold_b; no target-relative distance, validity, collision flag, likelihood, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8034 changed only training-time target-relative separation strength and complex-row symbolic weighting on fold_b; no symbolic probability, target-relative distance, validity, collision flag, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8035 changed only training-time target-relative separation strength on top of the E8034-style objective; no target-relative distance, validity, collision flag, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8036 changed only coordinate/lattice sampling scale while loading the E8034 train-loss-selected checkpoint in eval-only mode; no validity, bond score, collision flag, likelihood, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8037 added only training-time complex-row geometry NLL weighting; no geometry likelihood, validity, bond score, collision flag, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.
- E8038 changed only the scalar strength of the training-time complex-row geometry NLL weighting; no geometry likelihood, validity, bond score, collision flag, evaluator metric, or StructureMatcher outcome is used to order or filter inference candidates.

Forbidden routes are registered in `reports/dead_end_registry.md`.
