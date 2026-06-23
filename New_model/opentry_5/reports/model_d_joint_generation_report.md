# Model D Joint Generation Report

Model D is reserved for joint W/A + geometry generation when separate GT/OOF W/A and geometry components leave a train/inference mismatch.

## E8004/E8006 Gate0

MiniCFJoint-v2 target round-trip smokes passed on both grouped folds:

- fold_a: `reports/minicfjoint_v2_fold_a_gate0_smoke/00_target_roundtrip_audit.json`
- fold_b: `reports/minicfjoint_v2_fold_b_gate0_smoke/00_target_roundtrip_audit.json`

The fold_a small-overfit attempt E8005 wrote a checkpoint but was interrupted in the evaluator tail, so it is not counted as a passed gate.

## E8009-E8038 Fixed-Order Smoke

`scripts/train_eval_fixed_order_joint_smoke.py` trains a MiniCFJointV2Net smoke model and decodes candidates as single fixed generation_index/seed paths. It does not use beam-score ordering, candidate ranking, selector models, energy scoring, oracle replacement, or invalid-candidate deletion.

| experiment | fold | train/eval | match@1 | match@5 | match@20 | rows>=7 match@20 | status |
|---|---:|---:|---:|---:|---:|---:|---|
| E8009 | fold_a | 128 / 32 | 21.88% | 28.12% | 43.75% | 0.00% | path-check only |
| E8010 | fold_b | 128 / 32 | 9.38% | 21.88% | 34.38% | 0.00% | path-check only |
| E8011 | fold_a | 512 / 64 | 14.06% | 21.88% | 23.44% | 0.00% | scale-up failed rows>=7 |
| E8012 | fold_b | 512 / 64 | 28.12% | 34.38% | 43.75% | 0.00% | scale-up failed rows>=7 |
| E8013 | fold_a | 806 / 64 | 9.38% | 20.31% | 26.56% | 11.11% | rows>=7 curriculum partial |
| E8014 | fold_b | 806 / 64 | 10.94% | 21.88% | 31.25% | 0.00% | rows>=7 W/A improved, geometry failed |
| E8015 | fold_a | 806 / 64 | 28.12% | 37.50% | 39.06% | 11.11% | geometry-heavy CE partial |
| E8016 | fold_b | 806 / 64 | 17.19% | 23.44% | 28.12% | 0.00% | geometry-heavy CE not replicated |
| E8017 | fold_a | 806 / 64 | 7.81% | 23.44% | 42.19% | 11.11% | heteroscedastic geometry sampler top20 partial |
| E8018 | fold_b | 806 / 64 | 14.06% | 29.69% | 43.75% | 0.00% | heteroscedastic geometry sampler not rows>=7 |
| E8019 | fold_b | 2323 / 64 | 28.12% | 45.31% | 54.69% | 0.00% | balanced sampler strong overall, rows>=7 failed |
| E8020 | fold_a | 2323 / 64 | 15.62% | 32.81% | 51.56% | 0.00% | paired balanced sampler top20 only |
| E8021 | fold_b | 2323 / 64 | 21.88% | 32.81% | 53.12% | 0.00% | row-pair auxiliary top20 only, rows>=7 failed |
| E8022 | fold_b | 2323 / 64 | 12.50% | 25.00% | 53.12% | 0.00% | expanded-structure auxiliary improved complex W/A, geometry failed |
| E8023 | fold_b | 2323 / 64 | 18.75% | 37.50% | 59.38% | 0.00% | cartesian auxiliary strongest top20, complex W/A failed |
| E8024 | fold_b | 2470 / 64 | 31.25% | 40.62% | 57.81% | 0.00% | combined expanded+cartesian auxiliary near top1/top5 thresholds, complex geometry failed |
| E8025 | fold_b | 2470 / 64 | 18.75% | 31.25% | 54.69% | 0.00% | lower sample scale plus stronger cartesian loss regressed W/A and collision rate |
| E8026 | fold_b | 2470 / 64 | 23.44% | 34.38% | 57.81% | 0.00% | previous-coordinate decoder retained complex W/A, geometry still failed |
| E8027 | fold_b | 2470 / 64 | 29.69% | 45.31% | 51.56% | 0.00% | local-neighbor cartesian loss exceeded top5/top20 thresholds, rows>=7 failed |
| E8028 | fold_a | 2470 / 64 | 10.94% | 34.38% | 48.44% | 11.11% | paired local-neighbor fold_a near top20, one rows>=7 positive |
| E8029 | fold_b | 2470 / 64 | 26.56% | 45.31% | 60.94% | 0.00% | active-only local separation improved aggregate top20, collapsed complex W/A |
| E8030 | fold_b | 2470 / 64 | 21.88% | 31.25% | 54.69% | 0.00% | complex symbolic weighting plus weaker active separation regressed top1/top5 and did not restore W/A |
| E8031 | fold_b | 2470 / 64 | 21.88% | 42.19% | 56.25% | 0.00% | stronger local pair loss without active separation restored rows>=7 W/A/skeleton, geometry still failed |
| E8032 | fold_b | 2470 / 64 | 21.88% | 35.94% | 51.56% | 0.00% | eval-only lower sampling scale from E8031 preserved rows>=7 W/A but regressed aggregate top5/top20 |
| E8033 | fold_b | 2470 / 64 | 10.94% | 28.12% | 62.50% | 0.00% | target-relative cartesian separation improved top20 and bond scores but regressed early W/A/top5 |
| E8034 | fold_b | 2470 / 64 | 34.38% | 48.44% | 51.56% | 0.00% | weaker target-relative loss plus complex symbolic weighting exceeded all aggregate thresholds, rows>=7 still failed |
| E8035 | fold_b | 2470 / 64 | 28.12% | 40.62% | 56.25% | 0.00% | stronger target-relative loss regressed top1/top5 and did not create rows>=7 matches |
| E8036 | fold_b | 2470 / 64 | 34.38% | 48.44% | 51.56% | 0.00% | eval-only lower sampling from E8034 preserved aggregate thresholds but rows>=7 still failed |
| E8037 | fold_b | 2470 / 64 | 15.62% | 23.44% | 45.31% | 0.00% | complex geometry NLL weighting restored some rows>=7 W/A but regressed aggregate accuracy |
| E8038 | fold_b | 2470 / 64 | 14.06% | 28.12% | 57.81% | 0.00% | weak complex geometry NLL weighting preserved top20 but lost rows>=7 W/A and early accuracy |

Artifacts:

- `checkpoints/fixed_order_joint_smoke_fold_a/best_train.pt`
- `checkpoints/fixed_order_joint_smoke_fold_b/best_train.pt`
- `checkpoints/fixed_order_joint_smoke_E8011_fold_a/best_train.pt`
- `checkpoints/fixed_order_joint_smoke_E8012_fold_b/best_train.pt`
- `checkpoints/fixed_order_joint_smoke_E8013_fold_a/best_train.pt`
- `checkpoints/fixed_order_joint_smoke_E8014_fold_b/best_train.pt`
- `checkpoints/fixed_order_joint_smoke_E8015_fold_a/best_train.pt`
- `checkpoints/fixed_order_joint_smoke_E8016_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8017_fold_a/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8018_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8019_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8020_fold_a/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8021_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8022_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8023_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8024_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8025_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8026_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8027_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8028_fold_a/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8029_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8030_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8031_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8032_fold_b/eval_only_source.json`
- `checkpoints/fixed_order_geom_sampler_smoke_E8033_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8034_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8035_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8036_fold_b/eval_only_source.json`
- `checkpoints/fixed_order_geom_sampler_smoke_E8037_fold_b/best_train.pt`
- `checkpoints/fixed_order_geom_sampler_smoke_E8038_fold_b/best_train.pt`
- `eval/fixed_order_joint_smoke_fold_a/report.json`
- `eval/fixed_order_joint_smoke_fold_b/report.json`
- `eval/fixed_order_joint_smoke_E8011_fold_a/report.json`
- `eval/fixed_order_joint_smoke_E8012_fold_b/report.json`
- `eval/fixed_order_joint_smoke_E8013_fold_a/report.json`
- `eval/fixed_order_joint_smoke_E8014_fold_b/report.json`
- `eval/fixed_order_joint_smoke_E8015_fold_a/report.json`
- `eval/fixed_order_joint_smoke_E8016_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8017_fold_a/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8018_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8019_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8020_fold_a/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8021_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8022_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8023_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8024_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8025_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8026_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8027_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8028_fold_a/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8029_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8030_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8031_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8032_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8033_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8034_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8035_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8036_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8037_fold_b/report.json`
- `eval/fixed_order_geom_sampler_smoke_E8038_fold_b/report.json`

Conclusion: the no-ranking full-CIF generation path is executable on both grouped folds, but it is not a terminal result. E8009/E8010 proved the path; E8011/E8012 tested a plain train-scale/model-size increase and failed rows>=7. E8013/E8014 changed the training curriculum toward rows>=7. E8015/E8016 kept that curriculum and increased geometry/lattice CE weights; weighted CE alone did not replicate rows>=7 match on fold_b. E8017/E8018 changed the geometry objective to heteroscedastic free-param/lattice NLL with fixed-seed sampling. E8019/E8020 rebalanced symbolic CE against geometry NLL and increased model/data scale; top20 exceeded the target threshold on both folds, and E8019 also exceeded match@5 on fold_b. E8021-E8026 tested row-pair, expanded, cartesian, stronger cartesian, and previous-coordinate variants without terminal rows>=7 success. E8027 added a target-local/nearest-neighbor cartesian auxiliary and exceeded the top5/top20 thresholds on fold_b, but rows>=7 match@20 remained 0. E8028 repeated the same protocol on fold_a: top20 was near threshold at 48.44% and rows>=7 match@20 reached 11.11% with one new positive. E8029 made the local separation loss active-only and reached 26.56% / 45.31% / 60.94% on fold_b, but rows>=7 W/A, skeleton, and match remained 0. E8030 added complex symbolic loss weighting and weakened active separation; it restored one rows>=7 skeleton hit but not W/A or match, and top1/top5 regressed. E8031 removed active separation and raised target-local pair loss; rows>=7 W/A/skeleton@20 recovered to 66.67%, while match@5/top20 stayed above thresholds, but rows>=7 match@20 remained 0. E8032 re-evaluated the E8031 checkpoint with lower coordinate/lattice sampling scale; rows>=7 W/A remained 66.67%, but aggregate top5/top20 regressed and rows>=7 match stayed 0. E8033 added target-relative cartesian separation; top20 rose to 62.50% and some rows>=7 bond scores improved, but top1/top5 regressed and rows>=7 match stayed 0. E8034 balanced weaker target-relative loss with complex symbolic weighting and exceeded all aggregate thresholds, but rows>=7 match stayed 0. E8035 raised target-relative loss to 1.5; it kept top20 high but regressed top1/top5 and still had rows>=7 match@20 = 0. E8036 lowered E8034 inference sampling to 0.45/0.25; aggregate thresholds stayed passed, but rows>=7 match stayed 0. E8037/E8038 tested complex-row geometry NLL weighting at 1.5 and 1.2; both regressed early accuracy, and E8038 lost rows>=7 W/A entirely. This branch is stopped. The next work is a more targeted validity repair or returning to E8034/E8036 for architecture-level changes, not dev_gate or val512.
