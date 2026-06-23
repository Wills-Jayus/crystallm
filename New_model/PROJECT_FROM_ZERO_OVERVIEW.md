# New_model Project Overview From Zero

Created at: 2026-06-23T16:20:57Z

This document is a from-zero introduction to the crystal-generation work under:

`/data/users/xsw/autodlmini/model/New_model`

It is written for an AI or researcher who does not yet know the project. It summarizes the scientific problem, the evaluation protocol, the two major research lines, the historical experiments, the lessons learned, and the current `opentry_10` state. It is based on high-signal documents and reports under `New_model`, including `Paper/`, `data_exp/`, `symcif_experiment/`, `opentry/`, and `opentry_2` through `opentry_10`. Large candidate banks, per-sample JSONL shards, generated CIF pools, and low-level metrics shards were not read one by one; they are treated as data artifacts rather than explanatory documents.

## 1. Executive Summary

This project studies crystal structure generation. The central task is: given a chemical composition, and in some settings also the ground-truth space group, generate candidate CIF crystal structures whose atomic arrangement matches the reference structure under a StructureMatcher protocol.

There are two related but distinct research programs in this directory.

The first is the **SymCIF / structured crystal generation line**. This line asks whether crystal generation can be made more controllable by explicitly representing crystallographic symmetry: space group, Wyckoff assignment, multiplicities, skeletons, and geometry/free parameters. It produced strong diagnostic and manuscript-ready evidence on MP-20 under a symmetry-conditioned setting. The strongest SymCIF result is an MP-20 structured-test hybrid-prior run with match@1 = 60.90% and match@5 = 73.68%. After counting the non-structured original MP-20 samples as failures, the adjusted match@1/match@5 are 59.87%/72.43%. This is a strong top-1 recovery result relative to published CrystaLLM-a composition-only match@1, but it is not a same-input comparison because SymCIF uses composition plus oracle ground-truth space group, while published CrystaLLM-a is composition-only. The SymCIF line also shows that Wyckoff assignment candidate selection is a major bottleneck, and geometry quality remains a second hard bottleneck.

The second is the **opentry official-over-baseline line**. This line is stricter. Its active version is `opentry_10`, whose goal is not merely to report a diagnostic SymCIF result, but to produce at least one frozen strategy/fusion system or pure structural model that truly exceeds the CrystaLLM-a GT-SG anchor under the CrystaLLM Table 3 official full-test K20 protocol. The current anchor is much stronger than published composition-only CrystaLLM-a: on MP-20, CrystaLLM-a GT-SG has match@1/5/20 = 71.67/83.08/87.81; on MPTS-52, it has match@1/5/20 = 25.23/36.46/43.96. Under `opentry_10`, a real final report is forbidden until full validation gates, frozen configuration, and official full-test evaluation are complete.

The historical conclusion is conservative. Earlier fusion and pure-model attempts did not produce a meaningful official improvement over CrystaLLM-a GT-SG. The main positive evidence is that there is reranking headroom inside CrystaLLM K20/K100 candidates and that SymCIF candidates are complementary in some settings. The current best path is therefore not to claim success immediately, but to build a full validation CrystaLLM GT-SG K100 candidate bank, use it for leakage-free out-of-fold reranking and fusion decisions, and only then freeze a system for official test.

As of the latest status read in this document, `opentry_10` is still in the mandatory validation-anchor generation stage. MP-20 validation K100 generation has progressed beyond halfway: the supervisor state reports 83 completed shards out of 142, one running shard, and 58 pending shards. MPTS-52 is paused at 16 completed shards out of 79. No frozen system exists in `state/frozen_registry.json`.

## 2. Basic Concepts

### 2.1 What Is a CIF?

CIF means Crystallographic Information File. It is a text format used to describe a crystal structure. It contains cell lengths and angles, symmetry information, atom types, fractional coordinates, occupancies, and other crystallographic fields. A model can generate CIF text directly, but direct text generation is hard because small syntax or geometry errors can make a structure invalid or mismatched.

### 2.2 What Is the Crystal Generation Task?

The task is to generate crystal candidates from an input prompt. The prompt may be only the composition, such as `LiFePO4`, or it may include additional oracle information such as the ground-truth space group. The model outputs one or more CIF candidates. Each candidate is compared with the target crystal structure. If any of the top-k candidates match the target under StructureMatcher, the sample is counted as a match@k success.

### 2.3 What Are MP-20 and MPTS-52?

MP-20 and MPTS-52 are benchmark datasets used throughout this project. They have official train, validation, and test CSV splits under the CrystaLLM resources. `opentry_10` verified the official record counts:

| dataset split | records |
| --- | ---: |
| MP-20 train | 27136 |
| MP-20 validation | 9047 |
| MP-20 test | 9046 |
| MPTS-52 train | 27380 |
| MPTS-52 validation | 5000 |
| MPTS-52 test | 8096 |

MP-20 is smaller-complexity on average than MPTS-52 and has become the current priority because the user instructed subsequent new experiments to focus on MP-20 first. MPTS-52 is still important, but it should not be scaled in parallel until MP-20 evidence is strong enough or a go/no-go decision is reached.

### 2.4 What Is Ground-Truth Space Group Conditioning?

A space group describes the symmetry of a crystal. A composition-only model receives only the chemical formula. A GT-SG model receives the composition plus the true space group. GT-SG conditioning is easier and usually stronger. It is also a different input condition from composition-only generation.

This distinction is critical. The SymCIF paper line often uses composition + oracle GT-SG. The active `opentry_10` official target compares against CrystaLLM-a GT-SG, not composition-only CrystaLLM-a. Therefore, beating published composition-only CrystaLLM-a is not enough to satisfy `opentry_10`.

### 2.5 What Are Wyckoff Assignments and Skeletons?

In a space group, atomic sites occupy Wyckoff positions. A Wyckoff assignment maps elements to Wyckoff orbits and multiplicities. A skeleton is the orbit/multiplicity structure without necessarily assigning the exact elements. SymCIF treats crystal generation as a structured problem:

composition + space group -> Wyckoff skeleton -> Wyckoff assignment -> geometry/free parameters -> CIF.

This is different from free CIF language modeling, where all syntax, symmetry, atom counts, and geometry are entangled in one token stream.

### 2.6 What Are the Main Metrics?

The standard metrics are:

- match@1, match@5, match@20: whether the target is matched by the top 1, 5, or 20 candidates.
- RMSE@1, RMSE@5, RMSE@20: normalized RMS distance for matched candidates.
- rows>=7 subset: a harder subset defined from the number of `_atom_site` rows in the target CIF.
- validity/readability/formula/SG consistency: diagnostics only; they do not replace StructureMatcher match.

`opentry_10` uses StructureMatcher with ltol=0.3, stol=0.5, angle_tol=10, and normalized `get_rms_dist(...)[0]`. Missing candidates count as failures.

## 3. The Two Project Goals Must Not Be Confused

The project contains a paper-oriented SymCIF line and an official-over-baseline `opentry_10` line.

The SymCIF line can support a medium-conservative manuscript claim: explicit symmetry decomposition and hybrid-prior Wyckoff reranking produce strong MP-20 top-1 recovery under a ground-truth-space-group setting. This is useful and scientifically meaningful.

The `opentry_10` line has a harder success standard. It requires at least one frozen system on the official full test to exceed CrystaLLM-a GT-SG by at least 1 percentage point on at least one match metric, while match@20 does not drop by more than 0.2 percentage points and rows>=7 does not clearly degrade. It also forbids using test labels or test oracle selection. This is a production-style benchmark challenge, not a manuscript diagnostic exercise.

The most common historical mistake would be to take a positive SymCIF structured-test result and treat it as proof that the active `opentry_10` goal is solved. It is not. The input condition, candidate budget, validation gate, official full-test protocol, and baseline are different.

## 4. Evaluation Protocol and Leakage Rules

The strongest recurring lesson across the directory is that evaluation protocol matters as much as model design.

The official protocol requires full official train/validation/test splits, fixed candidate budgets, full sample coverage, and missing candidates counted as failures. Small sanity checks are useful only to verify scripts and generation mechanics. They cannot be used for model selection or final claims.

The project repeatedly forbids:

- using test target structures or test StructureMatcher labels for training, threshold selection, ranking, or route selection;
- freezing a strategy after inspecting test-only rescue patterns;
- claiming success from coverage repair;
- dropping difficult or missing samples;
- treating partial validation shards as a full validation gate.

The correct pattern is:

1. Build candidates on train/validation without using test labels.
2. Train or tune rankers using group cross-validation, where each sample's candidates stay in the same fold.
3. Evaluate all formal validation claims from out-of-fold predictions.
4. Freeze hyperparameters and model configuration.
5. Only then run the official full test.

`opentry_10` explicitly requires this discipline because earlier runs showed how easy it is to mistake a test-only diagnostic for a deployable strategy.

## 5. Chronological History and What Each Phase Taught

### 5.1 Early SymCIF Training Experiments

The earliest `symcif_experiment` training reports compared baseline CIF-like formats, CrystalFormer-like formats, and early SymCIF formats. The key result was not that SymCIF immediately won by token loss. In fact, `training_final_analysis.md` shows that all three early training runs overfit around step 750, while final-step checkpoints were worse. Baseline CIF had the lowest validation loss, but the report correctly warns that token loss is not a direct proof of structure-generation quality because each representation has a different semantic burden.

The early lesson was:

- token compression is useful but not sufficient;
- best checkpoint preservation matters;
- training loss alone cannot select a crystal generator;
- structure-level evaluation is mandatory.

The later `data_exp` experiment reinforced this. It trained two GPT-style models from scratch on MPTS-52: raw CIF text and SymCIF-v4 text. Under the same byte-level tokenizer and 3500-step schedule, raw CIF had better validation loss. SymCIF-v4 did not become better merely by changing the text format. This killed the naive hypothesis that a structured text format alone would improve language-model training.

### 5.2 SymCIF-v2/v3: Skeletons, Assignments, and Diagnostics

The intermediate SymCIF rounds introduced more crystallographic structure. They separated skeleton prediction, assignment search, and rendering. The repeated finding was that skeleton recall could be improved, but exact Wyckoff assignment recall remained difficult. Even when symbolic recall improved, matching final CIF structures still required good geometry/free parameters.

This is the central conceptual shift: correct symbolic structure does not automatically imply a StructureMatcher match. The system needs both:

1. correct or plausible Wyckoff assignment, and
2. continuous geometry that places atoms and lattice parameters close enough to the target.

### 5.3 SymCIF-v4 Table 3-Style Benchmark

The SymCIF-v4 Table 3 benchmark was a major formalization step. It built structured data for MP-20 and MPTS-52, trained geometry and WA-policy components, and ran a fair-budget WA20 x geom1 full evaluation.

The result was negative relative to published CrystaLLM-a:

| dataset | method | input | match@1 | match@5 | match@20 | RMSE@20 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MP-20 | CrystaLLM-a published | composition only | 55.85% | n/a | 75.14% | 0.0395 |
| MP-20 | SymCIF-v4 WA20xgeom1 | composition + GT-SG | 45.62% | 59.73% | 68.92% | 0.0848 |
| MPTS-52 | CrystaLLM-a published | composition only | 17.47% | n/a | 32.98% | 0.1197 |
| MPTS-52 | SymCIF-v4 WA20xgeom1 | composition + GT-SG | 14.46% | 22.55% | 30.43% | 0.1571 |

This result was important because it prevented overclaiming. Even with oracle GT-SG, early SymCIF-v4 was below CrystaLLM-a on match@20 and much worse on RMSE. The bottleneck was not only geometry; WA search recall itself was low. For MP-20, GT-WA@20 was only 18.40%; for MPTS-52, only 8.12%.

### 5.4 SymCIF-v4 MP-20 K<=5 Development Breakthrough

The MP-20 K<=5 development round found a much more promising mechanism. The baseline validation result was:

| metric | @1 | @5 |
| --- | ---: | ---: |
| match | 44.12% | 63.42% |
| RMSE | 0.0840 | 0.0828 |
| WA_hit | 38.63% | 65.11% |
| skeleton_hit | 47.32% | 70.39% |

A root-cause audit showed that raw top100 WA_hit was 87.85%, but selected top5 WA_hit was only 65.11%. That means the candidate pool already contained many correct assignments, but the ranking placed them too low. The project then tried a simple train-only hybrid-prior selector. Without using StructureMatcher labels or val/test match labels, this one-fix branch improved:

| metric | baseline @1 | one-fix @1 | baseline @5 | one-fix @5 |
| --- | ---: | ---: | ---: | ---: |
| match | 44.12% | 59.06% | 63.42% | 71.58% |
| RMSE | 0.0840 | 0.0591 | 0.0828 | 0.0646 |
| WA_hit | 38.63% | 56.91% | 65.11% | 79.52% |
| skeleton_hit | 47.32% | 69.78% | 70.39% | 82.51% |

This is one of the clearest causal-looking diagnostic results in the project: changing candidate ranking improves WA_hit and match simultaneously. It indicates that WA candidate selection is a first-order bottleneck.

### 5.5 SymCIF-v4 MP-20 Hybrid-Prior Test Result

The strongest paper-facing result is `symcif_v4_mp20_test_hybrid_prior_k5`. It used a frozen train-only hybrid-prior selector on MP-20 structured test samples. No test-label tuning was done.

| scope | match@1 | match@5 | RMSE@1 | RMSE@5 |
| --- | ---: | ---: | ---: | ---: |
| structured MP-20 test, 8893 samples | 60.90% | 73.68% | 0.0573 | 0.0615 |
| original-test adjusted, 9046 denominator | 59.87% | 72.43% | 0.0573 | 0.0615 |

The candidate coverage also improved:

| selector | WA_hit@1 | WA_hit@5 | skeleton_hit@1 | skeleton_hit@5 |
| --- | ---: | ---: | ---: | ---: |
| previous/current order | 49.80% | 71.27% | n/a | n/a |
| frozen hybrid-prior | 59.02% | 81.66% | 71.36% | 84.45% |

The honest interpretation is:

- This is a strong MP-20 top-1/top-5 result under composition + oracle GT-SG.
- It supports a manuscript claim about symmetry-conditioned structured generation and train-only WA reranking.
- It does not prove superiority over CrystaLLM-a under the same input setting.
- RMSE remains worse than CrystaLLM-a's published RMSE@1.
- The gap between WA_hit@5 = 81.66% and match@5 = 73.68% shows that geometry/rendering quality still loses correct symbolic candidates.

### 5.6 opentry_3: CrystalFormer/WyFormer-Style MPTS-52 Route

`opentry_3` tried to push a formula + GT-SG -> Wyckoff skeleton / assignment -> CIF route on MPTS-52. It made real progress in symbolic recall. For example, canonical W/A recall improved dramatically relative to older policy-search baselines, and a count-aware model-only skeleton generator reached skeleton@50 = 87.50% on val128.

However, validation CIF rendering was still not good enough for a full-test freeze. The best validation match numbers around the final summary were approximately match@1 = 32.03%, match@5 = 40.62%, match@20 = 45.31%, but match@5 and match@20 were still below the desired +5pp freeze target. Rows>=7 remained especially weak. The project correctly did not run full test from these configs.

The lesson: symbolic W/A stages improved, but W/A-to-match conversion was poor, especially for rows>=7. Geometry/free-parameter transfer was the main bottleneck.

### 5.7 opentry_4: Hard Negatives, Energy Models, Pairfield Proposals

`opentry_4` imported a frozen full-test result and trained hard-negative energy/proposal machinery. Its E718 full-test metrics on MPTS-52 were:

| metric | value | delta vs GT-SG CrystaLLM |
| --- | ---: | ---: |
| match@1 | 27.68% | +1.04 pp |
| match@5 | 35.85% | -0.73 pp |
| match@20 | 39.68% | -5.01 pp |

This did not satisfy the success target because match@20 degraded too much and two metrics did not improve. The energy model had useful AUC, and pairfield proposals showed that new rows>=7 positives exist, but direct ordering and anchor-safe insertion did not create a safe official improvement.

The lesson: energy/rejection scoring and proposal generators can identify useful regions, but ordinary reranking or conservative insertion was not enough. The system remained stuck in coupled lattice/free-parameter/site-mapping errors.

### 5.8 opentry_5 and opentry_6: MiniCFJoint and Geometry Refiner Attempts

`opentry_5` explored MiniCFJoint-style full-CIF models with grouped folds and no test leakage. Some fold-level aggregate metrics became strong, but rows>=7 remained the main blocker. The best non-oracle direction balanced local-neighbor geometry, weak target-relative repair, and complex symbolic weighting, but it was not eligible for a validation gate because rows>=7 match remained zero or inconsistent across folds.

`opentry_6` continued geometry/refiner-style stages. It explicitly concluded that the project should not return to ordinary selector/ranker, compatibility score, energy rejection, anchor-safe insertion, oracle selection, or match/RMSE/validity/logprob filtering as primary routes. The recommended direction was a CrystalFormer-style continuous geometry/refiner with broader canonical rows>=7 coverage or rebuilt full-train canonical geometry data.

The lesson: rows>=7 failures are not a cosmetic subset; they are the stress test for whether geometry and symbolic assignment truly generalize.

### 5.9 opentry_7: Official CrystaLLM Table 3 Test Reproduction

`opentry_7` established important official baselines under the unified evaluator.

For MP-20:

| system | match@1 | match@5 | match@20 | rows>=7 match@20 |
| --- | ---: | ---: | ---: | ---: |
| reproduced CrystaLLM-a composition-only | 62.17 | 74.98 | 81.84 | 76.36 |
| CrystaLLM-a GT-SG | 71.67 | 83.08 | 87.81 | 82.61 |
| best strategy/fusion line | 50.44 | 65.06 | 69.11 | 57.33 |
| best pure model line | 60.58 | 70.67 | 77.96 | 67.88 |

For MPTS-52:

| system | match@1 | match@5 | match@20 | rows>=7 match@20 |
| --- | ---: | ---: | ---: | ---: |
| reproduced CrystaLLM-a composition-only | 18.86 | 28.08 | 35.17 | 32.76 |
| CrystaLLM-a GT-SG | 25.23 | 36.46 | 43.96 | 41.04 |
| best strategy/fusion line | 17.63 | 27.99 | 32.99 | 29.61 |
| best pure model line | 17.18 | 24.35 | 31.52 | 27.88 |

The key lesson is that CrystaLLM-a GT-SG is a very strong anchor. Beating published composition-only CrystaLLM-a is much easier than beating GT-SG CrystaLLM-a.

### 5.10 opentry_8: Coverage Repair, Not a New Method Gain

`opentry_8` built a coverage-repaired frozen strategy that preserved the CrystaLLM-a GT-SG candidate order and used stablekey only to fill absent GT-SG slots. It reached exactly the GT-SG anchor on MP-20 and changed MPTS-52 match@20 from 43.96 to 43.97 by repairing one missing sample.

The report explicitly concluded:

- MP-20 strategy/fusion is identical to the GT-SG anchor.
- MPTS-52 improvement is coverage repair, not a method breakthrough.
- The pure structural line was not full-tested because no checkpoint passed validation-generation gates.

The lesson: using a stronger anchor or repairing missing samples can improve apparent results, but it is not scientific evidence of a new model or selector.

### 5.11 opentry_9: Feasibility Audit and Missing Validation Anchor

`opentry_9` audited whether a meaningful strategy/fusion system could be trained from available historical artifacts. It found that the required CrystaLLM-a GT-SG validation K20 candidate bank did not exist in earlier directories. Without that validation bank, oracle union and reranker validation could not be computed legally. Therefore, no selector/ranker was trained, no frozen strategy was produced, and no new official test was run.

`opentry_9` also reused SymCIF diagnostics:

- MP-20 SymCIF K<=5 baseline match@1/@5 = 44.12/63.42.
- one-fix prior selector match@5 = 71.58.
- GT-WA geometry match@1/@5 = 77.16/82.94.
- WA selection and geometry are both bottlenecks.

The lesson: the missing validation CrystaLLM candidate bank is not a reason to stop; it is a mandatory artifact to build.

### 5.12 opentry_10: Active Long-Run Completion Attempt

`opentry_10` is the active attempt to do the hard version correctly. It copied the mistakenly written `opentry_9` artifacts into `opentry_10`, but quarantined them. The copied `final_report.md` is explicitly invalid and must not be used as proof of completion.

`opentry_10` completed resource and provenance audit, rebuilt validation GT-SG CIF caches using symprec=0.1, prepared prompts, and started full K100 validation candidate generation.

The symprec=0.1 cache fixed an important prompt issue: direct CSV extraction can expose P1 symmetry, while the new cache conventionalizes validation CIFs for GT-SG prompt construction.

Validation cache status:

| dataset | records | non-P1 | P1 | conversion errors |
| --- | ---: | ---: | ---: | ---: |
| MP-20 validation | 9047 | 8900 | 147 | 0 |
| MPTS-52 validation | 5000 | 4980 | 20 | 0 |

Current `opentry_10` status:

- MP-20 validation K100 generation is active and past halfway.
- The latest supervisor state shows 83 completed MP-20 shards, one running shard, and 58 pending shards out of 142 total.
- MPTS-52 remains paused at 16 completed shards out of 79.
- No frozen systems exist.
- No real `opentry_10` final report is valid yet.

## 6. Main Scientific Lessons

### 6.1 Representation Helps Diagnosis More Than It Automatically Solves Generation

SymCIF's biggest success is not simply that a new text representation has lower loss. In fact, raw CIF byte-level models often had better token validation loss. The real value of SymCIF is decomposition. It lets the project ask: did the model fail because it chose the wrong space-group-consistent skeleton, the wrong Wyckoff assignment, bad free parameters, invalid CIF rendering, or poor geometry?

This diagnostic visibility is scientifically valuable. It prevents false explanations and allows targeted fixes.

### 6.2 Wyckoff Assignment Candidate Selection Is a Major Bottleneck

Multiple reports converge on this point. In MP-20 K<=5 validation, the raw top100 pool had WA_hit = 87.85%, while the selected top5 had only 65.11%. A train-only hybrid-prior selector raised selected WA_hit@5 to 79.52% and match@5 to 71.58%. On the test structured subset, hybrid-prior raised WA_hit@5 to 81.66% and match@5 to 73.68%.

This means candidate generation is not the only issue. Candidate ranking and selection matter strongly.

### 6.3 Geometry Is the Second Hard Bottleneck

Even with GT-WA, MP-20 validation geometry reached only match@1/match@5 = 77.16/82.94. Rows>=7 proxy match@5 was only 69.46. Therefore, even perfect WA selection would not automatically solve all samples. Continuous geometry, lattice/free parameters, coordinate quality, and validity remain central.

This explains why many symbolic improvements failed to become final StructureMatcher improvements.

### 6.4 Rows>=7 Is the Stress Test

Rows>=7 samples are repeatedly worse across opentry_3 through opentry_10 diagnostics. Many routes improved aggregate match while rows>=7 stayed weak or zero in small folds. The active official success standard also requires rows>=7 not to degrade. Any future system that improves only simple samples is risky.

### 6.5 Test-Only Gains Are Dangerous

Historical experiments show many tempting test-only observations: exclusive rescue, oracle union, stablekey overlap, and coverage repair. These are useful diagnostics, but they cannot train or freeze a system. The correct use is to motivate validation candidate generation and out-of-fold ranker design.

### 6.6 CrystaLLM-a GT-SG Is the Real Hard Baseline

Published composition-only CrystaLLM-a is not the active opentry baseline. The GT-SG anchor is much stronger, especially on MP-20:

- MP-20 GT-SG match@20 = 87.81.
- MPTS-52 GT-SG match@20 = 43.96.

Any final system must beat this anchor without sacrificing match@20. A rerank-only strategy is attractive because the candidate set remains the same, so match@20 should remain unchanged if only K20 ordering is changed.

## 7. Current Best Strategy

The current best strategy is not to train a new pure model first. It is to complete full CrystaLLM GT-SG validation K100 candidate generation and use that bank to test reranking/fusion safely.

The logic is:

1. CrystaLLM GT-SG already has strong candidates.
2. There is known post-hoc K20 internal rerank space: many top1 failures are top20 hits.
3. Rerank-only can improve match@1/match@5 while preserving match@20.
4. Expanded K100 candidates may reveal additional oracle headroom.
5. Validation OOF training can estimate whether this headroom is learnable without test leakage.

The immediate go/no-go gates are:

- Complete MP-20 validation K100.
- Assemble full candidate JSONL and metrics.
- Measure anchor K20, expanded K100, and source-union oracle coverage on validation.
- Train rerank-only OOF models.
- If rerank-only gives stable validation gains, freeze it.
- If oracle union is high but simple rerankers cannot select winners, train a stronger candidate correctness scorer.
- If MP-20 has no learnable headroom, shift major effort to MPTS-52 or a new candidate source rather than endlessly tuning MP-20.

## 8. Active opentry_10 Requirements

The `opentry_10` prompt requires:

- a resumable controller at `scripts/run_opentry10.py --resume`;
- state tracking in `state/controller_state.json`, `state/jobs.jsonl`, `state/artifact_registry.json`, and `state/frozen_registry.json`;
- full MP-20 and MPTS-52 validation K100 CrystaLLM GT-SG candidates;
- K1/K5/K20/K50/K100 validation metrics;
- rerank-only experiments regardless of oracle-union gate;
- candidate-feature extraction including rank, temperature, top-k, logprob if available, validity diagnostics, WA/skeleton keys, train frequencies, geometry proxies, duplicates, and consensus;
- 5-fold group cross-validation for any ranker or selector;
- conservative fusion only after validation evidence;
- official full test only after freezing.

The prompt forbids stopping because artifacts are missing. Missing candidate banks, checkpoints, validation labels, or scripts are tasks to build or recover, not reasons to end.

## 9. What Is Currently Complete

Completed or substantially complete:

- Official CSV counts and resource audit.
- Baseline provenance for historical CrystaLLM-a GT-SG generation protocol.
- Symprec=0.1 validation GT CIF cache.
- MP-20 and MPTS-52 validation prompt preparation.
- MP-20 K100 shard plan.
- MPTS-52 K100 shard plan.
- MP-20 K100 generation more than halfway complete.
- MPTS-52 K100 generation partially complete but intentionally paused.
- Historical candidate source manifest and negative/diagnostic reports copied and quarantined.

Not complete:

- Full MP-20 validation K100 candidate bank.
- Full MPTS-52 validation K100 candidate bank.
- Full validation assembly/evaluation.
- Validation candidate JSONL exports.
- Rerank OOF model search.
- Selector/fusion OOF search.
- Frozen strategy or frozen pure model.
- Official full test for any new `opentry_10` system.
- Valid `opentry_10` final report.

## 10. What Conclusions Are Already Safe

The following conclusions are safe:

1. Current SymCIF evidence supports a symmetry-conditioned MP-20 structured-generation claim, not a same-input universal SOTA claim.
2. WA candidate selection is a major bottleneck and hybrid-prior reranking is a promising mechanism.
3. Geometry remains a hard bottleneck even under GT-WA.
4. rows>=7 is the main stress subset and cannot be ignored.
5. Earlier opentry fusion gains were mostly coverage repair or anchor replacement, not new scientific breakthroughs.
6. A full CrystaLLM GT-SG validation bank is mandatory for a legal strategy/fusion decision.
7. The current `opentry_10` route is worthwhile as a strong discriminative experiment, but it does not guarantee success.

The following conclusions are not safe:

1. `opentry_10` has succeeded.
2. A frozen strategy exists.
3. SymCIF has beaten CrystaLLM-a GT-SG under official full-test K20.
4. Test-only oracle union proves a deployable selector.
5. Pure structural generation is ready for official test.

## 11. Recommended Path Forward

The practical path is:

1. Finish MP-20 validation K100 generation.
2. Assemble and evaluate MP-20 validation at K1/K5/K20/K50/K100.
3. Compute rerank-only OOF baselines on anchor K20.
4. Compute K100 oracle headroom and candidate-feature distributions.
5. If MP-20 OOF rerank gives stable match@1 or match@5 gains while preserving match@20, freeze a rerank-only system.
6. If MP-20 has no learnable headroom, avoid wasting cycles and move to MPTS-52 or candidate-source expansion.
7. After MP-20 gate, resume MPTS-52 validation K100.
8. Only after validation gates are clean should official full test be run.

For pure structural work, the best next direction is not another byte-level CIF model. It should combine:

- exact-cover or high-recall WA candidate generation;
- inference-feasible WA/ranker scoring;
- geometry-quality scoring;
- rows>=7-focused geometry/free-parameter modeling;
- full validation K20/K100 evaluation before test.

## 12. Directory Map

Important directories:

- `Paper/`: manuscript drafts, claim boundaries, evidence tables, figure planning.
- `data_exp/`: controlled from-zero raw CIF vs SymCIF-v4 language-model training.
- `symcif_experiment/`: main SymCIF representation, renderer, WA, geometry, and MP-20/MPTS-52 structured experiments.
- `opentry/`: earlier MPTS-52 W/A decoder and hybrid tests.
- `opentry_2/`: early policy/ranker and MPTS-52 candidate experiments.
- `opentry_3/`: CrystalFormer/WyFormer-style symbolic W/A and geometry route.
- `opentry_4/`: hard-negative energy and pairfield proposal experiments.
- `opentry_5/`: MiniCFJoint and grouped-fold non-oracle full-CIF attempts.
- `opentry_6/`: geometry/refiner continuation.
- `opentry_7/`: official CrystaLLM Table 3 reproduction and GT-SG baselines.
- `opentry_8/`: coverage-repaired strategy/fusion audit.
- `opentry_9/`: feasibility audit and partial validation-anchor start.
- `opentry_10/`: active long-run official-over-baseline attempt.

## 13. Key Source Documents Consulted

High-level documents read or used include:

- `Paper/words/symcif_paper_roadmap/evidence_table.md`
- `Paper/words/symcif_paper_roadmap/symcif_chinese_paper_battle_map.md`
- `Paper/manuscript/zh/00_claim_boundary.md`
- `data_exp/final_summary.md`
- `symcif_experiment/reports/training_final_analysis.md`
- `symcif_experiment/reports/symcif_v4_table3_mp20_mpts52_comparison/summary.md`
- `symcif_experiment/reports/symcif_v4_mp20_k5_dev/val_baseline_summary.md`
- `symcif_experiment/reports/symcif_v4_mp20_k5_dev/one_fix_experiment_summary.md`
- `symcif_experiment/reports/symcif_v4_mp20_k5_dev/final_root_cause_summary.md`
- `symcif_experiment/reports/symcif_v4_mp20_test_hybrid_prior_k5/mp20_test_hybrid_prior_k5_report.md`
- `opentry_3/final_summary.md`
- `opentry_4/reports/opentry_4_final_summary.md`
- `opentry_5/reports/opentry_5_final_summary.md`
- `opentry_6/opentry_6_final_summary.md`
- `opentry_7/opentry_7_final_report.md`
- `opentry_8/final_report.md`
- `opentry_9/final_report.md`
- `opentry_10/prompt.md`
- `opentry_10/reports/opentry10_progress_status.md`
- `opentry_10/reports/strategy_oracle_union_audit.md`
- `opentry_10/reports/pure_structural_val_report.md`
- `opentry_10/reports/pure_wa_decoder_val_report.md`
- `opentry_10/reports/pure_gt_wa_geometry_report.md`
- `opentry_10/state/frozen_registry.json`
- `opentry_10/state/mp20_until_done_status.json`

## 14. One-Page Mental Model

If a new AI has only one page of memory, remember this:

The project is about generating crystal structures as CIFs. Direct CIF language modeling is flexible but hard to control. SymCIF decomposes the problem into symmetry-aware symbolic choices and continuous geometry. This decomposition gives excellent diagnostics and a strong MP-20 structured-test top-1 result, but it has not yet beaten the much stronger CrystaLLM-a GT-SG official full-test anchor.

Historically, the project learned that:

- raw text-format changes do not automatically help;
- WA/skeleton recall can be improved;
- WA reranking is a major source of gains;
- geometry remains a hard bottleneck;
- rows>=7 samples expose failure;
- coverage repair is not method improvement;
- validation candidate banks are mandatory to avoid test leakage.

The active `opentry_10` goal is to build a legal, frozen system that beats CrystaLLM-a GT-SG. It is currently generating the missing full validation CrystaLLM GT-SG K100 bank, prioritizing MP-20. No frozen model exists yet. The next decisive experiment is leakage-free validation reranking/fusion after MP-20 K100 completes.

