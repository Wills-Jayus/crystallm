# Opentry-3 Wyckoff Model Strategy

## Scope

本轮使用 CrystalFormer/WyFormer 风格中间表示，而不是继续扩大 CIF next-token SFT、K50/RF ranker 或固定候选融合。

主线：

```text
formula + GT-SG -> Wyckoff skeleton -> element/W-A assignment -> rendered CIF
```

## Representation

- Canonical source: `structured_symcif_v4_mpts52/{train,val}.jsonl`
- Label fields: `canonical_skeleton_key`, `canonical_wa_key`, `wa_table`
- Canonical row order: `(multiplicity, letter, enumeration, site_symmetry, element)`
- Separate targets:
  - skeleton sequence: orbit-only rows
  - assignment sequence: element + orbit rows
  - direct W/A sequence: full canonical assignment
- `row_count` remains label/analysis only. It must not be used as a generation input or test-time sorting signal.

## Model Stages

### A. Skeleton Generator

Input:

```text
formula counts + GT-SG
```

Output:

```text
canonical orbit/skeleton sequence
```

Training and decoding:

- Cross-entropy sequence model or set/sequence hybrid.
- Composition-compatible mask through Wyckoff multiplicities.
- Complex weighting for rows>=7, atom_count>=12, many elements, rare SG.
- Report `skeleton@1/5/20/50`, unique skeletons, composition exact, rows>=7 metrics.

### B. W-A Assignment Generator

Input:

```text
formula counts + GT-SG + skeleton
```

Output:

```text
canonical element@orbit assignment sequence
```

Training and decoding:

- Composition-exact masking at each step.
- Hard negatives from legal but wrong element assignments.
- Report `W/A@1/5/20/50`, unique W/A, composition exact, rows>=7 metrics.

### C. Direct W-A Generator

Input:

```text
formula counts + GT-SG
```

Output:

```text
canonical W/A sequence
```

Use this as an independent generator after A/B works. It should improve recall/diversity rather than replace the gated A/B path too early.

### D. CrystaLLM Adapter

Only after A/B/C val W/A recall improves clearly:

- Add Wyckoff adapter/head on top of a CrystaLLM-style backbone.
- Do not resume plain CIF next-token SFT as the main training objective.

## Gates

Before CIF rendering:

- `W/A@50` must clearly exceed old policy-search baselines around 12-13%.
- `skeleton@50` must be healthy on full val and rows>=7 subset.
- composition exact rate must be high.
- unique W/A per sample must show real diversity, not duplicated sequences.

Before full test:

- Config frozen from val only.
- One full test per frozen config.
- Test GT is evaluator-only; no test labels for generation, sorting, filtering, tuning, or fallback.

## Current Validation CIF Finding

E09 symbolic gate locally passes on val128:

- predicted-skeleton `W/A@50=42.19%`
- predicted-skeleton `W/A@100=45.31%`
- rows>=7 `W/A@50=23.33%`

However, CIF rendering exposed a second bottleneck:

- collision-aware deterministic geometry gives perfect CIF closure but weak match (`match@50=21.09%`, rows>=7 `0%`).
- e08 row-conditioned geometry improves match (`match@50=30.47%`, rows>=7 `8.33%`) but collapses W/A diversity after composition-exact filtering (`unique W/A@50=3.23`).

Next decoder/renderer work should therefore:

- keep fixed-orbit duplicate masks and composition-exact filtering;
- combine e08 train-only row-conditioned geometry with collision-aware deterministic fallback;
- preserve W/A diversity instead of filling K with geometry variants of a few W/A keys;
- continue rows>=7-specific skeleton/W-A work before any full test.

## Count-Aware Skeleton Result

E35 changed the A-stage model rather than relying on train-prior skeleton augmentation:

- input now includes formula fractions plus absolute element counts;
- `row_count` is still not an input feature;
- train uses all MPTS-52 train records and complex/rows>=7 weighting;
- batched beam decoding is required because the full-train orbit vocabulary has 1,166 orbit tokens.

Validation skeleton result on val128:

| subset | skeleton@1 | skeleton@5 | skeleton@20 | skeleton@50 | unique skeleton@50 |
|---|---:|---:|---:|---:|---:|
| full | 45.31% | 70.31% | 81.25% | 87.50% | 45.23 |
| rows>=7 | 50.00% | 68.33% | 81.67% | 85.00% | 42.65 |
| atoms>=12 | 42.24% | 67.24% | 79.31% | 86.21% | 44.74 |

This means the E27 train-only skeleton prior can largely be replaced by a learned model-side A-stage generator.

Canonical W/A after budgeted assignment DP:

| subset | W/A@1 | W/A@5 | W/A@20 | W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|
| full | 45.31% | 66.41% | 73.44% | 73.44% | 10.98 |
| rows>=7 | 48.33% | 66.67% | 66.67% | 66.67% | 5.52 |
| atoms>=12 | 43.10% | 63.79% | 70.69% | 70.69% | 10.19 |

Compared with E29, this is better at early W/A ranks and rows>=7 recall, but slightly lower on full W/A@50. The assignment DP remains a bottleneck because many candidates are duplicate fixed-orbit or non-useful permutations.

## Current Renderer Finding

E40 rendered E35/E37 candidates with e08 train-only geometry and W/A-diverse ordering:

| subset | match@1 | match@5 | match@20 | match@50 | W/A@50 |
|---|---:|---:|---:|---:|---:|
| full | 17.97% | 27.34% | 35.94% | 39.06% | 73.44% |
| rows>=7 | 6.67% | 10.00% | 11.67% | 11.67% | 66.67% |
| atoms>=12 | 13.79% | 21.55% | 29.31% | 32.76% | 70.69% |

E42 tried geometry-interleaved ordering, giving early W/A multiple e08 geometry ranks. It reduced full match@5/20/50 to `21.88/27.34/36.72`, so geometry interleaving is not a fix.

Current conclusion:

- A-stage model-side skeleton generation is now strong enough for this validation scale.
- B-stage assignment must reduce duplicate fixed-orbit waste and improve canonical W/A diversity without losing early rank.
- CIF conversion is still bottlenecked by geometry/free-parameter transfer, especially rows>=7.
- No full-test config is frozen.

## Assignment And Geometry Update

E43 moved duplicate fixed-orbit masking into assignment DP:

| item | before | after |
|---|---:|---:|
| conversion dropped duplicate fixed candidates | 2,512 | 0 |
| canonical output candidates | 1,406 | 1,612 |
| full unique W/A@50 | 10.98 | 12.59 |
| rows>=7 unique W/A@50 | 5.52 | 6.55 |

The W/A hit-rate gain is small, but the constraint is correct and should remain part of the B-stage decoder.

E48-E51 tested hybrid geometry ordering. It does not solve the conversion problem:

| run | full match@5 | full match@20 | full match@50 |
|---|---:|---:|---:|
| W/A-diverse | 27.34% | 35.94% | 39.06% |
| hybrid5x3 | 24.22% | 36.72% | 39.06% |
| hybrid3x2 | 23.44% | 35.94% | 39.06% |

Failure diagnostic:

- E47 full W/A@50 is 74.22%, but match@50 is 39.06%.
- E47 rows>=7 W/A@50 is 66.67%, but match@50 is only 11.67%.
- rows>=7 W/A-to-match conversion is about 17.5%.

Next work should therefore prioritize learned or better-structured geometry/free-parameter proposal for rows>=7. Simple e08 geometry-rank scheduling is not enough.

## Geometry Proposal Update After E53-E57

Additional validation diagnostics narrowed the geometry bottleneck:

| run | change | full match@50 | rows>=7 match@50 | conclusion |
|---|---|---:|---:|---|
| E54 | train-only W/A/skeleton-priority source retrieval | 37.50% | 11.67% | worse than e08 row-conditioned retrieval |
| E57 | train-only one-shot geometry net, full lattice/free params | 22.66% | 0.00% | not viable as deterministic geometry replacement |

Retained findings:

- E47/E51-style e08 row-conditioned retrieval remains the strongest current renderer despite poor rows>=7 conversion.
- Exact W/A/skeleton source priority is not enough; picking a nominally closer train analogue does not solve free-param/lattice transfer.
- A deterministic MSE-trained geometry head is too brittle. It lowers validation regression loss, but creates low-SG-validity CIFs and fails completely on rows>=7.
- Any learned geometry continuation should be multimodal or constraint-aware:
  - predict multiple coordinate/lattice modes instead of one mean;
  - include collision/symmetry-validity penalties;
  - batch inference in the renderer before more lattice-only diagnostics;
  - train/evaluate first on val CIF conversion, not regression loss alone.

E59-E62 also tried deterministic multimodal jitter around e08 retrieval:

| run | free-param/lattice proposal | full match@50 | rows>=7 match@50 |
|---|---|---:|---:|
| E47 | e08 W/A-diverse baseline | 39.06% | 11.67% |
| E60 | hybrid5x5 wrapped jitter+lattice scale | 38.28% | 11.67% |
| E62 | hybrid2x3 wrapped jitter+lattice scale | 37.50% | 11.67% |

This rules out small deterministic jitter as a useful geometry mode. The issue is not a missing local perturbation around e08; the rows>=7 failures need a stronger proposal family or a better representation of geometry modes.

Near-term priority:

1. Keep E35 count-aware skeleton + E45 fixed-mask B-stage as the current symbolic backbone.
2. Do not freeze full test: best validation CIF remains below the target line.
3. For geometry, either improve e08 with a learned train-only proposal distribution or train a collision/symmetry-aware geometry model; avoid more source-priority, simple single-head regression, or small jitter runs.

## Neural Assignment Scorer Update

E63 added a train-only B-stage assignment scorer:

- input: `formula + GT-SG + skeleton orbit + remaining composition`
- output: element distribution for exact-cover W/A assignment
- training: MPTS-52 train only, 146,596 assignment states
- validation: MPTS-52 val only, 7,821 assignment states
- best val state metrics: top1 `60.84%`, top5 `86.86%`, MRR `72.72%`

The scorer is now wired into `diagnose_wyckoff_assignment_dp.py` as an optional score term:

- `--assignment-model-ckpt`
- `--assignment-model-weight`
- `--prior-weight`
- default remains prior-only for reproducibility

Reduced-budget exact-DP results on val128 show real assignment signal:

| config | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | rows>=7 W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| prior-only s10p10 | 45.31% | 68.75% | 71.88% | 71.88% | 68.33% | 8.80 |
| neural p=1,m=0.5 s10p10 | 49.22% | 70.31% | 75.00% | 75.00% | 75.00% | 9.68 |
| neural p=1,m=1 s10p10 | 47.66% | 70.31% | 75.78% | 75.78% | 76.67% | 10.00 |
| E45 full-budget reference | 45.31% | 66.41% | 74.22% | 74.22% | 66.67% | 12.59 |

Strategy implication:

- The next symbolic step should be neural exact-DP engineering, not another low-LR CIF SFT or RF ranker.
- Do not render CIF yet from E63-E74: full W/A@50 is only slightly above E45 and unique W/A is still lower.
- A CIF gate should require a full-budget neural DP run with both W/A@50 and unique W/A clearly above E45, especially rows>=7.

## Neural-First Merge Update

E75-E87 showed that the E63 scorer is useful when used as a symbolic decoder component, but complete-sequence rescoring alone is not enough.

Rescore-only:

| run | full W/A@5 | full W/A@50 | rows>=7 W/A@50 | full match@50 |
|---|---:|---:|---:|---:|
| E80 rescored union | 59.38% | 78.12% | 76.67% | 41.41% |

This improves tail recall but damages early rank.

The better decoder is E83 neural-first priority merge:

- source 0: neural DP `s10p10`
- source 1: prior full DP `s20p20`
- no labels, no StructureMatcher, no test
- keeps neural early order, fills tail diversity from prior full candidates

Current best symbolic candidate:

| run | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | rows>=7 W/A@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E45 fixed-mask reference | 45.31% | 66.41% | 74.22% | 74.22% | 66.67% | 12.59 |
| E85 neural-first merge | 47.66% | 70.31% | 77.34% | 78.91% | 76.67% | 13.74 |

Val CIF conversion with e08 row-conditioned geometry:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 |
|---|---:|---:|---:|---:|---:|
| E47 fixed-mask reference | 17.97% | 27.34% | 35.94% | 39.06% | 11.67% |
| E87 neural-first merge | 21.09% | 28.12% | 35.94% | 40.62% | 16.67% |

Strategy implication:

- E83/E85 is now the current B-stage symbolic backbone.
- E87 is the current best validation CIF at K50 and rows>=7 K50, but not a full-test freeze.
- The remaining bottleneck is still geometry/free-param conversion: rows>=7 W/A@50 is `76.67%`, but rows>=7 match@50 is only `16.67%`.
- Next experiments should use E84/E86 as input and improve geometry conversion, not return to ordinary CIF next-token SFT or RF ranking.

## Hybrid Geometry Recheck

E88-E89 retested `hybrid_top_wa` on the stronger E84/E85 neural-first W/A set:

| run | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | full W/A@5 | full W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E87 W/A-diverse e08 | 28.12% | 35.94% | 40.62% | 16.67% | 72.66% | 78.91% |
| E89 hybrid5x3 e08 | 27.34% | 35.94% | 40.62% | 16.67% | 57.03% | 78.91% |

This rejects simple geometry scheduling again. `hybrid_top_wa` preserves the K50 ceiling but sacrifices early W/A diversity and does not improve rows>=7 conversion.

Current strategy:

1. Keep E83/E85 neural-first merge as the symbolic W/A backbone.
2. Keep E87 W/A-diverse e08 as the current best validation CIF result.
3. Do not freeze full test.
4. Next geometry work should change the proposal/scoring family itself, using train/val-only non-StructureMatcher signals such as generated-CIF validity, symmetry consistency, collision/short-distance checks, and multimodal free-parameter/lattice proposals.

## Generated-CIF Self-Score Update

E90-E93 added a GT-free geometry ordering layer over E86 generated CIFs.

Self-score inputs:

- generated CIF readability and parser success
- formula / atom count / composition exact
- detected SG consistency
- generated-CIF min interatomic distance
- generated-CIF volume per atom
- e08 geometry distance and original rank as weak tie-breaks

No StructureMatcher match/rms is used for sorting. E91 uses `mode=diverse`, which first selects the best geometry for each W/A key and then fills additional geometry variants.

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | full W/A@5 | full unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E87 raw W/A-diverse e08 | 21.09% | 28.12% | 35.94% | 40.62% | 16.67% | 72.66% | 12.80 |
| E91 self-score diverse | 28.91% | 32.81% | 38.28% | 40.62% | 16.67% | 71.88% | 12.80 |

E93 global self-score reaches the same full match values but reduces W/A diversity (`unique W/A@5=3.02` vs E91 `4.45`), so E91 is the preferred ordering.

Strategy implication:

- E91 becomes the current best validation CIF ordering for early ranks.
- The remaining problem is candidate/proposal ceiling, not just ranking: match@50 is unchanged at `40.62%`.
- Full test remains unfrozen.
- Next geometry work should generate more plausible geometry modes per high-quality W/A and use this self-score layer to order them before validation.

## Larger Geometry Pool Update

E94-E96 rendered a larger e08 proposal pool from the same E84 W/A predictions:

- E94: `top_k=100`, `geometry_ranks_per_wa=10`, e08 row-conditioned KNN
- E95: GT-free diverse self-score selects top50
- E96: val128 StructureMatcher evaluation

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E91 K50-pool self-score | 28.91% | 32.81% | 38.28% | 40.62% | 16.67% | 12.80 |
| E96 K100-pool self-score | 29.69% | 32.81% | 39.84% | 39.84% | 16.67% | 13.20 |

Interpretation:

- Larger geometry pools help early rank and atoms>=12 match@20.
- They can hurt K50 tail under the current self-score.
- Track E96 as the early-rank validation best, but keep E91 as the K50-ceiling validation best.
- The next scoring/proposal change should explicitly preserve tail diversity while improving early geometry plausibility.

## Prefix-Tail Selection Update

E97-E100 tested a deterministic GT-free merge of:

- prefix: E96 larger-pool self-score ordering
- tail: E91 K50-pool self-score ordering

The best merged policy is E100 with `prefix_k=20`:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E91 K50 self-score | 28.91% | 32.81% | 38.28% | 40.62% | 16.67% | 12.80 |
| E96 K100 self-score | 29.69% | 32.81% | 39.84% | 39.84% | 16.67% | 13.20 |
| E100 prefix20-tail | 29.69% | 32.81% | 39.84% | 40.62% | 16.67% | 12.80 |

Strategy implication:

- E100 is now the best validation CIF ordering: it preserves E96 early ranks and E91 K50 ceiling.
- This is still below the full-test gate and should not be frozen.
- Further gains require new geometry proposal capacity, especially for rows>=7, not more deterministic selection over the same proposal family.

## Row-Prototype Geometry Update

E101-E106 added a train-only row-wise geometry prototype source:

- lattice: e08 row-conditioned source lattice
- free params: per-row train prototype selected by `(SG, orbit_id, element, free_symbols)` tiers
- fallback: train-only free-param quantile
- no StructureMatcher labels or test data in rendering/scoring

Standalone row-prototype is not viable:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E104 row-prototype standalone | 17.97% | 24.22% | 28.12% | 30.47% | 1.67% | 13.59 |

The row-wise free-parameter combination breaks global consistency, especially for rows>=7.

However, row-prototype tail-fill helps when used conservatively:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E100 prefix20-tail | 29.69% | 32.81% | 39.84% | 40.62% | 16.67% | 12.80 |
| E106 E100 + row-prototype tail-fill | 29.69% | 32.81% | 40.62% | 41.41% | 18.33% | 12.90 |

Strategy implication:

- E106 is the current best validation CIF result.
- New proposal capacity can help K20/K50, but it must be constrained; independent row mixing is too noisy.
- Next geometry work should use clustered or source-consistent row prototypes, not fully independent row-level composition.
- Full test remains unfrozen.

## Source-Consistent Row Alignment Update

E108-E113 implemented `row_aligned_knn`:

- source selection: train-only e08 row-conditioned KNN
- lattice: source-consistent e08 source lattice
- free params: align target rows to rows inside the same source by orbit/element/free-symbol/multiplicity/site-symmetry cost
- no StructureMatcher label or test data in proposal or ordering

This fixes the likely e08 row-index copy mismatch while avoiding fully independent row-prototype mixing.

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E106 previous best | 29.69% | 32.81% | 40.62% | 41.41% | 18.33% | 12.90 |
| E111 row_aligned_knn | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 13.20 |

Strategy implication:

- E111 is now the current best validation CIF match result.
- It nearly reaches the MPTS-52 match@1 +5pp target (`31.25%` vs `31.64%`) but does not pass it.
- The full-test gate remains closed: match@5 and rows>=7 are still far too low.
- Next work should improve source selection and lattice proposal for `row_aligned_knn`, because this is the first geometry change that substantially improves early ranks without relying on labels.

## Row-Aligned Source Selection And Self-Score Update

E114-E121 tested two follow-ups to E111:

1. `row_aligned_priority`
   - Keeps source-consistent row alignment.
   - Reorders train-only source candidates by target/source row-alignment cost before rendering.
   - Improves RMSE but hurts match versus E111.

2. global self-score over the E109 `row_aligned_knn` pool
   - Uses only generated-CIF internal signals: readability, formula/composition exactness, SG consistency, min distance, volume per atom, geometry distance, and rank tie-breaks.
   - Does not use StructureMatcher labels or test data.

Validation results:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E111 row_aligned_knn diverse | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 13.20 |
| E117 row_aligned_priority | 28.91% | 33.59% | 39.06% | 40.62% | 16.67% | - |
| E119 row_aligned_knn global | 31.25% | 35.16% | 41.41% | 43.75% | 18.33% | 10.62 |
| E121 global prefix20 + diverse tail | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 13.01 |

Current interpretation:

- E119 is the best validation K50 match ceiling: full `31.25/35.16/41.41/43.75`.
- E111 remains the best diversity-preserving row-aligned result: full `31.25/35.16/41.41/42.19`, unique W/A@50 `13.20`.
- E121 is a useful compromise with the same early ranks as E111/E119 and better RMSE@50 than E111, but it loses the E119 K50 gain.
- None passes the full-test freeze gate:
  - match@1 `31.25%` is still below the `31.64%` target;
  - match@5 `35.16%` is far below `41.58%`;
  - match@20 `41.41%` is far below `49.69%`.

Strategic implication:

- Source-consistent row alignment is still the correct geometry family.
- Simple source-priority reranking is not enough; it reduces match despite better RMSE.
- The next useful change should add new train-only lattice/free-parameter proposal capacity while preserving source consistency, especially for rows>=7.
- Avoid returning to RF ranker, CrystaLLM predictions as primary candidates, low-LR CIF SFT, or fixed-candidate fusion as the main path.

## Source-Cluster Lattice Quantile Rejection

E122-E125 tested a train-only `source_cluster_quantile` lattice mode:

- source pool: row-aligned train-only e08 sources;
- lattice: quantiles over volume-scaled source lattices;
- free params: unchanged row-aligned source transfer;
- sorting: GT-free diverse self-score;
- evaluation: validation StructureMatcher only.

Result:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | full W/A@50 |
|---|---:|---:|---:|---:|---:|---:|
| E111 row_aligned source lattice | 31.25% | 35.16% | 41.41% | 42.19% | 18.33% | 78.91% |
| E125 source_cluster_quantile lattice | 25.78% | 31.25% | 35.16% | 35.94% | 10.00% | 78.91% |

Interpretation:

- W/A recall is unchanged, so this is not a symbolic failure.
- Match drops sharply, especially rows>=7.
- Lattice quantile aggregation breaks source-specific compatibility between lattice and free parameters.
- Do not expand this mode to K100 or full test.

Updated geometry principle:

- Preserve source consistency more strictly, not less.
- Future geometry proposals should sample coherent source-conditioned modes, e.g. source-pair or source-cluster selection where lattice and free params remain compatible, or a train-only generative head that jointly predicts lattice/free params.
- Avoid independent lattice quantiles, small jitter, source-priority sorting alone, and one-shot MSE geometry heads as primary routes.

## Source-Health Aware Row Alignment Update

E126-E131 added `row_aligned_quality`, a conservative source-coherence selector:

- starts from train-only row-conditioned source pool;
- keeps lattice and free params from one coherent source;
- adds a small source-quality penalty from train structured fields:
  - `row_expansion_all_ok`;
  - `free_param_reextract_all_success`;
  - missing free params for declared free symbols.

This uses no StructureMatcher labels and no test data.

Validation results:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E119 row_aligned_knn global | 31.25% | 35.16% | 41.41% | 43.75% | 15.00% | 18.33% | 10.62 |
| E129 row_aligned_quality diverse | 31.25% | 35.16% | 40.62% | 44.53% | 15.00% | 18.33% | 12.80 |
| E131 row_aligned_quality global | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 12.80 |

Current best:

- E131 is the best validation match ordering:
  - full match@1/5/20/50 = `31.25 / 35.94 / 41.41 / 44.53`;
  - rows>=7 match@1/5/20/50 = `13.33 / 16.67 / 18.33 / 18.33`.
- It improves E119 full K5/K50 by `+0.78 pp` and rows>=7 K5 by `+1.67 pp`.
- It still does not pass the full-test gate:
  - match@1 remains below `31.64%`;
  - match@5 remains below `41.58%`;
  - match@20 remains below `49.69%`.

Strategy implication:

- Source health is a useful GT-free signal.
- The main geometry path should keep source consistency and add stronger source-conditioned multimodal proposals.
- Avoid spending more time on independent lattice aggregation or pure scorer reshuffling; the rows>=7 K50 ceiling is still stuck at `18.33%`.

## Wider Source-Health Geometry Pool Update

E132-E136 widened `row_aligned_quality` rendering:

- same E84 neural-first W/A candidates;
- K100/g10 coherent source-conditioned geometry modes;
- GT-free self-score top50;
- validation StructureMatcher only.

Results:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E131 K50/global | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 12.80 |
| E134 K100/global | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 10.66 |
| E136 K100/diverse | 31.25% | 34.38% | 39.84% | 43.75% | 13.33% | 18.33% | 13.02 |

Interpretation:

- Wider coherent geometry modes improve tail ceiling under global scoring: E134 is the best K20/K50 validation ceiling.
- E131 remains the best early K5 ordering.
- Diversity-preserving scoring on the wider pool does not improve match.
- rows>=7 K50 remains stuck at `18.33%`, so simple pool widening is not enough.

Current tracking:

- Report E131 as best K5/early validation result.
- Report E134 as best K20/K50 validation ceiling.
- Do not freeze full test; none meets two +5pp metrics.

Next direction:

- Increase rows>=7 source-conditioned geometry quality, not just pool width.
- Consider source-conditioned multimodal free-param/lattice generation or symbolic candidates that encode geometry compatibility, while keeping train/val/test separation.

## Train-Only Volume Prior Self-Score Update

E137-E140 tested a train-only volume-per-atom prior inside the generated-CIF self-score:

- source rendered pool: E132 K100/g10 `row_aligned_quality`;
- sorting: GT-free global self-score plus train-only VPA prior;
- train prior buckets: `(space_group, atom_bucket)`, `(crystal_system, atom_bucket)`, `atom_bucket`, and global;
- validation StructureMatcher only for evaluation.

The prior was motivated by rows>=7 diagnostics:

- W/A-hit and match rows had higher VPA than W/A-hit/no-match rows;
- no-W/A rows had much worse source health and geometry distance;
- therefore VPA could plausibly help ordering, but not symbolic recall.

Results:

| run | train VPA weight | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E131 K50/global | 0.0 | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 12.80 |
| E134 K100/global | 0.0 | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 10.66 |
| E138 VPA prior | 2.0 | 30.47% | 35.94% | 42.19% | 45.31% | 15.00% | 18.33% | 10.54 |
| E140 VPA prior | 0.5 | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 10.66 |

Interpretation:

- The train-only VPA prior is leakage-safe, but it does not improve the validation gate.
- Weight 2.0 trades away match@1 to recover E131-like K5 while keeping E134-like K20/K50.
- Weight 0.5 is effectively E134 with slightly better RMSE@5.
- rows>=7 match@50 remains `18.33%`; the bottleneck is not simple volume ordering.

Current tracking remains unchanged:

- E131 is the best early/K5 validation result.
- E134 is the best K20/K50 validation ceiling.
- Do not freeze full test.
- Next work should add new source-conditioned geometry proposal capacity for rows>=7, not more scalar post-hoc self-score tweaks.

## Hybrid Geometry Allocation Update

E141-E152 tested whether the renderer should allocate more coherent source-conditioned geometry modes to high-confidence W/A candidates instead of spreading rank budget across many W/A candidates first.

Variants:

- `hybrid10x10`: top 10 W/A each receive up to 10 source geometry ranks.
- `hybrid5x5`: top 5 W/A each receive up to 5 source geometry ranks.
- W/A candidates remain E84 neural-first.
- Geometry source remains `row_aligned_quality`.
- Sorting remains GT-free self-score.
- Evaluation is validation-only; no test data or StructureMatcher labels are used for sorting.

Results:

| run | plan | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 | unique W/A@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E131 | K50 baseline | global | 31.25% | 35.94% | 41.41% | 44.53% | 16.67% | 18.33% | 12.80 |
| E134 | K100 baseline | global | 31.25% | 35.16% | 42.19% | 45.31% | 15.00% | 18.33% | 10.66 |
| E146 | hybrid10x10 | global | 28.12% | 32.81% | 39.06% | 42.19% | 15.00% | 18.33% | 7.45 |
| E147 | hybrid10x10 | diverse | 28.12% | 32.03% | 38.28% | 42.19% | 13.33% | 18.33% | 8.66 |
| E151 | hybrid5x5 | global | 32.03% | 35.94% | 42.19% | 45.31% | 15.00% | 18.33% | 10.41 |
| E152 | hybrid5x5 | diverse | 32.03% | 35.16% | 40.62% | 43.75% | 13.33% | 18.33% | 13.02 |

Interpretation:

- `hybrid10x10` is too aggressive; it lowers W/A diversity and match.
- `hybrid5x5` with global self-score is the current best-combined validation ordering:
  - full match@1/5/20/50 = `32.03 / 35.94 / 42.19 / 45.31`;
  - it combines E131 K5 with E134 K20/K50 and improves match@1.
- However, rows>=7 match@50 remains `18.33%`, so the core complex-subset bottleneck is unchanged.
- Full-test gate remains closed because only match@1 clears the target; match@5 and match@20 remain far below target.

Strategy implication:

- A small amount of front-loaded geometry helps early-rank ordering.
- Larger geometry allocation is harmful because it displaces useful W/A diversity.
- Further gains require new geometry proposal capacity for rows>=7, not just rank-budget scheduling.

## Chemical Analogue Source Proposal Update

E153-E162 added `row_aligned_chem_quality`, a train-only source-conditioned geometry selector.

Motivation:

- The original e08 row distance uses exact formula overlap heavily.
- For rows>=7, a useful source may be a chemically similar analogue with different elements rather than an exact formula neighbour.
- This experiment changes source proposal, not post-hoc label ranking.

Method:

- Build a formula chemistry vector from periodic-table element features:
  - atomic number;
  - period;
  - group;
  - electronegativity;
  - atomic radius;
  - weighted means and variances by formula count.
- Use train-only source pools:
  - same `(SG, row_count)`;
  - same `(SG, atom_bucket)`;
  - same SG;
  - same `(crystal_system, atom_bucket)`;
  - train fallback.
- Keep lattice and free params from one coherent source.
- Use chemical row alignment for copying free params.
- No StructureMatcher label, test data, RF ranker, or CrystaLLM prediction source is used.

Results:

| run | source strategy | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@20/50 | unique W/A@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E151 | quality + hybrid5x5 | global | 32.03% | 35.94% | 42.19% | 45.31% | 15.00% | 18.33% / 18.33% | 10.41 |
| E156 | chem quality | global | 31.25% | 39.06% | 45.31% | 45.31% | 18.33% | 20.00% / 20.00% | 10.57 |
| E157 | chem quality | diverse | 31.25% | 39.84% | 44.53% | 45.31% | 18.33% | 20.00% / 20.00% | 13.23 |
| E161 | chem quality + hybrid5x5 | global | 31.25% | 39.06% | 45.31% | 45.31% | 18.33% | 20.00% / 20.00% | 10.39 |
| E162 | chem quality + hybrid5x5 | diverse | 31.25% | 39.84% | 44.53% | 45.31% | 18.33% | 20.00% / 20.00% | 13.23 |

Interpretation:

- Chemical analogue source proposal is the most useful recent geometry change.
- It improves K5/K20 and rows>=7 conversion, but not enough for the full-test gate.
- It hurts RMSE and does not preserve E151 rank-1 strength.
- Combining chem source with `hybrid5x5` does not improve over chem source alone.

Current best tracking:

- Rank-1 best: E151, full match@1 `32.03%`.
- K5 best: E157/E162, full match@5 `39.84%`.
- K20/K50 best: E156/E161, full match@20/50 `45.31% / 45.31%`.
- rows>=7 best: E156/E157/E161/E162, match@5/20/50 `18.33% / 20.00% / 20.00%`.

Gate:

- Full test remains closed.
- Need at least one more metric to clear target:
  - match@5 target `41.58%`;
  - match@20 target `49.69%`.

Next direction:

- Continue from chemical analogue source proposal.
- Add source-conditioned lattice/free-param calibration to improve geometry precision and RMSE.
- Avoid pure score tweaks, RF/test-label rankers, CrystaLLM candidate sources, or low-LR CIF SFT as the main route.

## Source VPA Calibration Update

E163-E172 tested train-only volume-per-atom calibration on top of `row_aligned_chem_quality`.

Method:

- Estimate expected target VPA from train-only chemical analogues and fallback buckets.
- Transfer lattice shape and free params from one coherent train source, then rescale lattice volume.
- Compare soft calibration (`strength=0.5`) and hard calibration (`strength=1.0`).
- Keep W/A candidates fixed from E84/E85; no StructureMatcher label is used for sorting.

Results:

| run | lattice mode | scorer | full match@1 | full match@5 | full match@20 | full match@50 | RMSE@1 | RMSE@5 | rows>=7 match@50 | unique W/A@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E156 | none | global | 31.25% | 39.06% | 45.31% | 45.31% | 0.1100 | 0.1273 | 20.00% | 10.57 |
| E157 | none | diverse | 31.25% | 39.84% | 44.53% | 45.31% | 0.1100 | 0.1332 | 20.00% | 13.23 |
| E166 | soft VPA | global | 32.03% | 40.62% | 44.53% | 45.31% | 0.0942 | 0.1295 | 20.00% | 10.59 |
| E167 | soft VPA | diverse | 32.03% | 39.06% | 42.97% | 45.31% | 0.0942 | 0.1219 | 20.00% | 13.23 |
| E171 | hard VPA | global | 32.03% | 39.84% | 44.53% | 45.31% | 0.0969 | 0.1252 | 20.00% | 10.55 |
| E172 | hard VPA | diverse | 32.03% | 37.50% | 42.19% | 45.31% | 0.0969 | 0.1154 | 20.00% | 13.23 |

Interpretation:

- Soft VPA calibration is retained as the current K5-best validation configuration:
  - full match@1/5/20/50 = `32.03 / 40.62 / 44.53 / 45.31`.
- It recovers E151 rank-1 while preserving most chemical-source K5 gain and improving RMSE@1.
- Hard VPA calibration is too strong; it improves some RMSE readings but reduces match@5.
- rows>=7 remains stuck at `20.00%` match@50, so the remaining bottleneck is free-param/source-mode compatibility, not only volume scale.

Current best tracking:

- Best match@1: E151/E166/E167/E171/E172, `32.03%`.
- Best match@5: E166, `40.62%`.
- Best match@20/50: E156/E161, `45.31% / 45.31%`.
- Best rows>=7 match@50: E156/E157/E161/E162/E166-E172, `20.00%`.

Gate:

- Full test remains closed.
- Only match@1 clears its target.
- match@5 is now close but still below target (`40.62%` vs `41.58%`).
- match@20 is still below target (`45.31%` vs `49.69%`).

Next direction:

- Keep `row_aligned_chem_quality + source_vpa_calibrated_soft + global` as the current validation default for K5.
- Continue with train-only source-conditioned free-param proposal or multimodal source selection for rows>=7.
- Do not run full test until a validation config clears at least two match targets.

## Soft-VPA Source Allocation Check

E173-E182 tested whether the soft-VPA chemical source family needs a few additional coherent train-source modes for high-confidence W/A candidates.

Results:

| run | plan | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 | unique W/A@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E166 | rank0-only | global | 32.03% | 40.62% | 44.53% | 45.31% | 20.00% | 10.59 |
| E176 | hybrid5x5 | global | 30.47% | 40.62% | 45.31% | 45.31% | 20.00% | 10.35 |
| E177 | hybrid5x5 | diverse | 30.47% | 37.50% | 42.97% | 45.31% | 20.00% | 13.23 |
| E181 | hybrid2x2 | global | 32.03% | 40.62% | 44.53% | 45.31% | 20.00% | 10.56 |
| E182 | hybrid2x2 | diverse | 32.03% | 39.06% | 42.97% | 45.31% | 20.00% | 13.23 |

Interpretation:

- Extra coherent source modes do not move match@5 over the target.
- `hybrid5x5` recovers K20 but damages rank-1.
- `hybrid2x2` is essentially a no-op relative to E166.
- E166 remains the best K5 gate candidate; E181 is a tie with marginally lower RMSE@1.

Bottleneck diagnostic from E166:

- full K5: 88 W/A hits but only 52 match hits;
- rows>=7 K5: 42 W/A hits but only 11 match hits;
- 44 full samples are W/A@5 true but match@5 false, including 32 rows>=7 samples.

Strategy implication:

- More W/A recall or simple source-mode allocation is not the main remaining bottleneck for the K5 gate.
- The next useful work should create train-only free-param/source-mode compatibility signals or a joint lattice/free-param proposal, especially for rows>=7.

## Free-Param Manifold Source Penalty Update

E183-E189 tried a train-only free-param manifold penalty during chemical source selection.

Method:

- collect train free-param value buckets by orbit/element/free-symbol context;
- sort buckets and use circular nearest distance as a source penalty;
- apply the penalty only after a chem-quality top-N prefilter to keep the renderer tractable.

Result:

| run | strategy | scorer | full match@1 | full match@5 | full match@20 | rows>=7 match@5 | rows>=7 match@20 |
|---|---|---|---:|---:|---:|---:|---:|
| E166 | chem quality + soft VPA | global | 32.03% | 40.62% | 44.53% | 18.33% | 20.00% |
| E188 | chem param quality + soft VPA | global | 29.69% | 36.72% | 40.62% | 16.67% | 18.33% |
| E189 | chem param quality + soft VPA | diverse | 29.69% | 37.50% | 40.62% | 18.33% | 18.33% |

Interpretation:

- This signal is leakage-safe but harmful.
- It reduces early W/A/skeleton ordering and does not improve rows>=7 conversion.
- The negative result suggests that scalar free-param-prior compatibility is not enough; it needs to be coupled to source mode and W/A ranking rather than added as an independent penalty.

Current active default remains:

- `row_aligned_chem_quality`;
- `source_vpa_calibrated_soft`;
- global self-score;
- no hybrid scheduling.

## Strict Physical Self-Score Update

E190-E193 tested a stronger GT-free generated-CIF physical self-score over the E163 soft-VPA candidate pool.

Method:

- add `--score-profile strict_physical` to `opentry_selfscore_rendered_cifs.py`;
- increase penalties for short interatomic distance and implausible VPA;
- keep renderer and W/A candidates fixed.

Results:

| run | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@5 | rows>=7 match@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 | standard global | 32.03% | 40.62% | 44.53% | 45.31% | 18.33% | 20.00% |
| E192 | strict global | 25.00% | 35.94% | 45.31% | 45.31% | 16.67% | 20.00% |
| E193 | strict diverse | 25.00% | 39.06% | 44.53% | 45.31% | 15.00% | 20.00% |

Interpretation:

- The strict profile improves min-distance/VPA surrogate values but damages early W/A and skeleton ranking.
- It is not adopted.
- Candidate-internal physical plausibility is useful as a sanity signal, but not enough as a primary ordering objective.

Strategy implication:

- Avoid further scalar self-score tuning unless it is paired with a genuinely new proposal family.
- Current best remains E166/E181.
- The next real move should jointly propose source/free params/lattice or explicitly condition geometry on W/A rows.

## Physical Source Selection Update

E194-E198 tested a local joint source/free-param/lattice selector.

Method:

- add `row_aligned_chem_physical_select` to the renderer;
- for the rank-0 geometry candidate, render the top 3 chemical train sources for the same W/A;
- choose among those rendered CIFs using GT-free generated-CIF health: parse/formula/atom/composition/SG flags, min-distance, VPA, and geometry distance;
- keep StructureMatcher labels out of source selection.

Results:

| run | method | full match@1 | full match@5 | full match@20 | rows>=7 match@5 | rows>=7 match@20 | W/A@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 | chem quality + soft VPA global | 32.03% | 40.62% | 44.53% | 18.33% | 20.00% | 76.56% |
| E197 | physical-select global K20 smoke | 30.47% | 34.38% | 39.84% | 16.67% | 20.00% | 77.34% |
| E198 | physical-select diverse K20 smoke | 30.47% | 33.59% | 39.84% | 15.00% | 20.00% | 77.34% |

Interpretation:

- Physical source selection improves rank-1 generated-CIF health, but loses StructureMatcher match.
- Because W/A@20 is stable, the problem is not W/A recall. The selected geometry is physically cleaner but less aligned to the true structure.
- Do not expand this branch to K100 or full test.

Updated strategy:

- E166/E181 remain the active validation default.
- Further progress likely needs an explicit learned geometry/free-param proposal conditioned on canonical W/A rows, not another scalar plausibility selector.

## Batched Geometry-Net Lattice Update

E199-E204 revisited the old lattice-only geometry-net diagnostic.

Implementation:

- E199 showed the previous per-candidate geometry-net lattice inference was still too slow.
- The renderer now precomputes geometry-net lattice predictions per sample and caches them by `(sample_id, canonical_wa_key)`.
- This fixes the runtime blocker only for `--geometry-lattice-mode geometry_net`; the default E166 path is unchanged.

Results:

| run | lattice | scorer | full match@1 | full match@5 | full match@20 | rows>=7 match@5 | rows>=7 match@20 |
|---|---|---|---:|---:|---:|---:|---:|
| E166 | source VPA soft | global | 32.03% | 40.62% | 44.53% | 18.33% | 20.00% |
| E203 | geometry-net | global | 28.12% | 32.03% | 36.72% | 8.33% | 10.00% |
| E204 | geometry-net | diverse | 28.12% | 32.03% | 36.72% | 8.33% | 10.00% |

Interpretation:

- The speed issue is fixed, but the old train-only geometry net lattice is not usable for the current gate.
- It damages rows>=7 geometry even though W/A recall stays similar.
- Direct mean lattice prediction should not be used as the next model route.

Updated strategy:

- Keep source-VPA soft as the active lattice default.
- If training a geometry model, train a source-conditioned residual/free-param+lattice transfer model rather than predicting lattice from W/A alone.

## Source-Conditioned Residual Geometry Update

E205-E213 implemented the first source-conditioned residual geometry model.

What changed:

- Added a train-only residual model:
  - input: formula, GT-SG, canonical W/A rows, selected train-source free params/lattice, source distance, alignment cost;
  - output: residual lattice and row free params.
- Added a paired renderer for predicted W/A candidates.
- Added a bounded fast source selector for this branch so train/inference use the same source proposal.

Small-scale results:

| run | setting | result |
|---|---|---|
| E205 | 512 train, 1 epoch | pipeline passes, but val lattice loss is very high |
| E209 | 1k train, 5 epochs | best val loss `9.2103`, still weak |
| E213 | val32/top5 self-scored | full `25.00/25.00`, rows>=7 `5.56/5.56` |
| E166 first-32 reference | existing best | full `34.38/43.75`, rows>=7 `22.22/22.22` |

Interpretation:

- The branch is aligned with the requested model-side direction, but the current one-source residual regression is not good enough.
- W/A@5 and unique W/A remain strong on val32; the failure is geometry validity/match.
- Composition/readability degradation means residual free-param updates need constraints or clipping.
- Uncached source-pair construction/rendering is too slow for val128+.

Updated strategy:

- Keep the scripts, but do not scale E209.
- Before retraining larger residual models, add cached source-pair datasets under `opentry_3`.
- Add validity-preserving residual constraints and clipping; do not add source-CIF fallback.
- Consider hard-negative source-pair training or an explicit no-move/source-mixture objective so the model learns when not to move a source geometry.

## Cached Source-Pair Residual Update

E214-E224 implemented the first cached source-pair residual workflow.

What changed:

- Added `opentry_build_source_residual_examples.py` to write train/val source-pair examples under `opentry_3`.
- Training can now load `--cached-examples-dir`, avoiding repeated online source selection.
- The residual model now exposes `lattice_delta_scale` and `coord_delta_scale`; old defaults are preserved, while new experiments can use tighter residual bounds.
- The renderer reads those scale values from checkpoint config.

Cache and model summary:

| run | setting | result |
|---|---|---|
| E214 | train1024 / val256 cache, selector from full train only | 1,024 train and 256 val examples, 0 skipped |
| E215 | delta scales 0.35 / 0.15 | val loss `4.985`; E219 val32 self-score full `31.25/31.25` |
| E220 | delta scales 0.15 / 0.05 | val loss `5.029`; E224 val32 self-score full `34.38/37.50` |

Same-prefix reference:

| method | full match@1 | full match@5 | rows>=7 match@1 | rows>=7 match@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | 22.22% | 22.22% | 78.12% | 2.88 |
| E224 tight residual | 34.38% | 37.50% | 16.67% | 16.67% | 81.25% | 4.38 |

Interpretation:

- Caching is useful and should be kept. It makes the residual route auditable and cheap to retrain.
- Tighter residual bounds recover rank-1 but not top-5.
- The problem is not symbolic W/A recall: E224 has higher W/A@5 and unique W/A@5 than the E166 first-32 reference.
- The issue is source-mode/free-param/lattice compatibility under predicted W/A candidates.

Updated strategy:

- Do not scale E215/E220 to val128 or full test.
- Continue only if the next residual branch changes the objective: cached predicted-W/A source pairs, source-mode/no-move classification, or a mixture model over coherent train sources.
- Avoid fallback and avoid scalar physical self-score tuning as a primary route.

## Source-Mode Selector Update

E225-E226 tested whether the train-source mode itself can be learned before changing free params or lattice.

Method:

- Build grouped examples with 8 candidate train sources per target.
- Label candidates by train/val lattice plus free-param transfer error.
- Do not use StructureMatcher match/rms, test data, or GT geometry as input features.
- Train a small listwise MLP to choose the source mode.

Oracle diagnostic:

| subset | heuristic best-source rate | heuristic error | oracle best error | oracle gain |
|---|---:|---:|---:|---:|
| val full | 23.83% | 1.7472 | 0.4064 | 1.3408 |
| val rows>=7 | 27.56% | 3.1732 | 0.6461 | 2.5270 |

E226 best validation result:

| subset | heuristic error | selected error | selected gain | hit best rate |
|---|---:|---:|---:|---:|
| val full | 1.7472 | 1.5610 | +0.1862 | 22.66% |
| val rows>=7 | 3.1732 | 2.7459 | +0.4272 | 25.20% |

Interpretation:

- Source-mode selection has large oracle headroom, especially in rows>=7.
- The first MLP learns a useful but weak signal. It reduces transfer error but does not yet approach oracle.
- This is a better-aligned direction than scalar physical self-score tuning because it changes the coherent geometry source proposal.

Updated strategy:

- Keep E225/E226 as source-mode infrastructure.
- Do not claim CIF improvement until the selector is integrated into rendering and evaluated on val StructureMatcher.
- Next model-side geometry work should either:
  - integrate E226 into a val32 renderer smoke; or
  - expand source-mode training and add a calibrated no-move/source-mixture objective.

## Source-Mode Renderer Integration Update

E227-E230 integrated the E226 source-mode selector into the residual renderer.

What changed:

- `opentry_render_source_residual_geometry.py` can now load a source-mode selector checkpoint.
- The renderer scores a small train-source pool and uses the selected source mode before residual geometry prediction.
- The resulting CIFs were evaluated only on MPTS-52 validation first 32 records.

Result summary:

| method | full match@1 | full match@5 | rows>=7 match@1 | rows>=7 match@5 | W/A@5 | unique W/A@5 |
|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | 22.22% | 22.22% | 78.12% | 2.88 |
| E224 tight cached residual | 34.38% | 37.50% | 16.67% | 16.67% | 81.25% | 4.38 |
| E230 source-mode + residual + self-score | 34.38% | 37.50% | 11.11% | 16.67% | 81.25% | 4.38 |

Interpretation:

- The first source-mode selector improves transfer-label error but does not improve CIF match after rendering.
- W/A recall remains strong, so the remaining failure is geometry/source compatibility, not symbolic candidate recall.
- Source-mode selection should stay as infrastructure, but E226 is too weak to scale.

Updated strategy:

- Do not run E227-E230 beyond val32 and do not freeze this config.
- Train the next geometry branch with a stronger source-mode objective: larger train-only source pools, explicit no-move calibration, and coherent source-mixture prediction.
- Avoid treating scalar self-score as the primary fix; it recovers rank1 in this smoke but does not move K5 or rows>=7.

## Source-Mode Margin Calibration Update

E231-E235 tested a conservative no-move margin on top of E226.

Method:

- Tune a score-margin threshold using E225 train/val transfer labels.
- If the selector prefers a non-rank0 source but its score advantage over rank0 is below the threshold, keep rank0.
- Use validation transfer error for threshold selection; StructureMatcher remains evaluation-only.

Transfer-label result:

| selector | val full selected error | val rows>=7 selected error |
|---|---:|---:|
| heuristic rank0 | 1.7472 | 3.1732 |
| pure E226 | 1.5610 | 2.7459 |
| E231 margin>=0.75 | 1.4991 | 2.6644 |

CIF smoke result:

| method | full match@1 | full match@5 | rows>=7 match@1 | rows>=7 match@5 |
|---|---:|---:|---:|---:|
| E166 first-32 reference | 34.38% | 43.75% | 22.22% | 22.22% |
| E230 pure source-mode | 34.38% | 37.50% | 11.11% | 16.67% |
| E235 margin-calibrated source-mode | 34.38% | 37.50% | 11.11% | 16.67% |

Interpretation:

- The margin is useful under the transfer label, but the transfer label is still too weak a proxy for rendered CIF match.
- Rank1 can be recovered by GT-free self-score, but K5 and rows>=7 do not move.
- The bottleneck is coherent geometry generation under predicted W/A, not only source choice confidence.

Updated strategy:

- Do not scale E231-E235.
- Next source-conditioned geometry work should train a mixture/no-move objective that predicts multiple coherent source modes and geometry deltas jointly.
- Keep thresholding as a diagnostic control, not the primary model route.

## Source-Mode Mixture Update

E236-E241 tested inference-time source-mode mixtures for the residual renderer.

Method:

- Use the same E84 symbolic W/A candidates and E220 residual geometry model.
- Score train-source candidates with E226.
- Render up to three source-mode proposals per unique W/A candidate, with rank0 included as a conservative no-move anchor.
- Reorder rendered CIFs with GT-free global or diverse self-score.

Result summary on MPTS-52 validation first 32:

| method | subset | match@1 | match@5 | match@20 | W/A@5 | W/A@20 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | 78.12% | - | 2.88 |
| E235 margin source-mode | full | 34.38% | 37.50% | - | 81.25% | - | 4.38 |
| E239 mixture global | full | 34.38% | 40.62% | 40.62% | 81.25% | 84.38% | 3.16 |
| E241 mixture diverse | full | 34.38% | 40.62% | 40.62% | 81.25% | 84.38% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | 83.33% | - | 2.72 |
| E241 mixture diverse | rows>=7 | 16.67% | 16.67% | 16.67% | 83.33% | 83.33% | 4.56 |

Interpretation:

- Mixtures recover some full K5 lost by the previous source-mode selector, but they remain below the E166 first-32 reference.
- Rows>=7 does not improve, despite high W/A recall. This points to geometry/source compatibility, not symbolic candidate recall, as the active bottleneck.
- Diverse self-score improves unique W/A without increasing match, so diversity alone is not the next lever.

Updated strategy:

- Keep source-mixture rendering support as infrastructure.
- Do not run E236-E241 at val128/full scale.
- The next aligned model should train source mixture/no-move and residual geometry jointly, preferably with a loss that rewards coherent rendered geometry under predicted W/A rather than only transfer-label distance.

## Train-Time Source-Mixture Residual Update

E242-E249 moved source mixtures from inference-time only into residual geometry training.

Method:

- Build residual training examples from E225 source-mode transfer-label groups.
- For each train/val target, include rank0 no-move source plus transfer-error top sources.
- Train the existing residual geometry net on these source contexts with tight delta scales.
- Render E84 W/A predictions with the same E226 mixture3 + rank0 protocol used in E236.

Result summary on MPTS-52 validation first 32:

| method | subset | match@1 | match@5 | match@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|
| E239 E220 mixture global | full | 34.38% | 40.62% | 40.62% | 81.25% | 3.16 |
| E241 E220 mixture diverse | full | 34.38% | 40.62% | 40.62% | 81.25% | 4.38 |
| E247 E243 mixture-trained global | full | 31.25% | 37.50% | 40.62% | 71.88% | 3.03 |
| E249 E243 mixture-trained diverse | full | 31.25% | 37.50% | 40.62% | 75.00% | 4.38 |
| E241 E220 mixture diverse | rows>=7 | 16.67% | 16.67% | 16.67% | 83.33% | 4.56 |
| E249 E243 mixture-trained diverse | rows>=7 | 11.11% | 11.11% | 16.67% | 72.22% | 4.56 |

Interpretation:

- E242 is useful infrastructure because it makes train-time source mixtures auditable and leakage-safe.
- E243 is not a better geometry model. Adding transfer-error top sources to the regression target reduces rendered CIF match under the same validation protocol.
- The transfer-error label is too weak as a proxy for rendered CIF quality. Training on multiple source contexts without a gating/no-op mechanism appears to average away useful source geometry.

Updated strategy:

- Keep E242 data builder.
- Do not scale E243 to val128/full.
- The next geometry model should predict a source-conditioned gate/no-op delta or learn a GT-free rendered-quality proxy for source contexts, rather than regressing all transfer-top sources toward the same target equally.

## Gated Source-Mixture Residual Update

E250-E256 implemented the source-conditioned gate/no-op residual head proposed after E242-E249.

Method:

- Keep the E242 train/val source-mixture residual cache: rank0 no-move source plus transfer-error top sources.
- Add optional lattice and coordinate delta gates to the source residual geometry net.
- Initialize gates toward small residuals and apply an L1 penalty on gate magnitude.
- Render E84 W/A predictions with E226 source-mode mixture3 plus rank0.
- Reorder only with GT-free global/diverse self-score; StructureMatcher is evaluation-only on validation first 32.

Training signal:

| item | value |
|---|---:|
| train / val examples | 2,645 / 672 |
| best val loss | 2.3394 |
| best epoch | 3 |
| epoch4 val lattice_gate_mean | 0.3448 |
| epoch4 val coord_gate_mean | 0.0315 |

Validation first-32 result:

| method | subset | match@1 | match@5 | match@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|
| E166 first-32 reference | full | 34.38% | 43.75% | - | 78.12% | 2.88 |
| E241 E220 mixture diverse | full | 34.38% | 40.62% | 40.62% | 81.25% | 4.38 |
| E249 E243 mixture-trained diverse | full | 31.25% | 37.50% | 40.62% | 75.00% | 4.38 |
| E254 E250 gated global | full | 37.50% | 43.75% | 43.75% | 75.00% | 2.91 |
| E256 E250 gated diverse | full | 37.50% | 43.75% | 43.75% | 78.12% | 4.38 |
| E166 first-32 reference | rows>=7 | 22.22% | 22.22% | - | 83.33% | 2.72 |
| E256 E250 gated diverse | rows>=7 | 11.11% | 16.67% | 16.67% | 77.78% | 4.56 |

Interpretation:

- The gated no-op residual is better than equal regression over transfer-top sources and recovers full K1/K5 on the small validation prefix.
- It does not solve the complex rows>=7 bottleneck; rows>=7 remains below the E166 reference.
- The low coordinate-gate mean suggests no-op behavior is learnable, but the remaining failure is source/W-A geometry compatibility under complex assignments.
- This is not strong enough for val128 or full test.

Updated strategy:

- Keep the gated residual head as the default residual-geometry experimental branch.
- Do not scale E250 directly.
- Next work should train a source/geometry quality proxy or mixture-of-experts selector that explicitly separates compatible and incompatible source contexts for complex W/A, using train/val only.

## Rendered-Health Source Selector Update

E257-E265 tested the next source/geometry proxy idea: choose source contexts using a GT-free rendered-CIF health score instead of lattice/free-param transfer error.

Method:

- Build train/val source-mode examples by rendering each candidate source context with the E250 gated residual geometry model.
- Score each rendered CIF with GT-free health features: readability, composition exactness, GT-SG consistency, pymatgen parse/min-distance/VPA, geometry plausibility, source distance, and rank penalties.
- Train the existing listwise `SourceModeScorer` on `combined_error = -render_health_score`.
- Evaluate on MPTS-52 validation first 32 only; StructureMatcher is evaluation-only.

Proxy data signal:

| item | value |
|---|---:|
| train / val groups | 512 / 256 |
| source_pool_k | 8 |
| residual geometry for labels | E250 gated residual |
| val full heuristic / best health | 4876.49 / 4944.31 |
| val full oracle health gain | 67.82 |
| val rows>=7 heuristic / best health | 4817.83 / 4922.07 |
| val rows>=7 oracle health gain | 104.23 |
| StructureMatcher labels used | 0 |
| test records used | 0 |

Selector result:

| item | value |
|---|---:|
| best epoch | 16 |
| best val full selected health gain vs rank0 | +1.42 |
| best val rows>=7 selected health gain vs rank0 | +3.06 |
| margin tuning best label | pure_model |

Validation first-32 CIF result:

| method | subset | match@1 | match@5 | match@20 | W/A@5 | unique W/A@5 |
|---|---|---:|---:|---:|---:|---:|
| E254 E250 gated global | full | 37.50% | 43.75% | 43.75% | 75.00% | 2.91 |
| E256 E250 gated diverse | full | 37.50% | 43.75% | 43.75% | 78.12% | 4.38 |
| E262 E258 render-health global | full | 34.38% | 40.62% | 43.75% | 81.25% | 2.78 |
| E264 E258 render-health diverse | full | 34.38% | 40.62% | 43.75% | 84.38% | 4.38 |
| E256 E250 gated diverse | rows>=7 | 11.11% | 16.67% | 16.67% | 77.78% | 4.56 |
| E264 E258 render-health diverse | rows>=7 | 11.11% | 16.67% | 16.67% | 83.33% | 4.56 |

Interpretation:

- E257 is useful evidence that complex rows have meaningful source-choice space under a GT-free rendered-health proxy.
- E258 does not recover that oracle space: the learned selector improves the proxy only marginally and reduces full K1/K5 versus E254/E256 on the same validation prefix.
- rows>=7 match is unchanged, so this does not address the active complex-geometry bottleneck.
- The bottleneck is no longer simply "which source candidates exist"; current numeric source features plus a shallow listwise classifier are not enough to infer source/W-A compatibility.

Updated strategy:

- Keep `opentry_build_source_render_quality_examples.py` as a diagnostic/data builder.
- Do not scale E258 to val128 or full test.
- Do not spend more time only tuning selector margins.
- Next work should either score rendered candidates directly with richer GT-free rendered-CIF features, or train a stronger per-rendered-candidate quality model that sees the actual rendered output rather than only source numeric features.

## Source-Expanded Rendered Candidate Update

E266-E279 tested direct rendered-candidate scoring without the E258 source selector.

Method:

- Extend the source residual renderer with `--source-expand-k`.
- For each predicted unique W/A, render multiple train-source contexts with the E250 gated residual model.
- Reorder rendered CIFs using the existing GT-free global self-score.
- Evaluate only on validation prefixes.

Key result:

| method | validation prefix | full match@1 | full match@5 | full match@20 | rows>=7 match@1 | rows>=7 match@5 | rows>=7 match@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E269 source-expand8 global | 32 | 34.38% | 40.62% | 46.88% | 11.11% | 16.67% | 16.67% |
| E275 source-expand8 global | 64 | 35.94% | 43.75% | 46.88% | 13.33% | 20.00% | 20.00% |
| E279 source-expand8 global | 128 | 27.34% | 35.94% | 41.41% | 11.67% | 16.67% | 18.33% |

Interpretation:

- Rendering multiple source contexts creates useful candidate diversity and can improve prefix K20.
- The effect is not stable at val128; E279 is below the E166/E181 val128 reference.
- rows>=7 remains the limiting subset, even though W/A@5 improves with source expansion.
- The GT-free self-score is good at selecting readable/composition/SG-valid CIFs, but not reliable enough as a StructureMatcher-positive proxy.

Updated strategy:

- Keep `--source-expand-k` as a diagnostic and future candidate-pool option.
- Do not scale source-expand8 + fixed self-score to full test.
- Do not spend the next iteration only tuning self-score weights on the same prefix.
- Next source/geometry work should train a per-rendered-candidate quality model using train/val labels and actual rendered-CIF features, or otherwise return to improving complex W/A generator recall before larger CIF evaluation.

## Canonical W/A Source-Priority Decoder Update

E280-E294 returned to the symbolic decoder and fixed a budget issue in candidate merging.

Method:

- Convert assignment-DP outputs to canonical renderer predictions per source before merging.
- Merge on `canonical_wa_key`, not raw `wa_key`, so source-specific W/A permutations no longer consume top-k budget.
- Use a priority decoder:
  - E69c neural assignment DP for early ranks;
  - E69d neural assignment DP for additional tail recall;
  - E43 prior DP for remaining canonical coverage.
- No StructureMatcher labels, no test data, no row_count sorting, no RF/ranker, no CrystaLLM candidates.

Best symbolic result:

| run | full W/A@1 | full W/A@5 | full W/A@20 | full W/A@50 | full unique W/A@50 | rows>=7 W/A@50 | rows>=7 unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E85 old raw merge | 47.66% | 70.31% | 77.34% | 78.91% | 13.74 | 76.67% | 8.37 |
| E289 canonical priority | 47.66% | 70.31% | 77.34% | 79.69% | 15.13 | 76.67% | 10.38 |

Validation CIF check with the current E166 renderer:

| run | scorer | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 |
|---|---|---:|---:|---:|---:|---:|
| E166 reference | global | 32.03% | 40.62% | 44.53% | 45.31% | 20.00% |
| E292 E289 candidates | global | 32.03% | 39.84% | 44.53% | 45.31% | 20.00% |
| E294 E289 candidates | diverse | 32.03% | 38.28% | 42.97% | 45.31% | 20.00% |

Decision:

- Adopt canonical merge infrastructure for diagnostics.
- Do not adopt E289/E292/E294 as the active CIF config.
- The active W/A problem is now rows>=7 recall and geometry compatibility, not full-set tail uniqueness alone.
- Full test remains closed.

## E69b-First Canonical Decoder Follow-Up

E295-E300 tested E69b as the first canonical source because its raw DP had better rows>=7 early assignment recall.

Symbolic outcome:

| run | full W/A@1 | full W/A@5 | full W/A@50 | rows>=7 W/A@1 | rows>=7 W/A@5 | rows>=7 W/A@50 | full unique W/A@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E289 E69c-first | 47.66% | 70.31% | 79.69% | 51.67% | 73.33% | 76.67% | 15.13 |
| E297 E69b-first | 49.22% | 70.31% | 79.69% | 53.33% | 73.33% | 76.67% | 15.23 |

CIF outcome with E166 renderer/global self-score:

| run | full match@1 | full match@5 | full match@20 | full match@50 | rows>=7 match@50 |
|---|---:|---:|---:|---:|---:|
| E166 reference | 32.03% | 40.62% | 44.53% | 45.31% | 20.00% |
| E292 E69c-first | 32.03% | 39.84% | 44.53% | 45.31% | 20.00% |
| E300 E69b-first | 31.25% | 39.06% | 44.53% | 45.31% | 20.00% |

Decision:

- E297 is the best symbolic W/A ordering record.
- It is not a CIF config: E300 loses match@1 and match@5.
- Do not continue rendering source-order permutations unless rows>=7 W/A@5/50 improves.
- Next iteration should improve complex assignment recall or learn rendered-candidate quality that preserves the symbolic gain through geometry selection.
