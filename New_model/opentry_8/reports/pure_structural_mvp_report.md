# opentry_8 Pure Structural MVP Report

Status: not official full-tested.

The opentry_8 pure structural line was not run on official MP-20 or MPTS-52 full test. This is an intentional gate decision, not a missing result.

Why:

- opentry_7 pure model was a byte-level CIF GPT fine-tune, selected by validation token loss. It is a valid fixed-order pure control, but it is not the requested structural W/A+geometry model.
- Historical MiniCFJoint-v2 fixed the 58,880-class action-space problem and passed small-overfit, but its GT-W/A geometry validation gate failed: match@1/5 = 55.73%/58.88%, RMSE@5 = 0.1684.
- SymCIF-v5 and stablekey are structured strategy/retrieval/index systems. They diagnose W/A coverage and geometry bottlenecks, but they are not pure neural models under the prompt restrictions.
- No new opentry_8 structural checkpoint exists with validation generation metrics exceeding opentry_7 pure or approaching CrystaLLM-a GT-SG.

Therefore, an opentry_8 pure structural official full-test would be misleading. The correct next step is to train a real composition+GT-SG structural generator and gate it on validation generation metrics before any full test.

Required architecture:

- `composition + GT-SG -> row_count / Wyckoff skeleton`
- `row_count / skeleton -> element assignment with exact-cover remaining-composition masks`
- `W/A -> lattice + free parameters with multimodal NLL or wrapped-coordinate loss`
- CIF renderer

Required validation gate:

- match@1/5/20
- RMSE@1/5/20
- rows>=7 match@1/5/20
- rows>=7 positive-any
- validity/readability
- fixed candidate order, no ranking or selector

