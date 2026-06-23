# Pure Structural Validation Report

Pure structural route did not pass a full K20 validation-generation gate in opentry_9.

Reusable diagnosis:

- CIF to SymCIF canonicalization coverage is high in the existing structured cache.
- MP-20 GT-WA geometry K<=5 is strong but not saturated: match@5 = 82.94%, rows>=7 proxy match@5 = 69.46%.
- WA candidate selection is weak at K<=5: baseline WA_hit@5 = 65.11%, while raw top100 WA_hit = 87.85%.

Conclusion: do not run official pure full-test. The next pure-model step should train an inference-feasible exact-cover WA decoder/ranker and a geometry quality scorer, then evaluate K20 on validation before any test.
