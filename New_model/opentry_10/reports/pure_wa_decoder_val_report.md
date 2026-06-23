# Pure WA Decoder Validation Report

No new exact-cover WA decoder checkpoint was trained in opentry_9.

Available MP-20 validation evidence from SymCIF-v4 K<=5:

- baseline WA_hit@1/@5 = 38.63% / 65.11%
- raw top100 WA_hit = 87.85%
- one-fix train-prior/hybrid-prior candidate selection raises selected WA_hit@5 to 79.52% and match@5 to 71.58%

Diagnosis: the candidate pool contains substantially more WA coverage than the selected top5 exposes, so WA selection/ranking is a major bottleneck. Because GT-WA geometry still reaches only 82.94% match@5, geometry remains a second hard bottleneck.
