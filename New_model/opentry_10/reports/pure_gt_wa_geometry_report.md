# Pure GT-WA Geometry Report

Input: composition + GT-SG + GT-WA. Output: rendered CIF geometry variants from the existing SymCIF-v4 MP-20 validation K<=5 artifact.

Key result: GT-WA geometry match@1/match@5 = 77.16% / 82.94%; RMSE@1/RMSE@5 = 0.0450 / 0.0388.

Rows>=7 proxy (n_sites>=6 in the historical report): match@1/match@5 = 59.22% / 69.46%.

Interpretation: even with correct WA, geometry is not saturated. The pure model bottleneck is both WA coverage and geometry, with geometry remaining a hard upper bound.

Limitations: no K20 GT-WA geometry run and no MPTS-52 GT-WA geometry artifact were available in opentry_9.
