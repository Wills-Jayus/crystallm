# Validation GT CIF Symprec Audit

- Created at: 2026-06-22T17:26:43+00:00
- Output root: `/data/users/xsw/autodlmini/model/New_model/opentry_10/cache/official_benchmark_cifs_symprec0p1`
- symprec: 0.1
- angle_tolerance: 5.0

| dataset | split | records | non-P1 detected | P1 detected | errors | CSV SG match rate | manifest |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mp_20 | val | 9047 | 8900 | 147 | 0 | 1.0000 | `/data/users/xsw/autodlmini/model/New_model/opentry_10/cache/official_benchmark_cifs_symprec0p1/mp_20/val/manifest.tsv` |
| mpts_52 | val | 5000 | 4980 | 20 | 0 | NA | `/data/users/xsw/autodlmini/model/New_model/opentry_10/cache/official_benchmark_cifs_symprec0p1/mpts_52/val/manifest.tsv` |

The earlier `cache/official_benchmark_cifs` direct extraction keeps `_symmetry_space_group_name_H-M P 1` for validation rows and is not valid for GT-SG prompt construction.
