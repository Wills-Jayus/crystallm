# SymCIF Canonicalization Audit

opentry_9 reused the existing SymCIF-v4 structured records from symcif_experiment and wrote local cache copies.

- total rows: 82784
- conversion success rows: 69654 (84.14%)

| output | rows | datasets | conversion success |
| --- | ---: | --- | ---: |
| cache/symcif_train.jsonl | 52627 | {'mp20': 26629, 'mpts52': 25998} | 83.99% |
| cache/symcif_val.jsonl | 13601 | {'mp20': 8874, 'mpts52': 4727} | 84.52% |
| cache/symcif_test_targets.jsonl | 16556 | {'mp20': 8893, 'mpts52': 7663} | 84.30% |

No failed conversion rows were silently dropped in opentry_9; the source structured files already encode conversion/extraction status fields such as row_expansion_all_ok and free_param_reextract_all_success.
