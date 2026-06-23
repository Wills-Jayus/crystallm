# Train-Inference Condition Gap Report

Created: 2026-06-19T02:47:58Z

opentry_5 stores both GT W/A targets and OOF/predicted W/A conditions so SFT or joint models can mix teacher-forced, predicted, and corrupted conditions. No val512/test true W/A is used as a training condition.

Files:

- `data/oof_wa_predictions_train.jsonl`
- `data/oof_wa_predictions_dev.jsonl`

Scheduled teacher forcing recommendation:

- early: 70% GT W/A, 20% OOF predicted W/A, 10% synthetic corrupted W/A;
- middle: 45% GT, 35% OOF, 20% corruption;
- late: 25% GT, 50% OOF, 25% corruption.
