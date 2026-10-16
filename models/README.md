# Private model artifacts

`selected/selected.pt` is the retained three-seed E5 MLP/character-overlap blend, SHA256 `43b16caca7c625d5513914da1eed338d40e41e11fd94e10381e2d8bd9859240e`. It contains train-only normalization and the fixed global semantic weight 0.25. `selected/baseline.pt` contains the lexical comparison; `selection.json` and `inference-proof.json` bind validation choice and actual text replay.

All encoder weights, fitted checkpoints, embeddings and raw text remain ignored. The model is trained on the complete available official training cohort, with development-only selection and a separately frozen final evaluation. See [Report](../Report.md) and the [frozen identity](../docs/frozen-policy.json).
