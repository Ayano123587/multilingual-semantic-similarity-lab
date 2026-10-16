# Real multilingual relatedness research

**Execution in progress; no trained score or gain is claimed yet.** The pipeline uses real human-rated Amharic, Hausa and Kinyarwanda SemRel2024 news pairs. The frozen study compares lexical/pretrained baselines, three ridge penalties, and three MLP configurations with three seeds each. Test gold remains sealed until an immutable validation winner is saved.

| Stage | Current evidence |
|---|---|
| Official source integrity | Nine pinned CSV hashes verified |
| Corrected cohorts | 3,456 train, 386 development, 996 sealed test inputs |
| Pair copies | 50 Kinyarwanda train and 23 development copies removed using text only |
| Sentence groups | Cross-split components disclosed; Hausa largest component has 1,364 sentences |
| Model/source code tests | Actual gradient, checkpoint, schema, mask-free test parsing and component tests |
| Encoder execution | Private cloud CPU v2 running; no local encoder allocation |
| Trained comparison/final test | Pending actual execution; no placeholder results |

The initial cloud attempt failed before model loading because Windows JSONL CRLF bytes differed from Linux LF. Canonical LF files now match cloud hashes exactly and preserve every parsed row; the failure remains documented. English/Marathi/Telugu were rejected during primary-source audit because of generated-paraphrase construction. No synthetic replacement data or pseudo-labels enter this study.

Relatedness is broader than paraphrase similarity and model output is not a calibrated probability. Official splits are not sentence-disjoint. Upstream encoder pretraining overlap is unknown, and clustered bootstrap uncertainty cannot remove source reuse or repeated validation-selection optimism. Source data and learned weights remain private; source code is MIT.

Use the [complete workflow notebook](notebooks/01_complete_relatedness_workflow.ipynb), [frozen protocol](docs/model-protocol-v1.json), and [provenance](docs/provenance.md). Optional notebook training/inference calls production modules and fresh paths. Final test release is separate and never automatic.
