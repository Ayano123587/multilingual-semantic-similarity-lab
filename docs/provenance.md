# Data and model sources

[SemRel2024 official repository](https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024) is pinned to `3c5941ddffdc9d833a5528f49dd13ed9087c0dba`. The [collection paper](https://arxiv.org/html/2402.08638v5) describes eight included source collections; its [earlier primary version](https://arxiv.org/html/2402.08638v1) identifies Punjabi's Varta news headlines, lexical-overlap pairing and human best-worst ratings. The [shared-task paper](https://arxiv.org/html/2403.18933v2) covers all 14 languages.

Included: Afrikaans, Amharic, Modern Standard Arabic, Algerian Arabic, Moroccan Arabic, Hausa, Indonesian, Kinyarwanda and Punjabi. Every available official split is used. The [coverage file](full-cohort-v3-coverage.json) pins all 23 CSV paths, byte hashes, schemas and row counts; [preparation counts](full-cohort-v3-preparation.json) record text-only removals. Source files give Punjabi 242 development/634 test pairs; these pinned file counts govern the run.

Excluded English/Hindi/Marathi/Telugu collections mix generated paraphrase sources without reliable per-row origin tags. Spanish mixes translated PAWS-X paraphrases with algorithmic PAWS ancestry; the [PAWS-X paper](https://aclanthology.org/D19-1382/) explains its construction. These five mixtures fall outside this project's real-source-text cohort because their per-row sentence origins cannot be separated reliably.

No explicit corpus license was found in the pinned release. Raw pairs, labels, caches and learned weights stay private and ignored; this repository does not relicense or redistribute them. Fetching scripts retain official attribution and immutable bytes.

The selected [multilingual E5 base](https://huggingface.co/intfloat/multilingual-e5-base) revision is `d128750597153bb5987e10b1c3493a34e5a4502a`, MIT. Safetensors SHA256 is `a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7`. Symmetric relatedness uses `query: ` on both inputs. The comparison [AfroXLMR base](https://huggingface.co/Davlan/afro-xlmr-base) is MIT, revision `25f27299c247a6b73a767bd82d12444138b19337`, weight SHA256 `f73b04316c9c2e76fcb213957616a09e1eb44112266717f649fc1897dac42953`.

Encoder revisions and weight bytes are checked before allocation. Masked mean pooling uses no random pooler. External pretraining contains multilingual retrieval/parallel mixtures and its benchmark exposure is unknown; this project adds no generated fine-tuning pairs. Original project code is covered by the root MIT license, separately from all upstream data/model terms.
