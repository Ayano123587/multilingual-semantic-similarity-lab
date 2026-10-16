# Prospective source-construction correction

The initial English/Marathi/Telugu proposal was rejected during source preflight,
before data ingestion, model inference, training or scoring. The primary dataset
paper identifies machine-translation paraphrases in English and paraphrase
generation in the Indic subsets. Human ratings alone do not make all underlying
sentences naturally authored. No such examples will be used.

Use Amharic, Hausa and Kinyarwanda instead. The paper's source-construction section
identifies news-text sources and lexical-overlap pairing for these subsets, with
human relatedness judgments. This satisfies the user's real-data requirement
without fabricating replacement text or translating another language's labels.
All other split, duplicate, validation-objective and reporting constraints remain.
The earlier plan is preserved as history, not silently rewritten as executed.

Primary source: [SemRel2024, section3.1.2](https://arxiv.org/html/2402.08638v5).
Original metadata revision3c5941ddffdc9d833a5528f49dd13ed9087c0dba remains pinned.
Original repository/HF card did not specify a dataset license at inspection;
authors explicitly release it for relatedness research. Source text remains
private and is not redistributed or relabeled under the code license.

GPU preflight also found the user's weekly Kaggle GPU quota exhausted. Use free
private CPU kernels, with prospective hardware/config amendments and measured
smokes. The user removed artificial GPU time caps; this is a service constraint.
