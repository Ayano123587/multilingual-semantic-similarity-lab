# Prospective multilingual relatedness study

Source: [official SemRel2024 repository](https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024),
commit `3c5941ddffdc9d833a5528f49dd13ed9087c0dba`, supervised Track A English,
Marathi and Telugu train/development/test files. The published task uses human
relatedness judgments. Verify construction and preserve source citations. No
machine-translated augmentation, generated text, generated labels or arbitrary
synthetic datasets are permitted. Original data license terms remain separate
from source-code licensing; no dataset text is redistributed.

Before any model scoring, attest source bytes and audit normalized unordered pair
duplicates across splits. Exclude train copies of validation/test pairs without
consulting heldout labels. Record sentence-level overlap and connected components,
without falsely treating a shared individual sentence as a leaked pair target.
Official development labels select procedures; all final test gold stays sealed.
Preparation may stage metadata/text programmatically for integrity checks, but
no test score or human test-case analysis guides training.

Primary objective: mean per-language Spearman correlation, equal language weights.
Also report each language's Spearman/Pearson, MAE and sample count. Baselines:
character/word overlap, pinned multilingual pretrained cosine, train-label mean.
Learn symmetric pair features from pretrained sentence embeddings with train-only
scaling and regularized regression or an MLP. Compare fixed seeds 42/1337/2026;
choose configurations by the deployed seed-ensemble validation result, never the
best individual seed. A promotion needs at least 0.005 macro-Spearman gain and no
language falling by more than 0.02. Keep all negative results and the baseline if
no candidate qualifies. Freeze detailed model/grid/training instructions after
source/preflight, before scoring, and prospectively register later hypotheses.

Dataset is relatedness, not strict synonymy, entailment, or a calibrated semantic
probability. Unknown upstream pretraining overlap and sentence reuse remain
limitations. These are portfolio research results, not an official competition
submission or a claim of globally maximum performance. Free private Kaggle
compute is authorized; service quotas and two concurrent GPU slots are respected.
