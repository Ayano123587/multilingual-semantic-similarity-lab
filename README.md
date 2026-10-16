# Multilingual relatedness models

Sentence-pair scoring for Amharic, Hausa and Kinyarwanda, trained on real human-rated SemRel2024 news text. Frozen AfroXLMR embeddings, symmetric pair features and lexical overlap feed a small regression MLP.

| Model | Development Spearman | Final test Spearman | Test MAE |
|---|---:|---:|---:|
| Character overlap | 0.44296 | 0.58204 | 0.23902 |
| Validation-selected ranking MLP | 0.56289 | 0.54864 | 0.16797 |

The learned model missed the heldout ranking goal. It reduced rating error, but character overlap ranked pairs better in all three test languages. The checkpoint was not replaced after seeing that result.

## Notebooks

- [Model and experiments](notebooks/01_complete_relatedness_workflow.ipynb): visible preprocessing, encoder, model, losses, training loop and scorer; genuine full-data replay; all learned comparisons and curves.
- [Data audit](notebooks/02_data_audit.ipynb): source checks, human rating distributions, language balance, text lengths and sentence-connected groups.
- [Results and inference](notebooks/03_results_and_inference.ipynb): per-language outcomes, uncertainty, development residuals and actual saved-text inference.

All three retain executed outputs and embedded plots. They never reopen final gold or rescore test. No raw news pairs or learned weights are published.

## Run

~~~bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,notebook]"
python -m src.data.make_dataset --prepare --output data/processed/semrel-lf-v1
python scripts/stage_kaggle_source.py --output data/source-staging --processed data/processed/semrel-lf-v1
python train.py --cache data/features/afro-xlmr-v1 --processed data/processed/semrel-lf-v1 --output models/my-heads
python -m src.models.ranking_followup --cache data/features/afro-xlmr-v1 --processed data/processed/semrel-lf-v1 --prior models/my-heads --output models/my-ranking-study
python -m src.models.predict --checkpoint models/my-ranking-study/selected.pt --input data/my-real-pairs.jsonl --output data/my-new-prediction.json
python -m pytest -q
~~~

Use the ordinary readable source files from the staging command with scripts/kaggle_encode.py in a private CPU kernel. It creates the authentic embedding cache and records loading keys, checksums and truncation. Encoder work needs substantial RAM; cached-head fitting needs much less.

Training uses train/development only. It rejects existing output directories and preserves every seed. Final evaluation is a separate action; evaluation_scripts/evaluate.py requires explicit gold release and an immutable selection, then prevents a second release for that selection.

## Sources and limits

[SemRel2024](https://arxiv.org/html/2402.08638v5) supplies the human news pairs. [AfroXLMR](https://huggingface.co/Davlan/afro-xlmr-base) supplies the frozen encoder. Exact revisions, source terms and hashes are in [provenance](docs/provenance.md) and the [experiment ledger](docs/experiment-ledger.json).

Official splits reuse individual sentences and are not sentence-disjoint. Upstream pretraining overlap is unknown. Small development cohorts and repeated selection limit the conclusions; the component-bootstrap rank-difference interval spans zero. [Report](Report.md) gives the recipe, failures and final numbers.
