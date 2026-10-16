# Multilingual sentence relatedness

A sentence-pair scorer trained on human-rated SemRel2024 text. A pinned multilingual E5 encoder feeds three trained ranking MLPs; their mean contributes 25% of the score and Unicode character overlap contributes 75%. One global rule serves the nine evaluated languages.

| Model | Development Spearman | Test Spearman | Test MAE |
|---|---:|---:|---:|
| Character overlap | 0.45429 | 0.50076 | 0.29204 |
| Validation-selected blend | **0.53164** | **0.55807** | **0.23779** |

The run includes every available official split from nine verified human-source-compatible languages: 5,640 training, 1,347 development and 3,969 unchanged test pairs after text-only duplicate guards. Five languages provide training; four are cross-lingual evaluation only. The shared task has 14 languages. Five mixed generated/translated-source collections are excluded in full. [Source audit and terms](docs/provenance.md) explain the boundary.

## Notebooks

- [Complete workflow](notebooks/01_complete_relatedness_workflow.ipynb): visible data, encoder, model, losses, training loop and scorer; actual source preparation, gradient updates, checkpoint replay and training/selection curves.
- [Data audit](notebooks/02_data_audit.ipynb): all source checksums, complete split counts, human ratings, scripts and sentence-connected groups.
- [Results and inference](notebooks/03_results_and_inference.ipynb): per-language evaluation, component uncertainty, development errors, seed variation and saved-model text/reversal proof.

The notebooks retain executed outputs and embedded plots. Replay requires locally prepared caches and trained checkpoints. They read the saved test aggregate without reopening gold or repeating test scoring. Raw text, embeddings and weights remain ignored.

## Train

Run from this repository in a separate virtual environment; the projects share a `src` package name.

```bash
python -m pip install -e ".[dev,notebook]"
python -m src.data.full_cohort --raw data/raw/semrel-full-v3 --output data/processed/semrel-full-v3
python scripts/build_embeddings.py --variant afro --output data/features/embeddings-afro
python scripts/build_embeddings.py --variant e5 --output data/features/embeddings-e5
python train.py --cache-afro data/features/embeddings-afro --cache-e5 data/features/embeddings-e5 --processed data/processed/semrel-full-v3 --output models/my-heads
python scripts/freeze_score_prior.py --prior models/my-heads --cache-afro data/features/embeddings-afro --cache-e5 data/features/embeddings-e5 --output data/reports/my-prior-protocol.json
python train.py --cache-afro data/features/embeddings-afro --cache-e5 data/features/embeddings-e5 --processed data/processed/semrel-full-v3 --prior models/my-heads --protocol data/reports/my-prior-protocol.json --output models/my-selected
python -m pytest -q
```

Use fresh output paths. Encoder work needs substantial RAM; the measured encoding/training run used private cloud CPU with two threads. [Plain-file cloud staging](scripts/stage_full_cohort.py) and the [readable runner](scripts/kaggle_sentence_comparison.py) support cloud execution. No private artifact is required for a cold-start run. Freeze a follow-up protocol for each fresh set of training artifacts.

## Inference

Input JSONL rows contain `sentence1` and `sentence2`, without labels:

```bash
python -m src.models.full_inference --checkpoint models/my-selected/selected.pt --input data/my-real-pairs.jsonl --output data/my-new-scores.json
```

The privately retained selected artifact is `models/selected/selected.pt`. Real-text inference matched cached scores within 1.79e-7; reversed pairs matched exactly. Final evaluation is separate and requires an explicit hash-bound freeze, inference proof and one-shot gold release; it is not part of training or notebook execution.

## Limits

Punjabi remains negatively correlated, and Indonesian MAE slightly increases. Official splits share sentences, upstream pretrained exposure is unknown, and development selection was repeated. Three language tests were already observed; six additional language tests were released once after the final policy freeze. The pooled score is not wholly untouched.

[Report](Report.md) gives per-language results and conditional uncertainty. [Reproducibility files](docs/frozen-policy.json) bind the selected model, source, caches and protocol. Code is MIT; upstream corpus terms are separate. No raw dataset or model weights are redistributed.
