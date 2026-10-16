# Multilingual Semantic Similarity Lab

An original project inspired by [Aditya's semantic-similarity work](https://github.com/tanalpha-aditya/Advanced_Semantic_similarity).
The task is graded **semantic relatedness**, which includes topical connections
and is broader than strict paraphrase similarity.

Research is in progress. Reproducible preparation, encoder caching, trained
ridge/MLP comparisons and guarded inference/evaluation are implemented. Cloud CPU
encoding is running; no trained result or gain is claimed yet. Use real human-annotated
SemRel2024 Amharic, Hausa and Kinyarwanda news-text pairs from the official task repository.
No generated translations, generated sentences, or pseudo-labels enter measured
training or evaluation. Models and data remain private/ignored.

The implementation compares lexical/pretrained baselines and learned symmetric
pair regressors using validation selection
before a single final test. Source and duplicate audits accompany every result.
[Prespecified research plan](docs/experiment-plan.md) records the initial scope.
[Source-audit amendment](docs/source-audit-amendment.md) excludes language subsets
with generated paraphrases before any model training or scoring.

Repository organization follows the user's reference: reusable code in `src/`,
root `train.py`, explicit `evaluation_scripts/`, exploration in `notebooks/`,
aggregate research evidence in `docs/` and `Report.md`. CI/tests and installable
metadata accompany the completed pipeline. This status will be updated after
actual execution; planned work is never reported as finished.

Start with the [complete executable workflow notebook](notebooks/01_complete_relatedness_workflow.ipynb).
It includes authentic source setup, component diagnostics, optional fresh training,
saved-model inference and result plots. Expensive cells are disabled by default;
production algorithms remain in Python modules. No fake outputs or automatic test
rescoring are included.

For a terminal workflow on Python 3.11 or later:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,notebook]"
python -m src.data.make_dataset --prepare
python scripts/stage_kaggle_source.py --output data/source-staging
python train.py --cache data/features/afro-xlmr-v1 --processed data/processed/semrel --output models/my-fresh-study
python -m src.models.predict --checkpoint models/my-fresh-study/selected.pt --input my-real-pairs.jsonl --output data/prediction-fresh.json
python -m pytest -q
```

The source-staging command creates ordinary readable files for a private versioned
Kaggle dataset; use `scripts/kaggle_encode.py` as the plain cloud CPU runner to
produce the authentic cache. No encoded source bootstrap is used. Encoder work
requires substantial RAM; use cloud CPU on a limited-memory host. Training opens
train/development only and refuses existing output directories. All three seed
checkpoints and histories are retained.

Final test release is a separate intentional action through
`python evaluation_scripts/evaluate.py --help`; it requires
`--release-test-gold`, verifies serialized selection and source hashes, and refuses
an existing final report. No final test gold has been parsed yet. See
[provenance and licensing](docs/provenance.md), the
[component audit](docs/component-audit.json), and the
[preparation correction](docs/line-ending-preflight-correction.json).
