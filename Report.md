# Sentence relatedness results

The validation-selected global blend achieves test macro Spearman **0.55807**, compared with **0.50076** for character overlap. Macro MAE is **0.23779** versus **0.29204**. All nine language rank correlations improve; Punjabi is still negative.

| Language | Test pairs | Selected Spearman | Character Spearman | Selected MAE | Character MAE |
|---|---:|---:|---:|---:|---:|
| afr | 375 | 0.79962 | 0.77011 | 0.24640 | 0.31544 |
| amh | 171 | 0.77002 | 0.72738 | 0.19021 | 0.24157 |
| arb | 595 | 0.55576 | 0.50238 | 0.33842 | 0.43616 |
| arq | 583 | 0.55672 | 0.53862 | 0.33432 | 0.44755 |
| ary | 426 | 0.75391 | 0.63185 | 0.14419 | 0.18525 |
| hau | 603 | 0.61322 | 0.53051 | 0.21971 | 0.25145 |
| ind | 360 | 0.53223 | 0.52590 | 0.22259 | 0.22052 |
| kin | 222 | 0.61163 | 0.48825 | 0.18041 | 0.22406 |
| pan | 634 | -0.17052 | -0.20820 | 0.26383 | 0.30634 |

## Data and recipe

The official nine-language compatible cohort has 5,691 training, 1,370 development and 3,969 test pairs. Text-only guards remove 51 training duplicate/protected pairs and 23 development pairs duplicating test; the complete test cohort remains unchanged. All 23 source CSV hashes are checked against revision `3c5941ddffdc9d833a5528f49dd13ed9087c0dba`. This is nine human-source-compatible languages, not the entire 14-language shared task.

Pinned E5 uses 512-token masked mean pooling, `query: ` on both sentences and unit normalization. No sentence in the full cache is truncated. Absolute difference/product/cosine plus word/character overlap give 1,539 symmetric features. Three ranking MLPs use 128 hidden units, dropout 0.2, AdamW LR 0.001/decay 0.01, batch 64, train-only scaling and equal-five-language weights. Ranking weight 0.1 compares human-order gaps of at least 0.1. Seeds 42/1337/2026 stop on development Spearman with patience six; their selected epochs are 2/1/1, after completed fits of 8/7/7 epochs.

The final score uses 25% mean learned prediction and 75% character overlap. Development Spearman rises from 0.45429 to 0.53164. Selection requires +0.005 macro gain and no language decline greater than 0.02; the worst selected change is -0.01706. Higher semantic weights fail that guard. This is a single global policy, with no language-specific fallback or test-driven switch.

## Evaluation and inference checks

All 3,969 test pairs were scored once after the hash-bound policy freeze and real-text proof. The six additional language tests score **0.50462** versus **0.46011**; the three previously observed language tests score **0.66496** versus **0.58204**.

A paired, language-stratified sentence-component bootstrap gives macro Spearman interval **[0.53335, 0.58272]** and model-minus-baseline interval **[0.04624, 0.06943]**, using 1,000 valid resamples. These are conditional descriptive intervals, not independent-document guarantees.

Selected checkpoint SHA256: `43b16caca7c625d5513914da1eed338d40e41e11fd94e10381e2d8bd9859240e`. Actual 18-pair E5 text/reversal inference matches cached predictions within **1.79e-7**, with exact reversal agreement.

The [frozen policy](docs/frozen-policy.json), [training aggregate](docs/training-results.json), [test aggregate](docs/final-test.json) and [text proof](docs/inference-proof.json) provide the exact hashes, histories and framework versions. [README](README.md) gives fresh training and prediction commands.

## Limits

Punjabi test Spearman is -0.17052; the model does not provide useful Punjabi ranking. Indonesian rating MAE is slightly worse despite its rank gain. Development cohorts vary sharply in size and selection was repeated. Sentence reuse connects official partitions, and pretrained exposure is unknown. Scores are not calibrated probabilities; independent-source transfer remains unmeasured.
