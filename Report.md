# Relatedness experiment report

The validation-selected ranking MLP reached final macro Spearman **0.54864**, below character overlap's **0.58204**. Rank correlation was lower in all three languages. Macro MAE improved from **0.23902** to **0.16797**. The frozen choice was kept; no test-guided switch was made.

| Language | Test pairs | Model Spearman | Character Spearman | Model MAE | Character MAE |
|---|---:|---:|---:|---:|---:|
| Amharic | 171 | 0.67256 | 0.72738 | 0.15294 | 0.24157 |
| Hausa | 603 | 0.50499 | 0.53051 | 0.19059 | 0.25145 |
| Kinyarwanda | 222 | 0.46838 | 0.48825 | 0.16039 | 0.22406 |

## Recipe and selection

Pinned SemRel2024 revision 3c5941ddffdc9d833a5528f49dd13ed9087c0dba: the prepared cohort contains 3,456 training, 386 development and 996 test pairs after text-only auditing removed 50 Kinyarwanda training duplicates and 23 development pairs duplicating test pairs; the test cohort stayed unchanged. Gold remained sealed during model choice.

Frozen AfroXLMR revision 25f27299c247a6b73a767bd82d12444138b19337 uses attention-mask mean pooling, no random pooler, unit embeddings and max 128 tokens. No encoded sentence was truncated. Symmetric absolute difference/product/cosine plus word/character overlap yield 1,539 features. Scaling and loss weights use training only.

The first study trained ridge penalties 1/10/100 and nine MLP heads. Equal-seed control MLP reached development Spearman 0.55548 versus character overlap's 0.44296; ridge, slower LR and stronger regularization failed to improve it.

A separately frozen follow-up trained nine more heads with ranking-loss weights 0.03/0.1/0.3. Architecture stayed 128 hidden units, dropout 0.2, AdamW LR 0.001/decay 0.01, batch 64, gradient clip 1, validation checkpointing and plateau stopping. Human rating order within language/minibatch supplied ranking terms. Weight 0.1 reached 0.56289 and met the +0.005 macro/no >0.02 language-regression rule. Hausa development rank fell by 0.01067; that was within the guard.

## Checks and uncertainty

The selected checkpoint is models/selected-v1/selected.pt, SHA256 293b99ca441829ba9cf36a60ce118384e54c0af8cf791dc222ea7e6eb286fc7d. Actual text inference matched cached predictions within 1.2e-7; reversed pairs matched exactly. The main notebook repeats one fixed selected fit and verifies its recorded state.

Test gold was parsed once after global policy freeze and real-text proof. Selection hashes stayed unchanged. Language-stratified sentence-component bootstrap used 1,000 resamples: model macro Spearman interval [0.49273, 0.60546]; paired model-minus-character interval [-0.09107, 0.02523]. This does not establish a rank gain.

[Ledger](docs/experiment-ledger.json) retains run versions, protocol hashes, source correction, failed comparisons and exposure timing. Both studies' learning curves remain in their aggregate selection reports.

## Limits

Hausa has a 1,364-sentence connected component crossing splits. Pair-copy removal does not make evaluation sentence-disjoint. Pretraining overlap is unknown, development cohorts are small, and the bootstrap is conditional on available groups.

The first encoder attempt failed before model loading because Windows CRLF bytes differed from Linux LF; canonical LF fixed the hash contract without changing any row. GPU quota was exhausted, so encoding used cloud CPU and heads used local CPU with different framework versions. Encoder fine-tuning was not run. These results claim neither an optimum nor independent document/location transfer.
