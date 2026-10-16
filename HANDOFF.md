# Current STS research checkpoint

Owner: leakage_toolkit subagent; root publishes. Branch codex/semantic-relatedness.
Preserved root source and immutable original protocol commit 10a8d70; no candidate
score, encoder fine-tuning, or final test result is claimed yet.

Private CPU kernel `aikyatansinha/portfolio-semrel-african-encoder-v1` version 2
is RUNNING. Plain runner scripts/kaggle_encode.py mounts ordinary readable source
files from private dataset `aikyatansinha/portfolio-semrel-african-source-v1`,
expected version 2/source revision 3cc2934. Launch manifest hashes in docs.
No local encoder or GPU job allocated. GPU weekly quota was already exhausted.
Root refreshed existing OAuth cache; no credentials copied, printed or committed.

Version 1 failed before encoder loading with processed hash mismatch: Windows
CRLF versus Linux LF. All raw official CSV hashes passed; canonical LF preparation
matches cloud bytes exactly and parsed contents match every original row. Old
ignored `data/processed/semrel` preserved, new canonical input under ignored
`data/processed/semrel-lf-v1`. Failure/correction receipts retained in docs.

Implemented exact frozen ridge alphas and nine actual MLP fits, language-balanced
MSE, train-only normalization, seed-ensemble comparison and promotion criteria.
Training has not started pending authentic encoder cache. Explicit saved-text
inference checks weights-only head/schema and actual pinned encoder weight bytes.
Separate final evaluation verifies serialized selection before first human test
score parsing and refuses existing reports. Bootstrap is language-stratified by
label-free sentence-connected components. Official split is not sentence-disjoint:
Hausa largest component 1,364 sentences; sentence reuse is not itself copied pair
target leakage. See docs/component-audit.json.

Tests: prior complete model suite 19 passed; canonical LF data regression 11 passed;
full suite including new regression currently running (exec session 96913).
Full notebook JSON/AST and all nine default lightweight cells executed without
training, encoder allocation, network fetch or data writes. No fake outputs.
Audio full-notebook commits f52b95f and README link b691067 ready for root to publish.

Next: collect cloud smoke/environment/cache receipts; confirm actual source revision
and loading keys, finite/unit embeddings and truncation. Retrieve private embeddings
to data/features/afro-xlmr-v1. Run root train.py on canonical inputs with a fresh
models directory. Inspect validation only; prospectively register any justified
follow-up before test release. Serialize final winner, evaluate once, bootstrap,
run actual saved text inference in cloud CPU, save aggregate evidence, update
Report.md/README/notebook links and deliver clean validated commits. Root handles
publication/CI and user goal scope. Do not fabricate missing results or allocate
the 1 GB encoder on the 8 GB host.
