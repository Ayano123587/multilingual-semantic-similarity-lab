# Private learned artifacts

The validation-selected release is selected-v1/selected.pt, SHA256 293b99ca441829ba9cf36a60ce118384e54c0af8cf791dc222ea7e6eb286fc7d. It averages three ranking-weight 0.1 MLP heads and uses the pinned AfroXLMR source/normalization.

All encoder weights, checkpoints, caches and raw text remain ignored. The character-overlap comparison is selected-v1/baseline.pt. No test-guided replacement was made. Reproduce the studies through train.py and src.models.ranking_followup; recipes and results are in Report.md.
