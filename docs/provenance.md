# Data and model sources

[SemRel2024](https://arxiv.org/html/2402.08638v5), sections 3.1.2 and 3.2, describes Amharic, Hausa and Kinyarwanda news pairs rated by native speakers using best-worst scaling. Relatedness includes topical association, beyond paraphrase equivalence. No generated text, translated augmentation or pseudo-label dataset enters this project.

The [official repository](https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024) is pinned to 3c5941ddffdc9d833a5528f49dd13ed9087c0dba. Nine CSV SHA256 values are enforced in src/config.py. The inspected source did not provide an explicit corpus license; authors release it for relatedness research. News text, ratings, caches and learned artifacts remain private. The source-code MIT license does not relicense them.

[Davlan/afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base/tree/25f27299c247a6b73a767bd82d12444138b19337) is pinned to 25f27299c247a6b73a767bd82d12444138b19337. Its card declares MIT and MLM adaptation on African languages including all three studied languages. It is not a sentence-similarity-fine-tuned encoder. Actual weights SHA256: f73b04316c9c2e76fcb213957616a09e1eb44112266717f649fc1897dac42953.

The loader disables the pooler and refuses missing/mismatched backbone keys. Only unused MLM-head keys remain. Attention-mask mean pooling and L2 normalization are recorded in the encoder receipt. Unknown pretraining exposure and sentence reuse remain evaluation limits.
