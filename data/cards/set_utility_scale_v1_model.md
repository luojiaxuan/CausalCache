# CausalCache Set Utility Predictors — Scale v1

Exploratory CPU-trained predictors for `n=4, |S|<=2` restoration utility.

- Git source: `CausalCache@ac0ef2766a7aac0d44f858070d23cda17849e787`
- Training data: 307 train / 24 tune states
- Development evaluation: 24 states
- Families: pairwise-additive, DeepSets, Set Transformer
- Config: `code/configs/causalcache_set_utility_scale_v1.json`

Best development result is DeepSets under fixed-B search at B2: exact recovery 0.8594 versus OCR/RGB 0.8541. This is an exploratory checkpoint, not a validated production model or a formal paper result.

