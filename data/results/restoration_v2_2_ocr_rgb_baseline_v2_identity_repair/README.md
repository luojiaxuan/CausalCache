# Restoration v2.2 OCR/RGB baseline v2 identity repair

本结果是 v1 zero-score identity-scanner failure 的 versioned repair。唯一 execution-contract 变化是：derived
trajectory 每行要求两个相同 `source_id` occurrence，OCR 每行要求一个 `image_member_path`；数量漂移或同一行
值不一致均 fail closed。parent scientific contract、immutable inputs、feature、selection、statistics 与 operation
ceiling 不变。v1 没有产生 scientific payload，因此这里不报告 v1/v2 scientific-value comparison。

本结果只覆盖 frozen primary `n=4,B=2` 的 15 条 train/development states。它使用 archived
`full_spatial_tokens` 和 deterministic 256×256 RGB histogram；没有 OCR inference、模型加载、GPU 或 policy
forward。confirm/test semantic access 与 feature score 均为 0。

| Split | States | Mean recovery | At-most-B exact match | Exact-B match |
| --- | ---: | ---: | ---: | ---: |
| Train | 10 | 0.771189 | 0.300000 | 0.300000 |
| Development | 5 | 0.019571 | 0.000000 | 0.000000 |
| Overall | 15 | 0.520650 | 0.200000 | 0.200000 |

- source commit: `a9bede85ab8bd10623c5755b944b3c26865c6485`
- repair contract SHA256: `d68cb032ef3c56c330d57329507d409b20f878b7510bd09d88e2eff1f3f898f3`
- parent contract SHA256: `08f57505d71e603d81d6915ef27cf008d0fe097a56d70a05f8b3519211e2e6f9`
- failed-attempt commit: `2870d8ae26542a184647e8b6d97b8c79e4e12641`
- scientific payload SHA256: `5942519bdff8f3e8a64bbc8b32a5d42a64abff37daf0804fcce0f098a63765b2`
- state score rows: `15` (`07e29538232620bbea3bb12d1c0ce2649ee505285f53daf006dea3f670ba2d0a`)
