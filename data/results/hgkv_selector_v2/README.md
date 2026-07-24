# HGKV selector V2 beam-4

## 状态

`CONTRACT_FROZEN_IMPLEMENTATION_IN_PROGRESS`。

V2 取代 `NO_GO_HGKV_SELECTOR_V1`，主方法为 full-history、empty-start、true-U teacher
beam-4、edge0–edge3、fresh unified set-conditioned student 与 learned beam-4 + STOP。
冻结协议见 [`docs/selector_v2_beam4_protocol.md`](../../../docs/selector_v2_beam4_protocol.md)，
config 见
[`code/configs/hgkv_selector_v2_beam4.json`](../../../code/configs/hgkv_selector_v2_beam4.json)。

## Source of Truth

| artifact | canonical / staging location | status |
|---|---|---|
| V2 code/config/docs/lightweight manifests/results | Git `main` | implementation in progress |
| V2 reusable state/render/score/feature dataset | intended HF dataset repo `gavinlaw/causalcache-hgkv-selector-v2` | `PENDING_HF_UPLOAD` |
| hg-s100 adapter | hyper01 `/data02/jaxan/runs/hgkv-eval/hg-s100.pt`；SHA256 `8f2cc49e1aa0b06ce231eb54937d813317f5274a799c97b09be7fdb22be46317` | local staging；`PENDING_HF_UPLOAD` |
| V1 exact-key cache candidates | hyper01 `/data02/jaxan/runs/hgkv-conditional-scores-v1/` | heldout 38,836 DONE；train 160,928 ABORTED；not yet imported |
| V2 model checkpoint | intended HF model repo TBD | not trained |

HF repo ID 是 intended destination，尚未创建或验证；不得当作已上传链接引用。

## Frozen gate

- B1：V2 beam-4 相对 Recent 不显著劣化；
- B2/B4：分别显著超过 Recent、OCR/RGB Similarity、Random；
- Avg(B1,B2,B4)：显著超过 Recent；
- 全部正式 utility 来自完整 selected set 的 hg-s100 重渲染/重打分和 frozen B0
  exact join；
- offline gate PASS 前不接 AndroidWorld/OSWorld closed loop。

## 当前下一步

1. 生成 longest-real-state full-history train/dev inventory 与 audit；
2. 建立 canonical coalition score cache；
3. 补齐 full-history singleton score/readout；
4. 按 true-U teacher beam-4 生成 edge0–edge3；
5. 训练 fresh unified student 并执行 learned greedy/beam-4 gate。
