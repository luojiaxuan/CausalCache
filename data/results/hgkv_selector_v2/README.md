# HGKV selector V2 beam-4

## 状态

`FULL_HISTORY_INVENTORY_DONE_IMPLEMENTATION_IN_PROGRESS`。

V2 取代 `NO_GO_HGKV_SELECTOR_V1`，主方法为 full-history、empty-start、true-U teacher
beam-4、edge0–edge3、fresh unified set-conditioned student 与 learned beam-4 + STOP。
冻结协议见 [`docs/selector_v2_beam4_protocol.md`](../../../docs/selector_v2_beam4_protocol.md)，
config 见
[`code/configs/hgkv_selector_v2_beam4.json`](../../../code/configs/hgkv_selector_v2_beam4.json)。

## Source of Truth

| artifact | canonical / staging location | status |
|---|---|---|
| V2 code/config/docs/lightweight manifests/results | Git `main` | implementation in progress |
| full-history state inventory | [`data/manifests/hgkv_selector_v2_train_states.jsonl`](../../manifests/hgkv_selector_v2_train_states.jsonl)、[`data/manifests/hgkv_selector_v2_dev_states.jsonl`](../../manifests/hgkv_selector_v2_dev_states.jsonl)、[`inventory-audit.json`](inventory-audit.json) | `DONE`；train/dev 835/165 |
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

## Full-history state inventory

source commit：
`97578b2c3ae59f0172689623415de10f566df23f`。Hyper01 staging：
`/data02/jaxan/runs/hgkv-selector-v2/inventory/`。

| audit | value |
|---|---:|
| successful trajectories / states | 1,000 / 1,000 |
| train / dev states | 835 / 165 |
| candidate min / median / p90 / p95 / max | 5 / 12 / 24 / 27 / 44 |
| candidate count ≥4 | 100% |
| candidate count >8 | 666 states |
| synthetic terminal | 0 |
| recent-8 truncation | 0 |

Action distribution 为 click 912、scroll 17、text 71。训练 low-fidelity schema 按冻结
纪律将 foreground app 保持 `unknown`，因此 app audit 的 1,000 行均为 `unknown`；没有
引入 app-specific feature。

| artifact | SHA256 |
|---|---|
| train states | `9c254298c23037436493bf449e1983975425345abf76f805331c8345face0137` |
| dev states | `3b545dc4de5aec129a919837b66db72712ef8b0e5ee2fb3930f6c2d1515f492c` |
| inventory audit | `aa24ea87fd5c2ce1b97a61ead00048e6bd6879371cc922c5eed17c1f4d3b3a3c` |

三个文件的 hyper01 staging 与 Git working copy SHA256 完全一致。

## 当前下一步

1. 建立 canonical coalition score cache；
2. 补齐 full-history singleton score/readout；
3. 按 true-U teacher beam-4 生成 edge0–edge3；
4. 训练 fresh unified student 并执行 learned greedy/beam-4 gate。
