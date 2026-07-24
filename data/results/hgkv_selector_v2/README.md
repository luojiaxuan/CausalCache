# HGKV selector V2 beam-4

## 状态

`INVENTORY_AND_INITIAL_EXACT_CACHE_DONE_IMPLEMENTATION_IN_PROGRESS`。

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
| V1 exact-key cache candidates | hyper01 `/data02/jaxan/runs/hgkv-conditional-scores-v1/` | heldout 38,836 DONE；train 160,928 ABORTED；exact hits imported |
| V2 coalition score cache | hyper01 `/data02/jaxan/runs/hgkv-selector-v2/coalition-cache/cache.jsonl`；[`coalition-cache-manifest.json`](coalition-cache-manifest.json) | 243,455 exact keys；`PENDING_HF_UPLOAD` |
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

## Initial canonical coalition cache

source commit：
`d1113cc961e651d414f5f3b43dcd69b46930c4ac`。Cache exact key 由
`pair_group/restored_set_key/target_action_sha256/prompt_revision/hgkv checkpoint
SHA/B0 policy SHA` 组成；冲突 fail closed。

- cache rows / unique keys：243,455 / 243,455；
- frozen B0 empty sets：10,680；
- V2 1,000 states B0 coverage：1,000/1,000；
- V2 full-history required singletons：13,680；
- V1 exact-key singleton hits：7,668；
- missing full-history singletons：6,012；
- singleton 已全覆盖 states：334/1,000；
- V2-state cache 中 set-size 0/1/2/3：1,000 / 7,668 / 10,732 / 5,372。

cache JSONL SHA256：
`b3bb9856eaa9cd32cd987a9507d6f769213cb0b05e2a61661e64082f0a6805b7`；
Git manifest 与 hyper01 SHA256 均为
`1c6b052cd5340d4fc7ba59ee30c68e03678a72e92db499ad776e94e00cecb152`。
B0 policy identifier 固定为 GUI-Owl snapshot manifest SHA256
`50b675ec31c5c46dbb0d44c137a808fffb9d054916d39b596648d4eb9df7cbc3`。

## 当前下一步

1. 渲染并打分缺失的 6,012 个 full-history singletons，同时补 readout；
2. 按 true-U teacher beam-4 生成 edge0–edge3；
3. 训练 fresh unified student 并执行 learned greedy/beam-4 gate。
