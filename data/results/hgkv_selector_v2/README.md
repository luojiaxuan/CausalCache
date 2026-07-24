# HGKV selector V2 beam-4

## 状态

`READOUT_DONE_TEACHER_EDGE1_RENDER_DONE_SCORING_PREPARING`。

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
| V2 coalition score cache | hyper01 `/data02/jaxan/runs/hgkv-selector-v2/coalition-cache/cache-singletons-complete.jsonl`；[`coalition-cache-manifest.json`](coalition-cache-manifest.json) | singleton-complete cache 249,467 exact keys；`PENDING_HF_UPLOAD` |
| V2 missing-singleton render/score | hyper01 `/data02/jaxan/artifacts/sft/hgkv-selector-v2-singletons/`；[`singleton-render-manifest.json`](singleton-render-manifest.json) | 6,012/6,012 rendered and scored；full inventory 13,680/13,680 |
| V2 candidate readout | hyper01 host `/data02/jaxan/runs/hgkv-selector-v2/readout/`（固定容器内 mount 为 `/data/runs/hgkv-selector-v2/readout/`）；[`readout-manifest.json`](readout-manifest.json) | 13,680/13,680 unique、1285 dims、finite/temporal/full-history validator PASS；`PENDING_HF_UPLOAD` |
| V2 true-U teacher beam | hyper01 `/data02/jaxan/runs/hgkv-selector-v2/teacher-beam-v2/` and `/data02/jaxan/artifacts/sft/hgkv-selector-v2-teacher/` | edge0 reduced；edge1 36,938/36,938 coalitions rendered and strict-validated；scoring pending |
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

1. 对已渲染的 36,938 个 edge1 coalition 打分，再按 true-U beam-4 迭代 edge2/edge3；
2. 对 development `n<=8` 子集完整枚举到 B4，报告 beam recovery/regret/Jaccard/utility
   gap；
3. 训练 fresh unified student 并执行 learned greedy/beam-4 gate。

## Full-history singleton render

source commit：
`7056fec474ef8e3813887785c8f0e5a0a54c4de2`。缺失项按 V2 inventory 与 initial
coalition cache 的 exact complement 渲染，train/dev 分别为 5,143/869，共
6,012 行。

- required/cached/rendered/complete：`13,680/7,668/6,012/13,680`；
- duplicate/unexpected/missing：`0/0/0`；
- referenced images：6,678，全部存在并完成 SHA256；
- image manifest SHA256：
  `c49654b92ee42718c99adc4876ce0c63f25e9b217929f4bec167bec40ba1e4d4`；
- hyper01 完整 `DONE` SHA256：
  `e30e39232d72e596a040fcafa3b2745d48f9b2ee71c51d143d8fe071bd71bf82`。

12-way hg-s100 scoring 已在固定容器 `sglang-omni-jaxan` 完成：train/dev
`5,143/869`，合计 `6,012/6,012`，全部 worker exit 0。重建后的 singleton-complete
cache 为 `249,467` rows/unique keys，V2 B0 `1,000/1,000`、singleton
`13,680/13,680`，缺失为 0；cache SHA256 为
`5578f5dd477996a4f9bfab12bca4b8d44d7f177e71d0d3ad488a679e484de4e3`。

## True-U teacher 与 exact-search validation

edge0 已从完整 singleton cache 归约。edge1 计划在 exact-key 复用后剩余 36,938 个
unique pair coalition；32-way CPU render 已完成，strict validator 得到
expected/observed/unique=`36,938/36,938/36,938`，
duplicate/unexpected/missing=`0/0/0`。13,920 张引用图片全部存在，image manifest
SHA256 为
`d8fb299913328ebf222e7b702741d17b480a3c29e82b8f758323abbf5c65467d`。

新增 `plan_hgkv_exact_search_v2.py` 与 `reduce_hgkv_exact_search_v2.py`，只对
development 且 `n<=8` 的冻结子集枚举 set size 1--4，并复用同一 canonical
exact-key cache、renderer 和 scorer。归约器以 at-most-B true utility 生成 exact
oracle，逐 B1/B2/B4 报告 teacher beam-4 recovery、regret、selected-set Jaccard 和
utility gap；该诊断不允许改变已冻结的 beam width。

## Full-history candidate readout

V1 feature cache 仅按冻结 `(pair_group,event_step_id)` exact key 复用 7,668 行；
其余 train/dev `5,143/869` 均由 hg-s100 fresh 抽取并分别生成 `DONE`。统一 validator
最终得到：

- expected/observed/unique：`13,680/13,680/13,680`；
- feature dims：`[1285]`（1280 HGKV + 5 temporal）；
- full-history `n>8` states：666；
- recent-8 truncation：0；
- 所有行长度、有限值与按 state 重算的 temporal feature 全部一致。

完整文件 SHA 见 [`readout-manifest.json`](readout-manifest.json)；remote `DONE`
SHA256 为
`c1c0b500e6c6b02be9fcbe20b3a33890d0f7a600e49ee25397bdf12a36def59b`。
