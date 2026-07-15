# Restoration v2 科学契约

## 当前决定

v1 的 `NO_GO_CURRENT_REFERENCE_STACK` 保持不变：它否定的是 UI-TARS 在 expert top-1
executable-match admission gate 下充当 reference 的路线，不是对 CausalCache restoration 假设的检验。

v2 改用冻结策略自身的稳定行为作为 attribution reference。科学配置已经在任何 v2 policy output 前冻结为
[`causalcache_restoration_v2.json`](../code/configs/causalcache_restoration_v2.json)，raw-file SHA256 为
`9b9b78d9e1902d6ba7c648c939809c56fe55cccc17de58d4e6eed8d9ddf746cc`。机器 validator 位于
[`restoration_v2_contract.py`](../code/causalcache/restoration_v2_contract.py)。当前仍没有 v2 policy output、
restoration label 或方法效果结果。

这不是降低 v1 的 50% threshold。v1 估计“与 expert top-1 对齐的 policy behavior 能否作为 reference”；
v2 估计“恢复视觉证据能否恢复冻结策略自身 parseable、finite、repeat-stable 的决策行为”。二者是不同
estimand，结果分别保留。

## Reference 与干预

Primary policy 固定为
`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`，使用
`do_sample=false`、BF16、`max_new_tokens=256` 和 native MobileUse envelope。生成 output 只用于得到
normalized action；teacher forcing 使用 processor 的原生 assistant prefix，随后固定加入非语义 carrier
`Action: Execute the selected mobile action.\n`，再加入 compact deterministic `<tool_call>`。primary KL
只从 `<tool_call>` 首 token 计算到 `</tool_call>` 末 token，carrier 仅作为 native-format context，不进入
distance，也不把 full-history 生成的 target description 泄漏给 coalition。
reference state 必须同时满足：

- 唯一 action 可按 restricted v2 grammar 解析；
- teacher-forced full-vocabulary logits 全部 finite；
- 两次 forward 的 canonical action 完全一致；
- repeat KL 被记录，memory sensitivity threshold 固定为
  `max(1e-4, 10 * mean_repeat_kl)`。

expert alignment 只作为独立报告轴，不能筛 label、development 或 confirm state。离线 GUIOdyssey 不声称
`successful_self_rollout`；没有 pre-output UI target annotation 时，也不事后人工声称 semantic match。

Confirm 每条 trajectory 固定取 decision step 6。history events 为 1--5，其中 event 5 的 post-state 与
current observation 是同一个 artifact observation，因此 event 5 始终保留 strong summary，但不重复添加
截图，也不是 selector candidate。reference 为 events 1--4 的四张 post-state image 加 current
observation；它覆盖该状态每个 non-current historical event，不把 recent-4 冒充 full history。若不同
artifact observations 恰好具有相同像素，它们仍按冻结 event identity 处理，而不是在看到模型输出后去重。

每个 memory 中所有 event 都保留同一份 byte-identical strong low-fidelity summary。把 event 升级为
high fidelity 时只增加一张 post-action state image，不增加 before image或额外 action text：

```text
step_id
action_type
action_argument
foreground_app
screen_text_added
screen_text_removed
screen_change
executor_result
```

因此 restoration gain 只归因于额外视觉证据，而不是文本 action 信息同时发生变化。主预算从四个等成本
candidate 中最多选两个；辅助容量为一个和三个，并另报实际 visual tokens、policy-visible text tokens 与
cardinality-matched ablation。

summary 是 deterministic 但不是固定 token 长度：screen-text added/removed 各保留 spatial order 中前 32
tokens，exact text/open/answer argument 不截断，任何 context overflow 在 policy forward 前记为 `INVALID`。
`foreground_app` 取 source app label、否则 executor package、否则 `unknown`，统一 NFKC/空白/casefold；
`executor_result` 只有 executor record 能产生 `accepted/failed`，不能从像素变化猜测。`screen_change` 是
before/after RGB 在 256x256 bilinear resize 后的 mean absolute difference，阈值固定为 0.005/0.05/0.20。
Pillow/resampling 与 OCR/accessibility 的 exact revision/hash 写入 execution config。
当前 implementation 选择、锁定参数、model/wheel SHA、uncapped OCR record schema 与 golden 进度见
[`restoration_v2_ocr.md`](restoration_v2_ocr.md)。synthetic golden 与 HF immutable model revision 已通过；
real-screen pre-output source/materializer/validator 已冻结，但 6-image OCR output、HF immutable re-download
和完成态 manifest 未闭合前，第 5 项 dependency 仍是 pending。

## Action contract

v2 prompt 只允许 `click`、`long_press`、`swipe`、`type`、`system_button`、`open`、`wait`、
`answer`、`terminate`；system button 只允许 `Back`、`Home`、`Enter`。接受模型 alias
`tap -> click`、`open_app -> open`，删除无法闭合到 executor 的 `key` 和 `Menu`。

normalized coordinate 固定为 `[0, 999]`，pixel 映射为
`floor(value * (extent - 1) / 999 + 0.5)`，必须严格落在有效像素范围。每种 action 和 alias 都要通过
`prompt -> native output -> parser -> canonical action -> AndroidWorld payload -> pinned JSONAction` 的
exhaustive fixture；raw native tool call 和实际 executor payload 都进入 archive。该 fixture 100% 通过前
禁止任何 v2 policy output。

实现层的 exact case、parameter、NFKC、swipe bin、screen-text tokenization 与 prompt ordering 已在
[`restoration_v2_interfaces.md`](restoration_v2_interfaces.md) 冻结，并由
[`restoration_v2_interfaces.json`](../data/manifests/restoration_v2_interfaces.json) 逐文件 hash。当前本地
CPU 检查通过 14 个合法 action、23 个非法 action、6,000 个 coordinate scalar checks 和 steps 4/5/6
共 28 个 coalitions（含全部 16 个 step-6 coalitions）；pinned AndroidWorld `JSONAction` constructor 已对
14/14 payload 通过，证据见
[`restoration_v2_constructor_preflight`](../data/results/restoration_v2_constructor_preflight/README.md)。
device-side executor dispatch 也已对 14/14 cases 正式通过，negative actuation control 为 HTTP 500，见
[`restoration_v2_executor_dispatch`](../data/results/restoration_v2_executor_dispatch/README.md)。action round
trip dependency 已闭合，但其余 pre-output dependencies 仍必须全部完成。

## 数据角色与 exposure

Parent artifact 保持 immutable：
`gavinlaw/causalcache-guiodyssey-independent-mobile@84c9f5a335e9612ccb4bd566f977574f359b2485`。
旧 8-trajectory reference 已有 UI-TARS outputs，只用于 action-contract development。旧 15-trajectory
oracle split 的 raw rows、images 和 expert actions 已被 builder 读取和打包，但尚无本项目 policy 或
restoration output；按原顺序拆为前 10 条 label-train 和后 5 条 development。

Label-train/development 对每条 source trajectory 只取可用的 step 4/5/6，保持 source order，不能 top-up。
Untouched confirm 从 v1 全序中排除 exact 8+15 后继续取首个满足 `decision_count >= 5` 的 20 条
trajectory，每条只取 step 6。不能依据 parse、stability、expert alignment、memory sensitivity 或结果 top-up。
exact IDs、overlap matrix 和 file hashes 必须在 inference 前 materialize。

Parent `manifest.json` 只保存原 8+15 条 selected trajectories，不能从其中恢复完整 111 条 eligible
records。正式 materializer 因而必须对 source-file manifest 中 pinned 16 个 Parquet 逐文件验证
size/SHA 后重跑 v1 inspection/salted order；只有 eligible count=111、pool SHA=`84d685...`、exclusion
counts 与 8/15 顺序全部一致，才允许继续选 confirm。`decision_count=len(actions)-1`，decision steps 为
`2..decision_count+1`，所以 frozen `decision_count>=5` 正好保证 step 6，而不是隐式收紧样本。

selection manifest 同时冻结 train/development/confirm 的 30/15/20 state IDs，以及每个 state 的 current
image、候选 event post-state、current-equivalence 和 validated-action hashes。exposure ledger 使用
append-only evidence events 再 reducer；confirm 会被 deterministic pipeline 读取 raw rows/images/actions，
所以准确术语是 `policy-output untouched`，不能声称 `raw unseen`。materializer/validator 位于
`code/causalcache/data/restoration_v2_selection.py` 与 `code/scripts/`。

Hyper00 canonical formal run 已从 pushed `main@30879c0` 复现完整 pool，并将 exact 20 confirm IDs、65 state
witnesses 与 exposure ledger 冻结到
[`restoration_v2_selection.json`](../data/manifests/restoration_v2_selection.json) 和
[`restoration_v2_exposure.json`](../data/manifests/restoration_v2_exposure.json)。两次全量构建 byte-identical，
两次均通过独立 validator；文件 SHA256 分别为 `292c7e52...` / `bc122482...`。dependencies 2/3 已闭合，
但这不解锁 policy inference：derived artifact、OCR、baselines 与 execution config 仍为 mandatory blockers。

所有 v2 derived summaries、OCR/UI delta、split manifests 和 attribution records 的 canonical destination 是
private HF dataset `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile`。在它获得 immutable revision、
从 revision 重下载验 hash，并把 revision 回写 Git 前，不允许运行 v2 policy。Git 只保存 config、代码、
exposure ledger、轻量 manifest/result 和进展。

## 两级 go/no-go

先只在 label-train/development 运行 substrate screening，至少 20 states：

- parse coverage 至少 0.99；
- finite-logit coverage 为 1.0；
- repeat canonical-action agreement 为 1.0；
- memory-sensitive states 至少 8。

失败输出 `NO_GO_V2_SUBSTRATE`，不会打开 confirm。通过后才运行固定 20-state confirm，primary capacity
`B=2`、`K=16`、seed `20270715`。通过 restoration gate 需要：

- memory-sensitive states 至少 8/20；
- mean oracle recovery 至少 0.30；每状态 recovery 固定为
  `(D(empty)-D(selected))/max(D(empty), epsilon)`，并对全部 20 个固定状态取均值；
- 相对 strongest baseline 的 mean gain 至少 0.10，且 paired 90% bootstrap lower bound 大于 0；
- `K=16` 相对 exact attribution 的 median Spearman 至少 0.80；
- median top-budget Jaccard 至少 0.75；
- median sampled-selector / exact-attribution-selector utility ratio 至少 0.90。

Oracle 在四个 candidate 的 0/1/2-event subsets 中找最小 distance；exact-2 另作 cardinality-matched
ablation。strongest baseline 定义为预注册 non-oracle baselines 中
dataset-mean recovery 最大者。paired bootstrap 同时比较 oracle 与每个 non-oracle baseline，所有 90%
lower bounds 都必须大于 0。

Baselines 的实现也在 output 前锁定：recent 取 events 3/4；random 是全部 2-of-4 subsets 的解析期望；
OCR+RGB score 是 OCR-token set Jaccard 与 16x16x16 joint-RGB-histogram cosine 的 0.5/0.5 加权；policy
vision score 是 spatial merger 后最后 visual output 的 mean-pool、L2-normalize、cosine。相似度均取 top 2，
tie 按较小 event step。exact formula source hashes 必须进入 execution config。

稳定性只在 memory-sensitive confirm states 上聚合并取 deterministic median。score ties 使用
`rel_tol=1e-9, abs_tol=1e-12` 和较小 step ID；Spearman 使用 average ranks，双方 constant 且选择集合相同
记 1、仅一方 constant 记 0；Jaccard 的 empty/empty 记 1。utility denominator 不超过 epsilon 时，双方
utility 都不超过 epsilon 记 1，否则记 0。

全部 pass 条件成立才输出 `GO_TO_GATE_TRAINING`。memory-sensitive 少于 4，或 mean oracle recovery
不高于 0.10，输出硬 `NO_GO_RESTORATION_V2`；其余未通过 pass 条件的合法 run，包括 confirm parse/
stability failure，输出 `INCONCLUSIVE_V2`。contract/runtime 错误为 `INVALID`。即使 offline oracle
通过，也只允许进入 gate training，不等于论文主张成立。

## Policy output 前的完成条件

以下八项缺一不可：

1. derived artifact immutable HF revision 与逐文件 hash；
2. exact confirm trajectory/state IDs（passed：`292c7e52...`）；
3. exposure ledger（passed：`bc122482...`）；
4. restricted action exhaustive round-trip fixture；
5. pinned accessibility/OCR backend revision 与 model hash；
6. baseline exact specification 与 source hashes；
7. v2 prompt/parser/bridge/runtime source hashes；
8. 引用本科学配置 SHA 的 execution config，并冻结 microbatch size。

验证命令：

```bash
make validate-restoration-v2 validate-restoration-v2-interfaces \
  validate-restoration-v2-selection
```

任何 v2 output 出现后不得修改 scientific fields。硬件、容器、batch size 与 source hashes 只写入单独
execution config；若 action semantics、data selection、reference、budget、distance 或 gate 改变，必须另立
versioned protocol，不能覆盖 v2。
