# Data 目录

本目录只保存适合 Git review 的小数据与轻量实验记录，不是大型 artifact 仓库。

```text
data/
├── fixtures/    # 单测和 contract 使用的最小确定性 fixture
├── manifests/   # 轻量 source/artifact index，不含 raw screenshots
└── results/     # README、summary.json、轻量 CSV/manifest
```

以下内容禁止作为唯一副本留在 Git 或共享机器本地盘：raw screenshots、完整 rollout traces、生成
dataset、model weights、checkpoint、adapter。它们以 Hugging Face dataset/model repo 为 canonical
source，并在顶层 `README.md` 和相关 result README 中记录 repo、revision/tag、schema 与生成命令。

`data/cache/`、`data/raw/`、`data/staging/`、`data/local/` 与 `data/tmp/` 已被 Git 忽略，只能用于
本地短期 staging。注意：仓库相对路径 `data/` 与容器持久挂载点绝对路径 `/data` 是两个不同概念。

当前 v2 interface fixtures：

- `fixtures/gui_owl_v2_action_roundtrip.json`：14 个合法、23 个非法 native action cases，覆盖全部
  canonical actions、aliases、system buttons 与 coordinate endpoints；
- `fixtures/restoration_v2_prompt_low_fidelity.json`：synthetic step-6 state，固定 events 1--5、四个
  candidates、八字段 summaries 与 current-equivalent event 5；同一 prefix fixture 还覆盖 development
  steps 4/5，validator 共穷举 28 个 coalitions，其中 step 6 为 16 个；
- `manifests/restoration_v2_interfaces.json`：上述 fixtures、v2 action/prompt/LF code 与接口说明的逐文件
  size/SHA256。它只证明本地 CPU interface validation；真实 AndroidWorld 运行状态由独立 result evidence
  给出，不事后改写 source manifest 的 pre-evidence sentinel。

冻结后的 constructor run 证据位于 `results/restoration_v2_constructor_preflight/`：14/14 payload 已被
pinned `JSONAction` 接受。formal device-side result 位于 `results/restoration_v2_executor_dispatch/`：14/14
cases 与 negative control 已通过独立 reducer。interface manifest 的 `pending` 是运行前 source snapshot，
不被事后改写；实际 run status 以 result summary 为准。

restoration v2 exact selection/exposure 产物：

- `manifests/restoration_v2_selection.json`：111-trajectory eligible pool、8/15/20 role proof、exact
  30/15/20 states 与 65 个 content witnesses，SHA256 `292c7e52...`；
- `manifests/restoration_v2_exposure.json`：append-only pre-output exposure evidence，SHA256
  `bc122482...`；
- `results/restoration_v2_selection/`：Hyper00 runtime、exact confirm IDs、失败 attempt 记录与两次
  byte-identical 全量构建结论。

当前 reusable artifacts：

- Independent GUIOdyssey gate：private HF dataset
  `gavinlaw/causalcache-guiodyssey-independent-mobile@v0.1.0`
  (`84c9f5a335e9612ccb4bd566f977574f359b2485`)；schema v0.4，reference 8 trajectories/75 decisions，
  oracle 15/132；
- Independent UI-TARS reference run：同一 private HF dataset
  `@reference-gate-v1` (`b3e1245c6c6a1723fe2ca3a861148008df39df46`)；raw per-decision summary 与
  GPU monitor 位于 `runs/independent-reference-gate-v1/`，immutable re-download verified；

- GUIOdyssey pilot：private HF dataset
  `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- GUI-Owl AndroidWorld validation traces：private HF dataset
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0`
  (`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)；旧 Instruct artifact 保留在 immutable
  `v0.1.0`。

最新轻量运行记录：

- `results/restoration_v2_selection/`：Hyper00 formal selection/exposure 通过独立 validator；20 条
  confirm trajectories 已按 fixed first-20/no-top-up 规则冻结，未加载 policy 或使用 GPU；

- `results/restoration_v2_executor_dispatch/`：Aries formal attempt 通过 14/14 cases，negative actuation
  control 为 HTTP 500，canonical verdict 为 `PASSED_EXECUTOR_DISPATCH`；未加载 policy 或使用 GPU；

- `results/independent_reference_gate_v1/`：正式独立 reference 得到 69/75 parsed、27/75 match、swipe
  0/2，合法输出 `NO_GO_CURRENT_REFERENCE_STACK`；oracle split 未运行；

- `results/ui_tars_hyper00_hardware_anchor/`：Hyper00 H200 精确复现旧 A6000 UI-TARS 9/9 parsed、4/9
  executable-match vector，允许独立 reference gate 留在 Hyper00；

- `results/gui_owl_1_5_8b_think_smoke_strict/`：Hyper01 首次 1/5-image Think checkpoint smoke；
  finite logits 通过，strict parser 因闭合 thinking prefix 按设计拒绝。
- `results/gui_owl_1_5_8b_think_smoke/`：format-only parser 适配后的原参数重跑；finite logits
  与 parse 均为 2/2，interface smoke 通过。
- `results/gui_owl_1_5_8b_think_androidworld_validation/`：Aries 42-checkpoint 数学 early-stop；
  parse 512/513，但 official-success 上界 29/62，candidate 被有效拒绝。
