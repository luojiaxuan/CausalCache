# CausalCache

**Restoration-Guided Memory for Long-Horizon GUI Action Prediction**

目标会议:AAAI。**当前主线 = History-Gated KV Adapter**(`exp/history-gated-mainline-v1`,
冻结契约见 [`docs/history_gated_mainline_v1.md`](docs/history_gated_mainline_v1.md)):在完全冻结
GUI-Owl 原始参数的条件下,只新增一个解释恢复历史视觉证据的 KV 接口(最后 8 层 k/v_proj、
mask 门控 residual、B0 逐位等价),用 benchmark-external 的修正版 GUI-Odyssey 训练,
零样本迁移 AndroidWorld 与 OSWorld。**paper 全线零样本**:端到端主表用两个 panel
分解科学问题——Panel A 固定 Recent,比较 Frozen / Full-layer LoRA /
Top-8 ungated KV LoRA / HGKV 在 B0/B1/B2/B4/B8 下的 history-use;Panel B 固定 HGKV,
只在 selector 契约支持的 B1/B2/B4 比较 Recent / marginal / set-conditioned,并报告
selected-4 相对 Recent-8 的差值。
AndroidWorld 报**全量 116-template × 2 fixed instances**的 template-macro success,
OSWorld 报 full fixed roster 的 mean normalized task score。全部 adapter/selector 仅在
GUI-Odyssey 训练与选择;未完成单元格在论文中显式标 `TBD`。此前在 AndroidWorld 上训练的
margin-SFT 线(v3-e1)已从 paper 移除(与全零样本叙事冲突),连同 restoration / gate v1 /
independent / set-utility(selector v0)各时代一并归档:摘要见
[`docs/archive/README.md`](docs/archive/README.md)。

## 当前结论

> **2026-07-25 新 benchmark substrate：MobileWorld-Memory + OSWorld 2.0。**
> MobileWorld 201-task inventory 已按冻结 GUI-Owl 可用 interface 锁定 117 个 GUI-only
> denominator，其中 pre-execution cross-app memory candidates / single-app controls =
> 62/55。Aries capacity 的 1/2/4/8-env wall time=`383.715/192.133/102.320/62.540s`；
> full campaign 从单 GPU 8 env 切换为 2 GPU × 8 env，真实 makespan
> **6:21:03.108**，得到 **114/117 有结果、27 success**；3 个 task 在三次尝试后仍
> policy-invalid missing。observed/strict-117 mean=`0.236842/0.230769`。结果见
> [`data/results/mobileworld_frozen_gui_owl_benchmark_v1/`](data/results/mobileworld_frozen_gui_owl_benchmark_v1/README.md)。
> OSWorld 2.0 的 108-task split 直接采用官方 task-construction phenomena：
> implicit-state memory core=43、dynamic/cross-source/implicit union=65、control=43；
> gated tasks/assets 尚未获批，正式 runner 当前 fail closed。协议与进展见
> [`docs/mobileworld_memory_osworld2_v1.md`](docs/mobileworld_memory_osworld2_v1.md)。
>
> **2026-07-25 主线变更:residual-only HGKV 判定 `NO_GO_RESIDUAL_ONLY_SPARSE_OBJECTIVE`。**
> v5 五臂留出集(494 组 / 176 episodes)显示主 claim `SA − RA` 从 identity 的 **+0.0335**
> 单调降到 epoch1 的 **−0.0006**(CI [−0.0037,+0.0025])。同一批数据上 `SA − R0` 从 +0.034
> 涨到 **+0.118** —— **只报后者会得出"方法有效"的相反结论**。差别在于 `RA`(最近窗口 +
> adapter 开)这个对照臂:adapter 对 recent 的增益(+0.118)**大于**对 sparse 的增益
> (+0.084),即它放大的是"有历史图"本身,不是"选对了历史"。
>
> **下方 2026-07-24 及更早的 HGKV / Selector 结论均以此为准被取代**,保留为负结果,
> 不得作为现行主张引用。详见 [`docs/sparse_history_v5_results.md`](docs/sparse_history_v5_results.md)。
>
> 仍然成立的正向结论:**冻结 policy 本身已存在可利用的稀疏选点优势** ——
> `frozen_selection_effect = S0 − R0 = +0.0335 [+0.0267,+0.0403]`,不含 0,且不依赖任何 adapter。
> 主线因此改写为"冻结 policy + set-conditioned selector",adapter 不再是前提。

- **[已被取代 · `NO_GO_RESIDUAL_ONLY_SPARSE_OBJECTIVE`] History-Gated KV Adapter 正式 gate v1 PASS,s100 冻结(2026-07-24)。**
  *2026-07-25 复核:该 gate 只比了 b0/shuffled/irrelevant,**缺少 `RA`(最近窗口 + adapter 开)对照臂**。
  当时如实记录的 wrong-history drift(+0.071/+0.067)与 v5 实测(+0.070/+0.057)几乎一致 ——
  同样的证据,补上 RA 才看出全部增益来自历史放大而非内容选择。原文保留如下:* 165 条 GUI-Odyssey
  heldout 轨迹、5,738 样本 × 2 条件、n=1,220 配对组、10k bootstrap:c−b0=`+0.1372
  [+0.1296,+0.1449]`、c−shuffled=`+0.0016 [+0.0003,+0.0030]`、c−irrelevant=`+0.0108
  [+0.0078,+0.0139]`,三条对照 CI 下界全正;B0 parity 漂移逐位 `0.000000`。冻结基线的两条
  内容对照 CI 认证为负(−0.0142/−0.0115),病症被修复并反转。如实记录的弱点:wrong-history
  drift(shuffled/irrelevant 也被抬高 +0.071/+0.067,内容辨别净边际小而显著)。
  [结果与 provenance](data/results/hgkv_gate_v1/README.md)。
- **闭环 dev canary 12/12 通过,sealed 零样本 policy 评测矩阵已发射(合同第 11-12 步)。**
  {Frozen, Full-layer(ody-margin-v2 s75), History-gated(hg-s100)} × {B0, Recent-B,
  Full-history} × 全量 116 template × 2 fixed instances 名单,协议冻结于
  [`docs/sealed_zero_shot_policy_matrix_v1.md`](docs/sealed_zero_shot_policy_matrix_v1.md);
  输出 hyper00 + hyper01 `/data02/jaxan/runs/sealed-matrix-v1/`。发射后不得按 benchmark
  结果改 adapter/checkpoint/协议。
- **HGKV-readout singleton 特征 75,628/75,628 完成。** 8 shard 均通过
  unique-key、统一 1,280 维、逐行长度与有限值验证；首轮唯一截断 PNG 已由两个
  SHA256 一致的持久化副本恢复，只续跑失败的 shard 1/5/6。逐 shard 哈希、恢复证据和
  完整命令见 [`data/results/hgkv_readout_v1/`](data/results/hgkv_readout_v1/README.md)。
- **HGKV selector V1 已封存为 `NO_GO_HGKV_SELECTOR_V1`。** 75,628 条
  readout 全量 join；heldout 1,621 states 上 realized `U=0.104780`，低于 Recent
  `0.113203`，差值 `−0.008423`（95% CI `[−0.010526,−0.006338]`）。V1 的
  set-conditioned B1 与 Stage-1 完全相同，而原 gate 要求 B1/B2/B4 每个预算都显著胜
  Recent，因此 V1 无论 Stage-2 如何都不能 PASS。V1 conditional train scoring 在
  `160,928/219,549` exact identities 停止，Stage-2 未启动；partial scores 只作 V2
  exact-key cache。后继主线改为 full-history、empty-start、true-U teacher beam-4、
  edge0–edge3 与统一 fresh student。结果、封存记录与 provenance 见
  [`data/results/hgkv_selector_v1/`](data/results/hgkv_selector_v1/README.md)。
- **[已被取代 · `SUPERSEDED_POLICY_NOT_FROZEN`] HGKV selector V2 beam-4 契约已冻结、实现中。**
  *V2 在定义上绑定 `hg-s100`:`U(S)` 用 `log p_hg-s100`、coalition cache key 含
  `hgkv_checkpoint_sha256`、候选特征是 1280 维 HGKV readout、正式 selected-set gate 也要
  重新送进 `hg-s100`。该 policy 已判 NO_GO,故 V2 的分数/readout/teacher beam/student
  checkpoint **一律不得进入最终 selector**。可复用的是 state inventory、图片资产、
  coalition renderer、beam planner/reducer、exact subset validator 与 bootstrap 代码。
  后继为 `selector_v3_final_policy_beam4`,须在最终 policy 冻结后重新生成全部标签。原文保留如下:* 主方法从空集合扫描完整真实历史，
  由 true-U teacher beam-4 生成 edge0–edge3，fresh unified student 使用 learned
  beam-4 + STOP；正式 B1/B2/B4 只认完整 selected set 的 hg-s100 真实 utility。首个
  full-history inventory 已完成：train/dev 835/165，候选中位 12、p95 27、最大 44，
  666/1,000 states 超过 V1 的 8-candidate cap，synthetic terminal/truncation 均为 0。
  initial exact-key cache 已合并 243,455 个 B0/singleton/pair/triple scores；V2 所需
  13,680 个 full-history singletons 已全部打分并重建为 249,467-key complete cache。
  edge0 已归约；edge1 的 36,938 个 cache-missing pair 已全部渲染、打分并完成
  36,938 unique-key 全量验收，9 个跨 hyper00/hyper01 shard 已合并为正式 `DONE`。
  edge1 score 已并入 286,405-key cache 并完成 true-U depth1 reduce；edge2 的
  39,599 个缺失 triple 已在 hyper00/hyper01 独立确定性渲染且得到完全相同的
  39,599-key/14,626-image validator，现由 hyper01 4 张 H200、10 个 scorer 打分。
  candidate readout 已完成
  13,680/13,680 unique、1285 维全量验证。development
  `n<=8` exact subset-search
  planner/reducer 已补齐，用于报告 beam recovery、regret、Jaccard 与 utility gap。
  [冻结协议](docs/selector_v2_beam4_protocol.md)与
  [结果 SoT](data/results/hgkv_selector_v2/README.md)。
- **GUI-Odyssey 外部训练零样本泛化 AndroidWorld 成立(基础能力层)。** 修正版 odyv2-s75:
  B0=6/45(frozen 3),配对净胜 +3、Δ+0.067 CI[+0.000,+0.133],parse 死亡 26→1;记忆剂量
  收益仍平(B8=6/45 vs frozen 6)。此前 s60 全量 0/90 两连败定性为动作分布负迁移。完整三轮
  记录见 [`data/results/odyssey_zeroshot_closed_loop_v1/`](data/results/odyssey_zeroshot_closed_loop_v1/README.md)。

- **AndroidWorld v3-e1 未在 30-task OSWorld paired pilot 中超过 frozen GUI-Owl。**
  frozen/v3-e1 mean score=`0.16667/0.13333`，score>0=`5/30` vs `4/30`；paired delta=
  `-0.03333`，95% CI=`[-0.13333,0.06667]`，1 win / 2 losses / 27 ties。v3-e1 将
  task-level HTTP 500 从 13 降到 9，但可执行稳定性改善未转化为更高 reward。它能完成部分
  desktop tasks，说明不是能力完全崩溃；但 AndroidWorld 的 `+8.9pt` 没有形成 OSWorld 正向迁移证据。
  当前不扩跑完整 361-task OSWorld，也不追加 α16。本实验不含 learned selector。
  [结果](data/results/osworld_v3_e1_transfer_pilot_v1/README.md)与
  [详细边界](docs/osworld_v3_e1_transfer_pilot_v1.md)。
- **terminal-s60 未在 30-task OSWorld closed-loop pilot 中超过 frozen GUI-Owl。** 相同 recent
  at-most-B4 memory 下，frozen/s60 mean score=`0.13333/0.07037`，score>0=`4/30` vs `3/30`；
  paired delta=`-0.06296`，95% CI=`[-0.22963,0.10370]`，3 wins / 4 losses / 23 ties。s60 将
  task-level policy HTTP 500 从 10 降到 1，但平均 completed steps 从 37.87 降到 20.13，
  可执行稳定性改善没有转化为更高 reward，表现出 terminal repair 的提前停止偏置。该 exploratory
  pilot 不含 learned selector，也不是完整 361-task benchmark；当前不扩跑完整 OSWorld。
  [结果](data/results/osworld_transfer_pilot_v1/README.md)与
  [详细边界](docs/osworld_transfer_pilot_v1.md)。
- **terminal-repair mobile LoRA 通过 OSWorld desktop grammar leakage gate。** official no-GDrive roster 等距
  取 30 个 prompts、覆盖 10 domains；frozen 与 terminal-s60 α32 均为 30/30 parser valid，single-image 与
  recent-B4 均 15/15，且 0 个 `mobile_use`/mobile-only hard leakage。s75 α32/α16 都因同一 Calc prompt 的
  coordinate 越界降至 29/30，α16 没有修复；terminal-state 续训后恢复。terminal-s60 相对 frozen 的
  normalized action JS divergence 仅 `0.00184`，因此从 grammar 角度可进入 OSWorld closed-loop，但该测试
  不构成 task-success 或 cross-platform performance 证据。见
  [`data/results/osworld_lora_leakage_v1/`](data/results/osworld_lora_leakage_v1/)和
  [`docs/osworld_lora_leakage_v1.md`](docs/osworld_lora_leakage_v1.md)。
- **OSWorld frozen GUI-Owl capacity study 已完成。** 8/8 正式点、每点 46/46 episodes、0 runner failure；
  6 replicas 下 environment throughput 在 24 envs 达峰值 `1835.24 fresh tasks/hour`，30 envs 回落至
  `1803.63`。固定 12 envs 时，1/2/4/6 replicas 的 throughput 基本不变，但 client p95 从
  `12.18s` 降至 `6.94/3.39/3.06s`，说明 one-step wall time 受 VM reset/setup 限制，而 GPU 数决定 policy
  尾延迟和多步余量。recent at-most-B4 的 5-image 真实请求也已通过。完整数字与边界见
  [`docs/osworld_gui_owl_capacity_v1.md`](docs/osworld_gui_owl_capacity_v1.md)和
  [`data/results/osworld_gui_owl_capacity_v1/`](data/results/osworld_gui_owl_capacity_v1/)。
- **OSWorld runner v1 已完成真实 KVM live smoke。** runner 绑定官方 pinned `DesktopEnv`，覆盖 10 domains / 369
  tasks 的 inventory、受限 desktop action 到 PyAutoGUI 的安全映射、mixed-fidelity policy HTTP boundary、
  task-level 可断点 suite sharding 与原子 episode evidence。Hyper01 与 H100 的真实
  reset→screenshot→WAIT→DONE→evaluate→close 均已通过，第二次运行均返回 `resumed_skips=1`；H100 需要显式
  `--docker-cpu-model qemu64`，避免默认 host CPU model 卡在 VM early boot。当前仍是跨平台 benchmark
  substrate，不是科学结果，
  learned selector、GUI-Owl desktop policy 和正式 OSWorld roster 尚未接入。复现见
  [`docs/osworld_runner_v1.md`](docs/osworld_runner_v1.md)，证据见
  [`data/results/osworld_runner_v1_smoke/`](data/results/osworld_runner_v1_smoke/)和
  [`data/results/osworld_runner_v1_h100_smoke/`](data/results/osworld_runner_v1_h100_smoke/)。
- **OSWorld benchmark acceleration v1 已接通 12-env / 2-GPU topology。** 官方 `test_nogdrive.json` 被
  fail-closed 验证为从 369 tasks 只排除 8 个 Google Drive `multi_apps` tasks，正式可运行 denominator 为
  361。H100 上 12 个并发 KVM environments 经动态任务队列轮询两个独立 H100 policy replicas，完成 12/12
  infrastructure episodes、24/24 requests、0 runner failure；重复运行 12/12 task-level resume。该 smoke
  使用 `WAIT→DONE`，不构成 policy score 或完整 benchmark 时间 claim。方案、排除清单和正式吞吐建议见
  [`docs/osworld_benchmark_acceleration_v1.md`](docs/osworld_benchmark_acceleration_v1.md)，轻量证据见
  [`data/results/osworld_benchmark_h100_smoke_v1/`](data/results/osworld_benchmark_h100_smoke_v1/)。
- **旧时代结论已整体归档。** restoration 反事实修复、gate v1 formal cache、independent
  confirm-20、set-utility / selector v0(labels、learning curve、held-out v1 NO-GO、contextual
  v3/v4、decision distillation v2、direct marginal v3 NO-GO、selector-side LoRA、budget-deferral
  停止)与 memory ceiling(`STORY_DEAD_CAPABILITY_BOUND`)各线的详细结论、文件与 artifact 指针
  见 [`docs/archive/README.md`](docs/archive/README.md)。贯穿性结论:离线 oracle 恢复 headroom
  始终确认(long-oracle `+0.467 [0.359,0.630]`),但没有 learned selector 稳定胜过 recent,
  且 frozen backbone 加记忆剂量无闭环收益——瓶颈在 policy 消化历史的能力,这是 history-gated
  主线的直接动机。

完整历史与失败记录:2026-07-23 起的主线条目在 [`docs/progress.md`](docs/progress.md),更早条目已逐字存档至 [`docs/archive/progress_2026-07-20_22.md`](docs/archive/progress_2026-07-20_22.md)。

## 方法(mainline)

- **History-Gated KV Adapter**:冻结 vision encoder / projector / 全部原 LM 参数 / tokenizer /
  tool schema / parser / executor;LM 最后 8 层(28-35)仅 k_proj/v_proj 加 mask 门控 residual
  (`K_l = W_K h + M_hist·ΔW_K h`,V 同理),rank 8 / alpha 16 / dropout 0;mask 只覆盖 restored
  history image tokens,B0(无历史)时 hook 直接返回原输出。行为契约:B0 bitwise parity 是
  架构不变量,训练前必须通过,失败不得开训。
- **训练目标**:L_gain / L_rank / L_ignore / L_norm(margins 0.01,L_CE 权重 0),变体
  correct/b0/shuffled/irrelevant 共享同一成功动作 target;数据为修正版 GUI-Odyssey
  (ody-sft-v2,37,635 样本),轨迹 ID 三切分 train/tune/heldout,checkpoint 只按 Odyssey
  tune/heldout 选,不触 benchmark。
- **Selector(adapter 过跨平台门后)**:`U_act(S) = log p_gate(a*|S) − log p_0(a*|B0)`。
  Stage 1 singleton-gain scorer 负责 B1/第一步、标签构造 shortlist 与 independent
  baseline；Stage 2 接收已选集合和剩余预算，在全部剩余候选上比较 learned rank 与
  STOP。query 已编码在 candidate readout 中，gain/marginal head 不直接驱动选择。正式
  B1/B2/B4 utility 必须将完整选中集合重新送入 hg-s100，禁止 singleton gain 求和冒充
  set utility。仅在 GUI-Odyssey 训练与选择、零样本部署；协议与结果见
  [`docs/selector_v1_protocol.md`](docs/selector_v1_protocol.md)和
  [`data/results/hgkv_selector_v1/`](data/results/hgkv_selector_v1/README.md)。
- 完整契约(token-role fail-closed 规则、数据协议、Outcome A-E 升级/回退决策树)见
  [`docs/history_gated_mainline_v1.md`](docs/history_gated_mainline_v1.md)。旧 set-utility 方法
  (`U(S)=D(∅)−D(S)` 监督的 set predictor)及其全部对照线已归档:
  [`docs/archive/README.md`](docs/archive/README.md)。

## 活跃文档

- [`paper/README.md`](paper/README.md):AAAI-27 HGKV draft、官方 author-year / arXiv
  bibliography、AuthorKit hashes、7+2 page rule 与独立 reproducibility checklist 状态；
- [`docs/history_gated_mainline_v1.md`](docs/history_gated_mainline_v1.md):主线冻结契约(2026-07-23);
- [`docs/history_gated_mainline_v1_provenance.md`](docs/history_gated_mainline_v1_provenance.md):分支 provenance;
- [`docs/sealed_zero_shot_policy_matrix_v1.md`](docs/sealed_zero_shot_policy_matrix_v1.md):sealed 零样本评测矩阵 v1(AW,预注册,执行中);
- [`docs/mobileworld_memory_osworld2_v1.md`](docs/mobileworld_memory_osworld2_v1.md):MobileWorld
  单 GPU 多模拟器 benchmark 与 OSWorld 2.0 task-construction memory split；
- [`data/results/hgkv_gate_v1/README.md`](data/results/hgkv_gate_v1/README.md):正式 gate PASS 判定表与 s100 provenance;
- [`data/results/hgkv_selector_v1/README.md`](data/results/hgkv_selector_v1/README.md):Stage-1/Stage-2 与 selected-set gate 的结果、provenance 和 artifact 状态;
- [`docs/androidworld_task_partition.md`](docs/androidworld_task_partition.md) 与 [`docs/androidworld_stack.md`](docs/androidworld_stack.md):AndroidWorld 冻结 partition 与 benchmark-native stack;
- `docs/osworld_*.md`:OSWorld runner / benchmark acceleration / capacity / leakage / transfer 线,runner 会话交接见 [`docs/osworld_lora_leakage_test_handoff.md`](docs/osworld_lora_leakage_test_handoff.md);
- [`docs/policy_selection.md`](docs/policy_selection.md):frozen policy(GUI-Owl)选型与候选记录;
- [`docs/shared_host_docker_rules.md`](docs/shared_host_docker_rules.md):共享 GPU 主机 Docker 规范;
- [`docs/progress.md`](docs/progress.md):主线时代进展(2026-07-23 起;更早条目见 [`docs/archive/progress_2026-07-20_22.md`](docs/archive/progress_2026-07-20_22.md))。

## Source of Truth

### 当前主线 artifacts

| 内容 | 位置 | 状态 |
|---|---|---|
| 代码、配置、论文、轻量结果 | 本 Git 仓库(主线分支 `exp/history-gated-mainline-v1`,保底线在 `main`) | canonical |
| MobileWorld-Memory / OSWorld 2.0 substrate | [`docs/mobileworld_memory_osworld2_v1.md`](docs/mobileworld_memory_osworld2_v1.md)；[`MobileWorld result`](data/results/mobileworld_frozen_gui_owl_benchmark_v1/README.md)；[`mobile manifest`](data/manifests/mobileworld_memory_split_v1.json)；[`OSWorld 2.0 manifest`](data/manifests/osworld_v2_memory_split_v1.json)；分支 `luojiaxuan/mobileworld-memory-osworld2` | MobileWorld campaign 完成但 114/117（3 policy-invalid missing）；OSWorld 2.0 gated substrate blocked |
| GUI-Owl snapshot | `mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc` | frozen |
| History-gated adapter hg-s100 | hyper00 `/data02/jaxan/runs/hgkv-formal-v1/lora-step100.pt`(sha256 前缀 `8f2cc49e1aa0b06c`);副本 hyper01 `/data02/jaxan/runs/hgkv-eval/hg-s100.pt`;config [`code/configs/causalcache_history_gated_kv_v1.json`](code/configs/causalcache_history_gated_kv_v1.json) | gate PASS(s100 冻结);`PENDING_HF_UPLOAD` |
| hgkv gate v1 认证 | [`data/results/hgkv_gate_v1/`](data/results/hgkv_gate_v1/README.md);原始行 hyper01 `/data02/jaxan/runs/hgkv-eval/cert/`(s100/frozen 各 5,738 行 jsonl) | 判定表入 Git;逐行 jsonl `PENDING_HF_UPLOAD` |
| 训练数据 ody-sft-v2 | hyper00 训练 run root `/data02/jaxan/runs/hgkv-formal-v1`(37,635 样本,165 heldout 轨迹不参训;provenance 见 gate README) | 本地;`PENDING_HF_UPLOAD` |
| HGKV selector runtime | hyper01 `/data02/jaxan/envs/causalcache-selector-v1`；lock [`code/requirements/hgkv_selector_v1_lock.txt`](code/requirements/hgkv_selector_v1_lock.txt) | persistent system-site venv；scikit-learn `1.9.0` |
| HGKV-readout singleton features v1 | [`data/results/hgkv_readout_v1/`](data/results/hgkv_readout_v1/README.md)；hyper01 `/data02/jaxan/runs/hgkv-readout-v1/` | 75,628 unique rows、1,280 dims、8 shard；`PENDING_HF_UPLOAD` |
| HGKV selector v1 | [`data/results/hgkv_selector_v1/`](data/results/hgkv_selector_v1/README.md)；hyper01 `/data02/jaxan/runs/hgkv-selector-stage1-v1/` | `NO_GO_HGKV_SELECTOR_V1`；Stage-2 未启动；partial scores 保留作 V2 cache |
| HGKV selector v2 beam-4 | [`data/results/hgkv_selector_v2/`](data/results/hgkv_selector_v2/README.md)；[冻结协议](docs/selector_v2_beam4_protocol.md) | contract frozen、implementation in progress；reusable artifacts `PENDING_HF_UPLOAD` |
| Sealed 矩阵 v1 run | hyper00 + hyper01 `/data02/jaxan/runs/sealed-matrix-v1/`(目标 2,088 局,116 template × 2 fixed instances × 3 policy × 3 arm；聚合器 `scripts.aggregate_sealed_matrix_v1`) | 执行中 |
| dev canary 记录 | hyper01 `/data02/jaxan/runs/hgkv-canary-v1/`(逐局 json) | 12/12;性能不读(canary 纪律) |
| Odyssey 零样本闭环三轮记录 | [`data/results/odyssey_zeroshot_closed_loop_v1/`](data/results/odyssey_zeroshot_closed_loop_v1/README.md);run root hyper00 `/data02/jaxan/runs/odyv2-zeroshot-full/` | 完成;s75 B0 翻倍,B8 平 |
| GUI-Owl policy LoRA s75 / terminal-s60 | Hyper00 `/data02/jaxan/runs/causalcache-odyssey-margin-v1/lora-step75.pt`、`...odyssey-margin-v1b/lora-step60.pt`；intended private HF model repo `gavinlaw/causalcache-gui-owl-policy-lora-mobile` | SHA `90a56b7e...d56e` / `cba455cb...7b18`；`PENDING_HF_UPLOAD`；[desktop leakage PASS](data/results/osworld_lora_leakage_v1/README.md) |
| AndroidWorld policy LoRA v3-e1 | Hyper00 `/data02/jaxan/runs/causalcache-margin-sft-v3/lora-epoch1.pt`；Hyper01 verified mirror `/data02/jaxan/runs/causalcache-margin-sft-v3-eval/lora-epoch1.pt`；intended private HF model repo `gavinlaw/causalcache-gui-owl-policy-lora-mobile` | SHA `7122d897...f9c73b8`；`PENDING_HF_UPLOAD`；AndroidWorld `+8.9pt`，但 [OSWorld transfer 无正向证据](data/results/osworld_v3_e1_transfer_pilot_v1/README.md) |

### 历史时代 artifacts(已归档线,链接与上传状态原样保留)

大 artifact 索引沿用旧表;对应文档与结果已移入 `docs/archive/` 与 `data/results/archive/`。

| 内容 | 位置 | 状态 |
|---|---|---|
| Processor substrate | [HF dataset](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/c20bab8df424dc9e45ece1084f3d1dc035dd1ed8/artifacts/processor-freeze-v2-image-contract-repair) | immutable，23 files / 18.73 GB |
| Dense image backfill | Hyper00 `/data02/jaxan/artifacts/causalcache-set-utility-dense-v1-backfill-d43a15c` | 42MB / 77 PNG；`PENDING_HF_UPLOAD` |
| recent-4 label smoke | Hyper00 `/data02/jaxan/runs/causalcache-set-utility-dense-v1-0d32187`；Hyper01 `/data02/jaxan/runs/causalcache-set-utility-dense-v1-9671e45-partition-01` | stopped；4,033 states；`DEPRECATED_SMOKE_ONLY_RECENT4` |
| Variable-history formal labels | Hyper00/Hyper01 `/data02/jaxan/runs/causalcache-set-utility-variable-history-labels-v1-969f2b9` | complete；11,721 completed + 25 skipped；`PENDING_HF_UPLOAD` |
| Variable-history full source | Hyper00 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-source-v1-a7213db` | 13.16GB / 256 shards；manifest `a37ce0b...042cde4`；`PENDING_HF_UPLOAD`；[census](data/results/archive/set_utility_variable_history_source_context_v1/README.md) |
| Variable-history full VLM tokens | Hyper00、Hyper01 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-tokens-v2-47161ca` | 256/256 shards、约 67GB、零失败；source revision `47161cac...`；`PENDING_HF_UPLOAD` |
| Variable-history exact context | [`data/results/archive/set_utility_variable_history_context_exact_v2/`](data/results/archive/set_utility_variable_history_context_exact_v2/README.md) | PASS；12,792/12,792 fit；max prompt+reserve 30,292/32,768 |
| Variable-history broad schedules | [`data/results/archive/set_utility_variable_history_schedules_v1/`](data/results/archive/set_utility_variable_history_schedules_v1/README.md) | 256/256 shards；11,746 states；461,040 coalition labels；`PENDING_HF_UPLOAD` |
| Variable-history formal label run | [`data/results/archive/set_utility_variable_history_labels_v1/`](data/results/archive/set_utility_variable_history_labels_v1/README.md) | complete；11,746/11,746 terminal states；state/microbatch atomic resume |
| Learning-curve full snapshot/cache | Hyper00/Hyper01 `/data02/jaxan/artifacts/causalcache-set-utility-variable-history-training-full-merged-5e71d05`；`...learning-curve-cache-5e71d05` | 11,721 states；cache SHA `ef82b077...23385`；`PENDING_HF_UPLOAD` |
| Variable-history partial train/tune snapshot | [HF dataset@771db37c](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/771db37c9ce6d3e7057b87730c400cbae66a5398) | immutable；1,501 states；2,105 visual / 2,256 text sequences |
| Variable-history partial predictor checkpoints | [HF model@b6bd8232](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/b6bd823221e2ec4b2e68518ad40efe7e7bd5d673/artifacts/set-utility-variable-history-partial-ca0383a) | immutable；DeepSets + Set Transformer；diagnostic only；[summary](data/results/archive/set_utility_variable_history_training_partial_v1/README.md) |
| 25% tuning snapshot/cache | [HF dataset@d8639bc8](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/d8639bc8b0a3237a8904aefee06d6d7f70ad7768/artifacts/set-utility-tuning25-v3-seeded-floor1e2-2d68154) | immutable；3,575 states；23.9GB；[summary](data/results/archive/set_utility_tuning25_v1/README.md) |
| 25% tuning checkpoints | [HF model@d628ecf6](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/d628ecf6e1783591a039080999f36635c36049fb/artifacts/set-utility-tuning25-v3-seeded-floor1e2-2d68154) | immutable；12 configs / 481.8MB；tuning-only |
| Learning-curve dataset/cache | [HF dataset@02a05ab1](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/02a05ab11fd3a5036b62244bf04f37aa5e41db59/artifacts/set-utility-learning-curve-v1-9863f43) | immutable；414 files / 77.4GB；nested splits + compact label archives |
| Learning-curve checkpoints | [HF model@5409e846](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/5409e846cc45a26b2ae617e39b3ebf7462d180e6/artifacts/set-utility-learning-curve-v1-9863f43) | immutable；8 checkpoints；[summary](data/results/archive/set_utility_learning_curve_v1/README.md) |
| Held-out selector v1 | [`summary`](data/results/archive/set_utility_heldout_v1/README.md)；[HF tag `set-utility-heldout-v1-d89263c`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/set-utility-heldout-v1-d89263c/artifacts/set-utility-heldout-v1-d89263c)；Hyper00 persistent mirror | 805 terminals：801 completed + 4 skipped；`INCOMPLETE/NO_GO`；result content `d89263c...8ef49`；immutable revision `df41e1d...be8c6` |
| Held-out scale diagnostic v1 | [`data/results/archive/set_utility_heldout_scaling_diagnostic_v1/`](data/results/archive/set_utility_heldout_scaling_diagnostic_v1/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-set-utility-heldout-scaling-v1-3b0be31` | 8 checkpoints；319 exact states；result content `c346bce...17c2`；`PENDING_HF_UPLOAD` |
| Small-history exact B4 v2 | [`summary`](data/results/archive/set_utility_b4_oracle_diagnostic_v2/README.md)；[HF dataset@9b53ec82](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/9b53ec82c12fefaba571233e5e0d78d0f6c599c0/artifacts/set-utility-b4-oracle-v2-195bcbb)；tag `set-utility-b4-oracle-v2-195bcbb` | 103/103 completed；8,628 distance rows；result content `195bcbb...0473b`；669 files / 4.7MB |
| Contextual hidden cache v3 | [`summary`](data/results/archive/set_utility_contextual_cache_v3/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-contextual-hidden-v3-e413e5b` | 8,813 contexts；1,174 chunks；37,771,051,973 bytes；content `af2b090d...91eb0`；`PENDING_HF_UPLOAD` |
| Contextual predictor training v3 | [`summary`](data/results/archive/set_utility_contextual_training_v3/README.md)；Hyper00 two persistent roots | DeepSets `0.32128`；Set Transformer `0.30910`；evaluation not loaded；`PENDING_HF_UPLOAD` |
| Contextual tune on-policy truth v3 | [`summary`](data/results/archive/set_utility_contextual_tune_on_policy_v3/README.md)；[HF dataset@d7a6e97e](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/d7a6e97e1b654cf17b8a06690adc55d55e3c53f7/artifacts/set-utility-contextual-tune-on-policy-v3-f34d368) | 1,063/1,063 completed；DeepSets `0.4492`、Set Transformer `0.4365`、recent `0.4519`；immutable |
| Contextual full inputs v4 | [HF dataset@268bae32](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/268bae32792c3b651541354f74d9a18b7b97ecd2/artifacts/set-utility-contextual-inputs-full-v4-af18388e)；Hyper00/Hyper01 mirror | 11,721 states；27,867 contexts；tag `set-utility-contextual-inputs-full-v4-af18388e`；immutable |
| Contextual full hidden cache v4 | Hyper00 `/data02/jaxan/runs/causalcache-contextual-hidden-full-v4-e97f2d4` | 27,867 contexts；119.16GB；content `44405c2c...97604`；`PENDING_HF_UPLOAD` |
| Contextual full predictor training v4 | [`summary`](data/results/archive/set_utility_contextual_training_full_v4/README.md)；[HF model@729a62fd](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/729a62fdaed53ccf04e641bea9ecac3ac7da3bcc/artifacts/set-utility-contextual-full-v4-44405c2) | DeepSets `0.30927`；Set Transformer `0.32233`；Set Transformer true-utility winner；immutable |
| Contextual full tune truth v4 | [`summary`](data/results/archive/set_utility_contextual_tune_on_policy_full_v4/README.md)；[HF dataset@014aee81](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/014aee81e0f95199163773440658d6eaf36d3ddb/artifacts/set-utility-contextual-tune-on-policy-full-v4-fb6de0d) | 1,063/1,063；Set Transformer `0.44985` vs recent `0.45192`；`FULL_DATA_NO_GO`；immutable |
| Train on-policy enrichment v1 | [合同/进展](docs/archive/set_utility_train_on_policy_enrichment_v1.md)；[HF dataset@9b436c9c](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/9b436c9c8ac645d20f2aa86ba0d519b14f5d6934/artifacts/set-utility-train-enrichment-v1-2711ab55) | 2,132 train states；52,744 scheduled rows；37,982 novel rows；tag `set-utility-train-enrichment-v1-2711ab55`；immutable |
| Contextual enriched fixed-tune result v1 | [`summary`](data/results/archive/set_utility_contextual_tune_on_policy_enriched_v1/README.md)；[HF dataset@bbee1ae7](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/bbee1ae7aae2a712aaf2a897a08fa69adcef4ee6/artifacts/set-utility-contextual-tune-enriched-v1-3f73e17) | 1,063/1,063；DeepSets/Set Transformer/recent=`0.44383/0.44141/0.45192`；`NO_GO_TRAIN_ON_POLICY_ENRICHMENT_V1`；immutable |
| Decision distillation v2 | [`summary`](data/results/archive/set_utility_decision_distillation_v2/README.md)；Hyper00/Hyper01 persistent run | 1,066/1,066 states、25,915/25,915 microbatches completed；`PENDING_HF_UPLOAD` |
| Long-history oracle diagnostic v1 | [`summary`](data/results/archive/set_utility_long_oracle_v1/README.md)；[HF dataset@8d5a5021](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/8d5a5021d8e69999ed944574bc8e243f386c2288/artifacts/set-utility-long-oracle-v1-179b0d8)；Hyper00 mirror | 250/250 states；25,032 labels；`HEADROOM_CONFIRMED`(+0.467 [0.359,0.630]);tag `set-utility-long-oracle-v1-179b0d8`；immutable |
| Decision v2 + long-oracle training source | [执行单](docs/archive/set_utility_decision_distillation_v2_long_oracle_training.md)；[versioned config](code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_training_v1.json)；Hyper00 `/data02/jaxan/artifacts/causalcache-decision-v2-long-oracle-training-inputs-v2-f7f6b14` | complete；content `3d011990...ad36ce`；5,550 decision-supervised states / Long+ 902；`PENDING_HF_UPLOAD` |
| Decision v2 distributed training | [DDP config](code/configs/causalcache_set_utility_decision_distillation_v2_long_oracle_ddp_v1.json)；Hyper00 `/data02/jaxan/runs/causalcache-set-transformer-decision-v2-long-oracle-ddp4-v2-ecd5579`；Hyper01 `/data02/jaxan/runs/causalcache-deepsets-decision-v2-long-oracle-ddp4-v4-1aafe4c` | complete；两模型 epoch 1 best、epoch 6 early-stop；evaluation labels loaded=false；checkpoints `PENDING_HF_UPLOAD` |
| Decision v2 epoch-best tune truth | [`summary`](data/results/archive/set_utility_decision_distillation_v2_epoch1_tune/README.md)；Hyper00 full result `/data02/jaxan/runs/causalcache-running-best-tune-evaluation-v2-650afbe/result.json` | 1,063/1,063、0 skip；Set base/recent macro=`0.47593/0.45192`、CI lower=`+0.00430`；B2 与 Long+ fail；`NO_GO_DECISION_DISTILLATION_V2`；`PENDING_HF_UPLOAD` |
| Set capacity + DeepSets prefetch v1 | [`summary`](data/results/archive/set_utility_capacity_and_prefetch_v1/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-set-transformer-decision-v2-long-oracle-l64-s4-ddp2-v1-8a6b8a0` | L64/S4 2×H200 running；epoch-1 tune truth macro=`0.45575`、p95=`114.19ms`；prefetch elapsed -11.22%；checkpoint `PENDING_HF_UPLOAD` |
| Fixed-tune Long+ oracle v1 | [`summary`](data/results/archive/set_utility_tune_long_oracle_v1/README.md)；Hyper00 `/data02/jaxan/runs/causalcache-tune-long-oracle-v1-b6a4643` | 249/249、0 skip；oracle/recent macro=`0.6949/0.3199`；`HEADROOM_CONFIRMED`；full payload `PENDING_HF_UPLOAD` |
| Direct marginal v3 Stage-A | [`v1`](data/results/archive/set_utility_direct_marginal_v3_fit_probe_v1/README.md)；[`rank repair`](data/results/archive/set_utility_direct_marginal_v3_fit_probe_v2_rank_repair/README.md)；Hyper00 persistent run | v2 Spearman/top-4=`0.291/0.951`；`NO_GO_DIRECT_MARGINAL_V3_STAGE_A`；fit-only checkpoints `PENDING_HF_UPLOAD`，不是正式模型 |
| Direct marginal v3 Stage-B v1 | [结果](data/results/archive/set_utility_direct_marginal_v3_stage_b_v1/README.md)；[config](code/configs/causalcache_set_utility_direct_marginal_v3_stage_b_v1.json)；Hyper00 `/data02/jaxan/runs/causalcache-direct-marginal-v3-stage-b-v1-8ccdb5c` | training complete；best epoch 12 / regret `0.2581`；checkpoint `d8abbe8c...1dd23`；`PENDING_HF_UPLOAD` |
| Direct marginal v3 final fixed-tune | [结果](data/results/archive/set_utility_direct_marginal_v3_fixed_tune_v1/README.md)；[HF dataset@76615721](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/766157217d99dc8c10d82349d9909ba30ceaa8e9/artifacts/set-utility-direct-marginal-v3-fixed-tune-794fb90) | 1,063/1,063、0 skip；direct/recent macro=`0.44583/0.45192`；CI crosses 0；B2/B3/Long+ fail；`NO_GO`；learned general-`B` stopped；immutable |
| Direct on-policy coverage v1 | [合同](docs/archive/set_utility_direct_on_policy_v1.md)；[结果](data/results/archive/set_utility_direct_on_policy_v1/README.md) | labels PASS：5,108 states / 359,189 sampled rows；merged input `6c9243a...1bfad`；9,287 optimizer states / 84,441 groups；`PENDING_HF_UPLOAD` |
| Per-epoch heldout training v1 | [结果](data/results/archive/set_utility_direct_on_policy_training_v1/README.md)；[合同](docs/archive/set_utility_direct_on_policy_v1.md)；Hyper00 Set root `...set-transformer-direct-on-policy-v2-a69c706`；Hyper01 DeepSets root `...structured-deepsets-direct-on-policy-v4-843e360` | complete；Set e2=`0.39491`、DeepSets e1=`0.39251`；均由 truth recovery 早停；evaluation/test sealed；`PENDING_HF_UPLOAD` |
| Budget-deferral candidate v1 | [冻结 config](code/configs/causalcache_set_utility_budget_deferral_v1.json)；[development truth result](data/results/archive/set_utility_direct_on_policy_deployment_v1/README.md) | B1/B2 recent + B3/B4 DeepSets direct；development delta=`+0.02689 [0.00039,0.07047]`；不是 evaluation；payload `PENDING_HF_UPLOAD` |
| Budget-deferral evaluation Stage-A | [执行 config](code/configs/causalcache_set_utility_budget_deferral_evaluation_stage_a_v1.json)；[状态](data/results/archive/set_utility_budget_deferral_evaluation_v1/README.md) | 用户在 selection/truth 前停止；truth read=0；Hyper00/01 保留 42/33 个 resumable receipts；转向 selector-side LoRA |
| Selector-side GUI-Owl LoRA v1 | [设计](docs/archive/set_utility_selector_lora_v1.md)；[extraction config](code/configs/causalcache_set_utility_selector_lora_v1.json)；[training config](code/configs/causalcache_set_utility_selector_lora_training_v1.json)；[状态](data/results/archive/set_utility_selector_lora_v1/README.md) | teacher/action policy frozen；top-4 parity PASS；boundary cache complete：23,714 contexts、3,065 shards、289.75GB、content `77dec757...5d16b`；e1 macro `+0.03205` vs recent 但 B2/Long+ 持平，e2 退化；保留 e1、停止 e3/e4；artifacts `PENDING_HF_UPLOAD` |
| Token predictor v2 partial snapshot/cache | [HF dataset@e1240bde](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/e1240bdeff500114097b71148ba65ae19e71e6e8/artifacts/set-utility-token-v2-partial-a73cc18) | 23.43GB / 1,878 files；tag `set-utility-token-v2-partial-a73cc18`；pilot 不含 evaluation；[summary](data/results/archive/set_utility_token_predictor_v2_partial/README.md) |
| Token predictor v2 partial checkpoints | [HF model@a55666c1](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/a55666c11da6b2848a6f066b4b05980704f1bf7a/artifacts/set-utility-token-v2-partial-a73cc18) | 155.44MB / 11 files；同名 tag；4 checkpoints |
| Anchor-only pilot labels/features | [HF dataset@a95ce68b](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-new-development-mobile/tree/a95ce68bd628daaec40a7575847c9db584f20dc4/artifacts/set-utility-scale-v1-ac0ef27) | deprecated pilot；仅保留复现 |
| Anchor-only pilot checkpoints | [HF model@365f3882](https://huggingface.co/gavinlaw/causalcache-set-utility-predictors-mobile/tree/365f3882658b40eccb64c9565ae8639986c59e82) | deprecated pilot；仅保留复现 |

大文件若暂时无法上传 HF，必须保存在个人 persistent storage，并在本 README 或结果文档记录精确路径与 `PENDING_HF_UPLOAD`。

## 仓库结构

- `code/`:package、scripts、configs、tests(全部时代共用,归档不移动代码);
- `data/manifests/`:机器可读 manifests(运行中 worker 依赖,路径冻结不动);
- `data/results/`:活跃结果目录(`hgkv_gate_v1`、`odyssey_*`、`osworld_*`);
- `data/results/archive/`:已归档时代的结果目录([索引](data/results/archive/README.md));
- `data/cards/`、`data/fixtures/`:dataset/model cards 与测试 fixtures;
- `docs/`:主线契约与活跃实验文档;`docs/archive/`:已归档时代文档([索引](docs/archive/README.md));
- `paper/`:AAAI LaTeX、references、figures/tables;
- `ablations/`:interaction 与 subset-search 分析设计(历史线)。

## 本地验证

```bash
PYTHONPATH=code .venv/bin/pytest -q \
  code/tests/test_history_gated_lora.py \
  code/tests/test_history_token_roles.py \
  code/tests/test_history_gated_generation_scope.py \
  code/tests/test_set_utility_dense.py \
  code/tests/test_set_utility_mvp.py \
  code/tests/test_set_utility_variable_history.py \
  code/tests/test_set_utility_variable_history_contract.py \
  code/tests/test_set_utility_token_models.py \
  code/tests/test_set_utility_label_inputs.py

PYTHONPATH=code python3 -m compileall -q \
  code/causalcache \
  code/scripts/materialize_set_utility_variable_history_inventory.py \
  code/scripts/census_set_utility_dense_states.py \
  code/scripts/train_set_utility_mvp.py
```

## Go / No-Go

- **Adapter 正式 gate v1(Odyssey heldout):PASS。** 三条对照 CI 下界全正 + B0 逐位 parity,
  s100 冻结;判定表见 [`data/results/hgkv_gate_v1/README.md`](data/results/hgkv_gate_v1/README.md)。
- **下一门:sealed 零样本 policy 评测矩阵**(AW 侧执行中,协议冻结,不得按结果改
  adapter/checkpoint);过跨平台门后才进入 selector 重标与闭环(合同第 13-15 步)。
- **升级/回退按合同 Outcome A-E**([`docs/history_gated_mainline_v1.md`](docs/history_gated_mainline_v1.md)):
  双平台正向→升级主方法,full-layer 降为 baseline;仅 AW→收缩 mobile transfer;V1/V2 全败→
  封存,`main` 保底路线(margin-SFT v3-e1)回归,history-gate 仅作失败分析。
- 旧 set-utility 时代的 held-out Go/No-Go(已消费,NO-GO)不再是当前执行清单;其判定原文与
  边界见 [`docs/archive/README.md`](docs/archive/README.md)。

## Paper Claim 边界

“Causal”只指对冻结策略行为进行受控 restoration intervention 得到的 counterfactual attribution；不声称识别环境结构因果，也不把方法包装成 world model。
