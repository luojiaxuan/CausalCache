# 项目进展

> 2026-07-22 及更早的全部旧条目已逐字存档至 [`docs/archive/progress_2026-07-20_22.md`](archive/progress_2026-07-20_22.md);本文件只保留 history-gated mainline 时代(2026-07-23 起)的条目。

## 2026-07-27:主张重冻结为固定预算重分配,corpus v3 构建完成,三行 v3 重训发射

- 用户裁定:主实验从"C_r + 单槽新增"改为**固定预算 B 内替换**——Recent-B 把全部
  槽位给最近观测,CausalCache 从完整历史选至多 B 个事件;k(窗口外图数)是结果
  统计量。四个冻结点:最老槽位替换规则、主 B={1,2,4}(B=3 不进任何主口径)、
  age ≥ B+2、B=8 只建不训(评测外推)。
- corpus v3(`build_desktop_did_corpus_v3.py`,55 项相关测试绿):B=1/2/4/8 =
  969/764/470/160 组;训练合并 B∈{1,2,4} = **2,203 组 / 11,015 行 / train units
  1,774 / heldout 211**,samples SHA `3114d4c9…66b6f`;9,925 行全过 trainer 校验,
  5,053 引用图 0 缺失。B=1 行即真前一帧探针口径的 v1。
- 三份 v3 config(300 步 / ckpt 50 / 协议块三行逐字一致)提交;v3 重训发射:
  hyper01 GPU0-3 HGKV(canonical 容器)、hyper00 GPU1-4 Full-LoRA;hyper01
  GPU4/5 被他人 125GB/100% 活跃负载占用,ungated 首启撤下,挂 Mac 侧单发监视器
  自动补位(hyper01 GPU4-7 或 hyper00 GPU0+5-7 先空者承接)。
- r-additive corpus v2(五分片 2,912 组,86M)归档为 recent-dose 附录分析材料,
  不训练;additive 版 claim 与实验节将由 fixed-budget 版在 paper 中取代。

## 2026-07-27(凌晨):Desktop DiD 三臂全部收官,HGKV 通过全部预注册 gate

- 三行 150 步全部 EXIT_0(HGKV/ungated 于 hyper01,Full-LoRA 于 hyper00),
  全 checkpoint(s25–s150)dev 94 组 gate 打分完成。
- **HGKV 选点 s150:did_select +0.0111 [+0.0074,+0.0150](ci_low>0 ✅),
  A_S +0.0139,|A_r| 0.0087、wrong drift 0.0085 均 < ε=0.02 —— s50 起全部
  gate 连续 PASS**。旧线两个失败模式(遗漏 RA / SA−R0 假阳性)未复现。
- matched ungated KV 也可选(s125,did_select +0.0076):层位/参数量本身有部分
  选择性;**门控净增益配对显著:+0.00345 [+0.0007,+0.0062],56/94 组占优**。
- Full-LoRA 选择性点估计最高(+0.0163)但 |A_r| 0.021–0.027 全程超 cap、
  wrong drift 两点超限 —— **无合格 checkpoint**:"能选择但锚不住",
  反证门控 = 在漂移包络内的选择性。
- 冻结基线 frozen_selection_effect = +0.0333 [+0.0118,+0.0567]:桌面冻结模型
  本就显著受益于目标相关旧帧(Odyssey 上测不出的现象层前提)。
- 完整表、§9 清单(6/7 两项生成式评测待跑,s150 未冻结)与 artifact 状态见
  [`data/results/desktop_did_policy_v1/`](../data/results/desktop_did_policy_v1/README.md)。
- 临时容器 `sglang-omni-jaxan-2` 已删;两台机器 GPU 已空,checkpoint/语料
  `PENDING_HF_UPLOAD`。

## 2026-07-26:MobileWorld frozen GUI-Owl B0 对照冻结

- 目标是给已有 B4 strict-117 结果提供同栈配对证据；唯一科学变量为
  `memory_budget: 4 → 0`，model revision、MobileWorld revision/image、roster、
  max 50 steps、auto-retry 2、2560 visual tokens 与确定性 decoding 均保持不变。
- Aries 预检于 `2026-07-26T17:06:23Z` 确认 A6000 GPU 0/1 均低于 1 GiB 且没有
  清理任何 workload；正式布阵为两张卡各一个共享 policy、各驱动 8 个独立 emulator，
  strict-117 按冻结 manifest 分成 59/58 个互斥 task。
- B0 的语义明确为“零张历史图”，不是 unlimited sentinel：
  `select_mobileworld_memory(..., budget=0)` 对 recent/full 均返回空集合；在线 policy
  另以 `--max-history-images 0` fail closed，并在 health/execution record 中累计实际
  `maximum_history_images`。任何 prompt 出现历史图都会使请求失败而不是静默继续。
- 启动前 focused regression=`7 passed`。冻结 shards 见
  [`data/manifests/mobileworld_b0_strict117_shards_v1/`](../data/manifests/mobileworld_b0_strict117_shards_v1/)；
  raw traces 将写 Aries persistent storage，结束后只回写 task-level paired
  B0/B4、strict score、discordant wins/losses、置信区间和完整 provenance。归约由
  `code/scripts/reduce_mobileworld_b0_b4.py` 固定执行，同时报告 observed intersection
  与 strict-117 missing-as-zero；含 focused regression 共 `9 passed`。

## 2026-07-26(夜):Desktop DiD 三臂 policy training 发射(Stage B)

- 发射前闭环:三份 config 过全部启动校验(commit `b88fb75`)、scorer 桌面 schema
  支持(`3085296`)、§8.6 机械审计 PASS(`5edeb6f`,B0 bitwise parity 0.0、mask
  2,584 tokens 恰覆盖恢复图、plain bypass parity 精确、969/969 target 回环)。
- 布阵(每行 torchrun DDP4、accumulation 4 → 16 组/步,150 步 ≈ 3.1 epoch,
  checkpoint 每 25 步,frozen score cache + 组内编码记忆化):
  - hyper01 容器 `sglang-omni-jaxan` GPU 0-3:HGKV,
    run root `/data04/jaxan/mw/runs/desktop-did-v1/hgkv/`;
  - hyper01 同容器 GPU 4-7:matched ungated KV,run root 同级 `ungated_kv/`;
  - hyper00 容器 `sglang-omni-jaxan` GPU 0,1,3,4:Full-layer LoRA,
    run root `/data02/jaxan/runs/desktop-did-v1/full_lora/`(容器内 `/data/...`)。
- 数据:hyper00 侧语料为 hyper01 直传的引用图片子集(2,754 张,2.5GiB)+
  samples/parity_b0/manifest,`samples.jsonl` SHA 与 config 冻结值逐字节一致。
- liveness:launch 脚本终态写 `EXIT_<code>` 哨兵;train.log 每 25 组打
  `sparse_history_diag`;会话侧 5 分钟轮询 diag/Traceback/OOM/EXIT 哨兵,另挂
  gpu-utilization-monitor(min-util 60%、连续 3 窗才告警——teacher-forced 编码
  间隙会周期性压低瞬时利用率)。checkpoint(每 25 步)即 resume 单元。
- 发射期两个排雷:① hyper01 canonical 容器只挂了 GPU 0-3(DeviceRequests),
  ungated 首启 "no GPUs found" 崩;临时容器 `sglang-omni-jaxan-2`(GPU 4-7,同镜像
  同挂载)承载该行,run 结束即删。② 首启崩溃的 EXIT_1 哨兵为 root 属主,host 侧
  rm 静默失败造成一次监控误报,已在容器内清除。
- hyper00 空闲 3 卡(5/6/7,GPU2 为他人工作负载)同时预热 dev-split(94 组)
  frozen + identity 基线分数:`score_sparse_history_arms` 3-shard 共享 cache
  (`/data02/jaxan/runs/desktop-did-v1/devscore-hgkv/`),HGKV checkpoint 的 gate
  归约将全量命中冻结通道。
- gate 打分按 §9 在 dev(heldout 94 组)执行;选点规则 all_must_pass +
  composite=did_select、earliest tie-break 已冻结。
- 运维:hyper01 仓库 origin 已切 `git@github.com:` + Mac agent 转发(`ssh -A`),
  不再走 bundle;hyper00 容器内仓库仍以 bundle 同步(agent socket 不进容器)。

主线 = History-Gated KV Adapter(合同见 [`history_gated_mainline_v1.md`](history_gated_mainline_v1.md),正式 gate PASS、s100 冻结,见 [`data/results/hgkv_gate_v1/`](../data/results/hgkv_gate_v1/README.md))。

## 2026-07-26:Desktop DiD 六臂语料 v1 构建完成,Stage A 停止条件通过

- prototype 升级为交接 §6 的 DiD schema(`causalcache.desktop_did_sample.v1`):
  每组 R0/RA/S0/SA/WA 五行(W0 由 trainer 对 WA bypass 重算),B0 独立
  `parity_b0.jsonl`;irrelevant donor 臂按 §6 冻结目标裁掉。
- trainer 完成交接 §8.2-4:`osworld_official` 编码路径(tools=[_TOOL_SPEC],与在线
  `generate_raw` 逐实参相同)、桌面 schema 契约/量名/必需 gate 按 schema 分派
  (负臂量名 `SA_minus_WA`)、`normalize_desktop_splits`(dev→heldout、test 剥离)、
  plain-LoRA ContextVar bypass 使 Full-layer/ungated-KV 与 HGKV 共用同一 DiD loss
  (新 adapter_type `ungated_kv_lora` = last_N k/v matched 结构对照)。
  相关测试 24+10 项新增,全套 155+ 回归绿。关键事实:v2.1 runtime 与桌面 OSWorld
  runtime 共用同一 GUI-Owl-1.5-8B snapshot 与像素公式(2560 tokens),trainer 无需
  换 runtime 类;端到端由 `audit_osworld_official_trainer_parity.py` 在真模型上锁。
- Hyper01 容器内全量构建:6,003 决策点 → **969 组**(train/dev/test =
  773/94/102,753 条轨迹),action 分布 left_click 56.4% / key 35.2% / type 3.1% /
  double_click 1.8% / right_click 1.4% / drag 1.1% / scroll 0.9%;positive age
  p50=5、p90=10;wrong 距 positive p50=1。**Stage A 停止条件通过**(≥500 组、
  click<85%、type/key/drag 有覆盖),无需先扩 AgentNet win/mac。
- staging:Hyper01 `/data04/jaxan/mw/desktop-did-corpus-v1/`(samples 21.9MB +
  parity_b0 4.1MB + manifest);Git 记录
  [`desktop_did_corpus_v1_manifest.json`](../data/manifests/desktop_did_corpus_v1_manifest.json);
  语料 + 后续 labels 归 `gavinlaw/causalcache-desktop-memory-training`,
  `PENDING_HF_UPLOAD`。
- **真模型 parity audit PASS**:Hyper01 GPU0、GUI-Owl-1.5-8B、2560 visual tokens,
  4 组 × 6 臂 = 24/24,token/image_grid/pixel 三层全部逐位一致、0 失败
  ([`desktop_did_corpus_v1_parity_report.json`](../data/manifests/desktop_did_corpus_v1_parity_report.json))。
  离线语料行 ≡ 在线部署 prompt 的链条(路径序列化→renderer→chat template→视觉
  预处理)已在真模型上锁死。
- 事故记录:corpus build 是纯 CPU 任务而跑在 GPU 容器里,`gpu-fleet-preflight` 的
  0%-5s 规则把 `sglang-omni-jaxan` 容器连同 build 一起收割;已重启容器并重跑
  (5 分钟)。教训:CPU-only 任务与 preflight 不能并行作用于同一容器。
- GPU 现状:hyper00 满载无可清理(0%-5s 规则下无 idle 容器);hyper01 除 mwb0
  emulator 舰队外 GPU 0-2 空闲。三臂 policy training 的发射还差:桌面训练
  config(三份,固定 manifest SHA/split salt/steps/cadence)、
  `score_sparse_history_arms` 桌面 schema 支持(gate 执行件)、§8.6 真模型机械
  测试(HGKV B0 bitwise parity、mask 只覆盖 restored tokens、DDP 单消费)。
- 其余(下一会话):witness 10 点训练臂物化并入语料、HF 上传
  (`gavinlaw/causalcache-desktop-memory-training`)。

## 2026-07-26:OSWorld witness round1 reduce 完成(Stage A.1)

- Hyper01 上对两 score shard(544+546+2 fingerprint = 1,094 行)执行
  `mine_osworld_v2_visual_witness.py --mode reduce`,1,092/1,092 单元全量落账,
  0 缺失;fingerprint 与 score 阶段逐字段一致(GUI-Owl-1.5-8B、480 visual tokens、
  tol 50/15)。
- 聚合层面 selected 增益极弱:`B_selected` mean Δlogprob=+0.0018、loose rate 与
  base 同为 0.1194——印证 witness 的价值在逐点筛选而非平均效应。
- 按交接 §5.2 严格筛(U>0 或修 base-wrong,且严格优于 nearby/random-old/
  next-recent):67 点中 **10 点合格,全部为 amplification,0 repair**;task 分布
  074×4、003×3、107×2、032×1。此即 OSWorld 高精度 seed 的真实规模,AgentNet
  recurrence 仍是训练主体。
- 产物入 Git:[`osworld_round1_witness_report.json`](../data/manifests/osworld_round1_witness_report.json)
  与逐单元 [`osworld_round1_witness_per_unit.jsonl`](../data/manifests/osworld_round1_witness_per_unit.jsonl)
  (Hyper01 staging 同路径)。

## 2026-07-26:停止 Odyssey prevalence 关键路径，冻结 Desktop→MobileWorld 交接

- Hyper00 上 8 个 `mine_rescue_tiers` shard 已只终止 workload process，canonical
  container 保留；8 张 H200 显存归零，`/data/artifacts/causalcache-rescue-v1/tiers/`
  cache/heartbeat 未删除。该线以后只作 prevalence/harm appendix，不阻塞训练。
- Hyper01 已准备 AgentNet Ubuntu 6,003 decision points（2,293 successful
  trajectories；原始数据 374 GB）和 OSWorld 2.0 visual-witness 67 decision points。
  witness 两 score shards 已完成 544/544、546/546，reduce 尚未执行。
- 新主线必须用 Desktop target-specific / recent / wrong 的 DiD 六臂，不能复用遗漏
  `RA` 的旧四臂 gate；policy ablation 固定 Frozen / Full-layer LoRA / matched ungated
  KV / HGKV，final policy 冻结后才可重标 selector。
- MobileWorld 保持完全 zero-shot，不参与 policy/selector/checkpoint/threshold 选择；
  OSWorld task `003/032/074/105/107` 已登记 contamination，只报 in-domain diagnostic。
- 完整执行顺序、停止条件、资源布局、Source of Truth 与 Claude 交接见
  [`desktop_memory_training_handoff_v1.md`](desktop_memory_training_handoff_v1.md)。

## 2026-07-23:terminate 饿死事故闭环——修复验证通过,零样本化验尺 v2 发射

- 事故:零样本化验尺 v1(ody-s75)0/90,零 policy_terminated;根因 = 训练目标 0 个 terminate
  (state 母表只含中段决策点)。修复:渲染器终止态合成(--terminal-states-only,decision=事件数+1,
  全历史+末观测→terminate)产 3,233 样本;混合续训集(终止全量 + 25% 常规)从 s75 续训 60 步
  (v1b,s20/s40/s60)。中途修图像根分叉(终止态涉及母表外轨迹,mix images 双源符号链接)。
- 冒烟验证:s40 版 10 局 B0,4/9 policy_terminated(修复前 0/90)——终止行为恢复;0 成功属难模板
  B0 预期(frozen 同条件 ~7%)。
- 零样本化验尺 v2 发射:v1b-s60 × B0/B8 × 45,retry 协议,frozen 侧沿用 assay-paired-v2。
- 渗漏测试已交接 OSWorld runner 会话(docs/osworld_lora_leakage_test_handoff.md,判定规则预注册);
  仓库工作流改为直推 main。教训入册:数据集验收必查动作类型分布,渲染 manifest 后续附分布统计。

## 2026-07-23:零样本化验尺 v2(s60)裁决——0/90 二连败,定性动作分布负迁移

- 三机联合舰队(hyper01 12w 正向 + H100 6w 反向 + hyper00 9w 中段,27 worker)~1h 收官,union 90/90
  (hyper01 独自跑完全集,余为冗余保险)。结果:s60 B0=0/45 B8=0/45;parse 死亡 0(语法完美);
  policy_terminated 15(终止行为已修复)但时机全错(3 局第 1 步宣告完成,其余 8-31 步散布,
  score 全 0)。frozen 对照 B0=3 B8=6。
- 定性:非机械 bug——GUI-Odyssey 人类演示的动作分布(201 应用/不同布局/操作习惯)在 135 优化步内
  把 policy 的 AW 任务执行力整体带偏,负迁移。teacher-forced 内容门禁过线与闭环 0/90 并存 =
  本文 dissociation 主题的又一实例(层级:teacher-forced 能力 ≠ 自由生成格式 ≠ 闭环任务能力)。
- 战略建议(待用户确认):Odyssey 闭环线停止迭代,资产转分析章(零样本 teacher-forced 认证 +
  闭环负结果 + 两层失败分析);主力转 selector 链,冻结 policy 用 v3-e1(唯一闭环验证过的
  history-aware checkpoint,+8.9pt CI);时间账:deadline ~4.5 天,selector 链需 ~3 天。
# 2026-07-23：OSWorld terminal-s60 cross-platform transfer pilot v1

- Hyper01 使用 2 × H200 并行跑 frozen GUI-Owl 与 terminal-s60，固定 official no-GDrive roster
  evenly-spaced 30 tasks、recent at-most-B4、50 steps、每臂 6 environments / 1 replica；
- frozen/s60 mean OSWorld score=`0.13333/0.07037`，score>0=`4/30` vs `3/30`；
  paired delta=`-0.06296`，bootstrap 95% CI=`[-0.22963,0.10370]`，3 wins / 4 losses /
  23 ties，判定 `NO_EVIDENCE_OF_POSITIVE_OSWORLD_TRANSFER_TERMINAL_S60_V1`；
- s60 task-level HTTP 500 为 1/30，frozen 为 10/30，但 s60 平均只执行 20.13 steps、20 个
  `policy_done`；frozen 平均 37.87 steps，说明 terminal repair 的 executable robustness 未转化为
  desktop reward；
- failure 按 0 分保留在固定 denominator；reducer 现同时读取失败 checkpoint steps、验证 profile 并
  汇总 latency/generation/queue/allocator peak；
- raw：Hyper01 `/data01/jaxan/osworld-runner/transfer-pilot-v1/raw`，715,756,714 bytes /
  1,925 files；Git summary SHA256=`42aff280...608d6`；
- 本 pilot 不含 learned selector，不扩跑完整 361-task OSWorld。结果见
  [`data/results/osworld_transfer_pilot_v1/`](../data/results/osworld_transfer_pilot_v1/)和
  [`docs/osworld_transfer_pilot_v1.md`](osworld_transfer_pilot_v1.md)。

## 2026-07-23(夜):修正版 Odyssey s75 全量零样本化验尺发射(过夜任务)

- 用户过夜指令:canary 完成后跑 90 局,验证坐标修复后的 GUI-Odyssey 训练能否直接泛化 AndroidWorld;
  本会话切回 main。
- canary 12/12:行为面无红旗(0 提前终止、终止时机正常),6 局环境故障为 emulator 启动后 30 分钟
  未稳期瞬态——2.5h 后 adb 验证 11 台全部 8 个关键应用在位(28203 台已死,弃用)。
- 90 局发射:11 worker(28200-28211 去 28203),odyv2-step75(坐标修复版,in-domain c−b0 CI 首次
  转正的那个),retry 协议,输出 /data02/jaxan/runs/odyv2-zeroshot-full/。frozen 对照沿用
  assay-paired-v2。完成后出配对裁决入库。
- 另:history-gated 分支(exp/history-gated-mainline-v1)契约与 provenance 已冻结推送;实现工作流
  因会话重启中断,待明日以 resumeFromRunId wf_1b21d1fe-782 续跑或重发。

## 2026-07-24(晨):过夜任务完成——修正版 s75 零样本裁决(0/90 → B0 翻倍)

- 90/90 收官(11 worker,infra 0):odyv2-s75 B0=6/45(frozen 3)B8=6/45(frozen 6);配对
  s75-B0 vs frozen-B0 净胜 +3(3W0L)Δ+0.067 CI[+0.000,+0.133];parse 死亡 26→1;正常终止 17→52。
  **GUI-Odyssey 外部训练零样本泛化 AndroidWorld 成立(基础能力层),记忆剂量收益仍平**。
  详表 data/results/odyssey_zeroshot_closed_loop_v1/README.md 第三轮。
## 2026-07-23:history-gated 新主线从零到正式训练(exp/history-gated-mainline-v1)

- 分支建于 main@43468dc,provenance + V1 契约冻结(docs/history_gated_mainline_v1.md)。
- 实现(工作流 5 agent):history_adapter_context(ContextVar)/history_token_roles(fail-closed
  mask)/history_gated_lora(B0 零运算 bypass)+ trainer/scorer 集成(full-layer 老路零改动),
  对抗性契约审查 7/7 PASS,单测 25/25。
- **B0 bitwise parity 真模型验证 PASS**(3 样本 max_abs_diff=0.0,16 注入模块),经完整打分管线
  复验漂移 +0.000000(n=76)——架构不变量端到端成立。
- debug 链(全部入 V1.1 备注):梯度检查点重算跑在 autograd 线程 → ContextVar(线程局部)读空
  → 张量数不一致崩溃;修复 = 两遍法(no-grad 求值+次梯度权重,再逐变体 scope 内带梯度前向并
  立即 backward,同时只活一张图)+ history 模式禁用梯度检查点(全激活单图 H200 可容)。
  另:scorer 加载需显式 --lora-rank 8 --lora-alpha 16。
- smoke(15 步)机械项全绿;正式训练发射(hyper00 六卡,ody-sft-v2 修正数据,6,807 组单元,
  165 heldout 轨迹,300 步上限,25 步一档,run root /data02/jaxan/runs/hgkv-formal-v1)。
- 滚动 gate(hyper01,v2 探针包+分支快照已铺):s25 三对照全部正向移动(c−b0 +0.0635→+0.0692,
  c−shuf −0.0169→−0.0160,c−irrel −0.0121→−0.0118),parity 精确;注意 v2 修正目标下冻结模型
  c−b0 本已为正(+0.0635),gate 判定点 = 内容对照 CI 转正处。s50/s75 探针滚动中。
- 同日:full-layer Odyssey salvage 线收档(离线 gate 全过 + canary 提前终止残留),为本主线
  动机证据;全 Docker 隐匿化(sglang-omni-jaxan 命名 + jaxanluo/sglang-omni:{dev,env} 镜像)。

## 2026-07-24:History-Gated KV Adapter 正式 gate PASS,s100 冻结

- 全量认证(165 heldout 轨迹,5,738×2 行,n=1,220 配对组):s100 三条对照 CI 下界全正
  (c−b0 +0.1372[+0.1296,+0.1449];c−shuf +0.0016[+0.0003,+0.0030];c−irrel
  +0.0108[+0.0078,+0.0139]),B0 parity 逐位 0.000000;冻结基线内容对照 CI 认证为负
  (−0.0142/−0.0115)→ 病症确证、修复且反转。合同第一阶段唯一问题答"是"。
- 弱点如实记录:wrong-history drift +0.071/+0.067(增益偏"有历史即放大",内容辨别净
  边际小而显著);详见 data/results/hgkv_gate_v1/README.md。
- 选点轨迹:探针 s25→s150 上升-平台-微降,s100 峰值;训练在 s175 后停止(用户执行)。
- 闭环接线落地(5c26008):worker/runtime 支持 history_gated_kv,K=0 臂 nullcontext 保
  parity,K>0 fail-closed;43 单测绿。
- dev canary 进行中(hyper01,复用 1750xx emulator,修正:真端口 28200/28201 高位映射、
  模板须取自 ceiling 15 名单——sealed-split 防护有效拦截了越界选择)。
- 认证期三连事故(截断 PNG/引号 bug/exec 连坐)全记录于结果 README,结论不受污染。

## 2026-07-24:history-gated 闭环 dev canary 12/12 通过(合同第 11 步 AW 侧)

- hyper01 2 worker × 6 局(RecipeAddMultipleRecipes/MarkorTranscribeReceipt/BrowserMaze ×
  {summary_B0, recent_B8} × instance 0/1),train plan v1 绑定;12/12 完成,infra 故障 0;
- B8 臂逐步实时构掩码零 fail-closed 报错,B0 臂 parity 路径正常;parse 219/241 与冻结水平相当;
- 记录:hyper01 /data02/jaxan/runs/hgkv-canary-v1/(逐局 json);性能不读(canary 纪律);
- 排雷三则:emulator 容器真端口为高位映射(docker port 查,28200+);--ceiling-plan 必须传带
  split 字段的 plan 清单(data/manifests/androidworld_train_plan_v1.json),不是模板 config;
  1750xx emulator 容器内 android_server 需手工复活(cd / && python3 -m server.android_server,
  容器内固定绑 5000,由 -p 映射出去)。
- 下一步 = 合同第 12 步 sealed 零样本评测矩阵:{Frozen, Full-layer(ody-margin-v2 s75),
  History-gated(hg-s100)} × {B0, Recent-B, Full-history} × sealed 名单;先冻结协议清单再发射。

## 2026-07-24:sealed 矩阵发射(合同第 12 步)

- 协议冻结 06722d4 后发射:hyper01 11 worker(gated×4/full×4/frozen×3,GPU 0/1/2/3/7,emulator 28200-28211 缺 28203),675 局 = 3 策略 × 225 cell(25 模板 × 3 instance × 3 臂),输出 /data02/jaxan/runs/sealed-matrix-v1/;预计 7-10h;
- 发射前事故:首版布阵脚本把 cell 按策略三分(总数 225),断言后置未拦住执行——11 台误配 worker 在模型加载期全部击杀,0 局消费,封存集未受损;修正版每策略全量 225 cell、断言前置、门控执行。

## 2026-07-24:fl75 selector 裁决(线路前缀规约生效)

- 用户裁定:s75 系 selector 全部归 full-layer 线(前缀 fl75-),非主线;主线 selector 挂
  history-gate(hgkv-)。hyper00 目录已改名,文档移至 data/results/fl75_selector_v1/。
- fl75 selector 科学结论:组内选择无信号(方差 0.8%,V1 多模态模型不敌 recent);状态级
  门控有信号(ridge Spearman 0.456)。落地形态 = gate + recency。详见结果目录 README。
## 2026-07-24:全量 116-template 零样本矩阵发射(取代 25×3 sealed)

- 用户裁定:paper 全线零样本(Frozen/Full-layer/History-gated 均 Odyssey 训练+选择,AW 不参与),
  margin-SFT 线从 paper 删除;主表 = 116 template × 2 实例(index 0/1)× 3 policy × 3 arm = 2,088 局;
- roster:androidworld_full_suite_plan_v1.json(232 实例,sha 1dd48941,三源 plan 合并,记 origin_split);
- 剔除三视图:headline 全 116;稳健子集显式剔 2 绘图(BrowserDraw/SimpleDrawProCreateDrawing)+
  7 trivial-verify(文献先例);硬删仅限 env-init 失败者(policy-agnostic,证据驱动,跑后列出);
- 复用旧 25×3 中 test-25×{0,1} = 327 局;缺失 1,761 局分 15 worker(每策略 5;h00 6 + h01 9),
  host-aware checkpoint,GPU 隔离;checkpoint 加载确认通过;扩容 9 台新 emulator 因并行装 APK
  冲突失败已清,15 台既有 emulator 稳跑;
- 旧 25×3 的 index-2 局保留作附录 within-template 方差检查。

## 2026-07-24:HGKV-readout 特征抽取接力与 8-shard wrapper 修复

- 接管时确认 `main=origin/main=exp/history-gated-mainline-v1=ad2625f`；conditional
  heldout 已 `DONE`(`failed_shards=0`,45,227 rows)，train 仍由 hyper00 CPU renderer
  运行；cheap selector 三件产物齐全，heldout model−Recent-1=`-0.0030`
  CI `[-0.0046,-0.0014]`，保持 appendix negative baseline 结论。
- hyper01 preflight 连续 5 秒确认 GPU 0-7 全部 free，零 idle container 需清理；8 个稳定逻辑
  shard 映射到物理 GPU 0-7。launch-time snapshot 保存于
  `/data02/jaxan/runs/hgkv-readout-v1/launch-gpu-snapshot.csv`，八卡均为 0 MiB。
- singleton 输入从 hyper00 直传 hyper01：`samples.jsonl` SHA256
  `68fef3bb6848f7c949640efa7f51123c14c783625862dfb7bf8f41a6dab2eaa0`，
  补齐 5,759 张缺图后的逐文件 manifest 聚合 SHA256
  `97839c274509bec4a8c0aa14177c21ed35b2ad00991658ef491ba65457f9da79`；
  最终 75,628/75,628 singleton、14,680 unique image refs、missing=0。
- 第一次容器 `sglang-omni-jaxan-07241348` 在 0-row 时 `Exited(127)`：cluster launcher
  外层 shell 提前展开内联 `$shard/$run_root`，所有参数变空；失败容器已删除，未污染输出。
  新增 Git-tracked `run_hgkv_readout_extraction_shards.py`，以独立 Python subprocess
  编排 8 shard，显式记录 input/checkpoint SHA、完整 child argv、GPU 映射、source commit，
  终态强制验证 75,628 unique keys、统一 feature_dim 与逐行 feature 长度后才写 `DONE`。
- wrapper 初次重发在 0-row 时因容器未设置 `PYTHONPATH=code` 退出，失败容器同样已清；
  最终 `sglang-omni-jaxan-07241352` 以 `env PYTHONPATH=code` 正常运行，run manifest
  固定 source commit `7ef68e4`。conditional train 同期完成：255,490 rows、
  `failed_shards=0`；heldout/train 两段现均具 `DONE`。

## 2026-07-24:全量零样本矩阵正式聚合器冻结

- 新增 `causalcache.sealed_matrix_v1` 与 `scripts.aggregate_sealed_matrix_v1`，权威 roster
  固定 116 template × index 0/1 × 3 policy × 3 arm = 2,088 cell；policy 身份由 frozen
  无 adapter、fl-s75 SHA `75670fa5...`、hg-s100 SHA `8f2cc49e...` fail-closed 解析；
- 三视图名单落成代码常量：headline 116；action-compatible 107（2 drawing + roster 中
  唯一 7 个 `*Verify`）；hard-delete 只认同一 instance 九个 cell 均有零步 env-init void
  且从未真实开跑的 policy-agnostic 证据；
- 聚合器忽略旧 sealed-25 index-2 appendix；零步 infra void 不计 cell，非零步 infra 保持
  待重试；非 infra 重复正式局、坏记录与 unexpected cell 均阻断 COMPLETE。指标先在
  template 内平均两个 fixed instances，再做 template-macro 与 10k paired 95% bootstrap；
- 两机真实记录联合 dry run schema 验收通过：1,046 个去重 attempt 中 417 个正式 cell、
  621 个零步 void、8 个开跑后 infra，缺 1,671；零 rejected、零 unexpected、零非 infra
  重复。这是首次执行中快照；后续 worker 推进会自然改变计数，不读取为科学结论。
- hostile review 对 manifest/test evidence 的质疑由 committed 232-instance roster
  (`1dd48941...`)与真实 dry run/单测证据排除；接受并修复 roster 必填字段的显式校验，
  同时新增 producer success 公式复核、policy/checkpoint mismatch、开跑后 infra 与九 cell
  policy-agnostic env-init hard-delete 覆盖。数值 success 保持现有 episode schema 的
  `0.0/1.0` fail-closed 类型，不接受 JSON bool；hard-delete 仅为敏感性视图，不能令缺失
  headline 获得 `COMPLETE`。
- 第二轮 hostile review 接受一项 operational 修复：聚合器改为始终写 `STATUS.json`、仅
  2,088 headline cell 全齐时写 `DONE`，避免 incomplete dry run 被监控器误判为完成；
  identity 预筛也只静默忽略已知 index-2 appendix，其余类型错或 roster 外记录全部进入
  rejected audit。两机复跑时 worker 已推进到 426 formal / 1,662 missing，policy breakdown
  frozen/full-layer/history-gated=`137/131/158`，arm breakdown
  B0/B4/B8=`107/209/110`；零 rejected、零非 infra 重复，且只出现 `STATUS.json`、没有
  `DONE`/`DONE.json`。这些仍是 liveness 证据，不是科学结果。
- 新增 Git-tracked `scripts.plan_sealed_matrix_relaunch_v1`，消费 formal aggregator 的
  `missing-cells.jsonl` 与当轮 preflight 后显式给出的 `POLICY:GPU:PORT`；worker 严格单
  policy，policy 内按 roster frozen `max_steps` 做 LPT 均衡并验证全覆盖、GPU/port 唯一。
  由此替代会随 session 消失的 `/tmp/relaunch.json` 分配逻辑；实际 Docker 发射仍在最后
  旧 worker 退出和新 preflight 后执行。
- 两机真实 CLI smoke（执行中快照）从 439 formal 得到 1,649 missing，六个假定 worker
  以 frozen/full-layer/history-gated 各 2 个分配 `277/278`、`282/283`、`264/265`
  cells；对应 frozen max-step load `5058/5056`、full `5167/5166`、gated
  `4730/4738`，全覆盖且 policy 隔离。正式发射会重新聚合最新结果，绝不复用本 smoke 清单。

## 2026-07-24:HGKV singleton selector Stage-1 架构与训练规则冻结

- 冻结 config：`code/configs/hgkv_selector_stage1_v1.json`。输入严格为 8 层 × 160 维
  HGKV counterfactual readout；共享 160→64 layer encoder + learned layer attention，
  不读取 cheap metadata 或原图，联合 gain regression、state 内 pairwise rank、positive
  与 state-level STOP 四个头。
- 固定训练规则：SmoothL1 + pairwise logistic rank + positive/STOP BCE；train episode
  `GroupKFold(5)` 只选 epoch，五折 best epoch 的中位数用于全 train refit；单一架构、
  单一 seed、30 epoch 上限、patience 5，不做 heldout hyperparameter search。
- heldout 读取纪律：full-train refit 后仅一次，输出 Spearman/top-1/sign precision/regret、
  exactly-B1 真实 U_act 相对 Recent/Random 的 10k paired bootstrap CI；它仍是 singleton
  训练诊断，不替代 B1/B2/B4 selected-set 正式 gate。

## 2026-07-24:HGKV selector persistent runtime 冻结

- 正式 `jaxanluo/sglang-omni:dev` runtime 提供 Python 3.12.3、torch
  `2.11.0+cu130`、numpy `2.3.5`、scipy `1.17.1`，但不含 Stage-1/2 的
  `GroupKFold` 依赖；在特征抽取容器不变的前提下，新建 persistent system-site venv
  hyper01 `/data02/jaxan/envs/causalcache-selector-v1`；
- venv 固定 scikit-learn `1.9.0`、joblib `1.5.3`、narwhals `2.24.0`、
  threadpoolctl `3.6.0`，完整 pip 侧 lock 写入
  `code/requirements/hgkv_selector_v1_lock.txt`，并同步为 `pyproject.toml[selector]`
  optional dependency；实机 import 版本核验通过。

## 2026-07-24:Stage-2 conditional base 契约阻断修复

- GPU conditional scoring 发射前的 hostile review 发现 renderer/trainer 真实错配：
  renderer 为第二层 anchor 合法产生二元素 `cond_base({i,j})`，trainer 却只接受 singleton
  base，正式 Stage-2 会在读首批真实 render 时 `ValueError`；因 scoring 尚未开始，没有消耗
  正式 conditional GPU 算力；
- 修复后 `cond_base({i,j})` 是 `cond_edge2` 的权威减数；它必须与同集合
  `cond_edge1` 分数 `atol=1e-8,rtol=0` parity。singleton `cond_base({i})` 也从仅报告
  drift 升级为强制对冻结 singleton score parity；任一不一致 fail closed；
- 回归 fixture 现包含真实的 1-event base、edge1、2-event base、edge2 全链，并新增 pair-base
  drift 拒绝测试。该修复保持真实 marginal 定义，不使用 singleton utility 求和。
- 对已完成 render 做全量 prompt-content 审计：heldout 3,242 个、train 18,118 个
  二元素 `cond_base` 全部能 join 同 `restored_set_key` 的 `cond_edge1`，且
  `messages + target_text + memory_config` SHA 逐对一致，missing/mismatch 均为 0；
  因而强制 score parity 有真实输入依据。
- 发射前 scorer resume 审计再修一处：同一 restored set 的对称 edge 共享
  `(pair_group,variant,singleton_event_step_id,restored_set_key)`，旧 scorer 只载入启动前
  output keys，本轮写后不更新，因而会重复 forward/落盘并被 wrapper 终态 duplicate 检查拒绝。
  新路径在进程内首次遇到即 claim；wrapper 同时验证一个 resume key 不得跨 physical input
  files，避免 8 个 file-shard process 之间产生不可见重复。
- 真实 render inventory 验证该分片不变量：heldout `45,227 raw / 38,836 unique /
  6,391 same-file duplicates / 0 cross-file`；train `255,490 / 219,549 / 35,941 / 0`。
  31 个 selector/scorer 聚焦测试全过后才允许发射 conditional scoring。
- conditional scoring 是独立 wrapper 路径，在 141GB H200 上按已测单进程约 18–20GB
  显存配置每卡 2 个 slot：16 logical shards 映射
  `0,0,1,1,...,7,7`，32 个 physical render files 每 process 2 个；feature extractor
  的一 GPU 一 shard 约束不变。slot 映射、每个 child argv 与 input manifest 全部进
  launch manifest，若真实长 prompt OOM 则保留 resume 行并降回 8 slot 诊断。
- 修复后的第三轮 hostile review verdict=`LOOKS_OK_WITH_RISKS`、无 must-fix。保留并要求
  结果侧披露的核心限制是 B4 第四个候选决策在 `|S|=3` 上结构外推（训练 edge 只覆盖
  `|S|=1/2`）；正式 gate 仍用完整集合真实 U(S)，不会用预测 marginal 冒充真值。新增
  B4 三元素 selected-mask/remaining-index/STOP 路径测试，并补 extractor 按升序 layer
  拼接 8×160 chunk 的显式 layout 回归。

## 2026-07-24:conditional hg-s100 打分器的流式 file-shard 路径冻结

- conditional renderer 的物理布局为每 split 32 个 `samples-shard*.jsonl`；heldout/train
  分别 45,227/255,490 rows、`failed_shards=0`。旧 scorer 只认单个 `samples.jsonl`
  且整表一次性载入内存，不适合 8 进程读取 2.9GB train JSON。
- `score_success_action_recovery.py` 新增向后兼容的 `--samples-glob` 与
  `--shard-by-file`：默认单文件/按行 modulo 行为不变；conditional 正式路径把排序后的
  32 文件按 8 worker 分成每卡 4 文件并流式解析，避免 8 路重复全表 I/O 与内存副本。
- 新增 `run_success_action_scoring_shards.py`：启动前冻结每个输入 shard SHA256、
  聚合 manifest、checkpoint SHA、GPU 映射与完整 argv；终态按 scorer resume identity
  `(pair_group,variant,singleton_event_step_id,restored_set_key)` 验证零缺失、零意外、
  零重复和有限分数后才写 `DONE`。该路径将在 HGKV-readout 8 卡抽取完成后复用同一 allocation。

## 2026-07-24:HGKV set-conditioned selector Stage-2 与 STOP 比较规则冻结

- 冻结 config：`code/configs/hgkv_selector_stage2_v1.json`。Stage-2 从 Stage-1
  `HGKVReadoutEncoder` 初始化，只新增 1 个 selected-set self-attention、1 个
  candidate-query attention、budget embedding 与 marginal/rank/positive/STOP 轻头；
  不引入完整 Set Transformer 或新 multimodal backbone。
- 条件标签按真实 policy 分数差构造：`cond_edge1=U({i,j})−U({i})`，
  `cond_edge2=U({i,j,k})−U({i,j})`；B2/B4 的第一层和 B4 的第二层按 config 显式复制
  remaining-budget 条件。训练数组只保存 singleton feature row index，不复制
  1280-d feature table，避免数十万 group 的多 GB 冗余。
- STOP 可比性在读标签前冻结：pairwise rank loss 同时训练 candidate↔candidate 与
  candidate↔STOP(STOP 真值固定为 0)；推理只在最高 candidate rank score 严格高于
  STOP rank score 时添加事件。gain/marginal 头保持真实 U 单位，不与 STOP probability
  混比；positive 头只作辅助校准。
- `select_hgkv_sets_v1.py` 固定 B1/B2/B4 at-most-B 推理：首步用 Stage-1，后续逐步用
  Stage-2，B4 的第三步后以同一 set-attention 结构外推到 3-element selected set；
  每步保存 candidate/STOP rank、预测 marginal、positive probability 与选择前集合。
- Stage-2 与 Stage-1 使用相同 train-only 5-fold epoch selection、full-train refit 和
  heldout-once 纪律；conditional edge 指标仍只是训练诊断，正式结论必须来自所选完整集合
  的重渲染、hg-s100 重打分与 B1/B2/B4 paired CI。

## 2026-07-24:selector V1 NO-GO 与论文候选 V2 裁定

- Stage-1 heldout realized `U=0.104780`，Recent `U=0.113203`，差值 `−0.008423`
  （95% CI `[−0.010526,−0.006338]`）；mean Spearman `0.022291`、top-1
  `0.173967`，没有学到优于 Recent 的 singleton 排序。`STOP accuracy=0.906848`
  约等于类别 majority baseline，不作正面证据。
- V1 set-conditioned selector 的第一张完全由 Stage-1 决定，B1 与 singleton 路径必须
  parity；原 gate 又要求 B1/B2/B4 每个预算都显著胜 Recent/Similarity/Random。因此 V1
  已由 B1 确定为 NO-GO，Stage-2 无法挽救原 conjunctive gate。
- 正在运行的 V1 不停、不调参：继续完成 conditional scores、Stage-2、selector inference、
  B1 parity 和真实 selected-set 重打分。结果定位为 interaction diagnostic，重点比较
  B2/B4 `set-conditioned − singleton`、集合冗余、后续 STOP 和完整集合 `U(S)`。
- 当前 edge1/2 coalition scores 是可复用 teacher labels；未来更换 student/loss 不重打
  已有集合。若 B4 进入正文，V2 增量增加
  `cond_edge3=U({i,j,k,l})−U({i,j,k})`，复用已有 triple denominator，只稀疏生成新的
  four-event numerator。
- 论文候选 V2 优先为 Recent-seeded set-conditioned selector：B1 与 Recent parity，
  B2/B4 从 Recent-1 seed 学习 edge1/2/3 与 STOP；B1 只做 parity 审计，B2/B4 才执行
  superiority gate。统一从空集合学习 edge0–edge3 保留为高风险备选。
- 该裁定只定义后续 versioned V2，不修改已冻结并执行中的 V1 config、数据、标签、选择或
  gate。

## 2026-07-24:V1 被 full-history beam-4 V2 supersede 并封存

- 用户以新 V2 code plan 取代“继续跑完 V1 diagnostic”的旧裁定。已停止本地 overnight
  supervisor、hyper01 固定容器内 V1 conditional wrapper 和 12 个 scorer；Stage-2
  training、selector inference 与 selected-set gate 均未启动，也不会自动恢复。
- 停止后 train partial score 为 `160,928/219,549` unique identities，12 个 JSONL shard
  duplicate=0、invalid JSON=0；heldout 保持 `38,836/38,836 DONE`。所有 render、partial
  score、Stage-1 checkpoint、singleton readout、B0/singleton score 原地保留。
- 四个持久化位置写入同一 `ABORTED.json`，状态
  `ABORTED_BY_SELECTOR_V2_SUPERSESSION`，SHA256
  `48510c1487ec266282cce15c32ae884322fc5dcd970bd7d85ca5462228078cdf`。
- V1 coalition score 数值只作为 V2 canonical exact-key cache 候选；不继续完成旧
  recent-8/singleton-proxy prefix 分布。V1 Stage-1 留作论文 negative independent
  baseline，V1 不作为主方法。
- 后续冻结 full-history、empty-start、true-U teacher beam-4、edge0–edge3、fresh unified
  student、learned beam-4 + STOP 的新 V2 契约；该 supersession 不修改 V1 历史 config
  或源码。

## 2026-07-24:HGKV selector V2 beam-4 契约冻结

- 新增 `docs/selector_v2_beam4_protocol.md` 和
  `code/configs/hgkv_selector_v2_beam4.json`，在读取任何 V2 development score 前冻结：
  longest-real-state full-history inventory、canonical exact-key coalition cache、
  1280-d HGKV readout + 5 temporal features、true-U teacher beam-4、edge0–edge3、
  fresh unified student、train-only GroupKFold checkpoint selection、learned beam-4
  inference、STOP 和正式 gate。
- teacher 从空集合开始，每层按完整集合真实 `U(S)` 保留 top-4 unique prefix；禁止
  singleton sum/proxy。student 也从空集合 fresh init，不加载 V1 Stage-1。
- gate 固定 B1 为相对 Recent 无显著劣化，B2/B4 对 Recent/Similarity/Random 全部
  superiority，Avg(B1,B2,B4) 相对 Recent superiority；统计为 episode 内先平均、
  10,000 次 episode-cluster bootstrap。
- 新增 `data/results/hgkv_selector_v2/README.md` 作为结果与 artifact SoT。V2 大规模
  dataset 暂定 intended HF dataset repo
  `gavinlaw/causalcache-hgkv-selector-v2`，当前仅为未验证的 `PENDING_HF_UPLOAD`
  destination，不冒充已创建或已上传。

## 2026-07-24:V2 longest-real-state full-history inventory 完成

- source commit `97578b2c3ae59f0172689623415de10f566df23f` 在 hyper01 固定容器
  `sglang-omni-jaxan` 内 CPU-only 执行；三输入为 packaged trajectory shards、
  annotations 和已验证 state-context，输出 staging
  `/data02/jaxan/runs/hgkv-selector-v2/inventory/`。
- 1,000 条成功 trajectory 全部得到一个最长真实 eligible decision；train/dev
  `835/165`。candidate count min/median/p90/p95/max=`5/12/24/27/44`，全部至少 4，
  其中 666 个 state 超过 8 candidates。
- synthetic terminal=0、recent-8 truncation=0、ineligible decision=0。Action 分布
  click/scroll/text=`912/17/71`；foreground app 按训练 schema 保持 `unknown`，不进入
  feature。
- Git-tracked train/dev/audit SHA256 分别为
  `9c254298...ce0137`、`3b545dc4...5f492c`、`aa24ea87...3b3a3c`，与 hyper01
  staging 完全一致。

## 2026-07-24:V2 initial canonical coalition cache 完成

- source commit `d1113cc961e651d414f5f3b43dcd69b46930c4ac` 摄取 frozen B0、
  V1 hg-s100 singleton、V1 heldout conditional DONE 和 V1 train conditional partial；
  只按包含 target/prompt/checkpoint/B0 policy SHA 的 canonical exact key 复用，冲突
  fail closed。
- cache 共 `243,455` rows/unique keys，含全部 10,680 个 B0 empty set；JSONL SHA256
  `b3bb9856...a6805b7`，staging 为 hyper01
  `/data02/jaxan/runs/hgkv-selector-v2/coalition-cache/`。
- 对 V2 1,000 states：B0 `1,000/1,000`；full-history singleton required/cached/missing
  `13,680/7,668/6,012`，只有 334 states 已全覆盖。V2-state cache 中 size 0/1/2/3
  coalition 为 `1,000/7,668/10,732/5,372`。
- manifest 入 Git 且本地/远端 SHA256 同为
  `1c6b052c...ecb152`；大规模 cache 仍为 `PENDING_HF_UPLOAD`，未冒充远端 canonical。

## 2026-07-24:V2 full-history missing singleton 渲染完成

- source commit `7056fec474ef8e3813887785c8f0e5a0a54c4de2` 在 hyper01 固定容器
  `sglang-omni-jaxan` 内执行；只渲染 V2 inventory 相对 initial exact-key cache 的
  complement。train/dev 为 `5,143/869`，合计 `6,012/6,012`。
- validator 对完整 1,000-state inventory、canonical singleton key、恢复集合、
  5-d temporal metadata 和 6,678 个引用图片逐项检查；duplicate/unexpected/missing
  均为 0。与已有 7,668 个 exact singleton 合并后，full-history singleton 覆盖
  `13,680/13,680`。
- hyper01 完整 `DONE` SHA256 为
  `e30e3923...71bf82`；轻量 Git summary 见
  [`data/results/hgkv_selector_v2/singleton-render-manifest.json`](../data/results/hgkv_selector_v2/singleton-render-manifest.json)。
- 6 张 H200、每卡 2 scorer 的 hg-s100 打分已在同一固定容器启动；必须等待 6,012
  unique score identities 全量 validator 后才能更新 coalition cache 和进入 teacher beam。

## 2026-07-24:selected-set 正式 gate 的 plan→render→score→reduce 路径冻结

- `select_hgkv_sets_v1.py` 同时物化 HGKV singleton independent 与 HGKV
  set-conditioned 两路 B1/B2/B4 at-most-B 选择；两路共用 Stage-1 首步，
  singleton 后续按独立 rank 填充，set-conditioned 后续逐步重算条件边际。
- `build_selector_gate_plan_v1.py` 将 learned 两路与 Recent、deterministic Random、
  OCR+RGB Similarity 合成 5-method plan。Similarity 沿用已冻结的
  `0.5*OCR token-set Jaccard + 0.5*256x256 joint-RGB-histogram cosine`，tie 取较小
  event id；Random seed 固定 20260724 并由 SHA256 排序取 exact-B 子集。
- Odyssey renderer 新增互斥的 `--selected-sets` 模式；同 state 内多个
  method/budget 若选出同一 coalition，只渲染一次，按
  `(pair_group,variant=selected_set,restored_set_key)` 打一次 hg-s100 分，再由 plan
  回填到各 method，禁止 singleton 求和。空集也真实走 B0/parity prompt，不特殊伪造分数。
- `reduce_selector_selected_set_gate_v1.py` 用 exact B0 join 计算
  `U_act(S)=logp_hg(S)-logp_frozen(B0)`；primary 统计单位为 episode，先在 episode
  内平均 state，再做 10k paired cluster bootstrap。冻结 PASS 条件：B1/B2/B4 每个预算上，
  set-conditioned 相对 Recent、Similarity、Random 的 95% CI 下界全部严格大于 0；
  否则 `NO_GO_SELECTED_SET_GATE_V1`。HGKV singleton 同表报告但不进入该 conjunctive gate。
- Oracle 不混入本 gate 的 PASS 判定：B1 可 exact；B2/B4 的小候选 exact 与长历史
  beam/approx oracle 需单独物化并明确标注，不能把当前近线性 conditional label coverage
  冒充全局 subset oracle。

## 2026-07-24:AAAI draft 主结果表与 selector 分析表同步

- `paper/main.tex` 按冻结零样本叙事加入两张正文表骨架:Table 1 为 AndroidWorld /
  OSWorld 上下 panel,统一六行 policy-selector 对照与 B0/B1/B2/B4/B8/Avg(B>0);
  Table 2 固定 HGKV policy,报告 GUI-Odyssey development 上 B1/B2/B4 真实 selected-set
  utility、oracle recovery/regret、STOP rate 与 latency。
- 所有未完成数值统一显式写为 `TBD`;HGKV 三行 B0 标为与 Frozen 相同,caption 记录 B0
  structural bypass。Full-layer 与 Top-8 ungated KV 的 B0 仍需独立评测。
- appendix 已为 closed-loop selector controls、HGKV 4/8/12 层、rank 4/8/16、set encoder、
  loss 权重与 corruption variants 建占位;正文主表不展开 policy × selector 组合。
- draft 的实验协议同步为 adapter/selector 全部只在 GUI-Odyssey 训练与选择、AndroidWorld
  116 templates × 2 fixed instances 的 template-macro success、OSWorld full fixed roster 的
  mean normalized score;Table 2 明确禁止用 singleton gain 求和替代完整集合重打分。

## 2026-07-24:AAAI-27 AuthorKit 格式对齐

- 用户提供的 `AuthorKit27.zip` SHA256 与仓库既有记录一致；`aaai2027.sty/.bst` 与官方
  文件逐 byte 相同。`paper/main.tex` 保持 `letterpaper`、`submission`、匿名作者块、
  `TemplateVersion (2027.1)` 与官方字体链，不引入禁用 package 或 page-layout 命令。
- 按 AuthorKit27 的 table 规则，将三张 table 的 caption 从表格上方移到下方；9pt
  `\small` 只作用于 table body，caption 回到官方 10pt。
- AAAI-27 官网当前规定主 PDF 最多 9 页、非 references 最多 7 页，第 8--9 页仅用于
  references；reproducibility checklist 必须单独上传。官方
  `ReproducibilityChecklist.tex` 已作为独立待填写模板纳入 `paper/`，不嵌入主稿。

## 2026-07-24:HGKV 论文重写与 bibliography 合入 main

- 将 `agent/rewrite-hgkv-paper-draft` 的三项论文提交合入 canonical `main`：
  `f544723` 重写 HGKV / set-conditioned selector 主叙事，`196c540` 重建 AAAI
  author--year bibliography，`05752ab` 将引用放到对应 claim 与实验设定处。
- `paper/references.bib` 当前 15 个 entries 与正文 15 个 cite keys 双向完全对应，无缺失、
  重复或未使用条目；BibTeX 实际使用 `aaai2027.bst`。
- 合并后将过宽的 HGKV counterfactual-readout 公式拆为两行，并等义压缩 Introduction
  首段一行。`make paper` 生成 6 页 US Letter PDF；最终日志无 overfull、undefined
  citation/reference、空 `booktitle` 或空 `journal` 警告。

## 2026-07-24:AAAI arXiv reference 类型纠正

- AuthorKit27 的 Reference Examples 明确规定 arXiv 论文使用 `@misc`，而不是
  `@article`。将 Mobile-Agent-v3.5、MementoGUI、AndroTMem、CMI 和 ATMem 五条
  2026 preprint 改为 arXiv 官方导出的字段结构，保留原 cite keys。
- 每条均记录 `eprint`、`archivePrefix={arXiv}`、arXiv API 核对的 `primaryClass`
  与 canonical abstract URL；Mobile-Agent-v3.5 / CMI 为 `cs.AI`，其余三条为 `cs.CV`。
- `make paper` 后五条均按 `aaai2027.bst` 渲染为普通文本 `arXiv:<id>.`；6 页 PDF
  保持不变，日志无 BibTeX warning、undefined citation/reference 或 overfull。

## 2026-07-24:HGKV-readout 截断图片恢复与 sparse shard resume

- 首轮 8-shard 抽取终态为 69,383/75,628：shard 0/2/3/4/7 完整，shard 1/5/6
  分别停在 7,372/7,371/7,372 行。三个 traceback 均指向同一输入
  `images/9471050986960951/observation-004.png`；该文件仅 8,704 bytes，PNG IDAT
  声明 8,192 bytes 时只剩 430 bytes。
- 对 singleton `samples.jsonl` 引用的 14,680 张唯一 PNG 做逐 chunk 长度与 CRC
  扫描，确认坏图恰为 1 张。`sft-labels/ody-labels-single` 与
  `sft-labels/ody-sft-v2` 的完整副本均为 231,296 bytes、SHA256
  `c6d222fca77bb3827c3dc6b1e19e5a1355b012cbc4d71dc81fdfa25bc8e6a70d`；
  原坏图已备份到
  `/data02/jaxan/runs/hgkv-readout-v1/recovery/9471050986960951-observation-004.png.truncated`，
  并以该一致副本原子替换。
- wrapper 新增 `--shard-indices`，允许只把失败逻辑 shard 映射到所选 GPU，同时
  保持最终对全部 8 个输出做 75,628 unique-key、统一维度、逐行长度和有限值验证。
  extractor + wrapper 相关测试 15/15 通过；实现与恢复记录由 commit `d6973a7`
  推送 canonical `main`。
- hyper01 5 秒 preflight 确认 GPU 0-7 全空；只取 GPU 0/1/2 映射 shard 1/5/6，
  launch-time 三卡均为 0 MiB。恢复容器
  `sglang-omni-jaxan-07240114` 使用精确 source commit
  `d6973a78cb3fb8fab19795b62a6ebeedcb197eca`，终态 exit 0、OOM=false，已按规则删除。
- `DONE` 于 `2026-07-24T08:43:56Z` 生成：75,628 rows / 75,628 unique keys、
  `feature_dim=1280`；shard 0-3 各 9,454 行、shard 4-7 各 9,453 行，逐行有限值
  验证通过。完整 8-shard SHA/大小、输入与 checkpoint provenance、恢复证据和命令见
  [`data/results/hgkv_readout_v1/`](../data/results/hgkv_readout_v1/README.md)；
  大 feature artifact 仍为 hyper01 local staging，`PENDING_HF_UPLOAD`。

## 2026-07-24:AAAI 主结果按 policy-use / selector 两个 panel 重构

- 将原先独立的 policy adaptation 表和 selector 表合并为同一个编号的 `table*`，但保留
  两套独立列头。Panel A 固定 Recent，比较 Frozen / Full-layer LoRA /
  Top-8 ungated KV / HGKV 的 B0/B1/B2/B4/B8 与 `Avg. B>0`；Panel B 固定 HGKV，
  比较 Recent / marginal / set-conditioned 的 B1/B2/B4、`Avg. B1--4` 与
  selected-4 相对 Recent-8 的差值。
- selector panel 不再暗示支持 B8；set-conditioned 的 B1 明确写为与 marginal 相同，
  因两者共用 Stage-1 首选。两种 Avg. 的预算集合分别在列头和 caption 中定义，避免不可比。
- Random 与 OCR/RGB Similarity 从端到端 headline panel 移到 selector mechanism /
  ablation 对照；ablation 文本同步冻结为 policy-use 与 selection 两条正交轴，不展开
  policy × selector Cartesian product。所有未完成结果继续使用 `TBD`。
- `make paper` 生成 6 页 US Letter PDF；逐页渲染检查确认两组 panel、caption、正文换页和
  references 无裁切、重叠或不可读列头，日志无 overfull、undefined citation/reference。
  同步补齐 `\method` / `\hgkv` 宏后的显式空格，消除 PDF 中的连字。

## 2026-07-24:HGKV selector Stage-1 全量训练完成

- 先验收 HGKV readout `75,628/75,628` unique rows、1,280 dims、8 shard 全量有限值，再
  exact-join singleton train/heldout `64,264/11,364` candidates 和 `9,059/1,621`
  states；无缺行、重复或 feature mismatch。
- frozen 5-fold epoch selection 得到 `[29,30,27,21,5]`，median 27 epochs 后在全
  train refit；hyper01 H200 运行完成，checkpoint
  SHA256=`775bfbca851abf535b286b674f7642a14a7915a8b2aea65a640c7088b5430fb3`。
- heldout singleton realized `U`：model/Recent/Random/Oracle =
  `0.104780/0.113203/0.103396/0.133664`；model−Recent=`−0.008423`
  (95% CI `[−0.010526,−0.006338]`)，model−Random=`+0.001384`
  (`[−0.000059,+0.002828]`)。该负结果只作冻结 Stage-1 诊断，不据此调参，也不替代
  B1/B2/B4 selected-set gate。
- Stage-1 仍按协议提供 Stage-2 encoder 初始化、首步和 independent baseline。conditional
  renders 已全量完成；hyper01 固定容器 `sglang-omni-jaxan` 已按 6 张 H200、每卡 2 个
  scorer 发射 heldout→train 流水线，终态须分别达到 38,836 / 219,549 unique identities
  才允许 Stage-2。
- train 的 32 个 physical render files 直接 round-robin 到 12 scorer 时 unique 负载为
  9,401–23,708，存在 2.5× 尾部不均。新增 deterministic score-identity hash rebalancer，
  raw 255,490 与 unique 219,549 均完全守恒、跨输出 key 重复为 0；12 个 balanced shard
  收敛到 18,001–18,602 unique identities。该视图只重排行，不改变 prompt/图片/标签；
  输入输出逐文件 SHA 和计数写入 persistent `DONE`。
- 冻结决策语义同步进协议与 paper：B1 两路相同；query 已编码进 candidate readout；
  inference 比较 candidate rank 与 STOP；shortlist 只用于标签构造；B4 第四步
  `|S|=3` 为结构外推。结果与 provenance 见
  [`data/results/hgkv_selector_v1/`](../data/results/hgkv_selector_v1/README.md)。

## 2026-07-24:Selector V2 singleton score、edge1 render 与 exact-search 基建

- V2 missing singleton 的 train/dev hg-s100 scoring 已分别完成
  `5,143/869`，12 个 scorer 全部 exit 0；singleton-complete coalition cache 为
  `249,467/249,467` unique exact keys，V2 1,000 个 B0 和 13,680 个 singleton
  全量覆盖，缺失为 0。
- true-U teacher edge0 已归约；edge1 在 cache 复用后产生 36,938 个 missing unique
  pair coalition，32-way render 与 strict validator 均完成，
  duplicate/unexpected/missing=`0/0/0`，等待 GPU scoring 后进入 edge2。
- V1 readout 仅按 `(pair_group,event_step_id)` exact key 复用 7,668 行；其余
  train/dev `5,143/869` 已由 fixed `sglang-omni-jaxan` 容器 fresh 抽取并 exit 0。
  统一 validator 得到 13,680/13,680 unique、1285 dims、666 个 `n>8` state、
  recent-8 truncation=0，逐行长度、有限值与重算 temporal feature 全部 PASS；
  remote `DONE` SHA256=`c1c0b500...def59b`。
- 补齐冻结协议要求的 development `n<=8` exact subset search：planner 完整枚举
  size 1--4 coalition 并记录 cache hit/miss；reducer 使用完整真实 U 生成 at-most-B
  exact oracle，报告 B1/B2/B4 的 teacher beam-4 recovery、regret、Jaccard 与 utility
  gap。独立 hostile review 后将 teacher/exact recovery 改为 canonical set 比较，并
  增加反序同集合、STOP→空集与 B0 缺失 fail-closed 测试；全套 V2 相关测试
  25/25 PASS。该诊断不得回写或调整 beam width 4。
- edge1 的 36,938 个 unique identities 已按 deterministic hash 平衡为 9 个逻辑
  shard。hyper00 复用本地 Odyssey 图片时，经 13,920-file SHA audit 补齐 160 个缺失
  episode，并覆盖 9 张同路径不同字节 PNG；最终 image-manifest SHA 与 hyper01 冻结值
  `d8fb2999...5467d` 完全一致后才启动 scorer。
- edge1 scorer 采用可恢复跨机布局：hyper00 物理 0/1/4 跑 shard 0/1/2；shard 3--8
  在完整 JSON 行边界迁到 hyper01 物理 0/1/5/6，并从各自 301--306 个 partial rows
  继续。迁移只改变基础设施分配，不改变 input shard、prompt、checkpoint、dtype 或
  科学参数；终态仍须回收到 36,938 unique-key 全量 validator。

## 2026-07-24:Selector V2 edge1 score 正式完成

- hyper00 shard 0--2 与 hyper01 续跑完成的 shard 3--8 已通过逐文件 SHA256 校验后
  原子合并到 hyper00
  `/data02/jaxan/runs/hgkv-selector-v2/teacher-edge1-hyper00-scores/`。
- 9 个逻辑 shard 行数为
  `[4095,3995,4235,4132,4087,4048,4112,4119,4115]`，合计
  `36,938/36,938`；全量 validator 得到 unique=`36,938`、
  missing/unexpected/duplicate=`0/0/0`，所有 `target_logprob_mean` 有限。
- 输入 manifest SHA256=`f6c011d6...a59`、hg-s100 checkpoint
  SHA256=`8f2cc49e...6317` 与启动记录一致。迁移触发的旧 wrapper `FAILED.json`
  仅是基础设施终止记录，在完整 validator PASS 并原子写入正式 `DONE` 后删除；
  `DONE` SHA256=`e2b3f227...3423`。
- 逐 shard SHA256、字节数和 provenance 已写入
  [`data/results/hgkv_selector_v2/teacher-edge1-score-manifest.json`](../data/results/hgkv_selector_v2/teacher-edge1-score-manifest.json)；
  大 score artifact 仍在本地持久盘，状态 `PENDING_HF_UPLOAD`。下一步先合并
  edge1 cache 并按 true-U beam-4 规划 edge2，而不是直接启动 student training。

## 2026-07-24:Selector V2 edge2 render 完成并启动 scoring

- edge1 score 已并入 coalition cache，得到 `286,405/286,405` rows/unique keys，
  SHA256=`346c2f4e...e2d9`。true-U depth1 reduce 完成 4,000 prefix groups、
  50,720 marginal rows、627 STOP-optimal groups，并生成 1,000 个 next-beam states。
- depth2 plan 含 42,354 个 unique triple child，其中 2,755 cache hit、39,599
  missing；plan SHA256=`0c34ba3c...0501`。39,599 个 missing triple 已 32-way
  渲染并严格验收，duplicate/unexpected/missing=`0/0/0`，14,626 张引用图片全部
  存在并完成 SHA256。
- hyper00 与 hyper01 从同一 plan 独立渲染所得 32 个 sample SHA 与 image manifest
  `7fa20db6...fba4` 完全一致；hyper01 `DONE` SHA256=`1846f0ee...f7cb`。
- 启动前按规则重新执行 5 秒 fleet preflight。hyper00 固定容器仍有 Frozen 主表
  policy workload，因此未停止、未重建、未叠加 selector；hyper01 GPU1 也被新 workload
  占用。最终使用 hyper01 host GPU `[0,2,5,6]`，固定容器 indices `[0,2,4,5]`，
  按 `3/3/2/2` 进程运行 10 个 scorer。
- scorer input 按 exact resume identity 平衡为 10 shard，3,875--4,117 rows/shard，
  总计 39,599/39,599。launch manifest、GPU snapshot 与调度记录已落在
  `/data02/jaxan/runs/hgkv-selector-v2/teacher-edge2-scores/`；轻量 provenance 见
  [`data/results/hgkv_selector_v2/teacher-edge2-launch-manifest.json`](../data/results/hgkv_selector_v2/teacher-edge2-launch-manifest.json)。
