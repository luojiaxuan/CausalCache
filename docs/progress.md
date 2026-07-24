# 项目进展

> 2026-07-22 及更早的全部旧条目已逐字存档至 [`docs/archive/progress_2026-07-20_22.md`](archive/progress_2026-07-20_22.md);本文件只保留 history-gated mainline 时代(2026-07-23 起)的条目。

主线 = History-Gated KV Adapter(合同见 [`history_gated_mainline_v1.md`](history_gated_mainline_v1.md),正式 gate PASS、s100 冻结,见 [`data/results/hgkv_gate_v1/`](../data/results/hgkv_gate_v1/README.md))。

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
