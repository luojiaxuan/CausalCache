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
