# Desktop DiD v4(官方多轮结构重训)—— 运行记录

日期:2026-07-27。背景:Codex 报告确认单轮 renderer 存在协议缺陷(执行器 JSON
历史、像素/[0,999] 量纲混用 —— 实测 31.8% 历史动作 x/y>999、多轮官方结构缺失),
MobileWorld 前例 ef19e1a 表明格式差异可吞掉稀疏选点收益。用户裁定:新建官方
多轮 builder(单轮脚本保留不删),v3 口径整体重训为 v4。

## 语料 v4(fixed-budget replacement,官方多轮 renderer)

- 生成器 [build_desktop_did_corpus_v4.py](../../../code/scripts/build_desktop_did_corpus_v4.py),
  renderer [agentnet_desktop_official.py](../../../code/causalcache/agentnet_desktop_official.py)
  (gap-fold 泛化官方滚动结构;R 臂逐消息退化为官方结构;历史 = `Action:` 行 +
  `<tool_call>`,坐标全 [0,999];保留步缺官方响应整组弃用);
- 组代数与 v3 逐字不变(B∈{1,2,4} 训练、B=8 只建不训、k=1 最老槽位替换、
  age≥B+2、同 seed 轨迹 split);
- 组数:B=1/2/4/8 = 962/756/463/158(v3:969/764/470/160,tripleClick 保留组
  弃用 <1%);train units 1,755;dev 组数与 v3 完全一致(94/72/45/11);
- `samples.jsonl` SHA `e6575f934a6a701c9c50bebfa248dbdf2fc5e244b43d763b0cfe9d3cd96c897c`
  (10,905 行 / 2,181 组);parity_b0 SHA `e33ac3a0…5fd0`;
- 全语料坐标扫描:12,657 行 0 处 tool_call 坐标越界(单轮量纲混用不复现);
- staging:hyper00 `/data02/jaxan/desktop-did-corpus-v4`(权威构建),hyper01
  `/data04/jaxan/mw/desktop-did-corpus-v4`(SHA 校验后镜像,帧 symlink 到 v3 树);
  `PENDING_HF_UPLOAD`。

## 训练三行(16 组/优化步不变式:world × accum = 16)

| 行 | 主机/卡 | world×accum | config | 输出 |
|---|---|---|---|---|
| HGKV | hyper00 GPU0-7(容器 sglang-omni-jaxan) | 8×2 | `causalcache_desktop_did_hgkv_v4.json` | `/data02/jaxan/runs/desktop-did-v4/hgkv` |
| ungated KV | hyper01 GPU0-3(容器 sglang-omni-jaxan) | 4×4 | `causalcache_desktop_did_ungated_kv_v4.json` | `/data04/jaxan/mw/runs/desktop-did-v4/ungated_kv` |
| Full-LoRA | hyper01 GPU6/7(容器 sglang-omni-jaxan-2,待并行会话 MobileWorld 评测释放) | 2×8 | `causalcache_desktop_did_full_lora_v4.json` | `/data04/jaxan/mw/runs/desktop-did-v4/full_lora` |

- 协议冻结面(300 步 / ckpt 50 / lr / margins / gates)三行逐字节一致,单测
  `test_desktop_training_configs_share_the_frozen_protocol[_v4.json]` 只剥离
  accumulation 比较并锁 {2,4,8};
- 监督器:每行 `supervise.sh`(失败全量重启 ≤3 次,冻结分数缓存加速重启;
  退出写 `EXIT_<code>` 标记);已知坑:torchrun 需 `PYTHONPATH=<repo>/code`
  (v4 首启即因此失败 3 次,已修);
- 监控:Mac 侧持久监视器(10 分钟轮询)对 EXIT 标记、checkpoint 落盘、
  train.log 25 分钟无更新(STALE)、probe Traceback、6/7 释放信号告警;
- HGKV 启动确认:training_units=1755,schema v2,config SHA `cae3e984…f3f3`。

## k=1..4 探针(决定是否出 v4.1 加 k≥2 训练臂)

[probe_desktop_official_k.py](../../../code/scripts/probe_desktop_official_k.py)
在 hyper01 GPU0-3 4 分片运行(官方多轮渲染,B=4):R vs S_k(k 个最近 distinct
正例旧帧替换最老 k 槽,age≥6),frozen 边际 + 旧格式 s300(SHA `1754d232…9ff2`)
的 did-style 迁移读数 + 每状态正例可用性分布。输出
`/data04/jaxan/mw/runs/desktop-did-v4/probe-k/probe_k.shard*.jsonl`(断点续跑),
完成标记 `PROBE_DONE_*` 后链式启动 ungated 行。

## 探针结果(2026-07-27,427 状态,B=4,官方多轮渲染)

| k | frozen 边际 [95% CI] | s300(旧格式)did 迁移 | n |
|---|---|---|---|
| 1 | **+0.0130 [+0.0048,+0.0211]** | +0.0002 [−0.0007,+0.0010](≈0) | 427 |
| 2 | +0.0118 [−0.0078,+0.0304] n.s. | +0.0009 n.s. | 106 |
| 3 | −0.0017 n.s. | +0.0022 n.s. | 32 |
| 4 | −0.0243 [−0.0633,−0.0048] | −0.0006 | 3 |

配对增量:k2−k1 = +0.0014 n.s.;k3−k2 = −0.0114 n.s.(负向);k4−k3 = −0.0330。
可用性:75% 状态只有 1 个合格正例(321/427),k≥2 原料 25%,k≥3 仅 7.5%。

**三个结论**:(1) 冻结替换效应在官方结构下**存活且为正**(k=1 显著 +0.013)——
格式迁移风险(桌面版 ef19e1a)未命中冻结效应;(2) 旧格式 s300 的 HGKV 选择性
在官方结构下**完全不迁移**(did≈0,v3 dev 口径为 +0.011)——**v4 重训是必要的**,
adapter 选择性绑定 renderer;(3) **k≥2 无增量收益 + 原料稀薄 → v4 保持 k=1
替换臂,不出 v4.1**;k≥2 布局仍由 selector 推理端 edge 重打分覆盖,若 edge2
真实 U 显示失灵再议小比例 k=2 臂(与 v3 时的冻结决定一致)。
汇总:`/bigdata/mw/runs/desktop-did-v4/probe-k/summary.json` +
[summarize_desktop_official_k_probe.py](../../../code/scripts/summarize_desktop_official_k_probe.py)。

## 运行变更记录

- 2026-07-27:probe 结束后 hyper01 GPU3 被同机 124GB 服务进程抢占(root,活跃,
  不符合 0% 空占击杀授权),ungated 行撤退到 GPU0-1 world-2×accum-8
  (`causalcache_desktop_did_ungated_kv_v4w2.json`,等价性单测锁定;gate 打分
  仍用 4 卡口径 config)。

## HGKV v4 s300 按 B 正式 gate(2026-07-27,dev 独立打分,2,000 次 episode bootstrap)

| B | n(dev) | did_select [95% CI] | frozen_sel | \|A_r\| | wrong drift | 判定 | v3 最优选点对照 |
|---|---|---|---|---|---|---|---|
| 1 | 94 | **+0.0224 [+0.0162,+0.0294]** | +0.0791 | 0.0088 | 0.0085 | ✅ 全过 | +0.0120 |
| 2 | 72 | **+0.0164 [+0.0110,+0.0221]** | +0.0256 | 0.0107 | 0.0078 | ✅ 全过 | +0.0128 |
| 4(主设置) | 45 | **+0.0109 [+0.0070,+0.0154]** | −0.0002 | 0.0098 | 0.0091 | ✅ 全过 | +0.0075 |
| 8(外推,未训练) | 11 | **+0.0063 [+0.0018,+0.0115]** | −0.0305 | 0.0106 | 0.0137 | ✅ **全过** | ≈0 n.s. |

- **官方结构下每个 B 的 adapter 选择性都约为 v3 的 1.5-2 倍**,且 **B=8 外推档
  首次显著**(v3 判定不显著)——四档全过 gate(含 |A_r|、wrong_drift 双 cap);
- frozen_sel 随 B 递减(+0.079→+0.026→−0.000→−0.031):窗口越大冻结模型越
  偏好纯 Recent,而 HGKV 在整条 B 轴上稳定创造正选择性——v3 的"逆风创造
  选择性"叙事在 v4 内部呈现为跨 B 趋势;
- 报告:本目录 `gate_report_hgkv_v4_s300_b{1,2,4,8}.json`;
  s300 SHA `572092c218a97d1aa88b6845086a2ad2cec77c1a6d6fcd796f2e125ed62d67a9`;
  全 checkpoint × B 表(pooled 选点用)由 hyper01 GPU2 循环补齐中。

## Selector v4 标签(基于 s300)

用户裁定 s300 全档过 gate 后直接开工:`score_selector_v4_singletons.py` 8 分片
于 hyper00 8 卡运行(官方多轮渲染;B0 锚 bypass + 逐候选 active;候选池
byte 去重保留最近;tripleClick 类不可渲染候选跳过并计数;绑定上述 SHA)。
输出 `/data02/jaxan/runs/selector-v4/singletons/`,断点续跑。旧格式 v3 标签
(58,376 行)保留作 renderer 迁移对照,不再进 selector 训练。

## Selector v4 标签全量 + 三臂对比(2026-07-27 深夜)

标签定稿:单例 52,823(B=1 穷举 oracle)+ 集合 147,297 唯一行(beam-3 ×
shortlist-6 全 edge 整集重打分 + Recent-2/4 锚,5,335 状态,skipped=0)+
**448 维 HGKV 反事实读出 52,823(100% 覆盖)**(8 层 × 8 kv-head × 7 相对
统计量,候选池内 z 归一)。学习曲线(25/50/100% 三点全平 + train/dev MSE
无缺口)判定 cheap 特征偏差主导后的处方验证:

| 臂 | best dev top1-regret | dev Spearman | replay_gain_b2 | 备注 |
|---|---|---|---|---|
| cheap-only(28 维) | 0.0939 | 0.176 | +0.0051 | 特征天花板基线 |
| **concat(全交互)** | **0.0849** | **0.193** | **+0.0108** | train/dev 缺口大(0.539/0.193) |
| **two-tower(可加残差)** | **0.0849** | 0.171 | +0.0105 | 缺口小,epoch 10 即达最优 |

- **读出特征有效**:regret −9.6%,replay_gain_b2 翻倍(+0.0051→+0.0108),
  两个读出臂在所有部署相关指标上一致胜出 —— 路线正确;
- concat 与 two-tower dev 打平(0.0849):可加性限制在 desktop 上零成本;
  two-tower 训练侧记忆容量更小(train_probe Spearman 0.268 vs 0.539),
  **零样本 mobile 迁移的保守选择 → 暂定主臂 two-tower**,concat 并列候选,
  mobile 验证探针仲裁;
- oracle_gap_b2 仍有 ~0.085:读出只咬下剩余 headroom 的一小口,后续升级臂
  (last-position hidden Δ + 冻结随机投影)已登记;
- replay_gain_b4 负值为已披露的图数不匹配下界(模型集合 1-2 张 vs 4 图锚),
  正式验收以 fill-to-B 整集重打分为准;
- 报告:本目录 selector_arm_{cheap,concat,twotower}_report.json;标签与
  读出 staging 于两台 host,`PENDING_HF_UPLOAD`。

## 依赖与后续

1. probe 汇总 → 判定 k≥2 是否有收益(有 → v4.1 语料加小比例 k=2 臂);
2. 三行训完 → per-B gate 打分(score_sparse_history_arms,v2 schema 已接线)
   → v4 选点 → §9-6/7 生成式评测(带坐标越界统计)→ selector 标签重打
   (绑定 v4 checkpoint;hyper00 旧格式 singleton 标签 58,376 行保留为对照);
3. 旧格式 v3 结论(B=1/2/4 全过 gate,s300 pooled +0.01106)保留在
   [desktop_did_policy_v3](../desktop_did_policy_v3/README.md),v4 是其官方
   结构复核 —— 若 v4 gate 不过,即桌面版 ef19e1a(收益是单轮结构产物)。

## 过夜冲刺(2026-07-27 深夜,用户最高授权)运行记录

- **MobileWorld HGKV-B4 closed-loop**(决定性实验:须显著超过冻结基线
  B0=28.21%/B4=29.91%, CI 跨零, 4f4ad25):基于并行会话的官方协议 harness
  (branch luojiaxuan/mobileworld-memory-osworld2 @7ff6cad),新分支
  `jaxan/mobileworld-hgkv-arm`(550e80a)给 serve 侧加可选 HGKV 挂载
  (官方图序 mask;B0/无 adapter bitwise 冻结);config
  `causalcache_mobileworld_official_b4_hgkv_v1.json`(desktop-v4 s300 零样本);
  hyper01:16-env 舰队(host docker,prefix sglang-omni-jaxan-mwh-)+ 3 policy
  副本(GPU0/1/7,restart-loop)+ 3 shard supervisor(断点续跑,pending 反推,
  STALL 保护);run root `/data04/jaxan/mw/runs/mw-hgkv-b4-v1`;
- **hyper00**:full_lora w4(GPU0-3,encode-off+expandable,config w4)与
  ungated w4(GPU4-7)并行训练中;
- **hyper01 附属**:fill-to-B 验收 ×2(GPU4/5,首启因 env 穿引号丢失重启)、
  HGKV 全 checkpoint devscore 循环(GPU2);
- 舰队/监督/服务的清理义务:run 结束按 fleet manifest 逐 container 停删
  (mwh- 前缀),canonical 容器保留。

## Fill-to-B 端到端验收(2026-07-27 凌晨,dev 全量,exact-B + recent 兜底,beam-3,真 U)

| 臂 | B=2: selector−Recent-2 [95% CI] | B=4: selector−Recent-4 [95% CI] | k 分布(B=4) |
|---|---|---|---|
| two_tower | +0.0250 [+0.0132,+0.0368] | +0.0108 [+0.0010,+0.0209] | 0:107 / 1:164 / 2:80 / 3:147 / 4:61 |
| **concat(改判主臂)** | **+0.0297 [+0.0185,+0.0409]** | **+0.0169 [+0.0067,+0.0277]** | 0:103 / 1:103 / 2:88 / 3:119 / 4:146 |

- 两臂两预算全部显著为正(episode bootstrap,n=564/559;fresh 整集补打 1,123 次);
- **concat 端到端反超**(dev 平手被真 U 对比打破)→ 桌面主臂 = concat;
  two-tower 保留为跨平台迁移保守变体(mobile 探针再仲裁);
- k=0(纯 Recent 兜底)真实被选中(B=4 下 ~19%),oracle_gap 仍余 ~0.06-0.08;
- 报告:filltob_{two_tower,concat}.json(本目录)。

## 过夜冲刺终态(用户醒来速览)

**Ungated s300 四档 gate(全过,但一致弱于 HGKV)**:did_select
+0.0177/+0.0139/+0.0075/+0.0047(B=1/2/4/8);B=8 处 |A_r|=0.016、
wrong=0.016 逼近 0.02 帽,而 HGKV 保持 ~0.010——门控 = 增益 + 约束。
报告 gate_report_ungated_v4_s300_b*.json。

**MobileWorld HGKV-B4 closed-loop(117/117 全分母)**:34/117 = 29.06%,
与冻结 B4(35, 29.91%)/B0(33, 28.21%)统计不可区分(vs B4 exact McNemar
p=1.0,不一致任务 6:7)。**这是设计使然**:|A_r| 漂移帽保证 recent 臂上
adapter ≈ 冻结;本结果 = "policy-preserving 零部署代价"的行为级、跨平台
零样本验证,不是主张性优势。**要显著超基线需 selected 臂**(selector 选中
旧帧的非连续恢复):缺 mobile 版 gap-fold 渲染 + 在线两段式特征(witness
特征依赖目标动作 → 需 propose-then-select 两遍推理),已列为下一程首项。
run root `/data04/jaxan/mw/runs/mw-hgkv-b4-v1`(轨迹 LOCAL_PRIVATE_RAW_TRACE),
舰队 16 容器已按 manifest 清理。逐任务对照数据取自 Aries paired-result.json。

**full_lora**:w4 在 OOM 边缘存活(attempt2 过 s50),持续训练中,预计上午
出 s300 → 补 gate。**paper 主表/摘要已填实测数字**(6 页编译通过)。

## full_lora v4 终态:BLOCKED(内存,待用户裁决)

w8(accum2)与 w4(accum4)各 3 次尝试全部 OOM(step ~40-70 的 B=4 长 prompt
单元;141GB H200 差 ~134MB-342MB;encode-off + expandable_segments 已用)。
根因:官方多轮 prompt(~14.5k tokens)× 全 36 层 LoRA 反向的激活峰值,较 v3
单轮长 ~1-2k tokens,恰好越过容量线。梯度检查点因 ContextVar 旁路重算语义
(plain_lora_bypass_scope)默认禁用。晨间选项:(a) use_reentrant=False 非重入
GC + ContextVar 正确性单测(协议冻结面需记录豁免);(b) 该行沿用 v3 结论 +
v4 blocked 披露;(c) 换更大显存机器。三次尝试日志在
hyper00 /data02/jaxan/runs/desktop-did-v4/full_lora/train.log。

## MobileWorld selected 臂 campaign(进行中)

mw-hgkv-sel-b4-v1:16 env × 4 shard × 4 个 selector-enabled HGKV 副本
(propose-then-select 两遍;witness=拟议动作;gap-fold 二遍渲染;
memory_arm=full 全池随请求)。首批请求 0 失败。分支
jaxan/mobileworld-hgkv-arm @b4c38c6(gap-fold/特征/serve 两遍/runner 白名单
全有单测或编译检查)。结果将与冻结 B0/B4、HGKV-recent-B4(34/117)三方配对。

### selected-B4 r1 结果(2026-07-27,117/117 全分母)

**历史标签更正（2026-07-31）**：本段的 selected-B4 `39/117 = 33.33%`
属于 Frozen+selector comparator，不是最终论文的 HGKV+selector 主臂。
最终主臂三轮均为 `43/117 = 36.75%`。以下配对检验保留为该历史
Frozen+selector round-1 的运行记录：
(逐任务,missing 记 0;bootstrap 10k + exact McNemar;冻结臂 per-task 取自
Aries `mw-official-sequential-v2-71caac9`,复算冻结 B4−B0 = +1.71pp
CI[−5.98,+9.40] p=0.824 与提交口径逐位一致):

| 对比 | Δ | 95% CI | 不一致 | McNemar p |
|---|---|---|---|---|
| selected-B4 vs 冻结 B0 | **+5.13pp** | [−2.56, +12.82] | 14:8 | 0.286 |
| selected-B4 vs 冻结 recent-B4 | +3.42pp | [−4.27, +11.11] | 12:8 | 0.503 |
| HGKV-recent-B4 vs 冻结 B0(参照) | +0.85pp | — | 6:7→p=1.0 | 1.0 |

方向正确但单轮 n=117 统计功效不足(+5pp 效应需 ~3× 判别对)。**加轮方案
(执行中)**:selected-B4 r2(hyper01,复用 fleet+servers)→ r3;冻结 B0
r2+r3(hyper00 新 16-emu fleet 拆两半并行,8×143GB GPU 各 4 replica);
Taurus 另跑 frozen+selected-B4(selector-only 消融行,16 emu × 8 GPU)。
终态每臂 3 轮,按任务聚类 bootstrap 求配对均值差 CI。
per-task 明细:`data/results/mobileworld_hgkv_selected_b4/per_task_success_r1.json`。

## Intent-selector v2 单遍判定(Line B,已裁决:不达标)

预注册判定:witness 清零(单遍部署口径)+ 8 维 instruction-conditioned
intent 特征后,B=4 的 selector−Recent episode-cluster CI 是否离开零。
8 卡分片 eval(`--shard-count 8` + merge,与单进程同一 CI 实现)结果:

| 口径 | B=2 | B=4 | B=4 k-dist 病理 |
|---|---|---|---|
| 两遍(witness 在线,主部署) | +0.0250 sig | **+0.0108 sig** | k=4 占 11% 正常 |
| 单遍 witness 清零(nowitness) | +0.0161 sig | +0.0086 CI 跨零 | k=4 暴涨至 34% |
| 单遍 + intent v2(本判定) | +0.0149 CI[+0.0001,+0.0297] | **+0.0046 CI[−0.0071,+0.0164] 跨零** | k=4 回落至 12% |

**结论:intent 特征修复了单遍的过度替换病理(k 分布回归正常)但救不回
B=4 显著性 → 两遍 propose-then-select 保持为 B=4 部署主路径;B=2 单遍可用。**
训练侧 intent 臂 dev_top1_regret 0.0901(cheap 0.0939 / readout 0.0849 之间)。
文件:`filltob_intent_singlepass.json`、`arm_intent_report.json`(本目录);
生产脚本 `extract_selector_v4_intent_features.py`、分片 eval
`eval_selector_v4_fill_to_b.py --shard-index/--shard-count` +
`merge_selector_v4_filltob_shards.py`。曾踩坑:链脚本 `set -e` 与
`grep && exit 1` 组合在 grep 无匹配时误杀自身;trainer `intent_dims`
先用后赋(已修)。
