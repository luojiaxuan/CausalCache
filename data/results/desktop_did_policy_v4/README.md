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
