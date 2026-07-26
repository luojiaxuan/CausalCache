# Desktop Memory 训练与 MobileWorld 零样本实验交接 v1

日期：2026-07-26  
工作分支：`luojiaxuan/mobileworld-memory-osworld2`  
Git remote：`https://github.com/luojiaxuan/CausalCache.git`

## 0. 一句话决策

停止把 GUI-Odyssey 的 `base-wrong prevalence` 诊断当作关键路径。下一条主线是：

1. 从 Desktop 成功轨迹构造**目标相关旧图 / recent / matched wrong** 的训练对；
2. 用带 `recent-active` 和 per-arm frozen anchor 的 DiD 目标训练 GUI-Owl
   History-Gated KV（HGKV），并同时训练 Full-layer LoRA 与 matched ungated KV
   对照；
3. policy checkpoint 完全冻结后，用该 checkpoint 在 Desktop 全候选池上重新计算
   真实边际效用，生成 selector 标签；
4. selector 只在 Desktop 训练与选 checkpoint；
5. 最后零样本迁移到未用于训练或选择的 MobileWorld memory split，并在未污染的
   OSWorld 2.0 tasks 上补充跨桌面评测。

**截至本文档提交时，没有启动新的 AgentNet/HGKV/Full-LoRA/selector 训练。**

## 1. 为什么改主线

GUI-Odyssey 对严格 image-only C-case 的自然覆盖太低。继续对 2,153 个组做
screening，回答的主要是“这种状态在 Odyssey 出现多少”，不是“如何得到足够训练数据”。
该 screening 已停止，cache/heartbeat 保留，可恢复但不进入当前关键路径。

另一方面，已有实验已经证明两个不能绕开的事实：

- Frozen GUI-Owl 技术上能接收不连续历史图，但 AndroidWorld success-anchor 数据上
  `correct − shuffled/irrelevant ≈ 0`，说明它没有稳定使用图像内容；
- 旧 HGKV gate 只比较 `correct/b0/shuffled/irrelevant`，遗漏
  `recent + adapter active`。补上 `RA` 后发现 adapter 对 recent 的增益大于对 selected
  的增益；这是“见到任何历史图都放大”，不是内容选择。

因此，新训练既要换到有规模的 Desktop 数据，也必须换成 DiD 目标，不能原样重跑旧四臂
目标。

## 2. 论文主张与实验对象

建议 headline claim：

> Historical GUI observations have highly non-uniform, context-dependent marginal
> utility. Under a visual-context budget, selectively restoring a small number of
> high-fidelity historical screenshots from compressed action-summary memory
> improves the correct-action probability and downstream task success.

这句话同时覆盖两个层次：

- 科学现象：恢复**目标相关**旧图能提高正确动作的概率；
- 方法结果：history budget 不应固定 recent，也不一定每次用满；selector 应在候选图和
  STOP 之间做条件化决策。

严格 image-only C-case 只保留为 diagnostic subset。训练正例允许两种机制：

- information addition：旧图携带 base context 没有的信息；
- evidence amplification / visual re-grounding：旧图没有新增语义事实，但提高正确动作
  的 margin、降低歧义或恢复 action-to-visual-state 绑定。

## 3. 冻结定义

对一个决策点 \(t\)，base context 定义为：

```text
C_r = task instruction
    + complete compressed action/result history
    + current screenshot
    + Recent-r screenshots
```

候选旧图 \(I_i\) 的 policy utility：

```text
U_pi(i | C_r) = log p_pi(a* | C_r, I_i) - log p_pi(a* | C_r)
```

其中 `a*` 必须是完整 desktop tool call；行为正确性必须比较 action type、参数和坐标，
不能只比较 `action_type`。坐标同时报告 `[0,999]` 空间 `τ ∈ {5,10,25}`，主分析固定
`τ=25`，其余为 sensitivity。

selector 的最终效用必须将**完整选中集合**重新送入冻结 policy：

```text
U_pi(S | C_r) = log p_pi(a* | C_r, S) - log p_pi(a* | C_r)
```

禁止把 singleton gain 求和冒充 set utility。

## 4. 数据源与污染边界

### 4.1 AgentNet/OpenCUA：规模主来源

Hyper01 local staging：

```text
原始数据：/data04/jaxan/datasets/agentnet
大小：374 GB
首批决策点：
  /data04/jaxan/mw/repo/CausalCache/data/manifests/
  agentnet_screening_manifest_ubuntu_v1.jsonl
```

首批 Ubuntu 数据统计：

- 5,000 trajectories seen；
- 2,293 successful trajectories kept；
- 6,003 decision points；
- full action parsing 覆盖 click/type/key/drag/scroll/terminal 等；
- manifest 约 26 MB，1,000 点图片抽验为 `11,894/11,894` 存在。

Git 中只保存轻量 summary：
[`data/manifests/agentnet_screening_manifest_ubuntu_v1_summary.json`](../data/manifests/agentnet_screening_manifest_ubuntu_v1_summary.json)。
完整 manifest、原始图片和后续 corpus 都是 reusable dataset，应上传 HF，不能把
374 GB 原始数据提交进 Git。

### 4.2 OSWorld 2.0 official successful trajectories：高精度 witness seed

第一轮从 13 条 evaluator `score >= 0.95` 的官方成功轨迹构造了 67 个 decision points；
每点包含 reference segment、selected reference frame、nearby、random-old 和完整候选池。

Hyper01 staging：

```text
manifest:
  /data04/jaxan/mw/repo/CausalCache/data/manifests/
  osworld_v2_visual_witness_round1.jsonl
scores:
  /data04/jaxan/mw/repo/CausalCache/data/manifests/
  osworld_round1_scores.shard000-of-002.jsonl
  osworld_round1_scores.shard001-of-002.jsonl
```

两 shard 已完成 `544/544` 和 `546/546` units；只差 reduce 与训练臂物化。它们不是
新训练结果。

用过的 task ids：`003, 032, 074, 105, 107`。污染记录：
[`data/manifests/osworld_v2_contamination_ledger_v1.json`](../data/manifests/osworld_v2_contamination_ledger_v1.json)。
这 5 个 task 后续只能报 in-domain diagnostic，不能计入 zero-shot OSWorld 结论。

### 4.3 MobileWorld：只做零样本 target benchmark

MobileWorld 的 201-task inventory 中，冻结 GUI-Owl interface 可跑 117 个：
62 个 cross-app memory candidates，55 个 single-app controls。任何 MobileWorld task、
trace、reward、图片或 selector truth 都不得用于：

- policy 训练；
- selector 训练；
- checkpoint/threshold/预算选择；
- prompt 或 equivalence 的事后修改。

优先单独报告这些天然 memory tasks：

1. `MattermostVisualInstructionResponseTask`；
2. `MastodonMultiInviteTask`；
3. `LocalFileManagementTask` / `Task2`；
4. `MastodonMallShareOrderTask`；
5. invoice / attachment / cross-app exact-copy tasks。

### 4.4 GUI-Odyssey：停止作为规模训练源

Hyper00 的 8 个 `scripts.mine_rescue_tiers` shard 已于 2026-07-26 停止，GPU 已释放，
输出未删：

```text
/data/artifacts/causalcache-rescue-v1/tiers/
```

以后若恢复，只用于 prevalence / harm / calibration appendix，不得阻塞 Desktop 训练。

## 5. Desktop 训练对如何构造

### 5.1 AgentNet target-action recurrence

对当前 gold action `a*`，在同一成功轨迹的旧动作历史中找完整等价动作：

- click / move / drag：同 action family，坐标偏差 `≤ τ`；
- type：文本完全一致；
- key / hotkey：key sequence 一致；
- scroll：方向一致；
- wait / terminal 不作为 recurrence positive。

若旧动作发生在 step `j`，将 step `j` 执行前的画面作为 target-specific visual
positive；在当前 event 存储语义里，它对应 event `j-1` 的 post screenshot。要求
`age >= 3`，避免把“最近一张”冒充非 recent memory。

对照：

- `recent`：与 positive 相同预算的 Recent-k；
- `wrong`：同轨迹、age 尽量匹配、但下一动作与 `a*` 不等价的旧状态；
- `irrelevant`：跨轨迹同分辨率 donor，替换同一 event slot 的像素，文字 history 不变；
- `b0`：无恢复图，只用于 HGKV structural parity 与总体能力报告。

已有 prototype：

```text
code/scripts/build_desktop_hgkv_corpus.py
code/tests/test_desktop_hgkv_corpus.py
```

它已通过与 AgentNet parser/renderer 合计 41 个单测，但当前只输出旧式
`correct/b0/shuffled/irrelevant` 四臂，且 `prompt_format=osworld_official` 尚未接入
trainer。**不要直接拿 prototype 开正式训练。**

### 5.2 OSWorld visual witness

每个 witness decision point 直接使用：

- selected reference segment frame 作为 positive；
- nearby / random-old 作为 matched controls；
- 另配跨轨迹 donor；
- target 是官方成功轨迹的下一步可约化单原语 action。

先执行 `mine_osworld_v2_visual_witness.py --mode reduce`，再只保留：

- correct frame 的 `U` 为正，或修复 base-wrong；
- correct 优于 nearby、random-old 和 next-recent；
- action parser / target span 完整。

OSWorld witness 是高 precision seed；AgentNet recurrence 是规模主体。训练时先报告各来源
group 数，不允许一个来源在未披露情况下被重复采样放大。

## 6. 必须使用的 DiD 六臂目标

对同一个图像条件 \(x\)，定义 adapter effect：

```text
A_x = log p_adapted(a* | x) - log p_frozen(a* | x)
```

每个训练组至少需要：

| arm | 图片集合 | adapter |
|---|---|---|
| `R0` | Recent-k | bypass |
| `RA` | Recent-k | active |
| `S0` | target-specific selected-k | bypass |
| `SA` | target-specific selected-k | active |
| `WA` | age-matched wrong-k | active |
| `W0` | 与 `WA` 同 prompt | 由 trainer 临时 bypass 重算 |

`B0` 另作 HGKV bitwise parity 与 policy-capability 评测，不混入内容选择主差值。

训练目标：

```text
L_select  = [m_select  - (A_S - A_R)]+
L_content = [m_content - (A_S - A_W)]+
L_gain    = [m_gain    - A_S]+
L_cap     = [|A_R|-eps]+ + [|A_W|-eps]+
L_l2      = lambda * ||LoRA||^2
```

初始冻结值：

```text
m_select = m_content = m_gain = 0.01 nats/token
eps = 0.02 nats/token
CE weight = 0
```

仓库的 sparse-history `did_ra_aware` 实现可复用，但要新增 Desktop corpus schema 和
`osworld_official` renderer，而不是把 Desktop 数据伪装成 GUI-Odyssey v6。

## 7. Policy ablation

同一 Desktop train/dev/test split、同一训练组、同一 visual tokens、相同 optimizer
steps 与 checkpoint cadence，至少跑四行：

1. `Frozen GUI-Owl`；
2. `Full-layer LoRA`：全部 LM layers 的 `q/k/v/o`，always active；
3. `Matched ungated KV LoRA`：最后 8 层 `k/v`、rank/alpha 与 HGKV 相同，active on all
   tokens；
4. `HGKV`：最后 8 层 `k/v`，residual 只作用于 restored-history image tokens。

默认结构：

```text
rank=8
alpha=16
dropout=0
vision encoder/projector frozen
base LM frozen
```

Full-layer LoRA 是必要 ablation；matched ungated KV 是判断收益来自“门控”还是仅来自
“参数量/层位”的关键结构对照。

不得复用旧 GUI-Odyssey `s75` 作为最终 Full-LoRA 行：它的训练数据、renderer 和 objective
都不匹配，只能作 pilot provenance。

## 8. Policy 训练前要补的代码

Claude 应按这个顺序改：

1. 将 Desktop prototype 从四臂升级为第 6 节的 DiD schema；
2. 在 `train_success_sft_lora.py` 和相应 scorer 中新增显式
   `prompt_format=osworld_official`：
   - 调用 `processor.apply_chat_template(..., tools=[_TOOL_SPEC])`；
   - 与 `GUIOwlOSWorldRuntime` 对同一 messages 做 token-by-token parity test；
3. 保持旧 GUI-Odyssey schema 和旧 checkpoint 路径可复现，不改旧 schema 的语义；
4. 为 Full-layer / matched ungated KV / HGKV 共用一个 DiD group loss，仅改变 adapter
   注入位置和 mask；
5. config 固定 dataset manifest SHA、base model revision、visual tokens、split salt、
   trainable parameter count、max steps 和 checkpoint cadence；
6. 先跑纯机械测试：
   - HGKV B0 exact parity；
   - history mask 只覆盖 restored image tokens；
   - `R0/RA/S0/SA/WA/W0` prompt 除 adapter mode 外逐字段对账；
   - DDP 每个 group 只消费一次；
   - parser target round-trip。

视觉 token 建议：正式主表与 MobileWorld 既有 runtime 统一用 `2560/image`。若为了快速估时
使用 `480/image`，必须标记为 throughput-only，不能把该 checkpoint 混入正式 2560-token
主表。

## 9. Policy checkpoint gate

checkpoint 只在 Desktop dev 选择。HGKV 至少同时满足：

1. `B0` 与 Frozen bitwise parity；
2. `A_S - A_R` 的 trajectory-cluster bootstrap 95% CI 下界 `> 0`；
3. `A_S - A_W` 的 95% CI 下界 `> 0`；
4. `A_S > 0`；
5. `|A_R|` 与 `|A_W|` 的 harm/drift 不超过预注册 `eps`，并报告尾部；
6. tool-call parser validity 不低于 Frozen；
7. Desktop heldout 的完整动作等价率不退化。

Full-layer 和 ungated KV 不要求 B0 parity，但必须报告 B0 drift、parser validity 与基础
动作正确率。

如果 HGKV 只提高 `SA-S0`，但 `A_S-A_R <= 0`，仍判定为旧问题复现，不能进入 selector。

## 10. Selector 数据与训练

只有最终 HGKV checkpoint 通过第 9 节并冻结 SHA 后才能生成 selector truth。

### 10.1 Candidate labels

对 Desktop train/dev 的每个决策点，在 `r ∈ {0,2}` 下：

1. 构造完整旧图候选池；重复/纯黑/逐 byte 相同图可先去重；
2. 单图枚举得到真实 `U_pi(i | C_r)`；
3. shortlist 只用于降低组合搜索成本，不是最终 oracle；
4. 从空集开始做 beam-4 + STOP；
5. 每个 edge 都将完整集合重新送入冻结 HGKV；
6. 同时保存 Recent、Random、same-app wrong、next-recent 对照。

正标签不要求 base action 错。分层保存：

- repair：base 完整动作错误，selected 后修复；
- amplification：base 正确但 selected 显著提高 gold margin/logprob；
- harm/control：wrong/irrelevant 或高置信 base 上乱加图。

### 10.2 Selector

复用 `selector_v2` 的 full-history inventory、exact subset key、beam planner/reducer、
STOP 与 bootstrap 代码；**旧 hg-s100 的 score/readout/checkpoint 不得复用**。所有 policy
dependent artifacts 必须用新 checkpoint SHA 重生成。

建议 student 输入：

- final-policy candidate readout；
- current/query readout；
- candidate age、action family、screen change、与 recent set 的关系；
- selected-set state；
- remaining budget；
- STOP token。

正式预算：`B ∈ {1,2,4}`，方法是 at-most-B。报告平均实际恢复图数、visual tokens 和
latency。

## 11. 评测矩阵

### 11.1 Desktop offline

Policy panel：

```text
{Frozen, Full LoRA, matched ungated KV, HGKV}
× {B0, Recent-B, target-specific/Oracle-B}
× B∈{1,2,4}
```

Selector panel固定最终 HGKV：

```text
{Recent, Random, Similarity, learned selector, Oracle}
× B∈{1,2,4}
```

指标：

- mean target logprob / token margin；
- full action equivalence；
- repair rate；
- amplification rate；
- wrong-history harm；
- parser validity；
- selected image count、visual tokens、latency；
- selector 相对 Recent 的 paired trajectory-cluster bootstrap CI。

### 11.2 OSWorld 2.0

- task `003/032/074/105/107` 只报 in-domain witness diagnostic；
- 其它 task 才能报 zero-shot；
- split、prompt、budget、threshold 在看到结果前冻结；
- 报官方 evaluator normalized score，不用 action logprob 替代闭环结果。

### 11.3 MobileWorld zero-shot headline

至少比较：

1. Frozen `B0`；
2. Frozen `Recent-B`；
3. HGKV `B0`；
4. HGKV `Recent-B`；
5. HGKV + Desktop selector（at-most-B）；
6. HGKV + random/wrong control；
7. Full-LoRA + Recent-B；
8. Oracle history（只作 headroom，不是可部署方法）。

分别报告：

- 62 个 cross-app memory candidates；
- 55 个 single-app controls；
- construction-level natural memory task 子组；
- strict-117 denominator 与 observed denominator；
- paired success delta、失败类型、平均步数、图片数和 wall time。

强故事的最低形态：

- selector 在 memory-needed tasks 上改善；
- controls 不能复现；
- non-memory tasks 不显著回退；
- learned selector 闭合一部分 base→oracle gap；
- 训练全在 Desktop，MobileWorld 完全未参与选择。

## 12. 算力与启动建议

当前事实：

- Hyper01 持有 374 GB AgentNet 原始数据，优先在 Hyper01 做 corpus build、policy training
  和 selector label scoring；
- Hyper00 的 GUI-Odyssey screening 已停，8 张 H200 已释放，但没有 374 GB AgentNet；
- 不要先复制整个 374 GB。只在 corpus 冻结后复制被引用的图片子集，或先上传 HF；
- B200 SSH 当前 `connection refused`，不作为首轮依赖；
- Aries/Taurus A6000 用于 MobileWorld/轻量闭环，不承担 8B 大规模训练。

正式 GPU job 前仍须执行 `$gpu-fleet-preflight`。Hyper01/Hyper00 本项目各获授权最多 8 卡；
H200 policy training 一进程一卡。跨 host label scoring 用独立 deterministic shards，不做
跨机 NCCL。

MobileWorld 使用已验证的“一个 policy GPU 服务多个 emulator”拓扑。若 Aries 至少两卡空闲，
直接用 2 GPU，每 GPU 驱动 8 个独立 emulator containers；不要退回一 GPU 串行跑模拟器。

## 13. 建议执行顺序与停止条件

### Stage A：数据冻结

1. reduce 67-point OSWorld witness scores；
2. 将 AgentNet prototype 升级为 DiD schema；
3. 生成 AgentNet + OSWorld-witness corpus；
4. trajectory/task 级 train/dev/test split；
5. 统计来源、action family、age、positive/wrong 图距离和 group 数；
6. 上传或登记 HF dataset revision。

停止条件：训练组少于 500，或 click 占比超过 85% 且没有 type/key/drag 覆盖时，不开正式
policy training；先扩 AgentNet win/mac 或 ProCUA，而不是回去做 Odyssey prevalence。

### Stage B：Policy

1. Frozen score cache；
2. 同时训练 Full-LoRA、matched ungated KV、HGKV；
3. 每个 checkpoint 跑第 9 节 gate；
4. 只按 Desktop dev 冻结 final checkpoint；
5. adapters 上传 HF model repo。

### Stage C：Selector

1. 用 frozen final HGKV 生成全部 Desktop labels/readouts；
2. beam-4 + STOP teacher；
3. fresh student；
4. Desktop heldout selected-set gate；
5. selector checkpoint 上传 HF。

### Stage D：Zero-shot

1. 冻结 MobileWorld/OSWorld protocol；
2. 先跑 OSWorld 未污染 development roster 的 grammar/closed-loop canary；
3. 再跑正式 MobileWorld 117 与 OSWorld 未污染 roster；
4. 回写 Git summary、raw artifact HF revision 和失败分类。

## 14. 当前 Git 实现状态

已提交并通过相关测试：

- [`code/causalcache/agentnet_actions.py`](../code/causalcache/agentnet_actions.py)；
- [`code/causalcache/agentnet_desktop_cr.py`](../code/causalcache/agentnet_desktop_cr.py)；
- [`code/scripts/build_agentnet_screening_manifest.py`](../code/scripts/build_agentnet_screening_manifest.py)；
- [`code/scripts/screen_agentnet_decision_points.py`](../code/scripts/screen_agentnet_decision_points.py)；
- [`code/scripts/mine_osworld_v2_visual_witness.py`](../code/scripts/mine_osworld_v2_visual_witness.py)；
- AgentNet/renderer/MobileWorld/OSWorld 相关测试。

待本交接提交：

- [`code/scripts/build_desktop_hgkv_corpus.py`](../code/scripts/build_desktop_hgkv_corpus.py)；
- [`code/tests/test_desktop_hgkv_corpus.py`](../code/tests/test_desktop_hgkv_corpus.py)。

prototype 当前测试命令：

```bash
PYTHONPATH=code .venv/bin/pytest -q \
  code/tests/test_desktop_hgkv_corpus.py \
  code/tests/test_agentnet_actions.py \
  code/tests/test_agentnet_desktop_cr.py
```

结果：`41 passed`。如第 5、6、8 节所述，它仍须升级 schema/renderer 后才可正式训练。

## 15. Source of Truth

### Git

```text
repo:   https://github.com/luojiaxuan/CausalCache.git
branch: luojiaxuan/mobileworld-memory-osworld2
Desktop mining import base commit: 616fa59
```

### Intended Hugging Face destinations

以下只是 intended destination，尚未验证创建或上传：

| artifact | intended repo | status |
|---|---|---|
| Desktop DiD corpus、score labels、selector labels | `gavinlaw/causalcache-desktop-memory-training` | `PENDING_HF_UPLOAD` |
| HGKV / Full-LoRA / ungated-KV adapters、selector checkpoint | `gavinlaw/causalcache-gui-owl-desktop-memory-adapters` | `PENDING_HF_UPLOAD` |

### Local staging

| artifact | path | status |
|---|---|---|
| AgentNet raw/images | Hyper01 `/data04/jaxan/datasets/agentnet` | local staging，374 GB |
| AgentNet 6,003-point manifest | Hyper01 `/data04/jaxan/mw/repo/CausalCache/data/manifests/agentnet_screening_manifest_ubuntu_v1.jsonl` | local staging，约 26 MB |
| OSWorld witness manifest | Hyper01 `/data04/jaxan/mw/repo/CausalCache/data/manifests/osworld_v2_visual_witness_round1.jsonl` | local staging，约 2.9 MB |
| OSWorld witness scores | Hyper01 同目录 `osworld_round1_scores.shard*.jsonl` | scoring complete，reduce pending |
| GUI-Odyssey rescue screening cache | Hyper00 `/data/artifacts/causalcache-rescue-v1/tiers/` | stopped/resumable，非关键路径 |

任何文档不得把上述 local staging 写成 HF canonical，直到记录可验证的 repo revision。
