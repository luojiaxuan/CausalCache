# Gate 1:prompt 格式 probe(冻结模型)—— 稀疏选点收益是单轮 renderer 的产物

日期:2026-07-26。**dev split(87 episodes / 240 groups)**,冻结 GUI-Owl,不加载任何
adapter,CI 为 episode-cluster bootstrap。confirmation 的 254 组由 `--episode-filter`
物理排除。报告:`artifacts/causalcache-prompt-format-probe-v1/dev_report.json`。

## 1. 管线可信度

probe 几乎精确复现了两个已发表量:`R_single − N0 = −0.0208`(已知 −0.0217)、
`S_single − R_single = +0.0335`(已知 +0.0335)。说明它测的是同一个东西。

## 2. 全量结果

| 派生量 | 值 | CI95 |
|---|---|---|
| `format_cost_multi` = R_multi − N0 | **+0.0261** | [+0.0211, +0.0307] |
| `sparse_gain_multi` = S_multi − R_multi | **+0.0027** | **[−0.0039, +0.0087]** |
| `net_effect_multi` = S_multi − N0 | **+0.0288** | [+0.0211, +0.0364] |
| `format_cost_single` = R_single − N0 | −0.0208 | [−0.0322, −0.0087] |
| `sparse_gain_single` = S_single − R_single | +0.0335 | [+0.0232, +0.0435] |
| `net_effect_single` = S_single − N0 | +0.0128 | [+0.0004, +0.0245] |

两个效应**不是可加的**。新格式不只是消掉了 −0.0217 的税,它比原生官方多轮还好 +0.0261。
但格式一旦修好,**冻结模型的稀疏选点收益基本消失**。

## 3. 按 K 分层 —— 排除"K=1 稀释"解释

| | K=1 | K=2 | K=3 | K=4 | 全体 |
|---|---|---|---|---|---|
| 单轮 `S−R` | +0.0283 * | +0.0418 * | +0.0567 * | +0.0196 * | **+0.0335 *** |
| 多轮 `S−R` | +0.0152 | −0.0000 | **−0.0109** | +0.0055 | **+0.0027** |
| 多轮 `R−N0` | +0.0270 * | +0.0363 * | +0.0271 * | +0.0179 * | +0.0261 * |

(`*` = CI 不含零)

**单轮下每个 K 都显著为正;多轮下没有一个 K 显著,K=3 点估计还是负的。**
所以不是高 K 有效被低 K 稀释,是整体消失。格式收益则在每个 K 都稳定为正。

## 4. 结论与它动摇了什么

我们一直当作基石的 `S0 − R0 = +0.0335`——"冻结 policy 本身已有可利用的稀疏选点优势"
——**是单轮 renderer 的产物**。合理机制:单轮格式对模型陌生、更难,而稀疏选点带显式
step 标签,**部分补偿了这种混乱**;多轮格式恢复熟悉度(并自带标签)之后,稀疏选点就
没有可弥补的东西了。

### 必须一起读的限定

`choose_sparse` 选的是**随机非相邻帧**,不是 oracle 选出的好帧。所以本结果说明的是
**随机稀疏 ≈ 最近窗口(在好格式下)**,而**不是**"任何选择都赢不了最近窗口"。
oracle selector 仍可能找到真正有用的集合 —— 这正是 Gate 3 要测的。

因此本结果不否定 selector,但它**抽掉了原来支撑 selector 的那条证据**,
把 Gate 3 从确认性实验变成了**承重实验**。

### 连带影响

DiD pilot 的 s50(`did_select = +0.0055`,dev 全 gate 通过)是在**单轮格式**下测的。
换 renderer 则 `S0/R0/SA/RA` 全变,adapter 需重训;且换格式后 `S0−R0 ≈ 0`,
adapter 要学的东西从"放大一个已存在的优势"变成"凭空创造优势",难度不同。

## 5. 冻结决定:暂缓

预注册规则是"`S_multi − R_multi > 0` **且** `S_multi − N0 > S_single − N0` 才换":

- 第二条满足(+0.0288 > +0.0128);
- **第一条不满足**(+0.0027,CI 跨零)。

严格执行应保留单轮。但该规则写作时未预见当前情形:它假设"换格式后选点收益仍在"是
好事的标志,而实际发生的是"选点收益消失、净效果反而更好"。

**决定(2026-07-26,人工):已冻结 official-style sparse multiturn。**
见 [`docs/renderer_freeze_v1.md`](renderer_freeze_v1.md)。

推翻预注册规则的不是重算口径,而是 Gate 3 测出了写规则时未测量的量
`recent_over_b0 = Q(Recent_B) - Q(空历史)`:单轮下它显著为负(B=2 时 −0.0615),
即 **Recent 在单轮格式下是病态基线**,于是随机选帧也能赢它(`random_gain` 三个
预算全部显著为正)。多轮下 `recent_over_b0` 三个预算全部显著为正、`random_gain`
全部跨零。预注册规则隐含假设"两种格式下 Recent 都是合理基线",该假设在单轮下不成立,
所以 `S_multi - R_multi ≈ 0` 不是"选择没价值",而是"基线终于正常了"。

## 6. 未做的检查(风险如实记录)

1. **`R_multi − N0 = +0.0261` 是一个 bundle**:稀疏声明 + 每图 step 标签 + 当前图标签。
   而在 recent-K 下根本不会产生 intervening 块,所以这个数很可能纯粹是
   "显式 step 标注有用"——**那对 N0 自己同样适用**。三个成分未做消融,
   "我们修好了格式"可能实为"标签有用"。
2. 全部是 teacher-forced logprob,**没有闭环 AndroidWorld 验证**。0.02~0.03 nats
   的格式效应未必转化为任务成功率。
3. 未验证 probe 的 no-adapter 路径与 scorer 的 inject-then-bypass 路径逐位一致,
   只通过复现已发表 delta 间接佐证。
4. trainer 的格式白名单现在允许任何 config 声明新格式,没有 fail-closed 检查阻止它
   出现在已冻结的五臂语料里。

## 7. 实现说明

`full_responses` **不在 v5 样本行里**(只有裸描述 `action_texts`;完整响应仅存在于
N0 的 assistant 轮,即只覆盖 recent-K 尾部,对更早的稀疏步无用)。probe 因此从
GUI-Odyssey 原始 annotation 用语料构建器自己的 `target_arguments()` + `official_response()`
**重建**完整响应,再与 N0 现有 assistant 轮**逐字节对账**,任何不一致即整组 fail-closed
丢弃。实测 dev 上 **240/240 组对账通过、0 步无法重建**,全程没有用裸描述凑数。
