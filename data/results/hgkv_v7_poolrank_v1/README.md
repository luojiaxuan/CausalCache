# v7 `did_pool_rank` vs `did_ra_aware`(同语料,只换 objective)

> **结论(单种子,待复现):去掉 `L_gain` 确实压低了绝对抬升,但代价是适配器
> 把 recent 臂打坏了。`did_select` 的改善 195% 来自 $A_r$ 下降,而稀疏臂自己的
> 增益 $A_c$ 反而降了 0.0029。这不是"更会用相关旧帧",是"更不会用 recent 窗口"。**

## 对照(单变量:同一份 v7pool 语料,world_size 均为 4,只差 objective)

dev = `desktop-did-corpus-v4/samples-b1.jsonl`,94 组 / 77 episode,
与 `v4baseline` 同一把尺子;选点用各臂 gate_report 自己的 `selection.selected`。

| 臂 | checkpoint | $A_c$ 稀疏臂 | $A_r$ recent 臂 | `did_select` | \|wrong drift\| |
|---|---|---|---|---|---|
| `didbase`(旧目标) | lora-step300 | **+0.01603** [+0.01260,+0.01990] | +0.00088 [−0.00142,+0.00319] | +0.01515 [+0.01070,+0.02004] | +0.00652 |
| `poolrank`(新目标) | lora-step250 | **+0.01310** [+0.00982,+0.01676] | **−0.00514** [−0.00731,−0.00274] | +0.01824 [+0.01374,+0.02311] | +0.00369 |

## 分解:改善来自哪一侧

$$\text{did\_select} = A_c - A_r \;\Rightarrow\; \Delta\text{did\_select} = \Delta A_c - \Delta A_r$$

| 项 | 值 | 读法 |
|---|---|---|
| $\Delta\text{did\_select}$ | **+0.00308** | 表面上是改善 |
| $\Delta A_c$ | **−0.00294** | 稀疏臂的绝对增益**降了** |
| $\Delta A_r$ | **−0.00602** | recent 臂**被打坏** |
| 来自 $A_r$ 下降的份额 | **195%** | 全部改善都出自这一侧,还倒贴 |

## 这正是 `L_gain` 当初存在的理由

v4 config 的 `gain_rationale` 原文:

> 只有 L_select 时,压低 $\ell_{RA}$ 与抬高 $\ell_{SA}$ 同样能减损失,而"把 recent 臂
> 打坏"不是要的能力。L_gain 要求 sparse 臂的绝对增量自己也为正。

我为了封死均匀抬升(`[m - A_c]_+` 是目标里唯一的绝对量,也是均匀抬升的唯一激励源)
把 `L_gain` 整个拿掉 —— **同时把这道闸门一起拆了**。

**预注册判据在这里失灵了**:两条都过($A_c$ 不高于 ✓、`did_select` 为基线 120% 且
CI 排除 0 ✓),但 criterion 1 写的是"$A_c$ 不高于",而 $A_c$ **下降**表面上正合本意,
实际却在说适配器整体变弱。写判据时没预见"通过降低 $A_r$ 来做大 DiD"这条通道。
教训:**差值型指标的判据必须同时约束分解出来的两侧,不能只约束差和其中一侧。**

## 修法(定向,不是回退)

保留"全是差"的结构(它确实封死了均匀抬升),另加一个**单边**约束禁止 $A_r$ 下降:

$$\mathcal L_{\text{floor}} = \lambda_{r}\,[-A_r]_+$$

它只惩罚"把 recent 打坏",**不奖励绝对幅度**,所以不会把均匀抬升的激励放回来 ——
与直接恢复 `L_gain`(奖励 $A_c$ 变大 → 均匀抬升重新有利可图)是两回事。

## 口径与边界

- **单种子,不足以下结论。** `didbase_seed2` 与 `poolrank_seed2`(同为新种子 20260802、
  world_size 4)正在跑,构成 {目标}×{种子} 的 2×2。$\Delta A_r = -0.006$ 这个量级
  是否跨种子稳定,等那两臂。
- 门控是 **development 口径**的过滤器,不是结论。离线全过而闭环无提升在本项目
  已经发生过(见 `hgkv_v6_capsweep_v1/PREMISE_REFUTED.md`)。
- 均匀抬升的**主战场不在这张表上**:$A_r$ 在 DiD 语料上本来就≈0,那 +0.0092 长在
  selector 的 set 打分语料上。要直接验证修法,得用 poolrank 的适配器重打一批 set 分
  再做 k 分层(`score_selector_v4_sets.py --limit-states` 抽样即可)。
  **那才是直接证据,本表是间接的。**

## 归约脚本的一个 bug(已修,值得记)

`reduce_hgkv_v6_capsweep.py` 原来找 `selection.selected_label`/`label`,而字段名是
`selection.selected`;取不到就**静默**退到"第一个 selectable"= `lora-step50`
(训练最早、几乎没学到东西),还把 pick_mode 标成 `selectable` 而非 fallback。
后果:didbase 被读成 $A_r$=+0.0036 / `did_select`=+0.0002,判定完全反过来。
现在三个键都认,label 对不上就抛错、不静默回退。

## 产物

- `gates/didbase.json`、`gates/poolrank.json`:两臂全 checkpoint 的门控原始报告;
- `capsweep.json`:归约中间量(其中 `verdicts` 用的是 v6 的判据,**不适用本页**);
- `PREREGISTERED.md`:结果落地前写定的判据与失败模式。
