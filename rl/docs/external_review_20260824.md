# ChatGPT 外审:固定 B 选帧 vs 自适应 B(2026-08-24)

外审设置:临时聊天、纯英文单条提示、推理档"极高"、思考 2m20s。
提交材料为当日全部实测数字与 MobileWorld 文本折叠机制。

## 结论一句话

**放弃 Option A 的现有形式,走 Option B,并改名** ——
不叫 "memory selection",而是
**"adaptive allocation of visual history on top of preserved semantic memory"**。

## 对 Option A(固定 B,学选稀疏重要帧)的三大攻击

1. **"你没有在选记忆,记忆一直都在"(构念效度问题)**
   未选中的帧以 conclusion/tool response 形式存活,两臂知道的
   *what happened* 基本相同,差别只在像素级细节(几何、视觉状态、图标、
   确切外观)。所以研究的是 **which historical states deserve visual
   fidelity**,不是 which deserve retention。

2. **"你自己的实验说这个干预几乎不起作用"** —— 比"不显著"更糟:
   116 配对中 **102/116 = 87.9% 的任务对策略不变**,只有 14 题提供信号。
   **即使一个只在 recent/random 之间按任务择优的 oracle 也只有
   40/116 = 34.5%,相对 recent 的 31.0% 仅 +3.5pp 上界。**
   叠加 AndroidWorld 全 null 与 OSWorld +0.9pp null,审稿人可以说:
   *三个 benchmark 没有可信的 selector 效应,你在推销唯一那次有利波动。*

3. **"固定 B 凭什么普适?"** MobileWorld 官方榜单本身就有反证:
   Kimi K2.5 在 1 张历史图的复跑中相对 3 张掉了几分,而 Claude Opus 4.7
   用 1 张是因为更多图会诱发幻觉。固定 B 看起来像一个被事后优化的
   任意系统参数。

## 对 Option B(自适应 B)的三大攻击

1. **"这是 context budgeting,不是 memory research"** ——
   外审明说**这个攻击是对的,应当承认而非辩解**。干净的提法是:
   *给定已压缩的语义轨迹,何时值得把历史视觉证据注入 executor?*
   若论文继续说 "memory retrieval",审稿人会指出**什么都没被遗忘**。

2. **"你的效率账可能是假账"**:若 selector 用 VLM 看完 11 张图再送 1 张
   给 executor,你减少的是 executor 的上下文,不必然减少总视觉计算。
   Pareto 的坐标轴必须诚实拆开:executor 历史图数/多模态 token、
   selector 计算、端到端延迟或成本、成功率。
   "同等成功率下 executor prompt 少 70% 图像"可辩护;
   "少 70% 计算量"不可,除非真去测。

3. **"你在证明有东西可适应之前就发明了自适应策略"** ——
   B=0 / recent-2 / full-history-11 是**硬 go/no-go**,必须先跑。
   且仅画 Pareto 曲线不够:学到的策略必须**推动前沿** ——
   同等平均历史图数下成功率高于任何合理静态策略,或同等成功率下
   图数显著更少。否则"自适应 B"只是绕远路取到一个静态 B 已有的点。

## 外审给的决策表(前置实验的判读规则)

| 静态实验结果 | 结论 |
|---|---|
| B0 ≈ B2 ≈ B11 | **在此 harness 下终止该课题**,历史像素不够重要 |
| B2 > B0,且 B11 ≈ B2 或 B11 < B2 | **强支持 Option B**:有些视觉有用,过量视觉无用/有害 |
| B11 > B2 > B0 | 视觉历史重要,用自适应 B + 选帧(涵盖 A) |
| B11 ≫ B2 | **Option A 更站得住**:稀疏选择有明确靶子——廉价地找回全历史增益 |

注意最后一行:外审**不永久否决 A**,而是要求
"先看到 full history 打败 recent-2,再相信 A 有值得解决的问题"。

## 跨 benchmark 泛化

* **Option A 的迁移:外审赌它不成。** "哪张旧截图重要"与 app 结构、
  任务分布、executor 行为、摘要格式、视觉惯例高度纠缠;用 117 个
  MobileWorld 任务 + 仅终局二元奖励去学,信号极弱。
  我们 OSWorld +0.9pp 可能已经在说这件事。桌面迁移尤其激进 ——
  AndroidWorld 论文自己报告过把桌面向的 web agent 适配到移动端效果更差。
* **Option B 的迁移:合理但远未确立。** "何时需要视觉"比"哪一帧重要"
  有更多不变结构,例如:动作没有产生可观察进展;摘要提到视觉上有歧义的
  状态;需要比较当前与先前布局;任务问的是外观/位置/图标/图像/颜色/状态;
  跨应用边界后需要视觉重定位;agent 在打转或不确定。这些现象手机与桌面共有。
* **建议:MobileWorld → AndroidWorld 作为主迁移测试**(同为 Android),
  MobileWorld → OSWorld 作为更强的 OOD,而非论文必须存活的那一个。

## ★ 外审提供的新情报:benchmark 本身可能没有足够记忆敏感任务

MemGUI-Bench 相关工作明确论证:**现有移动 GUI benchmark 中只有约
5.2–11.8% 是记忆相关任务**,因此他们构造了 89.8% 任务刻意压测记忆的
benchmark。这与我们 **76/116 双臂皆败**、AndroidWorld 全 null 高度一致:
executor 的失败可能主要是 grounding / planning / control / 任务知识,
**更好的 selector 修不了这些**。

外审对"前提是否薄弱"的回答:
> "GUI agent 受益于记忆"这个宽泛前提不弱;
> 但"在语义摘要历史下,截图选择会显著提升
> AndroidWorld/OSWorld/MobileWorld 成功率"这个窄前提**目前是弱的**。
> 这是两个不同的主张。

## 外审要求的四项证据(可发表的 B 论文)

1. **静态因果证据**:B0/B2/B11 清楚确立历史像素影响成功率;
2. **真实效率前沿**:冻结的自适应策略在**匹配的平均视觉上下文**下
   打败静态预算;
3. **零样本迁移**:MobileWorld 训练、全部冻结,在 AndroidWorld 上取得增益,
   executor 与历史序列化方式完全相同,**不得在目标 benchmark 上调参**;
4. **机制验证**:自适应的胜出必须**不成比例地**出现在缺失信息确实是视觉的
   场合,而不是因为 selector 偷学了任务长度、app 身份或步数。

若 (1) 失败,外审建议**改写成负结果论文**:
*语义轨迹摘要在标准 GUI benchmark 上基本消除了截图记忆的收益*,
配受控消融说明像素何时不再增加信息;再用 MemGUI-Bench 或刻意的记忆敏感
任务证明这个 null 是 benchmark/harness 的性质而非普遍事实。

## ★ 统计学纠错(外审最后一条,我方必须采纳)

> 不要用 p=0.34 声称 33.9% 的复现"匹配"官方 37.6%。
> **未能拒绝差异不等于等价性证据。**
> 应报告差异与置信区间;若复现等价性重要,需指定等价界并做等价性检验
> (如 TOST)。否则审稿人立刻可以攻击这一点。

此条直接推翻我方当日"p=0.341 → 复现成立"的措辞。台账需据此修订。
