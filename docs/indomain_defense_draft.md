# 同域结果的论文表述草稿(待你选定 main.tex / frozen_selector.tex 后再落地)

> 用途:回应"同域(desktop)没有提升,方法有何意义"这一最可能的审稿质疑。
> **未写入任何 .tex**——投稿版归属(`jaxan/trusting-zhukovsky-00d2bf` 分支的 main.tex
> vs main 分支的 frozen_selector.tex)尚未裁定,落地前需要你确认。
> 数据出处:`data/results/indomain_gap_v1/ROOT_CAUSE.md`。

## 论证结构(三步,按强度递增)

### 第一步:先说明单轮口径没有分辨力,而不是急着解释"为什么不涨"

这是最重要的顺序调整。当前论文把 +1.1pp 报成"小幅正向但不显著",读者自然读成
"方法在训练域上无效"。但我们现在能证明:**这个基准在单轮口径下根本测不出这个量级**。

> We first calibrate what a single OSWorld round can resolve. Two arms that issue
> identical actions on $98\%$ of steps---a selector configured to abstain almost
> always, and plain Recent-$B$---nonetheless disagree on $14.4\%$ of tasks
> ($52/361$) and differ by $-3.9$pp. Episode outcomes are therefore not a
> sufficiently sensitive readout at this scale: a single round cannot distinguish
> effects smaller than roughly four points, which is the regime every
> allocation contrast on OSWorld falls in. We report the $15$-step contrast for
> completeness but base the in-domain analysis on utility-level measurements.

### 第二步:用效用级证据证明"头寸存在",而不是"没头寸"

当前论文的解释是"OSWorld 相关状态多在近期窗口内,recent 已吃掉大部分收益"。
**这个解释与数据不符,建议撤换**——它把一个可修的工程缺陷说成了基准的性质。

> The absence of a task-level effect is not an absence of headroom. On the
> desktop development split the teacher finds at least one profitable promotion
> in $95.3\%$ of states ($68.4\%$ of singleton edges have positive marginal
> utility), and an oracle selector operating under the same budget improves
> utility over Recent-$2$ by $+0.102$. Our deployed selector captures $6.5\%$ of
> that. In-domain the instrument is right and the student is weak---a gap in
> selector quality, not an absence of exploitable structure.

### 第三步:说明预算 B 决定了可展示的空间

> Headroom also shrinks sharply with the budget: the oracle's advantage over the
> recent anchor falls from $+0.102$ at $B{=}2$ to $+0.037$ at $B{=}4$. Four
> recent frames already cover most of what the desktop tasks need, so $B{=}4$ is
> the operating point at which selection has least to add. This is consistent
> with the budget sweep, where the largest selector margin also occurs at $B{=}2$.

## 需要同时修正的既有表述

| 位置 | 现表述 | 问题 | 建议 |
|---|---|---|---|
| OSWorld 段 | "OSWorld 任务的相关状态多在近期窗口内(recent 已吃掉大部分收益)" | 与 oracle +0.102、95.3% 正边际状态矛盾 | 改为"头寸存在但 selector 只捕获 6.5%" |
| 五臂表 | LA +1.11pp "小幅正向但不显著" | 读者会读成"无效";实际是**测不出** | 加噪声地板脚注 |
| 限制段 | 无 | 缺少部署侧校准缺陷的披露 | 见下 |

## 建议新增的限制/诚实披露

> A deployment detail limits what the current selector can express. The scorer is
> an additive two-tower model whose second tower consumes probe readouts; because
> the probe does not ship, that tower is masked at deployment. It behaves almost
> exactly as a constant (mean $+0.065$, s.d.\ $0.002$), so masking it shifts every
> predicted marginal down by that amount and drives $100\%$ of deployment-time
> predictions negative. Ranking---and hence the argmax the deployed beam uses---is
> unaffected by a constant, but every threshold or early-stopping rule is. We
> correct this with an intercept fitted on development data; results using the
> uncorrected scorer should be read as argmax-only.

## 尚未验证、不要写进论文的

- 修复后的同域闭环是否真的涨:同批双臂正在跑,结果落
  `data/results/indomain_twoarm_v1/`。**在它出来之前,以上第二、三步只依赖离线效用
  证据,不要声称闭环改善。**
- 修复 3(特征增强)未做。新证据显示 readout 塔在可加结构下退化为常数,
  说明**架构可能无法利用高维特征**——若要写未来工作,应连架构一起提,而不是只说"加特征"。
