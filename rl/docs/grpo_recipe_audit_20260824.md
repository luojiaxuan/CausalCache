# GRPO recipe 外审结果与采纳的修订(v1→v2,2026-08-24)

外审(ChatGPT 极高档,思考 4m07s)对送审版 recipe
(grpo_recipe_20260824.md,commit df95a05)的判词:
**"I would not run this recipe unchanged"** —— 两个根本问题:
selector 估计量被夸大(oversold),任务筛选闸门与学习目标背道而驰。

## 一、被驳倒的两处设计(采纳,v2 修订)

### 1. "对照臂反事实隔离"是错误表述,且 4+4 拆分浪费一半预算

* 数学事实:对照臂均值作为 baseline **不改变 selector 梯度的期望,只改
  方差** —— 任何与动作无关的 baseline 都如此。称它 "counterfactual
  isolation" 是在邀请审稿人开火;正确名称是 **recent-policy control
  baseline**(control variate)。
* 实际害处三条:
  a. **50% rollout 预算花在不产生 selector 梯度的样本上**;RLOO 的优势
     恰是每个样本既当策略样本又当 leave-one-out baseline;
  b. selector 偏离 recent 后,对照均值对 selector 期望回报的估计越来越
     偏心(仍无偏但方差恶化),且 Var(r̄_c)=q(1−q)/4 的噪声灌进全部
     selector 轨迹;
  c. **二元奖励下 4 条 selector 样本的组统计极差**:组内非全同
     (才有梯度)的概率 = 1−q⁴−(1−q)⁴,q=0.1 时仅 34%、q=0.2 时 59%;
     8 条则为 57% / 83%。4+4 等于自愿把稀疏奖励变得更稀疏。
* **v2**:G=8 全部 selector 臂 + selector RLOO;recent 对照 rollout
  **低频周期性采集**(如每 5-10 step 一批)只用于**测量** selector-vs-
  recent 差距,不进梯度、不做因果声明。

### 2. ★ 最可能杀死训练的一条:recent 成功率 ∈(0,1) 的硬闸门

* 论证:"全败任务组内 advantage 恒 0" 只对**当前采样组自身全同**成立;
  用**另一个策略(recent)的历史成功率**做门,会**恰好删掉 selector
  能创造第一次成功的任务** —— recent 0/8 但某旧帧含关键瞬态信息、
  学出来的 selector 终能取回它,正是我们最想要的任务。
  76/116 双臂皆败之下,这个门可能把有效训练集塌缩到个位数任务。
* **v2**:撤硬闸门,改 **均匀采样地板(20-30%)+ 难度优先级**
  (按 selector 臂经验不确定度/混合组率加权);采到全同组时限次重采
  其它轨迹或换任务。无需任何 reward shaping。

## 二、审稿人会打的其它点(采纳)

1. **术语**:这不是 GRPO,是自定义混合体(selector: episodic REINFORCE
   + 跨臂 baseline + PL 结构化动作;executor: RLOO + PPO 裁剪/KL;
   自适应课程;双策略联动)。**如实命名**:
   *joint executor–memory policy optimization with arm-specific
   RLOO/control baselines*;
2. **cross-play 检查点评测是必做项**(归因矩阵):初始 sel×初始 exec /
   终版 sel×冻结 exec / 初始 sel×终版 exec / 终版×终版 —— 否则
   "记忆选择的增益"可能只是 executor 学会了容忍非连续截图(正对我们
   自己提出的 OOD 论证);
3. **selector 无信任域**:clip 只给了 executor;41M 头吃 50 步二元
   Monte-Carlo 回报 + 熵,一个幸运组就能任意跳。v2:对 **联合
   Plackett-Luce slate 概率**(非两个边际之积)做 PPO 式比率裁剪或
   old-policy KL;
4. **熵系数 0.01 未归一化**:候选帧数随步数涨,最大 PL 熵也在涨,同一
   系数在 step 5 与 step 45 是不同强度的正则。v2:按最大可行熵归一化;
5. **episode 级共享 credit 合法但残暴**:50 步失败会惩罚所有选帧
   (哪怕第 49 步 executor 犯了无关错误)。终局奖励**不禁止**状态依赖
   baseline / per-step critic 这类不改变奖励的方差缩减手段——留作
   全量阶段消融。

## 三、外审的最小改动版 recipe(v2 采纳为准绳)

1. 撤 recent-success 硬闸门,保均匀任务采样地板;
2. **6-8 条 selector 臂 rollout + selector RLOO** 为默认估计量;
3. recent 对照低频采集,只测记忆差距,不声称因果梯度隔离;
4. executor 主要在 selector 臂上下文上训练;若保留 recent 轨迹,
   降权并消融该权重;
5. selector 信任域:PL 联合比率裁剪或 old-policy KL + 归一化熵;
6. 显式监控:混合组率、selector 熵、选帧年龄分布、recent-vs-selector
   差距、train-vLLM logprob 误差、cross-play 表现;
7. 方法命名如实,不藏在 "GRPO-style" 后面。

外审结语(原文大意):当前 recipe 当然能跑出一条学习曲线;问题是
**失败时你分不清死因**(稀疏 credit / 移动闸门 / selector 塌缩 /
双策略共适应 / 训推失配),**成功时对照设计又支撑不了你想要的因果解释**。

## 四、保留意见(我方,不采纳外审的部分)

无实质保留。唯一补充:外审建议的 per-step critic 在 §9(终局奖励)下
合法,但 smoke 阶段不引入 —— 先用最简估计量把管线跑通,critic 属于
全量阶段的方差缩减消融。
