# 外审:审稿人视角的方法可靠性审计(2026-09-01,第二轮)

渠道:ChatGPT 临时聊天「极高」档,思考 1m 25s;输入 = 打包上传的方法摘要 +
作者自查启发式清单 H1–H10 + 关键代码原文(仓库私有,git 链接不可用)。

## 总判词

> "The core idea is defensible, but the current evidence does NOT support
> the two headline generalization claims as paper-ready claims. The biggest
> risks are not H4/H7-style engineering choices; they are label correctness,
> baseline fairness, test contamination, and claim/metric mismatch."

## 判定为致命(修完才许投稿)

1. **best-epoch 按 val 选、又在同一 val 报数** —— 37.5/45.1 全是开发数,
   不是泛化证据;必须 train/val/test 三向,test 只碰一次。
2. **recency 基线的帧序** —— 代码里 recency 序列化为 [k-1, k-2](新帧在前),
   selector 组合是正时序;自家实测顺序翻转 79% 的 argmax,这是基线混淆,
   全部重跑。【已核实为真,重标进行中】
3. **动作匹配器两处**:(a) 指控"alias 未生效"——核实为**假警报**
   (parse_action 在解析时规范化,审阅包漏了该函数);(b) **未知动作类型
   兜底 return 1**——核实为真:system_button/terminate/wait/interact 共
   5.2% 状态只比类型不比参数,已改为参数一致才判对(wait 除外)。
4. **任何 val/test 污染**:特征居中统计当前是全局拟合(含 val/test 轨迹),
   必须只用 train 拟合后冻结;checkpoint/超参选择同理。
5. **"parity" 一词无效**:45.1±0.6 vs 44.9 是"未拒绝差异",不是等价;
   要预登记等价边界(如 ±1pp)做配对等价检验。
6. **不得据此声称在线任务成功率提升**:现有证据是"成功轨迹离线状态上的
   单步解码正确性",在线 episode 是另一个实验。
7. 折叠摘要的因果来源需审计(在线时是否同样可得)。【自查:摘要仅含
   1..k-1 步的 executor 自产 conclusion,因果干净,将补文档声明】

## 视敏感性而定(可能致命)

匹配器容差(点击 80、滑动 150)、子串等价、单参考动作假设——+5.1 若在更
严评测器下坍缩,即评测器诱导的结果。success-only 采样对宽主张致命,
收窄为"成功轨迹的 held-out 状态上改进参考动作一致性"则可存活。

## 仅削弱(需消融/说明)

槽位混比、归一配方(改 train-only 后 2×2 消融:raw/center/unit/both)、
h=384、多基数联训 vs 仅对训练、两段式部署的 top-M(补:剪枝后最优对召回,
区分"头不行"与"检索结构不行")、checkpoint 选择(若 a priori 声明)。

## 它点名必须补的模型消融(我自查漏报)

**C3 元数据捷径**:候选特征里显式含 age/step/index/traj_len,query 里含
全部折叠摘要——不做 metadata-only / full-minus-metadata / text-only /
vision-only 的成分消融,就无法主张收益来自"学到的视觉/内容相关性"而非
"学了个时间位置启发式"。这是最重要的缺失消融。

## 其他要点

- 统计口径:3 个种子 ±0.2 度量的是训练随机性,不是评测总体不确定性;
  要按轨迹聚类的配对 bootstrap CI。
- "半量=全量 ⇒ 特征受限"是过度因果化,改写为字面观察。
- "set-energy" 名不副实:v(hc_i, hc_j) 拼接是有序非对称的——要么改名
  ordered-subset energy,要么对称化并对比一次。
- 措辞:槽内 oracle 应称 slate oracle,非 history oracle。

## 最强改写建议(采纳)

停止把方法写成"一串经验发现+补丁"(似然失败→随机有害→顺序敏感→居中
解锁→于是我们做了 X),改为**决策导向的结构化预测**统一框架:
"给定冻结 executor 与预算 B,有序历史子集 S 的效用由 executor 在按时序
序列化的 S 条件下的行为正确性定义;学习结构化代理 E_θ(S|x,B) 对该效用
排序(unary=单帧相关性,pairwise=非可加交互);推理即约束决策
argmax_{|S|=B} E_θ"。行为标签是 decision-focused supervision 而非补丁;
时序是输入定义的一部分;枚举是决策规则;归一是表示预处理;top-M 是
显式的近似推理扩展。

## 投稿前必做顺序(照单执行)

修匹配器 → 审 recency 时序 → 冻结三向切分与 train-only 归一 → 重跑主数 →
按轨迹配对 bootstrap → parity 等价检验 → metadata/text/vision 成分消融 →
success/error 状态适用域声明。
