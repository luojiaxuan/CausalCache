# 外审:selector 决策过滤应按策略版本而非墙钟年龄(2026-09-01)

渠道:ChatGPT 临时聊天,推理档「极高」,思考 3m 25s。全文以截图逐段读取后
转录(`get_page_text` 对本次长回复只返回了片段)。

## 我方送审的判断

热换只在轮次边界发生 → 一轮窗口内的决策全部同版本、对本轮起始权重完全
on-policy → `--max-age 900s`(短于实测轮次周期 1825–2416s)丢掉的是
on-policy 数据;真正陈旧的只有"episode 回报晚到、跨了轮次边界"那些。
拟改为按策略版本号过滤。

## 它同意的

- **"900 秒规则对 policy staleness 是错误的抽象;版本过滤才是正确的主机制。"**
  墙钟年龄不决定一个样本是否 on-policy——同一 selector 快照采出的决策,
  不会因为它是 901 秒还是 2000 秒前的就变成 off-policy。
- 时间阈值**两个方向都会错**:既拒绝"旧但同策略"的样本,又会在 reload 后
  立刻接受"很新但旧策略"的样本。
- 年龄可以保留,但只作为**大得多的运维 TTL**(损坏日志、僵尸 episode、
  重复回放、重启后的远古数据),不参与统计意义上的 on-policy 判定。

## 它反对、且我认为它对的(三个漏洞)

1. **"exactly on-policy" 说过头了。** 时间线是
   `read_k → train_k → reload_{k+1} → read_{k+1}`。在 `train_k` 期间推理仍
   在服务 `v_k`;这段时间产生的决策**不可能被第 k 轮消费**,它们首次出现是在
   `read_{k+1}`,而那时 learner 已在 `v_{k+1}`。所以"只有回报晚到的才陈旧"
   是错的——存在结构性的第二个来源。我的版本过滤会**正确地**拒绝它们,
   但我的口头模型没有涵盖这种情况。
2. **跨 reload 的 episode 比单条陈旧决策更糟。** 若一条 episode 用了
   `[v_k, v_k, v_{k+1}, v_{k+1}]` 四个版本的 selector 决策,而终局回报被
   赋给全部决策,那么"只保留 `v_{k+1}` 的决策"**并不能**让它们真的对
   `v_{k+1}` on-policy:它们的状态部分是由更早的 `v_k` 动作诱导出来的;
   反过来,早期动作的回报里也含有 `v_{k+1}` 的后续行为。
   **建议:要么把 selector 版本按整条 episode 钉死,要么整条丢弃混版本
   episode——把有效性单位定在 trajectory 而非单条决策。**
3. **只对轮次起始快照 on-policy。** 第 1 个 optimizer step 之后 θ 已变,
   第 2–8 个 minibatch 仍是 θ₀ 采的。TRL 的 RLOO 文档明确把"对同一批数据做
   多次梯度步"称为实质 off-policy,并因此使用裁剪的 behavior/current 比率
   ——与 PPO 的动机相同。(我方已有该裁剪,即 clip_frac 指标。)
4. **联合训练的额外警告:** 从 selector 的视角看,8B executor 实质上是
   环境的一部分。executor 若在 episode 内或收集窗口内独立更新,
   `selector_version == current_selector_version` **并不保证**回报是在同一个
   联合系统下产生的。这不算 selector 的 behavior-policy lag,但后果可能一样重。

## 它给的实现纠正

- **版本号要比"进程内整数"更强**,例如 `(run_uuid, selector_update_id,
  checkpoint_hash)`。
- **由推理服务标记它实际使用的版本**;不要从时间戳反推,
  **也不要仅仅因为 trainer 发了 `/reload` 就递增记录的版本——只有当服务副本
  确实加载成功之后才激活新版本号**。这在 reload 失败、多副本、请求在途时
  才不会错位。
- 重要性采样分两种情形:轮内(版本匹配)用存下的 behavior 概率对当前
  learner 概率做比率,是对"8 步漂移"的合理保护,至少要记录 ratio/KL 并在
  变大时裁剪或早停;**跨版本**的 IS 能复用陈旧样本,但不建议从这里起步
  ——PL 的动作密度虽可算,却需要在真实采样语义(**含顺序**、无放回、mask、
  候选集、温度)下的精确联合概率;多步 episode 里单条决策级比率也修不了
  状态分布的改变。

## 部署到在跑实验之前它要求做的检查

1. 用 `decision_time / inference_start,end / return_finalized_time /
   round_read_time / optimizer_start,end / reload_start,ack / 实际服务版本`
   重建几轮时间线,**量化 `read → reload` 区间内产生了多少决策**。
2. **记录决策时刻的 behavior log-probability**,轮次开始时用 learner
   checkpoint 对"版本匹配"的样本重算所选动作的 logprob:
   `log π_start(a|s) − log π_behavior(a|s)` 应当除数值噪声外约等于 0。
   **若不为 0,就存在版本计数器掩盖不住的 serving/training 失配。**
3. episodic 回报:要么按 episode 钉死 selector 权重,要么把有效性判定提到
   episode 级(不允许混版本);executor 若能独立热换,同样打版本并统计
   混 executor 的 episode 比例。
4. **版本过滤要发生在构造 RLOO 组与优势之前**,使每个 cohort 具备
   良定义的 behavior 语义。
5. 在跑的实验:**不要为无版本戳的旧数据猜版本**,除非有权威的 reload 时间戳
   且能证明不存在在途/多副本歧义。**从一个干净的 reload 边界开始打戳,
   把历史的歧义记录隔离掉。**
6. 监控:按版本的接受率、版本滞后直方图、跨边界 episode 数、reload 失败数、
   behavior/current log-ratio、每个 minibatch 后的 KL、梯度范数。
   **若我的判断成立,接受率应当显著跳升,而 behavior/start-policy 比率不应
   同步跳升。**

## 结论(原文)

> "replace the 900 s staleness test with actual behavior-version matching,
> but define the unit of validity as the trajectory/episode where necessary,
> not blindly as an individual decision. Keep importance ratios for learner
> drift; don't use wall-clock age as a proxy for policy identity."

## 我方处置

- **采纳主结论**:墙钟改版本过滤。
- **采纳"reload 成功后才激活版本号"**:我原实现是在 POST 之前就写版本文件,
  reload 失败会导致下一轮把全部决策判为版本不匹配而清零。改为收到成功
  应答后再落盘。
- **采纳"先量化再部署"**:跨 reload 的 episode 比例、以及 logprob 一致性
  自检,都在部署前跑。
- **暂不采纳**跨版本 IS 与 `(run_uuid, checkpoint_hash)` 复合版本号:
  当前是单副本单进程,复合 ID 的收益不抵改造成本;若将来多副本再上。
