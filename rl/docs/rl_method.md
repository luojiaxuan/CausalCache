# CausalCache-RL 方法文档:我们到底怎么做 RL

回答四个问题:训什么(LoRA RL,不是全参数)、GRPO 配置是什么、
怎么 rollout、怎么更新参数。代码指针全部给出,与实现逐行对应。

---

## 1. 训什么:LoRA RL,策略全程冻结

**不是全参数 RL。** GUI-Owl-1.5-8B 主干与视觉塔一个参数都不动
(装载即 `requires_grad_(False)`,且受冻结守卫保护:模型目录清单 +
transformers 5.6.0 源码哈希,漂移即拒绝启动)。可学参数只有两块:

| 可学件 | 参数量 | 作用 |
|---|---|---|
| **selector 打分头(v2 默认)** | ~4K(Linear(4096,1),吃策略自身 hidden state) | 决定"选哪些历史帧进 prompt";§2.5。旧 28 维 cheap 塔(~43K)仅作对照臂 |
| **HGKV LoRA** | ~2.6MB(last-8 层 k_proj/v_proj,rank 8,α 16) | 决定"选进来的历史帧怎么被读"。只作用于**历史图 token** 的 K/V(token 级门控 mask),当前屏与文本不受影响 |

为什么这样切:这是 CausalCache 的身份——记忆是唯一可学件;同时 8B 全参 RL
的算力与崩坏风险都不必要。LoRA checkpoint 与 v4/v7 离线训练形制互通
(`inject_history_gated_kv` / `history_gated_state_dict`)。

代码:`rl/code/causalcache_rl/action_scoring.py`(装载与注入)。

## 2. 优化目标:GRPO + KL 信任域 + 熵

一条轨迹的可学 log-prob 干净分解为两项之和:

$$\log p(\tau) = \sum_t \Big[\underbrace{\log \pi_{\text{sel}}(S_t \mid x_t)}_{\text{selector 选帧}} + \underbrace{\log p_{\theta_0+\Delta}(a_t \mid \text{prompt}(S_t))}_{\text{冻结策略+LoRA 出动作}}\Big]$$

对每个任务 $q$ 采 $G$ 条 rollout,**组相对优势**(GRPO):

$$A_i = \frac{r_i - \bar r_G}{\mathrm{std}_G + \epsilon_{\text{std}}},\qquad r_i \in \{0,1\}\ \text{(任务成败,无 shaping)}$$

损失(逐 episode):

$$\mathcal{L}_i = -A_i\big[w_s \log \pi_{\text{sel}}(\tau_i) + w_a \log p_\Delta(\tau_i)\big] + \beta\,\mathrm{KL}\big(p_{\theta_0+\Delta} \,\|\, p_{\theta_0}\big) - \lambda_H\, H(\pi_{\text{sel}})$$

三个正则项各挡一个已实测的失败模式:

- **组基线** $\bar r_G$:吸收任务难度差,是二值稀疏奖励下的方差控制;
  $\epsilon_{\text{std}}{=}0.1$ 防小样本 std 爆炸。全 0/全 1 的组优势恒 0,
  **自动跳过并计数披露**(可学带抽样就是为了少产生这种组);
- **KL-to-frozen**(β=0.05):在 rollout 真实状态的动作 token 上逐 token 算
  $\sum p_{\text{on}} (\log p_{\text{on}} - \log p_{\text{off}})$,off 侧 no-grad。
  **一个杠杆替代离线时代全部手工 drift cap**——任何没换来回报的分布偏移统一被罚。
  零初始化 LoRA = 恒等映射,第 0 迭代 KL 精确为 0(冒烟已验证);
- **selector 熵**(λ=0.01):防塌缩到"永远选 recent"(冻结策略在 recent 窗上
  训过,是强局部最优)。

### GRPO 配置总表(当前默认)

| 项 | 值 | 出处 |
|---|---|---|
| G(每任务 rollout 数) | 6(冒烟 3;正式建议 6–8) | `rl_iter_loop.sh` G |
| 每迭代任务数 | 8(从 47 个可学带任务确定性抽样,seed=迭代号) | TASKS_PER_ITER |
| 奖励 | 任务成败 0/1,无 shaping | collector |
| 优势归一 | 组内 (r−mean)/(std+0.1) | trainer `--adv-std-eps` |
| KL β | 0.05 | `--kl-beta` |
| 熵 λ | 0.01 | `--entropy-lambda` |
| w_sel / w_act | 1.0 / 1.0 | `--w-sel/--w-act` |
| LoRA lr / selector lr | 1e-5 / 1e-4(AdamW,梯度裁剪 1.0) | `--learning-rate/--selector-learning-rate` |
| grad accum | 8 episode | `--grad-accum` |
| rollout 步数 cap | 训练 30 步;评测 50 步(标准口径) | MAX_STEPS |
| 记忆预算 B | **2**(用户裁定;oracle 空间是 B=4 的 2.8 倍) | BUDGET |
| 采样温度 τ | 1.0(部署/评测 τ=0 即 argmax) | TAU |

## 2.5 Selector v2(默认):policy hidden-state 打分头,零手写特征

2026-08-05 用户裁定废除 28 维手写特征(它是最后一块启发式残留;witness 家族
实测无用,内容盲特征学不出有用性,"部署零成本"的理由随 pivot 已死)。
默认 selector 换为**索引遍 + 打分头**:全部历史图缩略(144 token/图)+ 各步
动作行 + 当前屏,过冻结策略一次前向,各历史图 token 段 mean-pool hidden
state(H=4096),Linear(H,1) 打分 → PL 采样。可学参数 ~4K;年龄/新近以
位置编码形式留在模型自身表征里,用不用由 RL 决定。部署成本 = 每步多一次
~5-8K token 的低清 prefill(诚实披露;低于旧 pass-2)。
详细设计与 v1 边界:`selector_hidden_design.md`。cheap 塔保留为对照臂
(SELECTOR_MODE=cheap)。

## 3. 怎么 rollout:探索来自 Plackett-Luce 选帧采样

环境 = 真实 OSWorld(Docker VM,worker 在宿主起 VM,server 在容器出动作)。
每一步:

1. worker 把(指令、历史动作、历史截图、当前屏)POST 给 policy server;
2. server 的 selector 对候选池做 **Plackett-Luce 顺序采样**(`--selector-temperature τ`):
   B 个槽位依次从 $\mathrm{softmax}(m/\tau)$ 无放回抽取,$m$ = cheap 塔对
   (候选特征 28 维,含已选集合上下文)的边际分。**这就是 RL 的全部探索来源**;
   τ→0 时退回部署的 beam-argmax,训练/部署同一参数化,没有 train-test 结构差;
3. 选中的 B 帧 + 当前屏组成官方 gap-fold prompt,冻结策略(+当前 LoRA)生成动作;
4. **审计落盘**(`--rl-audit-dir`):每步一行 jsonl,含逐轮
   `{candidates, 归一化特征矩阵, chosen, logprob}` 与 join 键
   `(task_id, step_index, response_sha16)`。

关键设计:**审计里存的是归一化后的特征矩阵**,trainer 复算 $\log\pi_{\text{sel}}$
只需特征 × 当前 selector 参数——训练侧完全不重建环境状态,这类重建是最大的
漂移源。收集器(`collect_rl_trajectories.py`)按三键 join,任何一步对不上
**整条 episode 弃用并计数**(错配的〈特征,动作,奖励〉是静默毒药)。

## 4. 怎么更新参数:两条梯度通道,一次 AdamW

每迭代(rollout 完成后)单进程 trainer:

**通道 A —— selector(便宜,CPU/GPU 皆可)**
从审计特征复算每条轨迹的 $\log \pi_{\text{sel}} = \sum_{\text{步}}\sum_{\text{轮}}
[\,m_{\text{chosen}}/\tau - \mathrm{logsumexp}(m/\tau)\,]$,乘 $-A_i w_s$ 反传;
熵项同一前向顺带算。整条轨迹一次 backward(图很小)。

**通道 B —— HGKV LoRA(重活,teacher-forcing)**
对 episode 的每一步,用**与 serve 同一个** prompt builder
(`build_official_messages_for_request`)+ 同一装载路径
(`GUIOwlOSWorldRuntime`,visual-tokens 2560)重建 prompt,把 rollout 里实际
生成的动作文本 teacher-force 一遍,取动作 token 的 $\log p$;同一次前向顺带算
KL(off 侧再来一次 no-grad 前向)。然后 **逐步 backward**:

```
loss_step = -(A_i · w_a / accum) · Σ log p(token) + (β/accum) · KL_step
loss_step.backward()          # 单步图立即释放;梯度在 optimizer.step 前自然累加
```

逐步反传是冒烟 OOM 的修法:整条 episode(12–15 步 × 多图长 prompt)攒计算图
必爆 139G;单步图只有单 prompt 大小,数学上与整条反传严格等价。
无历史的步(如 step 1)LoRA 不在图里,`requires_grad` 守卫跳过。

两条通道的梯度进同一个 AdamW(参数组各带各的 lr),每 8 条 episode
`clip_grad_norm(1.0)` + `optimizer.step()`。产物 `selector_bundle.pt`
(与部署 bundle 同构,serve 直接吃)+ `adapter.pt`(HGKV 形制)。

**DDP(2026-08-06 起默认)**:torchrun 多进程,episode 按全局序号
`e_idx % world` 分片;**手工 all-reduce 而非 DDP wrapper**——逐步反传是
"一次 step 前多次 backward",wrapper 的桶同步与之犯冲。累积窗口以全局
episode 计数划界,窗口边界所有 rank all-reduce(SUM)梯度(缺席 rank 补零
参与)再各自 clip+step;同种子同初值同梯度 → 各 rank 优化器轨迹恒等,
rank0 落盘,末尾跨 rank 校验 `trainable_param_sum` 防分叉。数学与单进程
严格等价(浮点加序除外)。单进程(不经 torchrun)自动走原路径。

## 5. 迭代闭环与算力

```
iter N:  [R] 3×server(τ=1, 上迭代权重) + 4×worker × G代  ≈ 30–45 min
         [C] collect(秒级)
         [T] GRPO 单卡(48 ep × ~25 步 × 2 前向 ≈ 1.6s/前向)≈ 40–60 min
         server 滚动重启(每迭代必重启 = RAM 泄漏纪律)
每 20 迭代:held-out 120 任务,τ=0、50 步、2 轮,只报方向(独立脚本)。
```

单机 3 卡串行两阶段实测 ≈ **2h10m/迭代**(iters 1-7,4 worker 单卡 train)。
已做的提速(2026-08-06):worker 4→8(8 任务/迭代一人一个,rollout 墙钟
≈ 单 episode 时长)+ trainer DDP 3 卡(train 段 ≈ 单卡/3)→ 预计
**~1h-1h15m/迭代**。进一步提速要加机器(rollout 双机分代)。

## 6. 与离线时代的对照(为什么这套能避开老坑)

| 老坑(已实测) | RL 里的对应机制 |
|---|---|
| relevant=动作复现启发式,假设站不住 | 没有任何"relevant"标签,有用与否由回报定义 |
| age≥B+2 拍脑袋 | 候选池 = 全部合法历史帧,无年龄阈值 |
| 离线 U 增益不迁移闭环 | 训练信号**就是**闭环成败 |
| 均匀抬升骗过 L_gain | KL 信任域罚一切无回报的分布偏移 |
| 手工 drift cap 调不对且致 DDP 发散 | 无逐臂 cap;KL 单杠杆 |
| 冻结策略没见过稀疏 B 图 prompt | LoRA 在 rollout 分布上现场学会读稀疏记忆 |

预注册停止判据(不许事后改):60 迭代训练集成功率无上行 → 停,负结果入论文。
见 `rl_pivot_contract.md`。
