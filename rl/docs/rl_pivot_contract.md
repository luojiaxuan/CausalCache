# CausalCache-RL 实验契约

> **v1 已结束(2026-08-04 → 08-08),结论:配置失败,非假设失败。**
> 20 迭代 + 双轮 held-out 评测:RL 81/240 vs recent-B 88/240(Δ=−2.9pp,
> p=0.345,CI 跨零)。复盘发现**两条可学通道都没被训动**(selector 位移
> 4.4%、与随机初始化头选择重合 86.7%;LoRA 的 `lora_b` 范数 0.04 vs
> `lora_a` 6.5,ΔW 约 0.1% 量级),根因是 20 迭代总共只有 67 次
> optimizer.step() + lr 过小。**v1 实际检验的是"几乎不更新参数的 RL"**,
> 对核心假设既未证伪也未支持。完整证据见 `../README.md`。
> v1 的预注册判据(60 迭代)因此**作废不再执行**——它检验的配置已被
> 证明不可能产生可测更新;下面是 v2 的全新预注册。

---

# v2 预注册(2026-08-08,用户批准重开)

## 与 v1 的唯一差别:有效步长

| 项 | v1 | v2 | 依据 |
|---|---|---|---|
| selector lr | 1e-4 | **1e-3** | 离线反事实:同数据量下位移 0.049→0.589;1e-2 起失控(位移 5.7) |
| LoRA lr | 1e-5 | **1e-4** | `lora_b` 范数 0.04 → 需同量级提升 |
| grad_accum | 8 | **4** | 每迭代优化步数 3-4 → 7-8 |
| 其余(G=6、8 任务/迭代、B=2、τ=1、30 步、β=0.05、λ=0.01、优势口径) | | **一律不变** | 保持改动可归因 |

**上线门槛(硬性,前 3 迭代内判定)**:`iter_report.json` 的
`selector_drift` 每迭代须 ≥ 0.02,且 iter-3 的 adapter `lora_b` 范数须
较 v1 同期显著增长。**不达标立即停机重调,不允许"先跑着看"**——
v1 的教训就是没有这道闸门,白跑了 20 迭代。

## v2 预注册停止判据

1. **前 3 迭代**未通过上线门槛 → 停,记录并重调超参(不算一次 campaign);
2. 通过门槛后跑满 **20 迭代** → 双轮 held-out 正式评测;
3. 若 20 迭代评测 Δ ≤ 0 且训练带无上行 → **停,负结果入论文**
   (结论口径:"在可产生实质参数更新的配置下,RL 联训 selector+HGKV
   在 20 迭代内不带来闭环收益");
4. 若 Δ > 0 但未达显著(检出下限约 9pp)→ 续跑至 40 迭代再评一次,
   **仅此一次延长**,之后无论结果如何都收口。

判据中途不改。检出力已知:单轮 120 任务需净胜约 11 个任务才达 p<0.05,
测量噪声下限为同系统重测翻转 11.7%、总数 ±4 任务——**报告必须同时给出
这两个数**,不得只报点估计。

---

# v1 原始记录(2026-08-04 pivot,存档)

## 为什么 pivot(用户裁定)

旧线两个不可辩护点:① relevant = "成功动作复现"是过强启发式(k 分层实测:
适配器只对这类帧有选择性,对部署实际选的帧无效);② `age ≥ B+2` 的 +2 无原则依据。
更深的病根(本 session 量化):**离线代理目标的增益不迁移到闭环**
(v7 双种子离线全胜、闭环三臂 2 轮无差异,跨轮翻转率 15–19%)。
且冻结 8B 从未见过稀疏 B 图 prompt——**联训的意义正是教会它读稀疏记忆**,
因此不做任何"零样本探针"式 go/no-go,直接上 RL。

## 设计

- **冻结**:GUI-Owl 8B 主干 + 视觉塔。**可学**:selector cheap 塔(~4.3 万参,
  readout 部署侧恒 mask 不训)+ HGKV LoRA(last-8 K/V r8,v4 形制)。
- **探索**:serve `--selector-temperature τ` → Plackett-Luce 顺序采样(exact-B
  无放回);τ→0 退回部署 argmax。动作侧采样解码。
- **目标**:GRPO。$A_i=(r_i-\bar r_G)/(\mathrm{std}_G+0.1)$;
  $\mathcal L=-\sum A_i[w_s\log\pi_{sel}+w_a\log p(a)]+\beta\,\mathrm{KL}(p_{\theta_0+\Delta}\|p_{\theta_0})-\lambda H(\pi_{sel})$。
  KL 在 rollout 真实状态逐 token 算(bypass 前向机制现成),
  **一个杠杆替代旧目标全部手工 cap**;熵项防塌缩到 recent。
- **奖励**:任务成败(0/1),不加人工 shaping。
- **rollout**:训练 cap 30 步、评测 50 步;G=8/任务、16 任务/迭代 ≈ 128 episode
  ≈ 30 分钟 @32 并行 VM(两 hyper 各 16,6 卡/机上限内)。

## 数据纪律

- **切分**(`data/manifests/osworld_rl_{train,heldout}_v1.json`):
  train 241 / held-out 120。fastdev 的 135 个**强制进 train**——它们已被
  2 轮×3 臂反复测量,充 held-out 会被质疑选择偏差;held-out 必须是没被看过的任务。
- 训练组内全 0/全 1 自动跳过并计数;rollout 调度优先冻结成功率 20–80% 可学带。
- 评测:每 20 迭代 held-out 固定协议(τ=0,50 步),**2 轮起报,只报方向**。
- join 纪律:collector 以 (task_id, step_index, response_sha16) 三键 join,
  任何一步对不上整条弃用——错配的 (特征,动作,奖励) 是静默毒药。

## 预注册停止判据

60 迭代后训练集成功率无上行趋势 → 停,结论为"B=4 记忆在该策略上闭环天花板
低于检出限",负结果入论文。中途不改判据;要偏离须另起一节写明。

## 组件与状态

| 件 | 路径 | 状态 |
|---|---|---|
| serve RL 模式(PL 采样 + 审计) | `serve_osworld_official_policy.py --selector-temperature/--rl-audit-dir` | ✅ |
| 轨迹收集器 | `collect_rl_trajectories.py` | ✅ |
| GRPO trainer | `train_causalcache_rl_grpo.py` | ✅(动作侧依赖 ↓) |
| **ActionScorer**(teacher-forced 动作 log-prob + KL,复用既有 runtime/adapter scope) | `causalcache/rl_action_scoring.py` | **待实现——下一件** |
| 迭代循环 runscript(rollout→collect→train→server 滚动重启) | `runscripts/rl_iter_loop.sh` | 待实现 |
| 任务切分 | `data/manifests/osworld_rl_*_v1.json` | ✅ |

ActionScorer 接口(trainer 已按此调用):`episode_action_logprob(ep, grad)`、
`episode_kl_to_frozen(ep)`、`adapter_parameters()`、`save_adapter(path)`;
prompt 重建必须走 `causalcache.osworld_official_online.build_official_messages_for_request`
同源函数,图片从 episode 的 `attempt_dir/attempts/*/step-*.png` 读。

## 2026-08-05 补充:tilde 实探与 B=2 裁定

- **tilde 计算节点无 /dev/kvm、docker socket 拒绝**(worker-7 实测)——OSWorld VM
  上不了 tilde;rollout 留 hyper,**训练步上 tilde 8×H100**(它才是瓶颈:
  ~10 GPU·h/迭代),跨站经 HF 私有仓库交换(轨迹上行 ~0.5GB,权重下行 ~3MB)。
- **预算固定 B=2**(用户裁定):oracle 空间是 B=4 的 2.8 倍,prompt 短 ~25%、
  PL 采样 2 轮,rollout/训练各省 20–30%。
- **估时**:流水线重叠后 ~1h/迭代;60 迭代判停线 ≈ 2.5–3 天,100 迭代 ≈ 4–5 天。
