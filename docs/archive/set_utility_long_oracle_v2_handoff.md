# Long-history oracle diagnostic v1 → decision distillation v2 交接

写给正在执行 decision distillation v2 的 session。v2 目前处于 train label 生成阶段
(365,043 coalitions,Hyper00 GPU 0--3 + Hyper01 GPU 2--5);本文档给出并行完成的长历史 oracle
诊断结论、对 v2 训练与 gate 的具体影响,以及可直接复用的标签 artifact。

## 1. 判定:HEADROOM_CONFIRMED

250 个 train long/very-long states(220/30,158 trajectories,`n_t` 17--43,确定性采样、
trajectory cap 3)上,candidate-complete singleton + true-greedy restoration 真值,
trajectory-equal normalized recovery:

| method | B1 | B2 | B3 | B4 | macro | very_long macro |
|---|---:|---:|---:|---:|---:|---:|
| oracle_greedy | 0.5279 | 0.6611 | 0.7368 | 0.7851 | **0.6777** | 0.6358 |
| additive top-B(纯 singleton 排序) | 0.5279 | 0.5035 | 0.5551 | 0.5322 | 0.5297 | 0.4797 |
| recent | 0.0325 | 0.2827 | 0.3282 | 0.1981 | 0.2104 | **-0.4183** |
| random | 0.1312 | 0.2171 | 0.2390 | 0.3040 | 0.2229 | 0.0876 |

`oracle_greedy - recent` macro = `+0.4674`,trajectory-clustered paired bootstrap 95% CI=
`[0.3591, 0.6304]`;预注册阈值 `lower>0.05` 以约 7 倍裕量通过。250/250 states、四 wave 0 skip、
跨 wave reference action 漂移 0。完整结果:
[`data/results/archive/set_utility_long_oracle_v1/`](../../data/results/archive/set_utility_long_oracle_v1/README.md)。

## 2. 对 v2 训练设计的直接含义

1. **Long+ 主攻方向被证实。** 长历史 oracle B4(0.785)与小历史 exact(0.825)同量级,信号无衰减;
   瓶颈确定在蒸馏,不在 estimand。保留(不要弱化)fixed-tune gate 里的 long-history slice 条款。
2. **条件边际监督是必需品,不是加分项。** additive top-B 在 B2--B4 平台化(macro 0.530 vs 条件贪心
   0.678),B2 甚至低于 B1(0.503 < 0.528)——top singletons 叠加存在明显冗余/干涉。若 loss 里有
   纯 singleton-marginal 回归项,不要期望它单独闭合 B2--B4;listwise/conditional 项的权重与覆盖
   (`S -> S∪{j}` given 已选 S)是成败关键。v2 现有的 beam-frontier expansion 标注方向正确,保持。
3. **表示必须允许"老事件"胜出。** best singleton 58.4% 落在 recent-4 之外,平均 age fraction 0.363。
   现有 numeric features 含 `age_fraction/step_fraction/reciprocal_age`;注意检查训练后模型是否对
   recency 特征形成过强先验(可用 age-bin 分组的 rank 指标做训练侧诊断)。
4. **recent 在长历史上约等于 random(very_long 为负)。** 部署对手很弱:模型只要在 Long+ 学到
   oracle 的一半(~0.44)就远超 recent(0.21)。反过来:此前 learned Long+ ≈ 0.33--0.35 输给 tune
   slice 的 recent ≈ 0.40,并不矛盾——tune Long+ 与本诊断的 train long 总体分布/加权不同;
   有效比较是同 state 的 paired 差,即上表。
5. **功效提醒(gate 设计)。** 1,063-state/100-trajectory fixed-tune 上 macro delta 的 95% CI 半宽约
   0.012--0.023;宣布胜出需要真实效应 ≥ ~0.015--0.03。鉴于 Long+ headroom 巨大而 short/medium 空间小,
   把可检验的主张压在 long-history slice + macro 双条款上是功效上最优的。

## 3. 可复用标签 artifact(合并进 v2 重训)

- **HF(immutable)**:dataset `gavinlaw/causalcache-set-utility-variable-history-mobile`,revision
  [`8d5a5021`](https://huggingface.co/datasets/gavinlaw/causalcache-set-utility-variable-history-mobile/tree/8d5a5021d8e69999ed944574bc8e243f386c2288/artifacts/set-utility-long-oracle-v1-179b0d8),
  tag=`set-utility-long-oracle-v1-179b0d8`,path=`artifacts/set-utility-long-oracle-v1-179b0d8`;
  `payload.tar.gz` SHA256=`9238f63dca61d7b18377a5df03448c2b15484522cb0e4a60cb32ffa9170c8b34`。
  Hyper00 mirror:`/data02/jaxan/artifacts/causalcache-long-oracle-v1-179b0d8`。
- **payload 结构**:`wave-{1..4}/terminals-hyper{00,01}/states/*.json`(runner 原生 terminal,
  `distance_rows` = coalition → `D(S)`)+ `wave-{1..4}/schedules/`(256-shard jsonl + receipts +
  manifest)+ 冻结 config 副本。
- **内容(250 states,共 25,032 rows)**:每 state 含 empty anchor、**全部 singleton**、recent B1--B4、
  每预算 1 个 random subset、true-greedy 路径的全部条件扩张(`S*_k ∪ {j}`,k=1..3 全量)、additive
  top-3/top-4。这正是 v2 需要的 candidate-complete + conditional 监督在 250 个 long states 上的完整版。
- **兼容性**:与 v2 labels 同 runner(`run_set_utility_variable_history_labels.py`)、同 scientific
  config(480 context-fit,SHA `90c72283...`)、同 execution config(SHA `9c655522...`)、同 GUI-Owl
  snapshot 与 source artifact(`a7213db`),distance 语义完全一致,可直接混表。与 v2 schedule 重叠的
  (state, coalition) 用既有 `1e-6` 绝对差 tolerance 去重(参照 enrichment merge 语义;历史重复漂移
  实测 ≤1.2e-7)。250 个 state_id 从 payload 里任一 wave 的 schedule shards 或 summary 的
  `completed_states` 重建。
- 合同显式允许该批 train 标签进入训练(`labels_reusable_for_training=true`);tune truth、untouched
  evaluation 均未触碰,v2 的 firewall 不受影响。合并属于新增输入,按仓库规范走 versioned config,
  不要改写已冻结的 v2 合同本体。

## 4. 若要扩样或复用 wave 机制

- module:[`set_utility_long_oracle.py`](../../code/causalcache/set_utility_long_oracle.py)
  (采样、wave coalition 构造、true-greedy 选择、reducer);
- CLI:`materialize_set_utility_long_oracle_schedules.py`(--wave N,N>1 时喂 `--wave-root k=path`)、
  `evaluate_set_utility_long_oracle.py`;
- config:[`causalcache_set_utility_long_oracle_v1.json`](../../code/configs/causalcache_set_utility_long_oracle_v1.json)
  (SHA `a8b8b7db...`);扩样应另立 v2 config 改 `bin_targets`/`selection_salt`,不复用本 salt。

## 5. 执行注意事项(与 v2 labels 同机共存)

- **全 long-history 状态下 3 lanes/GPU 必 OOM**(H200 141GB:每 lane 稳态 33--53GB + 30k-token
  microbatch 峰值);2 lanes/GPU 稳定,与外部进程共卡时降为 1 lane。
- **"preflight 时空闲"保鲜期只有几分钟**:本轮先后有其他用户 119.6GB / 68GB / 44GB 进程落到刚探测
  为空的卡上。每次 launch 前即时 `nvidia-smi` 定布局;runner 的 state/microbatch 断点让中途重启零损失。
- `run-gpu-cluster-job` launcher 的多卡 `--gpus device=a,b,c` CSV 引号 bug 已修(此前会被解析成
  Count+DeviceIDs 冲突;单卡不触发)。
- host 部署:`/data02/jaxan` 为 root 所有、SSH 用户无 sudo,写入走
  `tar | ssh docker run --rm -i -v /data02/jaxan:/data ... tar -x`;worktrees 是 `git archive` 快照,
  不能在 host 上 git fetch。

## 6. 边界

本诊断不推翻任何既有 NO-GO,不解锁 evaluation/policy replay/closed-loop,不修改 v2 冻结合同;它只
回答"长历史 headroom 是否存在"(答案:存在,且很大),并为 v2 提供可选的补充监督与训练侧诊断建议。
