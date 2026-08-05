# Session 总结(2026-08-02 ~ 08-05):HGKV 目标修复 → 闭环判定 → RL pivot

## 科学结论(全部有存档数据)

1. **v6 前提推翻**:drift cap 约束的 DiD recent 臂 A_r 本来就≈0(held-out −0.0008
   跨零),+0.0092 均匀抬升长在 selector set 语料上——收紧 ε 是拧没接上的旋钮。
   `data/results/hgkv_v6_capsweep_v1/PREMISE_REFUTED.md`
2. **v7 did_pool_rank 离线成功(双种子)**:去 L_gain + 池上排序后,同底 13,339
   集合上均匀成分 +0.0084→+0.0009/+0.0007(跨零),选择性 +23%/+33%,均匀占比
   68%→16%/12%。`data/results/hgkv_v7_poolrank_v1/`
3. **闭环快测(135×3 臂×2 轮)无可分辨差异**:排序两轮完全反转,跨轮翻转率
   15–19%,±4pp 内不可判。`data/results/osworld_fastdev_v1/AGGREGATE.md`
4. **用户裁定 pivot 到 agentic RL**(recurrence 启发式与 age≥B+2 不可辩护;
   冻结策略没见过稀疏 B 图,联训才能教会它)——不做 go/no-go,直接联训;B=2。

## RL 现状(rl/ 目录,自包含可迁出)

设计:冻结 8B,GRPO 联训 selector(PL 采样,τ 探索)+ HGKV LoRA,
奖励=任务成败,KL-to-frozen 替代全部手工 cap。契约与预注册停止判据:
`rl/docs/rl_pivot_contract.md`。切分 train 241 / held-out 120(fastdev 强制进 train)。

**冒烟已通到最后一关**:采样→审计→collect→prompt 重建→mask→Qwen3-VL 前向全通,
死于 OOM。两处修法(见 rl/README.md 2026-08-05 节):
① trainer 改逐步 backward(现在攒整条 episode 的图);KL 并入同一次带梯度前向;
② ActionScorer 对齐 serve 的 --visual-tokens 2560(不设则 token 数爆炸且分布不一致)。

## 基建要点(变更过的事实)

- **两台 hyper 的 `sglang-omni-jaxan` 被他人顶名重建**;h01 原容器 = 
  `sglang-omni-jaxan-2`(device-locked 4-7→容器内 0-3);h00 原容器视图丢失,
  宿主 /data02/jaxan 数据完好,要用需按 runscripts/README 第八条重建。
- tilde:计算节点无 KVM/docker,只能当 trainer;镜像/模型/语料已铺好(~/causalcache)。
- 数据搬运走 HF(gavinlaw),三机 token 已配;单机上限 6 卡(全局规则已更新)。
- RL 冒烟资产在 h01:server(GPU 序号1,端口 19501)可能仍在跑——**下一 session
  开始时先杀掉**;/bigdata/rl-smoke/*(审计、groups、fixture)。

## 下一步(按序)

1. OOM 两修(上面①②)→ fixture 重跑通 trainer → iter_report.json 出数;
2. rl_iter_loop.sh(rollout→collect→train→server 滚动重启);
3. 可学带任务表 + 正式迭代,训练曲线 & held-out 协议(契约已写死)。
