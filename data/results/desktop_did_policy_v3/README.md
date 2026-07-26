# Desktop DiD v3(固定预算替换)—— HGKV 按 B 分层正式 gate

日期:2026-07-27。语料 `desktop-did-corpus-v3`(fixed-budget replacement,
`samples.jsonl` SHA `3114d4c9…66b6f`,train units 1,774 / dev 211;
构造:R = Recent-B 不同帧窗口,S/W = 最老槽位替换,age ≥ B+2,k=1 替换接口,
B∈{1,2,4} 训练、B=8 只建不训);目标 `did_ra_aware`;300 步 / ckpt 50;
HGKV 行在 hyper00 以 world=8 × accum=2 训练(16 组/步与 4 卡×accum4 逐步等价,
config `causalcache_desktop_did_hgkv_v3w8.json` 有等价性测试)。
dev 打分:每 B 独立 `samples-b{B}.jsonl`,2,000 次 episode-cluster bootstrap。

## HGKV 按 B 终表(各 B 独立选点)

| B | n(dev) | 选点 | did_select [95% CI] | frozen_sel | \|A_r\| | wrong drift | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | 94 | s200 | **+0.0120 [+0.0075,+0.0170]** | +0.0013(≈0) | 0.0086 | 0.0099 | ✅ s100 起全过 |
| 2 | 72 | s300 | **+0.0128 [+0.0090,+0.0176]** | **−0.0022** | 0.0099 | 0.0064 | ✅ s50 起全过 |
| 4(主设置) | 45 | s250 | **+0.0075 [+0.0043,+0.0106]** | **−0.0020** | 0.0089 | 0.0042 | ✅ s150 起全过 |
| 8(外推,未训练) | 11 | 无 | ≈0(CI 全跨零) | +0.0091 | ≤0.0157 | ≤0.0133 | ❌ 不显著(n 太薄) |

- **主设置 B=4 达标**;B=2/4 的冻结基线为**负**(固定预算下冻结模型略偏好纯
  Recent),HGKV 在逆风上创造正选择性——"选择性是学出来的"的更强证据;
- B 递减趋势(+0.012/+0.013/+0.0075)是 Q_B(k) 曲线素材(窗口越大单张替换
  边际被稀释);
- **B=8 处置(用户 2026-07-27 裁定)**:offline 不判定(n=11 结构性稀缺,
  AgentNet Ubuntu 长轨迹就这么多),外推效果交由 MobileWorld/OSWorld
  closed-loop 的 B=8 stress 档检验;扩 AgentNet win/mac 挖长轨迹登记为
  可选 backlog,不挡 selector 关键路径。

## 统一冻结候选(selector 用单一 checkpoint)

**s300**:合并 211 个 dev 组 pooled did_select 最高
`+0.01106 [+0.00815,+0.01403]`(dp-cluster bootstrap 10k;s150-s300 四个
per-B 全合格候选中 pooled 最大),且在每个训练 B 上单独 all_must_pass。
SHA `1754d232aac440fd3ff158e19172eee8146714dfe33772b88480263a9fb59ff2`。
**正式冻结待 §9 第 6/7 项**(生成式 parser validity + 完整动作等价,
`eval_desktop_parser_validity_v3.py`,运行中);selector 单例标签已按用户
速度优先裁定提前发射(绑定该 SHA,若 6/7 失败按新选点重跑,风险已登记)。

## k 的覆盖范围(重要口径)

- policy 训练:**只有 k=1**(替换接口,冻结设计);
- selector 标签/推理:k=0..B 全覆盖(beam teacher 的 2/3/4 元集合含任意
  recent/archived 混合,每个集合完整重打分);k≥2 布局超出 policy 训练分布,
  edge2+ 的真实 U 即为其可用性检验,若失灵则以 v3.1 小比例 k=2 训练臂补救。

## Artifacts

| 内容 | 位置 | 状态 |
|---|---|---|
| 四份 per-B gate 报告 | 本目录 `gate_report_hgkv_b{1,2,4,8}.json` | Git canonical |
| HGKV v3 checkpoints(s50-s300+epochs) | hyper00 `/data02/jaxan/runs/desktop-did-v3/hgkv/`;s300 副本 hyper01 `/data04/jaxan/mw/runs/desktop-did-v3/hgkv-ckpts/` | `PENDING_HF_UPLOAD` |
| ungated v3 checkpoints | hyper01 `/data04/jaxan/mw/runs/desktop-did-v3/ungated_kv/` | 训练完,per-B 打分待 6/7 卡空闲 |
| full_lora v3 | hyper01 同级 `full_lora/`(链式接力中) | 训练中 |
| dev 分数缓存(per row×B) | hyper01 `/data04/jaxan/mw/runs/desktop-did-v3/devscore-*/` | local staging |
| selector v3 单例标签 | hyper00 `/data02/jaxan/runs/selector-v3/singletons/`(8 shard,断点续跑) | 生成中,绑定 s300 SHA |
| v3 语料 | hyper01 `/data04/jaxan/mw/desktop-did-corpus-v3/`;hyper00 镜像 + 26GiB train/dev 帧 `/data02/jaxan/agentnet-frames-v3/` | `PENDING_HF_UPLOAD` |
