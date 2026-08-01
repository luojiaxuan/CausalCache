# MobileWorld review ablations v1：arm 标签更正

本目录保留历史文件名以避免破坏已有 provenance，但 2026-07-31 已确认 Claude
生成汇总时对调了两个 deployment arm 的标签：

- `frozensel_tasks.json` 与 `reduce_frozensel.json` 实际对应论文主臂
  **CausalCache（HGKV + selector）**，不是 Frozen + selector；
- 该主臂三轮计数相同：full=`43/117`、memory-critical=`19/62`、
  single-app control=`24/55`，三轮均值分别为 `36.8%/30.6%/43.6%`；
- 历史目录 `data/results/mobileworld_hgkv_selected_b4/` 中的三轮
  `39/42/38` 汇总实际对应 **Frozen + selector**，均值为
  `33.9%/28.0%/40.6%`；
- HGKV + Recent-4 的三轮均值仍为 `30.2%/19.4%/42.4%`。

`reduce_*.json` 中原来的 `vs_causalcache_b4` 键已更名为
`vs_frozen_selector_b4`。历史输入文件名不再作为 arm 语义证据；论文、supplement、
claim ledger 和本 README 的显式映射才是当前 Git source of truth。

论文主比较的 paired bootstrap 统计为：full `+6.55pp`
`[+0.28,+13.11]`、`p=0.0347`；memory-critical `+11.29pp`
`[+2.69,+20.43]`、`p=0.0062`；control `+1.21pp`
`[-7.88,+10.30]`、`p=0.7529`。按 memory/control 分层独立重采样得到
split-by-method interaction `+10.08pp [-2.24,+22.93]`、`p=0.1144`，
因此 interaction 只能作描述性点估计，不能称显著。
