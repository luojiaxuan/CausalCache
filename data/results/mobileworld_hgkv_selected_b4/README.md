# MobileWorld Frozen+selected-B4 campaign（历史目录名保留）

> **Arm 标签更正（2026-07-31）**：Claude 生成论文汇总时对调了
> `HGKV+selector` 与 `Frozen+selector` 的标签。本目录名和下方部分历史运行说明
> 保留原样以维持路径 provenance，但这里的三轮 `39/42/38` 及其
> `33.9%/28.0%/40.6%` 分层汇总在当前论文口径中属于
> **Frozen+selector**。论文主臂 **HGKV+selector** 的三轮均为 `43/117`
> （memory-critical=`19/62`、control=`24/55`），汇总见
> `../mobileworld_review_ablations_v1/README.md`。不得再用目录名推断 arm 语义。

跨平台 closed-loop 历史 repeated comparator:GUI-Owl-1.5-8B +
selector-v4 two_tower(cheap 塔在线、readout 屏蔽,B=4,
beam=3,propose-then-select 两遍,mobile gap-fold 二遍渲染)。协议
mobile_agent_v3_5 官方忠实;分母 117(frozen_gui_owl_gui_only);config
`causalcache_mobileworld_official_b4_hgkv_sel_v1.json`;分支
`jaxan/mobileworld-hgkv-arm` @afe4dfd。

## 轮次与状态

| 轮 | 主机 | 布局 | 状态 | 成功 |
|---|---|---|---|---|
| r1 | hyper01 | 16 emu × 4 shard × 4 GPU | 完成 2026-07-27 | **39/117 = 33.33%** |
| r2 | hyper01 | 同上(复用 fleet/servers) | 完成 2026-07-28 | **42/117 = 35.90%** |
| 单遍臂 | hyper01 | 单遍 last-action `mw-hgkv-sel1p-b4-v1` | 完成 2026-07-28 | **34/117 = 29.06%** |

### 单遍 vs 两遍(mobile 零样本,2026-07-28)

单遍 sel1p vs 冻结 B0 三轮:+0.85pp CI[−6.55,+7.98](无效);
单遍 vs 两遍两轮:**−5.56pp CI[−11.11,−0.43](单遍显著更差)**。
结论:零样本 mobile 上 witness 参照必须用 proposal(两遍);last-action 参照
只在训练域(desktop)等效。paper 部署叙事改为"域内单遍、跨域两遍"。
明细:`per_task_success_sel1p.json`。

### 合并配对(2026-07-28 终版口径)

selected 两轮均值 **34.62%** vs 冻结 B0 三轮均值 **28.21%**(33/32/34,轮间
极稳;r2 有 1 个 evaluator 病态任务 MastodonUpdateContactsTask 按冻结先例
missing→0):**Δ=+6.41pp,CI95[0.00,+12.82],bootstrap p(Δ≤0)=0.024**——
显著超冻结基线(压线)。B0 per-task:`b0_per_task_r{2,3}.json`(r1 取
Aries 官方轮)。

### 两遍开销审计(r2 全程 2224 步,SELECT_AUDIT)

pass-2 触发率 **81.7%**(池>B 时 96.9%);中位 pass1 4.23s / selector 39ms /
pass2 3.26s;**平均每步开销 +65.9%**。两遍臂的真实成本高——这组数字是
"主线切单遍 last-action"的决定性依据(单遍开销仅 selector 的 ~7-39ms)。
文件:`two_pass_overhead_audit.json`、`per_task_success_r2.json`。

配套加轮:冻结 B0 r2/r3(hyper00,16 emu 拆两半并行,run roots
`/data/mw/runs/mw-b0-r{2,3}`);selector-only 消融 frozen+selected-B4
(Taurus `mw-selfrozen-b4-v1`,16 emu × 8 shard × 8 GPU)。

## r1 配对检验(bootstrap 10k / exact McNemar,missing=0)

- vs 冻结 B0(33/117):**+5.13pp** CI[−2.56,+12.82],不一致 14:8,p=0.286
- vs 冻结 recent-B4(35/117):+3.42pp CI[−4.27,+11.11],12:8,p=0.503
- 冻结口径复算校验:B4−B0 = +1.71pp CI[−5.98,+9.40] p=0.824(与 4f4ad25 一致)

单轮功效不足;判定推迟到 3 轮/臂后的任务聚类配对均值差。

## Provenance

- r1 run root:hyper01 `/data04/jaxan/mw/runs/mw-hgkv-sel-b4-v1`
  (轨迹 LOCAL_PRIVATE_RAW_TRACE,不入 git);r2:`…/mw-hgkv-sel-b4-r2`。
- 冻结臂 per-task:Aries `/data6/jiaxuanluo/runs/mw-official-sequential-v2-71caac9`。
- selector bundle:`marginal_scorer_twotower.pt` sha256 前缀 `c72974ab611c`。
- 模型:mPLUG/GUI-Owl-1.5-8B-Instruct @06d5faec,snapshot manifest 校验通过。
- 环境镜像:`ghcr.io/tongyi-mai/mobile_world@sha256:b680380e…`;
  MobileWorld 源 @8ae50648 只读挂载。
- 本目录:`per_task_success_r1.json`(逐任务 0/1)。
