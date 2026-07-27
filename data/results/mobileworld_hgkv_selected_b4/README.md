# MobileWorld HGKV+selected-B4 campaign(多轮)

跨平台 closed-loop 主实验:GUI-Owl-1.5-8B + HGKV adapter(desktop v4 s300,
sha256 5720…67a9)+ selector-v4 two_tower(cheap 塔在线、readout 屏蔽,B=4,
beam=3,propose-then-select 两遍,mobile gap-fold 二遍渲染)。协议
mobile_agent_v3_5 官方忠实;分母 117(frozen_gui_owl_gui_only);config
`causalcache_mobileworld_official_b4_hgkv_sel_v1.json`;分支
`jaxan/mobileworld-hgkv-arm` @afe4dfd。

## 轮次与状态

| 轮 | 主机 | 布局 | 状态 | 成功 |
|---|---|---|---|---|
| r1 | hyper01 | 16 emu × 4 shard × 4 GPU | 完成 2026-07-27 | **39/117 = 33.33%** |
| r2 | hyper01 | 同上(复用 fleet/servers) | 完成 2026-07-28 | **42/117 = 35.90%** |
| r3(改单遍) | hyper01 | 单遍 last-action 臂 `mw-hgkv-sel1p-b4-v1` | 进行中 | — |

两轮合并(selected 均值 34.62%)vs 冻结 B0 单轮:**+6.41pp CI[−0.85, +13.68]**
(任务级配对均值差,bootstrap 10k;B0 加轮 r2/r3 落地后基线侧方差还会收窄)。

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
