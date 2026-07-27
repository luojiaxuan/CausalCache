# OSWorld-Verified 三臂 closed-loop 终表(2026-07-28,361/361 全满员)

冻结 GUI-Owl-1.5-8B,官方 test_nogdrive 名册(361 = 官方允许豁免 gdrive 的
口径),docker provider(KVM),官方多轮渲染(desktop_official_multiturn,
gap-fold),每臂零缺失。

| 臂 | 成功 | 成功率 |
|---|---|---|
| 冻结 B0(无记忆) | 72/361 | 19.94% |
| HGKV recent-B4 | 119/361 | 32.96% |
| **HGKV + selector 单遍 last-action(B=4)** | **123/361** | **34.07%** |

配对(任务级 bootstrap 10k):

| 对比 | Δ | 95% CI | 不一致 | p(Δ≤0) |
|---|---|---|---|---|
| sel1p vs B0 | **+14.13pp** | [+9.42, +18.84] | 66:15 | <10⁻⁴ |
| recent vs B0 | **+13.02pp** | [+8.59, +17.73] | 61:14 | <10⁻⁴ |
| sel1p vs recent | +1.11pp | [−2.49, +4.71] | 25:21 | 0.30 |

**读法**:desktop(训练平台)上 HGKV 记忆本身是巨大收益(+13-14pp);
选择相对 recent 为小幅正向但任务成功率粒度下不显著。与 mobile 形成互补:
OSWorld 任务的相关状态多在近期窗口内(recent 已吃掉大部分收益);
MobileWorld 62 个跨应用记忆任务的相关帧在远处,只有 selected 恢复能拿到
(两遍 +6.41pp p=0.024)。"benchmark 的记忆结构决定哪个臂分化"。

预注册 steps≥6 分层无判别力(358/361 任务都 ≥6 步,中位 15 步顶满
step 上限),as-planned 报告:sel1p−recent +0.84 CI[−2.79,+4.47]。

## MobileWorld 按名册设计分层(补充,vs B0 三轮均值)

| 层 | 两遍 | 单遍 |
|---|---|---|
| memory_candidate(62,跨应用) | **+7.26** [−0.27,+15.32] | +4.84 [−4.30,+13.98] |
| single_app_control(55,对照) | +5.45 [−5.15,+16.36] | **−3.64** [−14.55,+7.27] |

单遍的 mobile 失效主要来自**对照组为负**(同应用内上一步动作平凡重现 →
误换帧);两遍在两层均为正。层内各自欠功效,方向与机制一致。

## Provenance

- run roots:hyper01 `/data04/jaxan/osworld/runs/verified-sel1p-b4`;
  hyper00 `/data02/jaxan/osworld/runs/verified-{recent-b4,b0}`(recent/b0
  为两机并集,分臂重组前 hyper01 部分 77/74 任务,first-wins,零冲突);
- OSWorld @0514b9a2,VM 镜像 happysixd/osworld-docker,qcow2 字节校验一致;
- policy:serve_osworld_official_policy.py(HGKV s300 sha 5720…67a9;
  selector two_tower c72974ab…,witness=last_executed_action 单遍);
- 本目录:`final_analysis.json`、`per_task_{sel1p,recent,b0}.json`。
