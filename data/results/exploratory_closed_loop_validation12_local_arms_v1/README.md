# Exploratory validation-12 本地臂 closed-loop(interim,36/60)

状态:`COMPLETED_36_OF_60_LOCAL_ARMS_INTERIM / OUTCOME_EXPOSED_DEV_PROBE_NOT_A_PAPER_RESULT`。

首次在 H100 上端到端执行 validation12 冻结合同的三个本地臂(`summary_B0`/`recent_B2`/`ocr_rgb_B2`,
12 templates × 3 arms = 36 episodes,runner revision `9f7a0a7`),36/36 episodes 原子完成、0 runner
错误、parse coverage 383/388(98.7%)。两个 formal-58 learned 臂未执行,**60-episode 固定分母未完成,
本记录只是 outcome-exposed 工程/方向证据,不进任何 paper 表**。

## 结果

| arm | official success | infra failures | parse failures | model steps |
|---|---:|---:|---:|---:|
| summary_B0(无高保真) | 3/12 | 3 | 2 | 134 |
| recent_B2 | 2/12 | 3 | 1 | 158 |
| ocr_rgb_B2 | 4/12 | 4 | 2 | 96 |

模板级配对(唯一合法比较单位):`recent_B2 - summary_B0` = **0 胜 / 1 负 / 11 平**(唯一分歧
`ExpenseAddMultiple`,summary 成、recent 败);`ocr_rgb_B2 - summary_B0` = 1 胜 / 0 负 / 11 平;
`recent_B2 - ocr_rgb_B2` = 0 胜 / 2 负 / 10 平。n=12 下全部远在噪声内(单分歧对的精确符号检验 p=1.0)。

## 失败分类

- `RetroPlayingQueue`/`RetroSavePlaylist`:三臂全部 HTTP 500(疑似 Retro Music app 在本 emulator build
  的 setup 问题,与臂无关、对称);
- `TurnOnWifiAndOpenApp`:500 ×2 + 一例"初始 reward 非零"——wifi 状态疑被更早的 environment smoke
  污染(suite reinitialize 未复位 wifi);
- 4 例 strict swipe 语法 parse failure(冻结 grammar 正确拒绝,分布在三臂);
- 1 例 CUDA OOM(worker 循环未在 episode 间清空显存,已知修法)。

## 诚实解读(重要)

1. **管线成立**:36/36 原子 episode、复用冻结选择器/OCR/2,560-token v2.1 policy、成功率量级(9/36≈25%)
  与历史 native validation 一致。
2. **在这个弱 backbone + 12 模板上,"恢复高保真记忆"的 closed-loop 收益不可见**:11/12 模板三臂同
  结果;唯一 recent-vs-summary 分歧方向为负。机理提示:三臂共享 decisions 1--5 的 summary-only 策略,
  而实际 episode 平均只有 8--13 个决策——**大量 episode 在臂分叉前或刚分叉即结束**;长 stratum 4 个
  模板中 2 个被 Retro infra 全灭。记忆干预真正"有机会起作用"的决策数太少。
3. 对 closed-loop 主实验的直接含义:要让记忆效应可测,需要(a)修复 Retro/wifi 环境问题,(b)更长
  horizon 的任务组合与更多模板(train-60),(c)并接受在 25% 成功率地板上配对功效天然有限。这些
  应写入 train-60 development contract 的设计输入。

## Artifact

- 本目录 [`summary.json`](summary.json)(content SHA `d2d75d1d...`);
- 逐 episode 原始 records(36 个,含逐决策 selection/action/audit):H100
  `/data/jaxan/causalcache/runs/causalcache-v12-local-arms-9f7a0a7/episodes/`,`PENDING_HF_UPLOAD`;
- 执行:3 workers(GPU 0/1/2)× 3 emulator containers(`causalcache-androidworld:11cea575` 原生重建,
  digest 见 build log),派生策略镜像 `causalcache-policy-ocr:v1`(冻结 OCR 包版本);
- smoke 链:environment smoke(`SystemWifiTurnOn` 0→1,36.2s)与单 episode 实测
  (`recent_B2:MarkorDeleteNote` official_success=1.0,92.3s)先于正式 36 episodes 通过。
