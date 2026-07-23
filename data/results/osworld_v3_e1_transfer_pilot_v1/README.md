# OSWorld AndroidWorld-v3-e1 transfer pilot v1

## 结论

AndroidWorld 上获得 paired closed-loop `+8.9` percentage points 的 v3-e1 LoRA 可以在 OSWorld
执行 desktop actions，并完成 4/30 个任务；但它没有超过 frozen GUI-Owl：

| Arm | Mean OSWorld score | Score > 0 | Policy HTTP 500 | Mean completed steps |
|---|---:|---:|---:|---:|
| frozen GUI-Owl | 0.16667 | 5/30 | 13/30 | 34.47 |
| AndroidWorld v3-e1 | 0.13333 | 4/30 | 9/30 | 36.07 |

成对差值 `v3-e1 - frozen = -0.03333`，task-level paired bootstrap 95% CI 为
`[-0.13333, 0.06667]`；v3-e1 为 1 win、2 losses、27 ties。正式判定：

> `NO_EVIDENCE_OF_POSITIVE_OSWORLD_TRANSFER_V3_E1_V1`

CI 跨 0，因此不能声称 v3-e1 必然损害 desktop；但本 pilot 没有提供正向迁移证据，点估计也低于
frozen。v3-e1 将 HTTP 500 从 13 降到 9，但可执行稳定性的改善没有转化成更高 reward。

## 固定设置

- official OSWorld no-Google-Drive roster 中 evenly-spaced 30 tasks，覆盖 10 domains；
- 两臂使用完全相同的 task identities、recent at-most-B4 memory、`max_steps=50`；
- 每臂 6 environments / 1 policy replica，在 Hyper01 GPU0/1 同时执行；
- v3-e1 checkpoint SHA256：
  `7122d8978a8c3a86e98a49c83f154e71bbe0b426dadb37e1f77efca88f9c73b8`；
- LoRA rank 16、alpha 32、LM `q/k/v/o`，视觉塔冻结；
- task-level failure 不从 denominator 删除，按 score 0 计入。

本实验隔离 policy LoRA 的 cross-platform transfer，不包含 learned CausalCache selector，也不是
361-task 完整 OSWorld benchmark。

## Source of Truth

- 轻量结果：[`summary.json`](summary.json)，SHA256
  `b274b5357d6df59b98bef9f278250a8bff62d72b2cb3454f6d845b7d294d0f2b`；
- config：
  [`code/configs/causalcache_osworld_v3_e1_transfer_pilot_v1.json`](../../../code/configs/causalcache_osworld_v3_e1_transfer_pilot_v1.json)；
- episode/reducer Git commit：`571ea60c536fe46199e69d32ee014b5fecd3eb97`；
- OSWorld revision：`b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf`；
- raw diagnostic mirror：Hyper01
  `/data01/jaxan/osworld-runner/v3e1-transfer-pilot-v1/raw`，
  951,036,579 bytes / 2,301 files；
- v3-e1 model 当前仍是 local staging artifact，状态 `PENDING_HF_UPLOAD`；
- 详细解释：[`docs/osworld_v3_e1_transfer_pilot_v1.md`](../../../docs/osworld_v3_e1_transfer_pilot_v1.md)。
