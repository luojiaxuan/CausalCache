# Direct marginal v3 Stage-A rank repair

状态：`COMPLETED / NO_GO_DIRECT_MARGINAL_V3_STAGE_A`。

唯一一次 rank-loss repair 在 Hyper00 单 H200 上完成 40 epochs（321.7s，exit 0）。它只读取 250 个
train long-oracle states，不读取 tune/evaluation，也不生成 restoration labels。

| metric | v1 | rank repair | frozen gate |
|---|---:|---:|---:|
| singleton Spearman | 0.2384 | 0.2905 | `>0.5`，FAIL |
| true-best top-4 recall | 0.9873 | 0.9515 | `>0.6`，PASS |
| top-1 exact-best rate | 0.9040 | 0.8407 | diagnostic |
| learned B1 recovery | 0.4732 | 0.4375 | oracle 0.5014 |
| outside-recent4 best 的 top-4 recall | 0.9811 | 0.9591 | diagnostic |
| singleton sign accuracy | — | 0.5364 | diagnostic |
| STOP accuracy | 1.0000 | 1.0000 | diagnostic |

提高 ranking loss 后 Spearman 只增加约 0.052，仍远低于门槛，同时 top-1 与 B1 recovery 下降。因此失败
不能再解释为简单的 rank-loss 权重不足。按预注册 stop rule，不创建第三个 probe，不启动完整 conditional-
marginal training、fixed-tune v3、untouched evaluation、policy replay、closed-loop 或 matched-NLL。

这不否定 restoration teacher：同 denominator oracle 仍显著超过 recent。它否定的是当前
`contextual resampler + Set Transformer + direct marginal head` 作为 learned general-`B` student 的这条实现路线。

## Artifact

- [`summary.json`](summary.json)：Git 轻量摘要，SHA256=
  `7f5ea8411f45f0013609b4ebcc34581152778332c00a9a064a45d4ac4536cb25`；
- source revision=`8f127b4`，config SHA256=
  `1f519d972d39a0c892ac43029651e7256ce7b7d6c12c67b4fa9c22403cbca87b`；
- Hyper00 full run：
  `/data02/jaxan/runs/causalcache-direct-marginal-v3-rank-repair-8f127b4`；
- full summary content SHA256=
  `f7846a3b46bfe6416bc8219761bc1359d06533b622a7ff1d9b4617f462c1f03b`；
- full run/checkpoints 状态=`PENDING_HF_UPLOAD`；fit-only weights 不作为正式模型。

合同与停止边界：
[`docs/archive/set_utility_direct_marginal_v3.md`](../../../../docs/archive/set_utility_direct_marginal_v3.md)。
