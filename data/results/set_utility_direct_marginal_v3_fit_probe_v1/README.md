# Direct marginal v3 Stage-A fit probe v1

状态：`COMPLETED / NO_GO_SINGLETON_SPEARMAN_GATE`。

250 个 candidate-complete train long-oracle states、189 trajectories 上，direct marginal head 训练 30 epochs，
只读取 empty/singleton truth，没有生成新 labels，也没有访问 tune/evaluation。

最佳 epoch 30：

| metric | value | gate |
|---|---:|---:|
| singleton Spearman | 0.2384 | `>0.5`，FAIL |
| true-best top-4 recall | 0.9873 | `>0.6`，PASS |
| top-1 exact best rate | 0.9040 | diagnostic |
| learned B1 recovery | 0.4732 | oracle 0.5014（94.4%） |
| outside-recent4 best 的 top-4 recall | 0.9811 | diagnostic |
| STOP accuracy | 1.0000 | diagnostic |

因此冻结的 conjunctive gate 正式未过，不能直接进入 full conditional training。与旧 scalar in-sample
B1=`0.178` 相比，direct head 已经把 top decision 拟合到接近 oracle；失败集中在大量非最优 tail candidates
的完整顺序，而不是 best-event retrieval。为区分“全排序 loss 权重不足”与表示上限，允许一次在同一 train-only
probe 上、保持 Spearman 门槛不变的 rank-weight repair；该 repair 失败即终止 learned general-`B` 路线。

## Artifact

- [`summary.json`](summary.json)：30 epochs 完整 history；content SHA256=
  `0f1ac9647ed39d922c921bc6fccc5a3604c7e8eac002ff66e06698d731418667`，Git file SHA256=
  `229a1ec0bdc0a097995893330a465ac9033894a5454031cda76706c101d2d439`；
- Hyper00 run：`/data02/jaxan/runs/causalcache-direct-marginal-v3-fit-e6727d5`；
- source revision=`e6727d5`，elapsed=`239.6s`，container exit 0；
- checkpoints 当前只在 Hyper persistent path，状态=`PENDING_HF_UPLOAD`；fit-only weights 不作为正式模型。

合同：[`docs/set_utility_direct_marginal_v3.md`](../../../docs/set_utility_direct_marginal_v3.md)。
