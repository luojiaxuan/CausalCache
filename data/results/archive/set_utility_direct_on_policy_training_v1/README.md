# Direct on-policy per-epoch truth selection v1

状态：`COMPLETED_TRUE_RECOVERY_EARLY_STOP`。

两条模型线共享同一 merged input、contextual GUI-Owl cache、900 条 optimizer trajectories、9,287 个
optimizer states、84,441 个 candidate-complete groups，以及 trajectory-disjoint 的 256-state checkpoint
denominator。每个 epoch 保存 checkpoint，运行真实 at-most-`B` selector，并在补齐实际访问 coalition 的
policy restoration truth 后才继续训练。选模主指标为 trajectory-equal B1--B4 macro recovery；
`minimum_delta=0.005`、`patience=3` 在训练前冻结。未读取 evaluation/test。

## 结果

| 模型 | Epoch | Macro | Long+ | 相对当前 best | 决策 |
|---|---:|---:|---:|---:|---|
| Set Transformer | 1 | 0.38508 | 0.36343 | first | best=e1 |
| Set Transformer | 2 | **0.39491** | **0.38618** | +0.00983 | best=e2 |
| Set Transformer | 3 | 0.00610 | 0.00000 | -0.38881 | stale=1 |
| Set Transformer | 4 | 0.39529 | 0.36765 | +0.00038 | stale=2；低于 0.005 |
| Set Transformer | 5 | 0.19033 | 0.20821 | -0.20458 | stale=3；early stop |
| DeepSets | 1 | **0.39251** | **0.37929** | first | best=e1 |
| DeepSets | 2 | 0.00000 | 0.00000 | -0.39251 | stale=1 |
| DeepSets | 3 | 0.39581 | 0.36887 | +0.00330 | stale=2；低于 0.005 |
| DeepSets | 4 | 0.31853 | 0.33968 | -0.07398 | stale=3；early stop |

最终选择：

- Set Transformer epoch 2：`ed890219f5c765ad1c5098b691658f5bc1b5c4fd6b89114e944b5e2dd2ad3da1`；
- DeepSets epoch 1：`5f5e21db7af1cc7785ba99ca9e7a7ecabdecbafa24ff70e0657ffea97ea9cd3e`。

Set Transformer 的冻结主指标比 DeepSets 高 `0.00239`，但差距很小，尚不能只凭该点估计宣布部署赢家。
下一步必须在同一 256-state denominator 上补 recent、safe-fallback hybrid 与 selector latency；再决定进入
policy replay 的 Pareto-optimal checkpoint。

训练 loss 在后续 epoch 继续下降，而真实 recovery 出现崩塌或不足 `0.005` 的微小波动。因此本结果直接验证：
不能按 loss 或最后 epoch 选 checkpoint，per-epoch truth gate 与早停是必要的。

## Artifacts

- Set root：Hyper00 `/data02/jaxan/runs/causalcache-set-transformer-direct-on-policy-v2-a69c706`；
  final summary content SHA256=`de2027b5fe8108b278f3114a4920e9dd430c1a5934cd94cbf2b8ff9b8a09a41c`；
- DeepSets root：Hyper01 `/data02/jaxan/runs/causalcache-structured-deepsets-direct-on-policy-v4-843e360`；
  final summary content SHA256=`a54d3505cfd054f01cfa31a3a8293050c36789deaac19315920dc9b30f6b7b48`；
- epoch 1 union truth manifest content SHA256=`a6ba7c267052de75e3771915e8de1730c967f0202d632a73eb170d207e9efa36`；
- Set epoch 2/4 supplemental truth manifest content SHA256=
  `b7225a0091faacad366ffe7e6c38bf1f07ae68673c3facfc457767ed8cc0c7fe` /
  `d94fde97d7280bab2eef3e2a36933005356457e4ba30b090a96376eaeb276d0e`；
- DeepSets epoch 3/4 supplemental truth manifest content SHA256=
  `e61d14ef2570f4b206b0b8afd2e4a6509230c6ad791b7f57b072dd7ab315e172` /
  `5ec77d84c6c4deeaa8ad6db790ed9b02fa1777c1da8059834a837a285949b6a5`；
- training input content SHA256=`6c9243a2a2846180f717ea59e3692a3005ae153b8d206f7ad1ec4a8bf1a1bfad`；
- contextual cache content SHA256=`44405c2cf96f468e6d0f087e8249bdb504c7efd150673e9bda836d2f52597604`；
- heldout manifest content SHA256=`6d5d4767578ac12131e22b15bda88a2475a22d8d25778350b353f5d1e590c554`；
- checkpoints 与大型 truth payload 当前保存在上述 persistent roots，状态 `PENDING_HF_UPLOAD`。

两个 root 的 `summary.json`、`training-progress.json`、epoch rollout 和 sealed truth 是本轮 authoritative
selection record。历史 `recovery-checkpoints.json` 只登记 immutable checkpoint bytes，尚未回写 delayed
truth，不能用于读取最终 best；发布 HF 前将做确定性 metadata reconciliation，不改变模型或科学数字。

机器可读数字见 [`summary.json`](summary.json)。
