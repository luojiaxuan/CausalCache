# Fixed-tune Long+ conditional-greedy oracle v1

状态：`COMPLETED / HEADROOM_CONFIRMED_AUTHORIZE_DIRECT_MARGINAL_V3`。

在与 decision distillation v2 完全相同的 fixed-tune Long+ denominator 上，249/249 states、35
trajectories（241 long、8 very-long）完成四个依赖 wave，0 skip、0 reference drift，共 24,122 个
coalition labels。

| method | B1 | B2 | B3 | B4 | macro | very-long macro |
|---|---:|---:|---:|---:|---:|---:|
| oracle greedy | 0.5392 | 0.6762 | 0.7597 | 0.8046 | **0.6949** | 0.6186 |
| additive top-B | 0.5392 | 0.5511 | 0.5737 | 0.5466 | 0.5527 | 0.4381 |
| recent | 0.1605 | 0.3592 | 0.4372 | 0.3227 | 0.3199 | -0.2115 |
| random | 0.0454 | 0.0494 | 0.2607 | 0.2747 | 0.1576 | 0.1300 |

- `oracle_greedy - recent` macro=`+0.3750`，trajectory-clustered paired bootstrap 95% CI=
  `[0.2627,0.5656]`；同时满足 point `>0.10` 与 lower `>0.03`；
- machine verdict=`HEADROOM_CONFIRMED_AUTHORIZE_DIRECT_MARGINAL_V3`；
- `additive - recent=+0.2328 [0.1518,0.3586]`，但 B2--B4 明显低于 conditional greedy，再次证明
  singleton/additive student 不足以承载 general-`B`；
- true oracle B4=`0.8046`，而现有 scalar Set student Long+ 仅 `0.3885`：同 denominator 下仍有约
  `0.416` 的巨大蒸馏 gap；
- 该结果只授权一次 direct conditional-marginal + explicit STOP v3，不授权 evaluation、policy replay
  或 closed-loop。

## Artifact

- [`summary.json`](summary.json)：完整 reducer 输出；content SHA256=
  `852cc7d8fcf9ebf591c705d0803432419213ed0363b96138e9a6cd312c4c0e3a`，file SHA256=
  `33d6a83bfb7199b657198a562bf5ce54e6c9c93cb21a5024cfae8b6a8c26ae28`；
- Hyper00 active result：`/data02/jaxan/runs/causalcache-tune-long-oracle-v1-b6a4643/result.json`；
- 四 wave schedules/terminals：Hyper00/Hyper01
  `/data02/jaxan/runs/causalcache-tune-long-oracle-v1-b6a4643/wave-{1,2,3,4}`；
- source revision=`b6a4643818054c8a920d168df3079b679d4ce077`，config SHA256=
  `94d35bff6f6192a9c445138265f996c86a675e732ec83bcade2ad814d63d5313`；
- 完整 tune-truth payload 当前状态：`PENDING_HF_UPLOAD`；目标 private dataset repo=
  `gavinlaw/causalcache-set-utility-variable-history-mobile`。这些 tune labels 的
  `labels_reusable_for_training=false`，不得进入 v3 optimizer。

复现合同：[`docs/set_utility_tune_long_oracle_v1.md`](../../../docs/set_utility_tune_long_oracle_v1.md)。
