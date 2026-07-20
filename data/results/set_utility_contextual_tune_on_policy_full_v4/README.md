# Full-data contextual tune on-policy truth v4

状态：`COMPLETED / FULL_DATA_NO_GO`。

1,063/1,063 tune states、0 skip，真实 `D(S)` 覆盖 10,368 个去重 coalitions。结果按 100 条
trajectory 等权聚合：

| Method | B1 | B2 | B3 | B4 | B1--B4 macro | Long+ | Total p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSets full | 0.1566 | 0.3842 | 0.5308 | 0.6272 | 0.4247 | **0.3933** | 10.65 ms |
| Set Transformer full | 0.1822 | 0.4120 | **0.5569** | **0.6484** | **0.4499** | 0.3542 | 19.53 ms |
| Recent | **0.1911** | **0.4361** | 0.5543 | 0.6263 | 0.4519 | 0.4008 | -- |

Set Transformer 是 learned winner，并把 25% checkpoint 的 macro recovery 从 `0.4365` 提高到 `0.4499`；
但仍比 recent 低 `0.00207`，trajectory bootstrap 95% CI=`[-0.01339, 0.00964]`。DeepSets 显著低于
recent：差值 `-0.02722`，95% CI=`[-0.05459, -0.00446]`。

因此增加同分布训练数据确实缩小了 Set Transformer distillation gap，且 B3/B4 已超过 recent；但 B1/B2 与
long-history 泛化仍不足，当前不授权 untouched evaluation、policy replay 或 closed-loop。下一步是只用 train
split 生成 predictor-search 实际访问的 on-policy/conditional-marginal coalitions，再重训并用固定 tune truth
判定。

Source of Truth：

- full result：Hyper00
  `/data02/jaxan/runs/causalcache-contextual-tune-on-policy-evaluation-full-v4-e97f2d4/result.json`；
- result content SHA256：`fb6de0dc0b29fc9ffedb801205ee3f50ed4d116de5b5e272430226fecbc05d79`；
- result file SHA256：`46e871aff7d817ff7cb817a81d295774d00d2d80307cae1b7a737371fb557bb4`；
- raw labels：Hyper00
  `/data02/jaxan/runs/causalcache-contextual-tune-on-policy-labels-full-v4-e97f2d4`，39MB；
- HF 状态：`PENDING_HF_UPLOAD`，intended dataset repo=
  `gavinlaw/causalcache-set-utility-variable-history-mobile`。
