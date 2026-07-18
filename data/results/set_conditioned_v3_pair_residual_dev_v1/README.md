# Set-conditioned v3 pair-residual development result

状态：`NO_GO_SET_CONDITIONED_V3_PAIR_RESIDUAL_DEVELOPMENT_V1`。

这是 non-mainline、consumed-development 结果。fresh16 已被消费，不能再作为 holdout；confirm20、matched-NLL、
closed-loop、policy forward 与 GPU operation 均为 0。本结果不改变另一个 worktree 中的
restoration-guided independent gate 论文主线。

## 结论

- Formal58 的 oracle interaction headroom 明确存在：exact 相对 oracle additive 的 mean normalized gain 为
  `+0.122118`，trajectory bootstrap 90% lower=`+0.069350`，53/58 trajectory 为正；
- 但 learned residual 没有稳定蒸馏该 headroom：Formal58 OOF 上 safe residual 只在 1/58 state 接受 pair
  candidate，并略低于 additive；
- fresh16 上 safe residual 与 additive 在 48/48 state 完全相同，`used_pair_candidate=0`，所有冻结继续条件均失败；
- unguarded residual 只改变 3/48 state：1 个改善、2 个恶化，mean normalized delta=`-0.285271`。其中一个
  baseline distance 只有 `0.000323` 的 state 放大了 normalized harm，但该 state 未被过滤；
- 因此 v3 按冻结规则停止，不调 4/5 guard、不打开 confirm20。

## Source of Truth

- Machine-readable summary：[`summary.json`](summary.json)
- Git Source-A：`785b542a9d538bbfc80de95d5f8d956a04234e48`
- Git Execution-B：`e9262338ee96f404ca37babaa2f29a24f6c6cb17`
- Private HF model：[immutable revision](https://huggingface.co/gavinlaw/causalcache-set-conditioned-v3-pair-residual-exploration-mobile/tree/79b53aaa6017458d92e626bc4801d3b3bef9facd)
- Private HF development artifacts：[immutable revision](https://huggingface.co/datasets/gavinlaw/causalcache-set-conditioned-v3-pair-residual-development-mobile/tree/bcf7c7c8057f60b655736b5c63ec43551b3e3121)
- Immutable tag：`set-conditioned-v3-pair-residual-exploration-v1`

两个 HF repo 都已逐 manifest fresh-download，并复验所有 payload 的 SHA-256 与 size。evaluate 与 label-free
validate 的 stdout byte-identical，SHA256 均为
`e2e1367d710b052670e0281de756cb5ecbb99c021b68afa6d10cd9b1df6219b5`。
