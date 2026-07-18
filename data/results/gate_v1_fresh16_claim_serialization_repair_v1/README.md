# Gate v1 fresh-16 primary result

这是 frozen formal-58 ensemble 在独立 fresh-16 slice 上的首个完整、可重放 primary result。执行与
immutable validation 都成功，因此这是有效的科学 `NO-GO`，不是实现失败。

## 结论

- `go_selector=false`：conditional ensemble 相比最强 heuristic 有正增益，90% paired-bootstrap lower bound
  为 `0.0490`，但未达到预注册的 oracle proximity、跨 seed 稳定性与 12/16 trajectory support；
- `go_set_conditioning_primary=false`：conditional 相比 parameter-matched independent 的 mean normalized
  delta 为 `-0.1224`，仅 1/16 trajectory、1/5 paired seed 为正；fresh-16 不支持“set conditioning 优于
  independent gate”这一主张；
- conditional raw utility / exact raw 为 `0.9431`，说明 selector 仍捕获了明显 restoration signal；但
  normalized recovery / exact 只有 `0.6942`，5 个单 seed 均未达到 `0.75`，不能按 frozen gate 宣布 GO；
- 当前不打开 confirm-20、matched-NLL 或 closed-loop。先冻结结果并设计不复用 fresh-16 调参的下一阶段。

完整机器可读指标见 [`summary.json`](summary.json)。完整 13-target artifact 位于 private Hugging Face dataset：

- repo：[`gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile)
- tag：`gate-v1-fresh16-claim-serialization-repair-v1`
- payload commit：`9f0c61b9437773ca5d3f7e0cabd6e2a987e3c908`
- report commit：`3541fe1ea2c46e555c29cc53483e6f3b809f8f81`
- annotated tag object：`34d5928db4e82532f46f7596702aec3f018c4339`

tag 已验证指向 report commit；独立 immutable replay 的 remote mutation count 为 0。
