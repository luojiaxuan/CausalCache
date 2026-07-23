# Restoration v2.2 exact-label v1 attempt

本目录只保存第一次 formal label attempt 的轻量 failure binding。该 attempt 已被 fail-closed ledger 永久封存，
不得删除、resume、retry 或在同一 identity 下 top-up。

## 结论

`INVALID_RESTORATION_V2_2_EAGER_LABELS_V1`

formal source 与 immutable parent/derived inputs 已通过 authorization，global ledger 随后 durable claim。两个 worker
在任何 state marker、teacher forward 或 KL measurement 之前发现指定的 GUI-Owl snapshot directory 不存在，并同时
退出。global terminal 记录：

- attempted / completed states：`0 / 0`；
- restoration teacher forward / KL / raw `D(S)` rows：`0 / 0 / 0`；
- exact-subset oracle / conditional-marginal labels：`0 / 0`；
- retry / resume / top-up：全部禁止；
- confirm / gate / matched-NLL / closed-loop：均未运行。

根因不是 scientific estimand 失败，而是 execution preflight 缺陷：旧 runner 只在 durable claim 前确认
`--model-dir` 是 absolute path，真正的 snapshot existence 检查发生在 worker runtime load。后续 attempt 必须使用新
identity，并把 snapshot existence/revision/inventory 校验前移到 global claim 之前。

## 运行与证据绑定

- source：`main@3942d687d03bf63ea683fe8ad906a161eb10dc27`；
- contract SHA256：`56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87`；
- run-contract SHA256：`4dd470cbde3184decb0f38133dc9347b528304f44b13762c881540aefa4ccebf`；
- Hyper00 / container：`node-radixark-16-0000` /
  `39749a3bd0875f3c15216211f875220a4934fe62313487b537c1116e82040ff0`；
- image digest：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- execution window：`2026-07-16T16:39:43.874890Z`--`2026-07-16T16:40:14.012087Z`；
- global ledger SHA256：`36afef01861304647290bbac5f7ef9d662c8a92e42aabcd02c54835a41233af6`；
- even / odd sibling ledger SHA256：`73c6b6ec505349cec922c38020b89f177670c0a41b5bb7bcc7deed36e280201d` /
  `12041df247d302505e6a5e2615ea3fcc5cafea3561b704e7b2073084a4ddb20b`；
- run manifest SHA256：`0b9a4a826364b8d4a4260b31ba24a0b21bba8c9771e39acfdb3e09f47fe70a66`；
- even / odd terminal SHA256：`baf06414e766571991d99fe471eb8f2aacb9408dfbf6839e5709d34586ae0603` /
  `499df2158da84afb3092d11fd6042161578f6440c42191bd4fc0978ddcbd77e9`。

没有 raw labels 可作为 reusable dataset，因此没有创建或上传 HF label artifact。canonical invalid root 与 ledgers
保留在 Hyper00 `/data/experiments/causalcache/` 作为不可变本地 forensic evidence；Git 中的
[`summary.json`](summary.json) 是团队交接用 compact binding。
