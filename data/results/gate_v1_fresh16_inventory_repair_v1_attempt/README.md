# Gate v1 fresh-16 inventory-repair v1 首次执行

本目录封存 2026-07-18 在 Hyper00 上进行的唯一 inventory-repair v1 formal attempt。该执行已完成
fresh-16 label-blind semantic materialization、10-checkpoint replay 与双 H200 policy-vision workers，但在
持久化 `label-access-claim` 之前因 Python `mappingproxy` 序列化错误退出。它没有读取 fresh labels，也没有生成
report 或修改 Hugging Face，因此不能产生 GO/NO-GO 结论，且旧 namespace 不能续跑。

- Source-A：`6fb3e868e293bce191ce30a5c6a15ecc544c4591`；
- Execution-B：`c22734ffc85935882f57ddb081c9194d6dae92d0`；
- config SHA256：`5ba1b2d433c01defa0faae92a163dc9a9a916d967218abdc7607bd3724829a44`；
- formal start：`2026-07-18T04:27:04Z`；exit code `1`；
- failure：`pretty_json_bytes(claim.claim)` 收到不可 JSON 序列化的 `mappingproxy`，发生在
  `heuristic_local_seal` 已完成、`label-access-claim` 尚未写入的边界；
- 已完成 16 trajectory 与 80 OCR semantic decode、80 selected images、48 feature states、144 candidates、
  10 checkpoint loads、2 policy workers 和 97 vision forwards；
- fresh label semantic decode、label access claim、primary report、HF mutation、旧 dev-5、confirm、matched-NLL
  与 closed-loop 均为 `0`；
- ordered state receipts 精确停在 ordinal `0..7`；独立 `heuristic-local-seal.json` 存在，而
  `label-access-claim.json` 不存在；
- artifact tree 保留 88 files / 51,508,707 bytes，顶层仅 `fresh16-eval` 与 `selected-images`。失败执行器冻结/
  报告的 canonical inventory digest 为
  `5443db6df16234c29b32501bc9f766670d428141fb02c4a665be936e1ca2587c`；它对按 relative POSIX path 排序的
  `{path, mode=stat.S_IMODE, size_bytes, sha256}` records 做 compact/sort-keys UTF-8 JSON 后计算；
- 原 v1 与 inventory-repair v1 的 planned private HF repositories 均不存在，remote mutation 为 `0`。

完整 receipt、seal、artifact、log/start/exit 与 Docker receipt 的 mode、size 和逐字节 SHA256 见
[`summary.json`](summary.json)。原始 mode-0600 evidence 保留在 Hyper00 `/data02/jaxan`，不删除、不覆盖、不复制进
Git，也不上传到 planned result repo。

下一步必须另立 versioned claim-serialization repair，修复 claim 的 plain-JSON 持久化接口，并使用新的
state/artifact/runtime-receipt/HF identities。新 repair 在授权前必须先验证本目录绑定的旧 evidence 与 successor
absence；不得删除旧 state 后原 namespace 续跑。
