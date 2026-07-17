# Invalid-forensic publication v1 fail-closed attempt

唯一 P1 invocation 已完成 private repo、single-pair commit 与 tag creation，但在 completion seal 前因真实
Hugging Face annotated-tag ref interface 与冻结 parser 不一致而 fail closed。本记录不删除或覆盖 remote，
不把原 expansion exact-label v1 改写成 PASS，也不解锁 repair/gate。

## 已验证事实

- publication source：`main@3941d82a8ad83daaec55726a799d4d200735428d`；
- config SHA256：`affb54cf44a656394c983cebe5d004d3e1ff216537c43da942dc031fa6baaccb`；
- claim：mode `0600`、1,383 bytes、SHA256
  `0a07692b7f29fb0e9f6a9da23925135c8911f45d4f10f2edd2d277ceed0da5de`；
- archive SHA256：`8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489`；
- sidecar SHA256：`c67914d05ce5e93518c710cd99132548f4bf5b51250e506c996a9450a369b561`；
- pair commit：`5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416`；direct predecessor
  `1e1bb828d196779e4fc855890922c4865fcd6458` 只有 `.gitattributes`；
- frozen tag 的 `dataset_info(tag).sha` 解析为 pair commit，但 `list_repo_refs().tags[].target_commit`
  返回 `ca652858c3d59eab066eb7b31399690a65ea5f44`；后者是本次冻结 parser 未建模的 annotated-tag object identity；
- pair commit 的 archive/sidecar immutable readback 与 P0 strict reader 已通过；最终 tag/immutable pre-post fresh
  attestation 未完成，completion seal 不存在。

## 结论与下一步

这是 publication validation-interface failure，不是 archive byte mismatch，也不是第二次 data generation。
同一 P1 source 不重跑；remote pair commit 与 tag 原样保留。下一步冻结独立、read-only、versioned
tag-resolution repair：同时绑定 tag object identity 与 tag-resolved commit，重新做 force-download、P0 strict
readback 和 pre/post stability，再写新的 child completion。该 child 完成前，P0 archive 仍不能作为 formal labels。

机器可读记录见 [`summary.json`](summary.json)。
