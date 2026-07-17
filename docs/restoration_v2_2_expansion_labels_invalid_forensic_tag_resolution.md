# Restoration v2.2 invalid-forensic tag-resolution child

> 状态：source-only frozen，尚未执行真实 remote validation。本协议只修复 P1 对 annotated-tag 接口的建模，
> 不修改 Hub remote、不补写原 P1 completion，也不解锁 formal labels。

## 为什么需要独立 child

P1 source `3941d82a8ad83daaec55726a799d4d200735428d` 已经创建正确的 private single-pair
commit 与 annotated tag，但冻结 parser 把 refs API 的 annotated-tag object identity 当成 tag-resolved commit，
因此在原 completion 前 fail closed。failure evidence 固定在
`dcc37e5ce62b844a1814a9f0fe29ec4827f2547a`。

child config 为
`code/configs/causalcache_restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution_v1.json`，SHA256：
`f257dcdebb218533a080739e8a5067cc44fe35fd533ad57808db5ce789cfd956`。

它把两种 identity 分开冻结：

- annotated-tag object：`ca652858c3d59eab066eb7b31399690a65ea5f44`；
- `dataset_info(tag).sha` resolved commit：`5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416`。

二者不同是预期事实，不再要求相等。`main` 与 immutable pair revision 必须分别解析为同一个 pair commit。

## 冻结 transport 与 provenance

- private dataset：`gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-invalid-attempts-mobile`；
- tag：`v2.2-expansion-exact-labels-v1-attempt-1-forensic-v1`；
- pair commit：`5efe1ae861d16e2ee144ed5f4c7b5ad25a28b416`；
- direct predecessor：`1e1bb828d196779e4fc855890922c4865fcd6458`；
- pair title：`Publish invalid expansion exact-label forensic archive`；
- predecessor files 必须精确为 `{.gitattributes}`；
- pair files 必须精确为 predecessor 加 archive、sidecar 两个冻结 path；
- archive SHA256：`8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489`；
- sidecar SHA256：`c67914d05ce5e93518c710cd99132548f4bf5b51250e506c996a9450a369b561`。

sidecar 的 `publication_source` 仍绑定原 P1 source，而不是 child source：原 P1 config SHA256 为
`affb54cf44a656394c983cebe5d004d3e1ff216537c43da942dc031fa6baaccb`，五文件 source inventory
SHA256 为 `d4ec445884bc2296c824e5f90718a1ede0be13023d5d969f77adf379a1c29cfa`。child 自己的 clean
pushed-main source identity 单独写入 child claim/completion。

## 严格只读 remote validation

实现只调用四个只读 `HfApi` 方法：`dataset_info`、`list_repo_refs`、`list_repo_commits`、
`list_repo_files`；唯一额外 remote read 是 `hf_hub_download(force_download=True)`。模块和 CLI 不暴露
`create_repo`、`create_commit`、`create_tag`、delete 或 move 路径，记录的 remote mutation count 必须为 0。

validation 窗口前后各读取并 exact compare：

```text
(main resolved commit,
 annotated tag-object identity,
 tag-resolved commit,
 immutable revision resolved commit)
```

pair title、direct history、pair/predecessor exact file inventories 也在下载前后复验。archive 与 sidecar
使用 pair commit 这一 immutable revision force-download 到新建的 absolute、初始空、non-symlink directory；
returned path 必须 lexical exact，所有中间 component 必须经 `lstat` 验证。archive 随后重新通过 P0 strict USTAR
reader，并核对 406 members 与 tree inventory
`bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1`。

## Parent 与 child local state

在任何 remote read 前，child 必须验证原 P1 claim：mode `0600`、size 1,383、SHA256
`0a07692b7f29fb0e9f6a9da23925135c8911f45d4f10f2edd2d277ceed0da5de`，并严格解析其 destination、
archive/sidecar 与原 P1 source binding。原 P1 completion 必须在 remote replay 前后都不存在；child 从不创建、
修改或删除它。

child 使用全新的 state paths：

```text
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-tag-resolution-v1.claim.json
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-tag-resolution-v1.completion.json
```

claim 在第一个 remote read 前写入；completion 只在全部 replay 与 pre/post stability 成功后写入。两者都用 mode
`0600` 的 `O_CREAT|O_EXCL|O_NOFOLLOW` staging、`fsync` 和 hard-link no-replace 原子发布；重跑只接受 byte-identical
state。read/response error、identity drift、byte mismatch、split pair、unexpected 原 P1 completion 或 orphan child
completion 都只留下 claim，不会 seal。

## 命令

离线 contract validation 不导入 HF transport、不读 token：

```bash
cd /absolute/path/to/CausalCache/code
PYTHONPATH=. python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution \
  validate-contract \
  --repository-root /absolute/path/to/CausalCache
```

正式只读 replay 只有 `validate-remote` command 可以读取显式 token。token path 必须 absolute、regular、
non-symlink，mode 只能为 `0400` 或 `0600`：

```bash
PYTHONPATH=. python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic_tag_resolution \
  validate-remote \
  --repository-root /absolute/path/to/CausalCache \
  --fresh-download-parent /data/tmp \
  --hf-token-file /data/.secrets/hf_key.txt
```

正式执行前必须先 push child source 到 canonical `main`，并在 clean checkout 中满足
`HEAD == origin/main`、canonical origin URL、failure/parent commits 为 ancestor、required source files 与 committed
bytes 一致。当前 source-only 工作没有触网，也没有读取 token。

## Claim 边界

child completion 只证明冻结 private transport 的 tag resolution、single-pair provenance、byte identity 与 P0 strict
readback。`formal_label_loader_eligible=false`、`gate_training_unlocked=false`、原 producer attempt 仍永久为
`INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`。CPU scientific repair 必须继续使用独立协议与新 artifact identity。
