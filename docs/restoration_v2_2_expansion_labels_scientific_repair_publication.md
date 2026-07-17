# Restoration v2.2 repaired-label private-HF publication

> 当前状态：publication source 已冻结并完成本地 fake-HF recovery 验证；尚未执行真实 remote mutation，
> `formal_label_loader_eligible=false`、`gate_training_unlocked=false`。只有本协议的 completion seal 与 immutable
> fresh replay 同时闭合后，才解除 formal-58 的 label-data prerequisite。

## 目标与边界

该步骤只把已经 `VALID + REVALIDATED` 的 ledger-neutral scientific-repair child 作为 reusable dataset artifact
发布到新的 private Hugging Face identity。它不重算 GUI policy、不重新运行原 GPU producer、不修改原
invalid-forensic repo，也不重分类原 attempt：

- 原 expansion exact-label producer 永久保持 `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`；
- 原 invalid-forensic payload 永久保持 formal-loader ineligible；
- publication sidecar、claim 或 remote pair 本身不能解锁 gate；
- 只有本协议最后写入的 completion seal 才声明 repaired payload 可被 formal loader 消费；
- 该 completion 只清除 label-data blocker，不代表 gate 已训练，也不授权 matched-NLL、closed-loop 或 confirm。

冻结 config：

```text
code/configs/causalcache_restoration_v2_2_expansion_labels_scientific_repair_publication_v1.json
SHA256 547b291022e2430fe6af5158d9350132f405c4b6019198488386dd7fb50be0aa
```

## 已绑定的 producer output

publication 只接受以下 completed child bytes：

```text
archive:
  /data/artifacts/causalcache/restoration-v2-2-expansion-exact-labels-scientific-repair-v1.tar
  3,020,800 bytes
  SHA256 1a9fdcc08aeb83f88bcd50957c3d890e3b103f2063a2aecbe08610d450950e01
  tree SHA256 09bb681b2715997df326ab9648d8501955ec6ba127631a97141484631a98f44b

producer claim:
  mode 0600
  SHA256 f2a19d6f2b444eb000027abc8b2358e409be22713870bb7e84e62a82998d70f5

producer completion:
  mode 0600
  SHA256 6a4775357de392aef5d9ce7d9999db6cfd3a27f8948fd532e9341c7acacb4905

repair source commit:
  b14f489fe55b51a83917b57e6a54fb73d268342a

lightweight result commit:
  534bd7aecdb858decf4756b3934fe4f6acc989e7
```

archive 必须是 deterministic USTAR，并且只有 `audit.json`、`derived_labels.jsonl`、`manifest.json` 与
`raw_states.jsonl` 四个 member。publisher 对 archive bytes、size、member inventory、tree hash、manifest/audit
status、runner config、producer claim/completion 和 Git result summary 做联合 readback。该 readback 是
transport-level exact binding；scientific payload 已由 producer 的 formal run 与完整 revalidation 重算验证。

## Remote identity 与 exact-pair provenance

唯一 destination 是：

```text
private dataset: gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-repaired-mobile
tag: v2.2-expansion-exact-labels-scientific-repair-v1
archive: repaired/v2.2-expansion-exact-labels-scientific-repair-v1/repaired-labels-v1.tar
sidecar: repaired/v2.2-expansion-exact-labels-scientific-repair-v1/artifact-manifest-v1.json
```

只接受三个 crash-recovery state：

1. `EMPTY`：两个目标 path 都不存在；
2. `BYTE_IDENTICAL_UNTAGGED`：exact pair 已由同一个 commit 首次加入，但 tag 尚不存在；
3. `TAGGED_BYTE_IDENTICAL`：exact pair 与 annotated tag 均已存在且 byte-identical。

claim 在任何 remote mutation 前先持久化。repo 存在且目标 path 为空后，publisher 在 content commit/tag 前再
写入 mode-0600 remote-base receipt；它绑定 base main、base 的完整 reachable commit-id 集合、recursive
folder/file tree、每个 blob/tree identity、size、LFS/Xet identity 与 inventory hashes。receipt 一旦存在，后续
replay 不再调用 `create_repo`；没有 receipt 的 non-empty remote 不能被接管。

pair commit 的 reachable history 必须恰好是 base history 加一个 pair commit，不依赖 Hub 按日期排序的 commit
列表位置。pair tree 去掉 archive/sidecar 后，所有既有 file blob identity、size 与 LFS/Xet identity 必须和 base
完全相同；因此同时修改 `.gitattributes` 也会失败。partial pair、不同 bytes、split/interposed commits、目标
path 已在 base 中存在、pair commit 同时加入第三个 path、tag 指向其他 commit 或 lightweight tag 都 fail
closed。新 commit 使用 receipt 绑定的 base `main` 作为 `parent_commit` 做 CAS；archive 与 sidecar 只能在
同一个 frozen-title commit 中新增，不允许 overwrite。

Hugging Face refs API 对 annotated tag 返回 tag-object SHA，而 `dataset_info(tag).sha` 返回 resolved commit。
本协议分别记录并验证这两个 identity，要求二者不同，同时要求 resolved commit 等于 exact pair commit；不会再
复用旧 P1 parser 把 object SHA 与 commit SHA 强制相等的错误假设。

`create_repo`、`create_commit` 或 `create_tag` 丢失响应时，publisher 只允许重新查询上述状态；只有已存在的
byte-identical exact pair/tag 才可恢复，不能盲目重发 mutation。

## Local state 与 completion-last

publication state 使用独立路径：

```text
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-scientific-repair-publication-v1.claim.json
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-scientific-repair-publication-v1.remote-base.json
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-scientific-repair-publication-v1.completion.staged.json
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-scientific-repair-publication-v1.completion.json
```

四者都必须是 mode 0600 regular file。claim 在任何 remote mutation 前写入；remote-base receipt 在 content
mutation 前写入；两者都明确保存 `formal_label_loader_eligible=false` 与 `gate_training_unlocked=false`。完成
remote pair/tag 后，publisher 从新空目录 force-download immutable revision，前后重复验证：

- private repo；
- `main` resolved commit；
- annotated-tag object；
- tag-resolved commit；
- immutable commit；
- sealed base history/tree 与 exact-pair history/blob provenance；
- archive/sidecar bytes；
- repaired USTAR strict transport readback。

completion payload 先原子写入并保留在 deterministic staging path；final seal 由该 stage 做 no-replace
hard-link，二者必须是同一 device/inode。final hard-link 是首次成功执行的最后一个 local namespace mutation，
不会再 unlink staging；其后没有网络或 semantic validation。只有该 seal 同时绑定 remote-base receipt SHA、
immutable revision、tag object、tag-resolved commit、archive/sidecar hashes 与 fresh attestation 后，才写入
`formal_label_loader_eligible=true` 和 `gate_training_unlocked=true`。crash-after-stage 可以通过相同 inode
hard-link 恢复；已有 completion 的 replay 不调用 `create_repo`、`create_commit` 或 `create_tag`。若 remote 被
删除或漂移，只能 fail closed，不能重建。

## Git source 与执行命令

`prepare`/`publish` 都要求 absolute、non-symlink clean checkout，并验证：

- branch、local `HEAD`、`origin/main` 与实时 `git ls-remote origin refs/heads/main` 完全相同；
- publication source、repair source 和 lightweight result commit 具有冻结 ancestry；
- 六个 required source files 与 pushed commit byte-identical；
- result summary 与 runner config 在各自冻结 commit 上 byte-identical。

source-only validation 不读取 token、不访问 Hub：

```bash
PYTHONPATH=code python3 -m \
  scripts.manage_restoration_v2_2_expansion_labels_scientific_repair_publication \
  validate-contract \
  --repository-root /data/<clean-checkout>
```

clean pushed source 上的离线 prepare：

```bash
PYTHONPATH=code python3 -m \
  scripts.manage_restoration_v2_2_expansion_labels_scientific_repair_publication \
  prepare \
  --repository-root /data/<clean-checkout>
```

唯一允许 remote mutation 的命令：

```bash
PYTHONPATH=code python3 -m \
  scripts.manage_restoration_v2_2_expansion_labels_scientific_repair_publication \
  publish \
  --repository-root /data/<clean-checkout> \
  --hf-token-file /data/.secrets/hf_key.txt \
  --fresh-download-parent /data/tmp
```

token path 与 fresh parent 必须以 lexical absolute path 传入；CLI 不先解析 symlink，底层 `O_NOFOLLOW` 与
component checks 负责拒绝 symlink escape。publication 是 CPU/network-only 工作，不申请 GPU，也不重启任何
model runtime。

## Source-freeze 验证

- focused fake-HF recovery tests：24/24 PASS；
- scientific-repair wildcard：72/72 PASS；
- `py_compile` 与 `git diff --check`：PASS；
- Hyper00 `huggingface_hub==1.16.1` 的真实 API signature 已核对；
- 既有 private annotated tag 实测 `list_repo_refs().target_commit` 与 `dataset_info(tag).sha` 不同，证明本协议的
  object/resolved 双 identity 处理符合当前 Hub 行为；
- 本里程碑没有读取 repaired destination、没有创建 repo/commit/tag，也没有写 publication claim/completion。

下一步必须先 commit/push 本 source freeze，再从独立 clean checkout 执行 prepare 和 publish。真实 immutable
revision、tag object、claim/completion SHA 与幂等 replay 结果只能在 execution milestone 中回写。
