# Restoration v2.2 invalid-forensic private-HF publication P1

> 当前状态：P1 source-only contract、实现和 fake-HF recovery tests 已冻结；本步骤没有读取 token、没有访问
> Hugging Face、没有上传。P0 local archive 仍是 staging，原 producer attempt 仍永久为
> `INVALID_EXPANSION_EXACT_LABEL_ATTEMPT`。

## 固定输入与目的地

P1 只传输 P0 已严格验证的 deterministic USTAR，不生成或修复 labels：

- P0 archive SHA256：`8e205d73196604c0d8d342f9415755b090b96e83555ca55c386b12b8427ca489`；
- size：3,880,960 bytes；
- members：406；
- tree inventory SHA256：`bc481dd77e26764dad3458e91a3e0247ade6049af7adb1744672aa6f2fb437d1`；
- P0 config SHA256：`5290a51a250e31be4fcf0a970c77ef31c08c92c5892edd19a12ecb14d9d5a6a2`。

P1 frozen config：
`code/configs/causalcache_restoration_v2_2_expansion_labels_invalid_forensic_publication_v1.json`，
SHA256 为 `affb54cf44a656394c983cebe5d004d3e1ff216537c43da942dc031fa6baaccb`。

private-HF destination 固定为：

```text
repo: gavinlaw/causalcache-restoration-v2-2-expansion-exact-labels-invalid-attempts-mobile
tag: v2.2-expansion-exact-labels-v1-attempt-1-forensic-v1
archive: attempts/restoration-v2-2-expansion-exact-labels-v1/attempt-1/raw-evidence-v1.tar
sidecar: attempts/restoration-v2-2-expansion-exact-labels-v1/attempt-1/archive-manifest-v1.json
```

archive 与 sidecar 必须在同一个 commit 中创建。任何已有 remote path 或 tag 都不能被覆盖。
pair commit title 也固定为 `Publish invalid expansion exact-label forensic archive`。

## Clean materialized origin/main source binding

真正执行 publication 前，CLI 强制验证：

- 当前 branch 为 `main`；
- `HEAD == origin/main`；
- origin URL 精确匹配 canonical GitHub repo；
- worktree 包括 untracked files 在内完全 clean；
- P0/P1 module、P0/P1 config 和 P1 CLI 五个 source files 的本地 bytes 与 `git show HEAD:<path>` 相同。

validator 本身不对 GitHub 做 live `ls-remote`，不能单独证明 remote-tracking ref 没有被本地伪造。formal checkout
必须来自 canonical push 完成后的 bundle/fetch：执行前先在联网 Mac 用 `git ls-remote origin main` 核对 exact
commit，再物化同时包含 `main` 和 `refs/remotes/origin/main` 的 checkout，并把 receipt 写入 execution record。在此
操作前提下，sidecar 绑定该 P1 source commit、五个 source files 的 canonical inventory SHA256、P0 config、archive
SHA/size、member count 与 tree inventory。由此 remote bytes 不只绑定数据，也绑定实际执行 publication 的代码版本。

## Crash-recoverable remote state machine

publisher 只接受三种 remote state：

| 状态 | remote 事实 | 允许动作 |
| --- | --- | --- |
| `EMPTY` | 两个目标 path 均不存在，tag 不存在 | 以 observed `main` 为 parent CAS，一次 commit 两个文件，再建 tag |
| `BYTE_IDENTICAL_UNTAGGED` | 同一 pair commit 首次加入两个文件、bytes 相同，tag 不存在 | 不重复 commit，只建 tag |
| `TAGGED_BYTE_IDENTICAL` | tag 指向该 exact pair commit，两个文件 bytes 相同 | 不做 remote mutation，只复验并 seal |

以下情况全部 fail closed：只存在 archive 或 sidecar、任一 byte 不同、tag 指向冲突内容、dataset 不是 private、
tag target 与 tag-resolved immutable SHA 不同、下载路径逃逸、非 regular/symlink 文件、P0 strict USTAR readback
失败。

每次接受已有 pair 前，还会调用 `list_repo_commits(revision=immutable)`：history head 必须就是 immutable revision，
title 必须等于 frozen title，direct predecessor 必须同时缺少两个目标 path，而 immutable snapshot 必须同时包含两者。
因此“先提交 tar、后提交 sidecar，但最终 snapshot bytes 一样”仍会被拒绝；tag 也只能指向这个 exact pair commit。

这里的 `EMPTY` 不是 0-commit repository。新 Hugging Face repo 的 system `.gitattributes` initial commit 仍被保留；
`EMPTY` 只表示两个冻结目标 path 与 tag 都缺失。publisher 读取该 initial/main immutable SHA，并把它作为
`parent_commit`，因此第一次 pair commit 也走 compare-and-swap。

如果 `create_commit` 已在 server 生效但 response 丢失，本次运行可以失败；下一次运行会识别
`BYTE_IDENTICAL_UNTAGGED`，不会产生第二个 data commit。如果 tag request 的 response 丢失，当前运行立即重新查询
tag：只有 tag 已精确指向预期 immutable commit 才继续，否则保留错误并失败。

## Local claim、completion 与 fresh replay

运行在任何 network call 前，先以 `O_CREAT|O_EXCL|O_NOFOLLOW`、mode `0600` 写同目录 staging file，`fsync`
后用 hard-link no-replace 原子发布 deterministic claim：

```text
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-publication-v1.claim.json
```

这样进程在写 claim 中途崩溃时不会暴露半写 final file。重跑只允许复用逐 byte 相同且 mode 仍为 `0600` 的
claim。remote 闭合后，从 tag 和 immutable revision
分别在下载前后重新解析 SHA，并把 archive/sidecar `force_download` 到新建的初始空目录。下载 archive 必须再次通过
P0 strict reader；returned path 必须 lexical 等于目标 path，且 fresh root 到目标的每一级 path component 都经过
`lstat` 验证为非 symlink directory/final regular file。tag 与 immutable revision 在整个下载窗口保持不变后，才以
相同 O_EXCL-staging + hard-link 规则写 completion seal：

```text
/data/experiments/causalcache/.restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-publication-v1.completion.json
```

completion 绑定 claim SHA、archive/sidecar SHA、immutable revision、进入 publisher 时的 remote state 和 fresh replay
attestation。已有 completion 的重跑仍重新读取 remote 并做 fresh replay，但不会再次 create repo、commit 或 tag；若
remote 已删除则只读失败，不会偷偷重建空 repo。

claim/completion parent 必须是预先创建、只有执行账户可写的 trusted local state root。实现对 final parent、staging
和 final file 做 non-symlink/O_EXCL 检查，但不把祖先目录 race 建模成对抗性 shared-filesystem threat；formal run
固定使用 `/data/experiments/causalcache`，不允许把 state path 改到不可信目录。

`--fresh-download-parent` 同样必须是已存在的 absolute non-symlink directory；relative path 会在 claim 和任何
network call 前失败。若 fresh download 窗口内 tag 被移动到其他 commit，pre/post revision check 会失败且不会写
completion。

## 离线验证与正式命令形状

本地只验证 frozen contracts，不导入 HF transport，也不访问网络：

```bash
cd /path/to/CausalCache/code

python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic_publication \
  validate-contract \
  --repository-root ..
```

P1 source commit 已推到 `origin/main` 且 worktree clean 后，可离线构建并验证 sidecar binding：

```bash
python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic_publication \
  prepare \
  --repository-root .. \
  --archive /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-v1.tar
```

只有显式 `publish` 子命令才惰性导入 `huggingface_hub`、读取显式指定的 token file 并允许 remote I/O。token
不会写入 log、sidecar、claim 或 completion；CLI 以同一个显式 token 构造 `HfApi(token=token)` 和
`partial(hf_hub_download, token=token)`，不依赖 ambient identity。token file 必须是 non-symlink regular file，
mode 只能是 `0400` 或 `0600`，并在 fd read 前后保持 inode/size/time/mode 稳定。本轮 source/test 工作没有读取
真实 token：

```bash
python3 -m scripts.manage_restoration_v2_2_expansion_labels_invalid_forensic_publication \
  publish \
  --repository-root .. \
  --archive /data/experiments/causalcache/restoration-v2-2-expansion-exact-labels-v1-invalid-forensic-v1.tar \
  --fresh-download-parent /data/tmp \
  --hf-token-file /data/.secrets/hf_key.txt
```

正式 Hyper 执行只使用预先复制为 mode `0600` 的 `/data/.secrets/hf_key.txt`；Mac 上的原始 key file
不直接传给 publisher，也不得写入 Git、log、sidecar、claim 或 completion。

fake-HF 回归测试不触网：

```bash
cd /path/to/CausalCache/code
PYTHONPATH=. python3 -m unittest -v \
  tests.test_restoration_v2_2_expansion_labels_invalid_forensic_publication
```

## Claim 边界

P1 即使未来成功发布，也只把 invalid-attempt forensic bytes 迁移到可复用的 private immutable storage。sidecar、
claim 与 completion 中的 `formal_label_loader_eligible` 始终为 `false`；本协议不解锁 gate training、matched-NLL、
closed-loop，也不把 producer attempt 改写成 PASS。后续 CPU-only validation repair 必须是独立 child protocol，并产生
新的 artifact identity。
