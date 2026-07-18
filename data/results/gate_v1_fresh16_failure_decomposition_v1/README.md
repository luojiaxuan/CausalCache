# Gate v1 fresh-16 failure decomposition：NO_V2_CONDITIONAL_RESCUE

这是 frozen fresh-16 primary artifact 上的正式 reporting-only failure decomposition。唯一 formal `run`
与独立只读 `validate` 分别返回 `COMPLETED_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1` 和
`REVALIDATED_GATE_V1_FRESH16_FAILURE_DECOMPOSITION_V1`。结果是有效的诊断性
`NO_V2_CONDITIONAL_RESCUE`，不是实现失败；parent v1 的 selector / set-conditioning `NO-GO` 不被重分类。

## 路由结论

| Gate | 正式结果 | 判断 |
| --- | ---: | --- |
| `G/E` normalized | `0.956636` | 通过 `>=0.95` |
| `G/E` raw | `0.975350` | 通过 `>=0.95` |
| `n=3` / `n=4` `G/E` raw | `0.936503 / 0.988067` | 均通过 `>=0.90` |
| `G-J` normalized delta | `+0.081684` | 通过 `>=0.02` |
| `G-J` 90% bootstrap lower | `+0.002816` | 通过 `>0` |
| `G-J` positive trajectories | `10/16` | **失败**，要求 `>=12/16` |
| `C/G` normalized / raw | `0.725699 / 0.966926` | normalized 触发 material student gap |

因此 `search_pass=true`、`oracle_set_headroom_pass=false`、`material_student_gap=true`，冻结 routing 在
`oracle_set_headroom_pass` 停止。true conditional greedy 已经较好逼近 exact subset，主要剩余误差位于 learned
student；但 oracle set-conditioning headroom 只有 `10/16` trajectory support，未达到预先冻结的稳健性门槛，
不能据此再开一次 conditional v2 rescue。

诊断进一步显示：small-denominator effect 满足 frozen `denominator_dominated` 条件，但它只能解释 normalized
metric 的敏感性，不能删除样本或翻转 v1；positive distillation regret 中 completion/stop share 为
`0.780398`，高于 `0.60` dominance 边界；seed instability 分类为 `systemic`。完整 aggregate、round、seed、
denominator 与 per-`n` 数值见 [`summary.json`](summary.json)。

## Source of Truth

48 条 state diagnostics 和完整 16-trajectory rows 不进入 Git。canonical exact-three 位于 private Hugging Face
dataset [`gavinlaw/causalcache-gate-v1-fresh16-failure-decomposition-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-gate-v1-fresh16-failure-decomposition-mobile)：

- tag：`gate-v1-fresh16-failure-decomposition-v1`；
- exact-three commit：`9aa2540088c575f5c34dfb208e4f429fa53d355b`；
- annotated tag object：`4914c5add7ea27b89b185e240ea2670abe116473`；
- tag-resolved commit：`9aa2540088c575f5c34dfb208e4f429fa53d355b`；
- `bundle-manifest-v1.json`：3,796 bytes，SHA256
  `54fe7b7c4e7b7fc9840aefb3a59de7d77f9d9de061aa759f40abd128bd4de41b`；
- `failure-decomposition-report-v1.json`：63,609 bytes，SHA256
  `7a9d30015ea2476efd5fc03ab8fdc9ef6f59f3da82d700ec252de0bccd081feb`；
- `state-decomposition-v1.jsonl`：173,701 bytes / 48 rows，SHA256
  `0e048b85fdd84c94b7cab2a3ad78d948b85c07e26d804bb8a7781ba3a8e84069`。

tag 已验证解析到 exact-three commit；formal `run` 的 remote mutation count 为 `3`，独立 `validate` 为
`0`。`validate` 从 immutable identity 重新下载并重算同一 exact-three，local write count 也为 `0`。

## Execution provenance

- Source-A：`718a08b3d1ab78d5c2770e870cefabb2a8667f08`；
- direct-child Execution-B：`21b775020318f40941bdaa1d9fc9c8f99b1e5c40`；
- config SHA256：`fa2cd3759150f838fce78b72a987d7a889ef23f5eec5ec41286ae69090104f1f`；
- runner-freeze SHA256：`cc43d1cf609e5851543e5a0c3e256d9a6e82441d0bc9a55f1a5a1de8c0009a0c`；
- host / container：`hyper00` / `node-radixark-16-0000` /
  `sglang-omni-jaxan-07181414`；container id
  `ed83f5a9ff38b67f72974a63ee11851a33dffff8c0e1319f204ac77b73a5dc5c`；
- image digest：`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- CPU-only、unprivileged `runc`；Docker `DeviceRequests=null`、device list 为空、GPU count `0`；
- CPython `3.12.3`、`huggingface_hub 1.16.1`；bootstrap seed `271828`；
- run bracket：`2026-07-18T06:16:45.690306096Z` -- `2026-07-18T06:16:50.021356852Z`；
- validate bracket：`2026-07-18T06:16:58.528456548Z` -- `2026-07-18T06:17:01.283488835Z`。

实际应用命令在 container 内、working directory
`/data/CausalCache-fresh16-failure-decomposition-21b7750-clean` 执行：

```bash
PYTHONPATH=code python3 -m scripts.manage_gate_v1_fresh16_failure_decomposition run \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_failure_decomposition_v1.json \
  --execution-b-git-commit 21b775020318f40941bdaa1d9fc9c8f99b1e5c40 \
  --data-root /data \
  --fresh-download-parent /data/tmp/fresh16-failure-decomposition-v1 \
  --hf-token-file /data/.secrets/hf_key.txt

PYTHONPATH=code python3 -m scripts.manage_gate_v1_fresh16_failure_decomposition validate \
  --repository-root . \
  --contract code/configs/causalcache_gate_v1_fresh16_failure_decomposition_v1.json \
  --execution-b-git-commit 21b775020318f40941bdaa1d9fc9c8f99b1e5c40 \
  --data-root /data \
  --fresh-download-parent /data/tmp/fresh16-failure-decomposition-v1 \
  --hf-token-file /data/.secrets/hf_key.txt
```

Local completion 是 7 条 ordinal `0..6` durable state records，加同 bytes、同 inode 的 staging/final hard-link
pair；总计 9 个 filesystem entries，不是 9 条独立 records。completion SHA256 为
`5284a915e4ead3fd2daa4f416806ff74dd829dbba67826ab7dbd902fe4029fc0`。

## 结论边界

本 child 只读取 parent report commit 上的 bundle manifest、48-row label table 与 48-row primary state records。
raw GUI trajectory、截图/OCR、feature cache、GUI-Owl、gate checkpoint 和上游 raw label 均未读取；model/policy/
teacher forward、generation、gate training、旧 dev-5、confirm、matched-NLL 与 closed-loop operation 均为 `0`。
fresh-16 已作为 diagnostic split 消费，不能再作为改模后的 holdout。当前结果不授权 conditional v2 rescue，
也不打开任何 post-GO stage。
