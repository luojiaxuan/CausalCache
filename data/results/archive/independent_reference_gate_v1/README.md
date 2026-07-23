# Independent UI-TARS reference gate v1

## 结论

`NO_GO_CURRENT_REFERENCE_STACK`。在任何 independent policy output 产生前冻结的 8 条 trajectory、75 个
decision 上，UI-TARS full-history deterministic generation 得到 69/75 parsed、27/75 executable match，
overall coverage 为 36.0%，低于预注册的 50%。required action types 中 tap 为 21/58、type_text 为 5/9，
但 swipe 为 0/2，因此 overall gate 与 action-type gate 同时失败。

6 个 parse failure 全部是 `open_app(...)`：冻结 prompt 将它列为合法 action，但冻结 parser 未实现对应
映射，这是 v1 reference stack 的已知 interface mismatch，不能表述为实现完全无缺陷。其中对应 reference
action 为 5 个 `home` 和 1 个 `tap`。即使把这 6 项全部反事实地算成 match，上界也只有 33/75（44.0%），
仍然达不到 50%，且 swipe gate 仍失败；所以 no-go 对该 mismatch 稳健。其余错误可分为 33 个 action type
相同但坐标/文本不匹配、9 个错误 action type。8 条 trajectory 中只有 1 条超过 50%，3 条恰好 50%，其余
4 条低于 50%，不是单个 trajectory 主导的边界失败。

按冻结 protocol，reference gate 失败后立即停止当前 UI-TARS reference route。disjoint oracle split 的
132 个 decisions 未运行；不得从该 split 生成 restoration labels、训练 gate，或根据本结果修改 prompt、
equivalence、threshold、sample split 后重报同一 gate。

若另立 v2，必须在未观察的新 split 上预注册一致的 action contract：要么实现 executor-compatible
`open_app` 映射，要么从 prompt action space 删除它；不能回改 v1。source 来自 GUIOdyssey train shards，
当前 provenance 也不能排除 UI-TARS 训练数据与其重叠，因此本结果只声称对本项目未观察过 policy output，
不声称严格排除了 backbone 训练数据污染。

## Frozen execution

- Git：`585fd2aa8061f069008d985552ddaec4bbfd5246`，clean `main`；
- config SHA256：`7b62aa31c80536f28bc4e8a3d684ff535bc2315d44a31cb5f002ed6dcb493fd8`；
- dataset：private HF
  `gavinlaw/causalcache-guiodyssey-independent-mobile@84c9f5a335e9612ccb4bd566f977574f359b2485`；
- dataset manifest/tar SHA256：`3870900442dd0c8f037c9127e0c61c53c9d08c3735164c3c57c857f2c65eb949` /
  `b16bd9c631c3e0ad2576715db6aad72be82ca9dda71612576c5f8b8ab52b0fe2`；
- policy：`ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`；
- hardware anchor：旧 9-decision A6000 vector 已在 Hyper00 精确复现，随后才运行本 split；
- host：Hyper00 `node-radixark-16-0000`，physical GPU 1 / container `cuda:0`，NVIDIA H200；
- container：`sglang-omni-jaxan-07151458`，image
  `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- runtime：Python 3.12.3、PyTorch 2.11.0+cu130、Transformers 5.6.0、BF16；
- UTC：2026-07-15 07:00:15--07:05:12；wall time 296.75 s，generation latency 合计 185.26 s，
  peak allocated 18.20 GB。

正式任务前 GPU0 被其他 workload 占用，因此按 10 秒 preflight 选择 physical GPU 1，并用非 privileged
container 验证 PyTorch 只看到这一张卡。monitor 的 10 个 active windows 平均 window utilization 为
20.4%，sample max 为 83%；逐 decision 的 variable-history preprocessing 与短 generation 仍是明显瓶颈。
当前已经是单 GPU，未为提高利用率修改 frozen runner。若未来另立新 reference protocol，必须先完成
batching/concurrency；本路线已经因科学 gate 失败，不进入 coalition-scale oracle。

## Artifacts

- Git 轻量机器可读结论：[`summary.json`](summary.json)；
- raw per-decision summary 与 GPU monitor：private HF dataset
  `gavinlaw/causalcache-guiodyssey-independent-mobile@reference-gate-v1`
  (`b3e1245c6c6a1723fe2ca3a861148008df39df46`)，路径
  `runs/independent-reference-gate-v1/`；
- raw summary SHA256：`0a5af5b1ef4478c9ccf3b7c9ba59a7d01bc46ba73b330d281ab23dff6b93eb52`；
- monitor SHA256：`cec257fc075e60dedafff728c2669816eb80ab6cc4886e7cf6b20648ca7ece18`。

四个 HF payload files 已从 immutable revision 强制重下载并逐个核对 SHA256；tag
`reference-gate-v1` 已验证指向同一 revision。
