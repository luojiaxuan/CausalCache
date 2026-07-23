# Independent confirm-20 restoration continuation v1

本协议只修复 confirm-20 v1 的 execution runtime 边界。它不重试或重分类旧 attempt，也不修改 independent
gate、数据、阈值、seed、comparator、confirm denominator 或 GO 判据。

机器可执行 Source-A 是
[`code/configs/causalcache_independent_confirm_continuation_v1.json`](../../code/configs/causalcache_independent_confirm_continuation_v1.json)。
其 SHA256 为 `d54988fa05693a42574db2f9ec492d55405bcd2c2e902ccb79897f971c6fc7a9`。
父科学合同仍是
[`causalcache_independent_confirm_closed_loop_v1.json`](../../code/configs/causalcache_independent_confirm_closed_loop_v1.json)，
SHA256 `33e5c0f856a9074cd5460f0d6f7cdda2108b120d50b208234253ff3136d27b63`。

## 为什么允许 continuation

v1 已经把 20-state label-blind payload 发布为 private HF commit
`6d0cd95997186293e01c65276f3c082c11a9f52d`，但 reference generation、teacher forward、`D(S)`、exact oracle
和 report 均为 0。失败来自 formal parent 在 POSIX `fork` 前由 Transformers runtime verification 间接调用
`torch.cuda.is_available()`，触发 `cuInit`；restoration child 随后不能重新初始化 CUDA。这是已定位的执行器
错误，不是 selector 结果。

旧 failure、summary、A/B、HF base→payload history 与 8 个 payload bytes 都已冻结。continuation 的科学输入
没有新增自由度：selector truth 只能来自该既有 payload commit。

## 单向执行顺序

1. 从 clean pushed continuation Execution-B 验证 Source-A inventory、旧 v1 failure tombstone 与父科学合同；
2. 在 formal CLI 外重新运行四卡 topology smoke，并用 Execution-B commit 与 runner-freeze nonce 包装 fresh
   envelope。challenge 同时绑定 model/snapshot、logical devices 与 physical GPU UUID；`policy_vision_spawn` 和
   `teacher_forced_fork` 各 4 个实际 worker 都必须返回 response，formal CLI 逐条重算并要求 8/8 覆盖。response
   是软件审计绑定而非硬件 attestation，因此还显式拒绝旧 v1 inner receipt SHA256
   `8e06034e…2f13`；手工包装或重序列化旧 receipt 不能授权 continuation；
3. formal coordinator 在读取/验证该 envelope 以及 HF/model/semantic access 前独占创建 `attempt.json`；任一后续
   失败都必须留下 terminal，不能无痕重试。receipt 仍须绑定同一 container image、model snapshot、显式 CUDA
   devices 与 physical GPU UUID；
4. 只读下载固定 8 个 payload targets，校验逐文件 size/SHA、全 8-file inventory
   `861fa54d2b46509a62b3cfb7d376c9ab3298c18656ae2c4e80418f86593fa779`，并做 canonical semantic replay；
5. 只读采用现有 HF payload：要求 private、`main==6d0cd959…9f52d`、history 精确
   `base c66a67c3…574f → payload 6d0cd959…9f52d`、report/tag absent，fresh download 前后 remote snapshot
   不变；此阶段禁止 `create_repo`、`create_commit`、`create_tag`；
6. 完整重放 parent derived artifact；feature、dynamic-recent 与 OCR/RGB 只作 exact deterministic replay
   绑定，不能替换 sealed payload；independent 与 policy-vision 只从 sealed bytes 读取，不加载 gate checkpoints、
   不运行 scorer、不运行 policy-vision；
7. formal coordinator 必须是未 import `torch` 的 fresh exec。它在任何 topology/HF/model work 前安装 tripwire，
   阻断 parent 的 `torch.cuda.is_available()`、`device_count()` 与 `_lazy_init()`；`fork` child 通过 at-fork handler
   恢复原入口。model snapshot verification 仍在独立 `spawn` child 完成，只返回 canonical plain mapping；
8. restoration `fork` 紧前再检查 `torch.cuda.is_initialized()`、`_is_in_bad_fork()` 与 active tripwire；任一异常都在
   创建 multiprocessing context、Queue/Process 前 fail closed，显式传入 fork context 也不能绕过；
9. 4 个 worker 各运行原冻结 5 states、同一两次 reference generation 与 17 teacher forwards schedule；
10. 使用父合同的原 evaluator 生成固定 20-state report；report 必须是既有 payload commit 的 direct child，随后
    创建原 annotated tag `independent-confirm20-v1` 并 fresh replay payload+report。若 commit/tag/replay 中途失败，
    只读 reconciliation 必须记录 remote mutation count 或保守下界；不回滚、不重试、不再写成 0；
11. 只有有效 report 为 `GO_TO_PAIRED_CLOSED_LOOP`，才解锁 paired train-60。科学 retry 仍为 0；
    `runtime_continuation_count=1` 只记录执行器修复链，不把旧 attempt 改成成功。

## 禁止项

- 不重新下载或加载 formal-58 的 5 个 gate checkpoints；
- 不调用 independent scorer、policy-vision 或 payload publish；
- 不覆盖、删除或创建第二份 payload commit；
- 不根据 continuation 结果改 threshold、normalization、bootstrap、data、model 或 selection；
- 不在有效 confirm GO 前运行 closed-loop、matched-NLL 或 sealed AndroidWorld test；
- 不把旧 v1 `INVALID` 写成 scientific `NO-GO`，也不因 continuation 成功而追认旧 attempt。

## Source-A / Execution-B

Source-A 包含 continuation config、只读 adoption、sealed-decision replay、isolated model verifier、fresh-exec
CUDA tripwire、durable attempt terminal、partial-publication reconciliation、restoration-only runner、测试与本说明。
Source-A 不下载 payload、不解码 confirm、不加载模型、不运行 GPU、不访问 HF。

Execution-B 必须是 Source-A 的单一 direct child，且只新增
`code/configs/causalcache_independent_confirm_continuation_runner_v1.json`。它绑定 Source-A source/module
inventory，并只授权：采用 exact payload 后运行 restoration/report。selector reexecution authorization 固定为
false。

## 执行结果：有效 NO-GO

Source-A=`8c0d3aeb22079c77fd66a8aa70861d5e80a141ac`，唯一 direct-child
Execution-B=`68e71fd6397ee85f758fbb9e7e9332860ddf0466`。Hyper00 四张 H200 上的 fresh topology envelope SHA256 为
`0c25b5b052c0050238f0bee7bb47558f750b16774f02733acfb11782c8292aeb`；旧 inner receipt 被拒绝，新 inner
receipt SHA256 为 `ba7bd83713c4eb39303d673adb680e86755e3aecfca4848139c853397b15df2a`。4 个 restoration worker 各完成
5 states；正式 operation counts 为 40 generations、340 teacher forwards、320 KL measurements 与 320 raw
distance rows，scientific retry/top-up/filter 均为 0。

20/20 states reference 成功且 memory-sensitive。exact raw utility sum=`0.856431`，independent raw utility
sum=`0.703385`，因此 independent/exact raw ratio=`0.821298`，低于冻结门槛 `0.85`。retained baseline mass
`0.730356` 通过；5 个 seed 的 exact raw ratios 均超过 `0.75`，population std=`0.021896` 也通过。但
independent 相对 dynamic-recent、policy-vision、OCR/RGB 的 mean raw delta 分别为
`-0.001039/-0.002759/-0.003994`；最强 heuristic 是 OCR/RGB，其 paired 90% bootstrap lower=`-0.012737`，
positive support=`7/20`。有效 verdict 因而是 `NO_GO_INDEPENDENT_CONFIRM`，`confirm_go=false`。

canonical report 位于 private HF dataset
[`gavinlaw/causalcache-independent-confirm20-mobile@a0b408e`](https://huggingface.co/datasets/gavinlaw/causalcache-independent-confirm20-mobile/tree/a0b408e58d629299be334a74ecbd0ec2fa2ed1fc/independent-confirm20/v1/report)。report commit
`a0b408e58d629299be334a74ecbd0ec2fa2ed1fc` 是原 payload commit
`6d0cd95997186293e01c65276f3c082c11a9f52d` 的 direct child；tag `independent-confirm20-v1` 的 annotated
object 为 `48921cafa00d7102d8b4c709d58061951f90e42c`，immutable fresh replay 逐字节一致。Git 轻量结果见
[`../../data/results/archive/independent_confirm20_continuation_v1/`](../../data/results/archive/independent_confirm20_continuation_v1/)。

durable completion、HF report/tag 与 replay 全部成功后，CLI 在最后把 immutable completion 打印到 stdout 时
因 `json.dumps(mappingproxy)` 抛出 `TypeError` 并非零退出。该 post-terminal transport bug 不改变已经持久化和
发布的科学终态；后续只修 CLI 序列化并加回归测试，没有重跑 confirm。按父合同，paired closed-loop、
matched-NLL 与 sealed AndroidWorld test 均未执行并继续 locked。本 v1 路径到此停止。
