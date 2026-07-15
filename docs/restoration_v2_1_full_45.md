# Restoration v2.1 full-45 substrate contract

## 本步骤回答什么

fixed-15 pilot 已证明 GUI-Owl 1.5 8B 在 official `tools=` interface 下可以稳定地产生可解析、可转换的
restricted action，但它没有测量 reference stability、teacher logits 或 memory sensitivity。full-45 substrate
只回答下一层问题：在冻结的 30 个 `v2_label_train` 与 15 个 `v2_development` states 上，同一 high-fidelity
history 的两次 greedy action 是否完全一致、teacher logits 是否 finite，以及 summary-only 相对 full-history
是否产生超过 repeat noise 的 action-token KL。

本步骤不是 restoration attribution、gate training、confirm evaluation 或 AndroidWorld closed-loop success
实验。即使通过，也只能授权先冻结下一阶段的 restoration source；不能直接运行 confirm。

## 冻结身份

- protocol：`causalcache_restoration_v2_1_full_45_substrate`；
- contract：`code/configs/causalcache_restoration_v2_1_full_45.json`；
- contract SHA256：`0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b`；
- state projection：30 个 `v2_label_train` 后接 15 个 `v2_development`，SHA256
  `65e7085f01bc26e425bab7f5148c6729bc8c6d0a62802ebc5e8dfeaed6439249`；
- attempt：`restoration-v2-1-full-45-substrate-v1`；
- output：`/data/experiments/causalcache/restoration-v2-1-full-45-substrate-v1`；
- sibling ledger：`/data/experiments/causalcache/.restoration-v2-1-full-45-substrate-v1.attempt.json`；
- raw archive：`/data/experiments/causalcache/restoration-v2-1-full-45-substrate-v1.raw.tar`；
- private HF dataset：`gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile`，tag
  `v2.1-full-45-substrate-v1`，path `raw/restoration-v2-1-full-45-substrate-v1.tar`。

fixed-15 的 root、ledger、runner 与 raw archive 都不能复用。正式运行必须绑定 clean、已 push 的 `main`
commit，并在 runtime import 前从两个 private HF immutable revisions fresh-download 和验证 fixed-15 PASS 与
90-prompt processor PASS evidence。

## 每个 state 的固定计算

每个 state 先 durable 写 attempt marker，然后按固定顺序执行：

1. 对同一个 full-history reference input 作两次 fresh greedy generation；
2. 两份完整输出分别经过 strict whole-output parser、model-emitted closer 检查和 AndroidWorld payload
   conversion；
3. 两个 canonical restricted actions 必须完全相同；
4. 以该 canonical action 为同一个 teacher target，依次计算 reference teacher forward 1、reference teacher
   forward 2、summary-only teacher forward；
5. 在 GPU 上计算 reference repeat KL 与 summary-to-reference KL。

全 45 states 成功到达 logits 阶段时，计划调用数也是全 attempt 的硬上限：90 次 generation、135 次
teacher forward、90 次 KL。没有 retry、top-up、sample mutation、expert-action read、restoration coalition、
baseline selection 或 gate example construction。confirm state/prompt/image/decoder/teacher/generation 的计数都
必须为 0。

parse mismatch、两次 canonical action 不一致或 non-finite logits 是固定 45 分母内的科学失败，不能过滤或
补样。contract/runtime invariant、bridge implementation failure、OOM 或 evidence/source binding failure 是
`INVALID`，不能伪装成科学 `NO_GO`。

## Gate

从逐 state terminal records 独立重算：

- parse coverage 至少 0.99；在 45 个固定 states 上等价于 45/45；
- finite-logit coverage 为 1.0，即 45/45；
- repeat canonical-action agreement 为 1.0，即 45/45；
- `epsilon = max(1e-4, 10 * mean(repeat_reference_kl))`；
- 至少 8 个 states 满足 `summary_reference_kl > epsilon`。

全部满足时为 `PASS_V2_1_FULL_45_SUBSTRATE`；合法完成但任一条件失败为
`NO_GO_V2_1_FULL_45_SUBSTRATE`；运行或证据不完整为 `INVALID_V2_1_FULL_45_SUBSTRATE`。

## Resume 不是 retry

`--resume` 只允许跳过已有、schema 完整的 terminal state prefix，并继续第一个完全没有 marker 的 state。
任何 marker 存在但 terminal record 缺失的 state 都使整个 attempt 进入 `INVALID`；不能重新生成该 state，
也不能删除 root/ledger 后换路径重跑。sibling ledger 还维护 root 外的 durable attempted/completed high-water
journal：attempted 在 marker 前以 atomic replace + file/directory fsync 推进，completed 在 terminal record
落盘后推进；resume 时 journal 与两个目录的 inventory 必须逐项相等。因此同时删除某个已完成 state 的 marker
和 terminal record、或删除整个目录，也只会得到 canonical `INVALID`，不会把该 state 伪装成“未尝试”。已完成
state 永远不能再生成。

## Artifact 与 Source of Truth

canonical output 与 sibling ledger 打成 normalized deterministic USTAR。packager 必须从 raw records 重算
45-state denominator、调用计数、gate 和 outcome，并把 formal source inventory 的 Git blob hashes、run
contract、parent immutable evidence 与 archive tree 绑定起来。reader 会按同一 deterministic USTAR serializer
在内存重建 archive，并要求原始 bytes 完全一致，因此 GNU tar、trailing bytes 或非 canonical header 都会被
拒绝。raw archive 上传 private HF；Git 只保存轻量
manifest、compact reduction、immutable revision 和 fresh-download byte hash。

正式结果只有在以下链条全部完成后才成立：run terminal outcome → deterministic archive → private HF upload
与 immutable tag → fresh immutable download byte verification → Git manifest commit/push → clean descendant 上的
committed-binding validation。

## 当前状态

唯一正式 attempt 已在 source commit `7a5b6d5710fe4d054936b5aa474648149f725edb` 上完成，结论为
`NO_GO_V2_1_FULL_45_SUBSTRATE`：

- 45/45 states 的两次 generation 都 strict parse；90/90 outputs 都有 model-emitted closer 并通过
  AndroidWorld bridge；
- 32/45 states 的两次 exact canonical action 完全一致，低于冻结的 45/45 gate；13 个 mismatch 为 12 个
  `click→click` 与 1 个 `swipe→swipe` coordinate jitter；
- 只有上述 32 个 agreement states 按冻结 schedule 进入 teacher forcing：96 次 teacher forward、64 次 GPU
  KL，没有 non-finite failure；
- 32 个 states 的 repeat KL 都为 0，故 epsilon 为 `1e-4`；32/32 summary-reference KL 均高于 epsilon，
  summary-reference KL 的 min/median/mean/max 为 0.000783/0.031472/0.044022/0.196197；
- retry、top-up、sample mutation、expert read、restoration、baseline、gate training 与 confirm work 全为 0。

raw 94-file deterministic USTAR 已上传 private HF dataset
`gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile`，tag
`v2.1-full-45-substrate-v1` 固定到 immutable revision
`814506ef1450838d4bc6ed3d89fe53e0773d92fb`。fresh immutable download 的 962,560 bytes 与 SHA256
`8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4` 已逐 byte 复核；Git 轻量结果见
`data/results/restoration_v2_1_full_45_substrate/`。manifest push 后，clean
`main@554c51e417d702a6bc759b1f592979a4c38c5283` 从该 fresh archive 返回
`VALID_RESTORATION_V2_1_FULL_45_ARTIFACT` 与 `archive_hash_verified=true`；committed binding 已闭合。

本结果只否定 exact coordinate-level canonical equality 下的 v2.1 substrate admission，不等价于 restoration
oracle 失败。当前 contract 不授权 restoration 或 confirm；若继续，必须先冻结新的 executable/UI-element
equivalence protocol，不能对当前 45 states 事后加容差或 retroactive PASS。
