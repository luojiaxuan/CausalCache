# CausalCache go/no-go 路线

## 当前结论边界

仓库没有通过历史 v1 expert-coverage gate 的 teacher；v1 UI-TARS 已正式输出
`NO_GO_CURRENT_REFERENCE_STACK`。restoration v2 已另立为 stable self-behavior estimand 并在任何 v2
policy output 前冻结，但目前只完成科学契约，还不是 `GO`。已观察的 GUIOdyssey decision step 4/8 仍有
post-selection bias，只能作为历史 existence diagnostic，不能成为 v2 label 或推翻 v1 rejection。

第一阶段使用 `code/configs/go_no_go_diagnostic_v1.json`。配置在任何 restoration forward 之前冻结，
只允许四种输出：

- `INVALID`：prefix alignment、full-history executable match、visual-token cost、finite KL 或重复 forward
  噪声检查失败；
- `NO_GO_DIAGNOSTIC`：两个状态都不对 memory fidelity 敏感，或在 1024-token 预算下两个 exhaustive
  oracle 都无法产生超过数值阈值的恢复；
- `INCONCLUSIVE_POSITIVE`：至少一个状态敏感、1024-token oracle normalized recovery 不低于 0.25，
  且至少一个非 recent event 有正 exact restoration gain；
- `INCONCLUSIVE_NEGATIVE`：其余合法结果。

这里的 `NO_GO_DIAGNOSTIC` 只否定“当前 low-fidelity/high-fidelity 表示 + Qwen diagnostic stack”，不是
对任意 GUI policy 或 CausalCache 机制的一般否定。第一阶段明确禁止输出 `GO`。

## 第一阶段：selection-biased existence diagnostic

固定输入：

- GUIOdyssey private artifact
  `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- rejected diagnostic policy
  `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；
- decision step 4 与 8，不根据 restoration 结果继续筛状态；
- processor 参数仍以每图 256 visual tokens 作为 resize target；在任何 restoration forward 前，使用
  pinned Transformers 5.6.0 对 artifact 中统一的 2208×1840 图像做静态检查，`smart_resize` 实际得到
  476×392，因此每图 effective cost 为 238、每个 high-fidelity event 的两张图合计 476；预算上限仍
  固定为 512 与 1024，分别最多容纳 1 与 2 个事件；
- 对最多两个恢复事件的全部可行 coalition 做 exhaustive rerun。step 4 为 7 个 coalition，step 8 为
  29 个 coalition，加两个 full-history reference，共 38 个 teacher-forced forward；重复 reference
  仅用于估计数值噪声。

full-history deterministic generation 必须先与 recorded action executable-match。其首个 JSON action
object 使用 `sort_keys=true`、UTF-8、无空格重新序列化为唯一 canonical action；不从 coarse target
反推坐标，不把 explanation 或 generated special token 纳入路径。对 full prompt 和每个 coalition
prompt teacher-force 同一 action token prefix，计算 full-vocabulary float32 log-softmax 的逐 token KL，
并对 action tokens 取平均。该量只称为 `teacher-forced action-path KL`，不称为完整 sequence-action KL。

每个状态的数值阈值为：

$$
\epsilon_t=\max(10^{-4},10D_{\mathrm{repeat},t}).
$$

只有 $D_t(\varnothing)>\epsilon_t$ 才称为 memory-sensitive。normalized oracle recovery 定义为：

$$
R_t(B)=\frac{D_t(\varnothing)-D_t(S^*_{t,B})}
{\max(D_t(\varnothing),\epsilon_t)}.
$$

baselines 固定为 recent、全部等大小 coalition 的 uniform-random exact expectation、event after-image 与
current image 的 RGB histogram cosine similarity、global exhaustive oracle 和 budget-conditioned
restoration selector。所有 selector 读同一个 coalition-distance cache，不追加选择性 policy forward。

## 历史第二阶段：v1 paper-level go/no-go

第一阶段即使得到 `INCONCLUSIVE_POSITIVE`，也只表示值得扩展。v1 当时要求使用没有观察过
restoration 的独立多轨迹 split，并在 policy inference 前完成 manifest 与判据 commit。当时冻结路线是
固定旧 pilot 中最接近 gate 的
`ByteDance-Seed/UI-TARS-1.5-7B@683d002dd99d8f95104d31e70391a39348857f4e`，保持原 prompt、parser、
10×10 coordinate bin、deterministic decoding 和以下 gate 不变：full-history executable-match coverage
至少 50%，且 split 中出现的 `tap`、`swipe`、`type_text` 各至少命中一次。

新轨迹必须与旧 `0054832199799795` 完全不重叠，由 pinned GUIOdyssey source rows 按固定 salt 的
`SHA256(source_id)` 排序选择，并在 policy inference 前拆成不重叠 `reference_gate` 与 `oracle_pilot`。
若 reference gate 失败，立即输出 `NO_GO_CURRENT_REFERENCE_STACK`，不得换 subset、prompt、equivalence
或 threshold。它只说明当前 reference 获取路线不成立。

可执行预注册已经冻结为
[`code/configs/independent_reference_gate_v1.json`](../code/configs/independent_reference_gate_v1.json)：从
固定 transport revision 的前 16 个 mobile/use train shards 读取全部 rows，先按显式结构规则做
policy-blind eligibility，再以 `SHA256(UTF8(salt + NUL + source_id))` 唯一排序。`reference_gate` 是满足
8 trajectories、48 decisions、3 个 normalized app labels 与三类 required action 都出现的最短前缀；
`oracle_pilot` 是移除前者后满足 15 trajectories、60 decisions、3 apps 与相同 action presence 的最短
前缀。整个 frozen reference denominator 都要评测，parse error 计为 non-match；任何 policy output 后都
禁止补样本或重排。

旧结果来自 Aries A6000，而独立 gate 计划在 Hyper00 H200 执行。为隔离硬件变量，Hyper00 必须先在旧
9-decision artifact 上复现冻结的 9/9 parsed、4/9 match 及逐 decision boolean vector。若 behavioral
anchor 不一致，就把未修改的 policy/config 转到 Aries；在 anchor 通过前不得在 Hyper00 打开独立 split
的 policy outputs。新 artifact 还必须先上传 private Hugging Face、把 immutable revision 与 tar/manifest
SHA 回写并 push，之后才允许 inference。

### v1 执行结果

该 gate 已在 Git `585fd2aa8061f069008d985552ddaec4bbfd5246` 上完整执行，结果为
`NO_GO_CURRENT_REFERENCE_STACK`：69/75 parsed、27/75 executable match（36.0%），tap 21/58、swipe
0/2、type_text 5/9。整体至少需要 38/75，且 swipe required-action gate 独立失败，因此按上述规则停止，
oracle split 未打开任何 policy/restoration output（raw artifact 已由 builder 打包）。完整轻量结论见
[`data/results/independent_reference_gate_v1/`](../data/results/independent_reference_gate_v1/)，raw run 位于
private HF `gavinlaw/causalcache-guiodyssey-independent-mobile@reference-gate-v1`
(`b3e1245c6c6a1723fe2ca3a861148008df39df46`)。

冻结 prompt 允许 `open_app`、parser 却未实现它，导致 6 个 parse failure；这是 v1 stack 的已知 action
contract mismatch。它不能在看过结果后修正：即使把 6 项全部乐观计为 match 也只有 33/75（44.0%），且
swipe 仍为 0/2，所以 v1 no-go 不依赖该 mismatch。若未来另立 v2，必须在新 untouched split 的任何 policy
output 前先解决并提交一致 action contract。GUIOdyssey source 来自 train shards，当前 provenance 不能
证明它不在 UI-TARS 训练语料中，因此“独立”只指本项目此前未观察这些 policy outputs，不声称严格训练集
去污染。

v1 当时规定：若 gate 通过，paper-level oracle 至少需要 20 个预先冻结的 executable-matched states，覆盖至少 10 条
trajectory 与 3 个 app。扩展集的 `GO` 条件预先固定为：

- memory-sensitive states 至少 8/20；
- 1024-token mean oracle recovery 至少 0.30；
- 相对 strongest recent/similarity/random baseline 的 mean recovery gain 至少 0.10，paired 90%
  bootstrap lower bound 大于 0；
- $K=16$ 对 exact attribution 的 median Spearman 至少 0.80、top-budget Jaccard 至少 0.75、oracle
  utility ratio 至少 0.90。

若不多于 3/20 states 敏感，或 oracle mean recovery 不高于 0.10 且没有 executable recovery，则为
paper-level `NO-GO`；其余为 `INCONCLUSIVE`。AndroidWorld closed-loop success、matched-NLL 与 distilled
gate 仍是后续阶段，offline oracle `GO` 不等于论文主张已经成立。

## Restoration v2 当前 gate

v2 不重开或放宽 v1 expert gate。它把 reference 改为 GUI-Owl-1.5-8B-Instruct 的 parseable、finite、
repeat-stable self-behavior；expert alignment 只作为独立质量轴报告，不能决定 admission、drop 或 top-up。
科学配置为 [`causalcache_restoration_v2.json`](../code/configs/causalcache_restoration_v2.json)，完整干预与
data exposure 见 [`restoration_v2.md`](restoration_v2.md)。当前尚未运行任何 v2 output。

第一层只在 label-train/development 做 substrate screening，至少 20 states：

- action-contract parse coverage 至少 0.99；
- finite-logit coverage 为 1.0；
- 两次 deterministic forward 的 canonical-action agreement 为 1.0；
- `D(summary) > max(1e-4, 10 * mean_repeat_kl)` 的 memory-sensitive states 至少 8。

任一失败输出 `NO_GO_V2_SUBSTRATE`，untouched confirm 不打开。通过后，confirm 固定使用 frozen hash order
中的 20 条 trajectory，每条 decision step 6 一个 state；parse/stability failure 计入固定分母，不能换样本、
按 expert/quality/sensitivity 过滤或事后 top-up。

Confirm primary 固定四个 visual candidates、容量最多两个，`K=16`、seed `20270715`。进入 gate training 必须
同时满足：

- memory-sensitive states 至少 8/20；
- mean oracle recovery 至少 0.30；每状态使用
  `(D(empty)-D(selected))/max(D(empty), epsilon)`，对全部 20 个固定状态取均值；
- 相对 strongest baseline 的 mean gain 至少 0.10，paired 90% bootstrap lower bound 大于 0；
- median Spearman to exact 至少 0.80；
- memory-sensitive states 上的 median top-budget Jaccard 至少 0.75；
- median sampled-selector / exact-attribution-selector utility ratio 至少 0.90。

Oracle 在 0/1/2-event subsets 中取最小 distance，exact-2 作为 cardinality ablation；paired bootstrap
要求 oracle 对每个 non-oracle baseline 的 90% lower bound
都大于 0。全部 pass 条件成立才输出 `GO_TO_GATE_TRAINING`。memory-sensitive 少于 4，或 mean oracle
recovery 不高于 0.10，输出硬 `NO_GO_RESTORATION_V2`；其余合法但不满足 pass 的结果，包括 confirm
parse/stability failure，输出 `INCONCLUSIVE_V2`。contract/runtime 错误为 `INVALID`。offline
`GO_TO_GATE_TRAINING` 只表示 oracle 值得蒸馏，仍不等于 gate、closed-loop success 或 matched-NLL
paper claim 已成立。
