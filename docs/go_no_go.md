# CausalCache go/no-go 路线

## 当前结论边界

仓库目前没有通过冻结 coverage gate 的 reference policy。因此，已观察到 full-history executable
match 的 GUIOdyssey decision step 4/8 存在明显的 post-selection bias：它们可以验证真实视觉
mixed-fidelity rerun、teacher-forced KL 和 oracle ceiling，也可以给出当前表示的硬负信号，但不能单独
产生论文意义上的 `GO`，不能生成 gate training labels，也不能推翻已经记录的 policy rejection。

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

## 第二阶段：真正的 paper-level go/no-go

第一阶段即使得到 `INCONCLUSIVE_POSITIVE`，也只表示值得扩展。真正的 `GO` 必须使用没有观察过
restoration 的独立多轨迹 split，并在 policy inference 前完成 manifest 与判据 commit。当前推荐路线是
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

若 gate 通过，paper-level oracle 至少需要 20 个预先冻结的 executable-matched states，覆盖至少 10 条
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
