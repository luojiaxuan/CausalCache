# Gate v1 fresh-16 failure decomposition v1

> 当前状态：本文件随 Source-A 冻结；runner-freeze B 在该 milestone 中必须 absent，尚未授权读取 sealed parent
> artifact、生成正式诊断结果或修改 Hugging Face。
> fresh-16 primary 的 selector 与 set-conditioning verdict 继续保持 `NO-GO`；本步骤只解释失败来源，不能重分类
> v1，也不会打开旧 dev-5、confirm、matched-NLL 或 closed-loop。

## 目的

fresh-16 primary 已经证明 frozen conditional gate 的 aggregate 恢复量高于三个 heuristic，但它只达到
`0.694229` normalized exact ratio，且 conditional 比 learned independent 低 `0.122394`。这个结果还不能区分：

1. `B=2` 的逐步 greedy 本身无法逼近 exact subset；
2. set-aware oracle 有空间，但 learned student 没有学到 conditional marginal；
3. raw utility 较好而 normalized recovery 较差主要由小 `D(empty)` state 放大；
4. conditional 模型的第一步 event ranking、第二步 completion/stop，或 seed stability 是主要错误源。

本 child 固定以下可加分解：

\[
U(E)-U(C)=\underbrace{U(E)-U(G)}_{\text{search gap}}
+\underbrace{U(G)-U(C)}_{\text{student gap}},
\]

其中 `E` 是 at-most-`B` exact subset，`G` 是读取真实 conditional marginal、每轮重排且遇到非正 gain
停止的 greedy，`C` 是 parent 已封存的 conditional ensemble。所有 raw 与
`U(S)/D(\varnothing)` normalized 数值都使用与 primary 相同的 trajectory-equal reducer。

## Immutable input 与最小读取面

parent 固定为 private HF dataset
`gavinlaw/causalcache-gate-v1-fresh16-claim-serialization-repair-mobile`，tag
`gate-v1-fresh16-claim-serialization-repair-v1`，report commit
`3541fe1ea2c46e555c29cc53483e6f3b809f8f81`。child 只允许 force-download：

1. `fresh16-eval/v1/manifests/bundle-manifest-v1.json`；
2. `fresh16-eval/v1/caches/label-states-v1.jsonl`；
3. `fresh16-eval/v1/reports/primary-state-records-v1.jsonl`。

bundle manifest 继续绑定后两个文件的 exact size/SHA。读取面为 48 个 sealed states、448 个完整 powerset
`D(S)` 值、parent 已落盘的 ensemble round trace、seed final sets、independent 和 heuristic selections。以下对象
都不允许读取或重算：raw GUI trajectory、截图、OCR、feature cache、GUI-Owl、gate checkpoint、上游 raw label、
旧 dev-5 与 confirm。因而本步骤为 CPU reporting reduction；policy/gate/model forward、training、teacher KL、
generation、matched-NLL 与 closed-loop operation 都固定为 0。

## 方法别名与 oracle-independent comparator

- `E`：在 `|S|<=2` 内最小化 `D(S)`；依次按 distance、较小 cardinality、较小 event-id tuple 破 tie；
- `G`：true conditional greedy；每轮在当前 coalition 上计算真实 marginal，只有严格正 gain 才加入；
- `J`：oracle budget-conditioned independent。对 event `j` 使用

  \[
  \frac{1}{2}\left[\Delta_j(\varnothing)
  +\frac{1}{n-1}\sum_{i\ne j}\Delta_j(\{i\})\right],
  \]

  然后选择严格正分数的 top-`B`。这与 `gate_v1_data` 中 independent raw target 完全相同；
- `C` / `I`：parent 已封存的 learned conditional / independent ensemble。

`G-J` 测量 oracle teacher 层面的 set-conditioning headroom；`C-I` 只复核 learned architecture 的已观察
差异，二者不能混为一谈。

## 冻结 routing rule

本诊断不是新的 paper GO。它只决定是否值得进行**至多一次** v2 conditional distillation rescue。

1. Search PASS：`G/E` 的 normalized 与 raw ratio 都至少 `0.95`，且 `n=3`、`n=4` 的 raw ratio
   分别至少 `0.90`；否则停止 distillation rescue。
2. Oracle set-headroom PASS：`G-J` 的 trajectory-equal normalized delta 至少 `0.02`，16-trajectory paired
   bootstrap 90% lower bound 严格大于 0，且至少 `12/16` trajectories 为正；否则停止 set-conditioned
   contribution，最多保留 restoration-guided independent 前置模块。
3. Material student gap：`C/G` normalized ratio 严格低于 `0.90`，或 raw ratio严格低于 `0.95`。
4. 只有 `Search PASS + oracle set-headroom PASS + material student gap` 同时成立，输出才可为
   `ONE_V2_CONDITIONAL_RESCUE`。其余路径均不能靠本诊断打开 post-GO stages。

第一选择造成的不可逆 prefix regret 与 completion/stop regret 以 `60%` 为 dominance 边界，用来决定 v2
优先修 event/query representation 还是 coalition encoding/conditional target。bootstrap 固定 10,000 次、seed
`271828`、confidence `0.90`。

## 只解释、不翻案的诊断

denominator-dominated 必须同时满足：`C/E` raw ratio 至少 `0.90`、raw/normalized ratio 差至少 `0.15`、
每个 `n` 内 bottom-`D(empty)` quartile 贡献至少 `50%` positive normalized regret、但最多贡献 `20%` exact
raw utility，并且移除 bottom quartile 后 normalized ratio 提高至少 `0.10`。即使满足，v1 primary 仍是
`NO-GO`，也不能把删 state 后的数值当主结果。

seed 只有在 median exact-normalized ratio 至少 `0.75`、至少 `4/5` seeds 达到 `0.75`，且移除最差 seed 后
population std 不高于 `0.08` 时才可称为 isolated instability；否则记为 systemic student failure。seed artifact
只有 final sets，没有逐轮 trace，因此只报告 final-set agreement/utility，不声称 seed-specific step ranking。

## Output 与 source-of-truth

详细 artifact 计划发布到新的 private HF dataset
`gavinlaw/causalcache-gate-v1-fresh16-failure-decomposition-mobile`，tag
`gate-v1-fresh16-failure-decomposition-v1`，exact-three targets 为：

- `state-decomposition-v1.jsonl`：48 条 state-level diagnostics；
- `failure-decomposition-report-v1.json`：aggregate、16 trajectory rows、routing verdict 与 operation counts；
- `bundle-manifest-v1.json`：parent/child inventory、Source-A/Execution-B/config、operation contract 与
  prohibited-access 声明；host、Python、argv 与时间边界另由 local completion 和 Git 轻量 summary 记录。

Git 只保存轻量 aggregate summary、README 与本协议；48-row artifact 不进入 Git。Source-A validator 必须
保持 network/file-write/semantic-decode/model/HF-mutation 全为 0，并要求 runner-freeze B 不存在。Execution-B
只能是 Source-A 的 direct single-parent child，且唯一 tree diff 为
`code/configs/causalcache_gate_v1_fresh16_failure_decomposition_runner_v1.json`。

report 中的 operation counts 明确以
`pure_reducer_after_two_sealed_48_record_decodes` 为 scope；parent/child downloads、local writes 与实际 HF
mutation 另由 durable local receipts 和 Git summary 记录，二者不混报。local completion 由 7 条有序 state
records 加同 bytes 的 completion staging/final hard-link pair 组成，并非 9 条彼此独立的 records。若第一次 remote
mutation 后进程崩溃，该 attempt 必须 fail closed 并另立 versioned repair；本协议不声称跨 remote mutation 的
crash-resume，只允许 mutation 前的 byte-identical restart。

## Fresh split 消费边界

fresh-16 已经被 v1 primary 和本 failure analysis 消费。即使 routing 输出允许 v2，也只能用这 16 条轨迹
定位 failure mode，不能据此选超参后再次把它们作为 holdout。任何 v2 都必须先冻结 source、模型、阈值和新的
untouched holdout；若一次 v2 仍未通过原 selector 全部门槛与 set-conditioning 门槛，就停止独立 AAAI
CausalCache story，不执行 matched-NLL 或 closed-loop。
