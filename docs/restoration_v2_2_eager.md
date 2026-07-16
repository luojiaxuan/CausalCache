# Restoration v2.2-eager fresh-45 substrate contract

## 本步骤回答什么

v2.1 full-45 的 13 个 exact repeat mismatch 已由 bounded spatial audit 定位为
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`：同一批已暴露状态上，BF16 auto 为 7/13 exact stable，
BF16 eager-control 为 13/13。该结论只支持冻结一个新的 eager runtime substrate；它不改写
`NO_GO_V2_1_FULL_45_SUBSTRATE`，也不授权 restoration、gate training 或 confirm。

v2.2-eager 因而只回答：在 official-tool interface、同一 45-state projection、同一 operation schedule 与同一
gate 下，把 runtime 固定为 BF16 eager numerical control 后，fresh 45-state reference 是否达到 substrate
gate。它不是 CausalCache 方法效果实验，也不读取 AndroidWorld sealed test split。

## Source-only 冻结身份

- protocol：`causalcache_restoration_v2_2_eager_full_45_substrate`；
- config：`code/configs/causalcache_restoration_v2_2_eager.json`；
- attempt：`restoration-v2-2-eager-full-45-substrate-v1`；
- canonical root：`/data/experiments/causalcache/restoration-v2-2-eager-full-45-substrate-v1`；
- global sibling ledger：
  `/data/experiments/causalcache/.restoration-v2-2-eager-full-45-substrate-v1.attempt.json`；
- raw archive：
  `/data/experiments/causalcache/restoration-v2-2-eager-full-45-substrate-v1.raw.tar`；
- planned private HF dataset：
  `gavinlaw/causalcache-restoration-v2-2-eager-full-45-substrate-mobile`，计划 tag
  `v2.2-eager-full-45-substrate-v1`，计划 path
  `raw/restoration-v2-2-eager-full-45-substrate-v1.tar`。

config SHA256 与 clean pushed source commit 只在最终 source freeze 完成后记录真实值；本 source-only 文档不写
预测值。source-only validation 不读取 processor/model，不运行 GPU，也不授权正式 policy execution。

## 唯一允许的 scientific delta

v2.2-eager 继承 v2.1 full-45 的 official tool schema、system prompt bytes、chat template、prompt builder、
canonical parser、teacher target serialization、teacher-forced action-token KL、45-state projection、gate 和
per-state schedule。允许变化仅限：

1. attention backend 固定为 observed `eager`；
2. seed 固定为 0，cuDNN deterministic on、benchmark off，CUDA matmul/cuDNN TF32 off，float32 matmul
   precision 为 `highest`；
3. execution transport 固定为同机同容器的两个 H200 worker；
4. 双 worker attempt 不支持 resume；任何中断都终止为 `INVALID`；
5. attempt、ledger、archive 与 artifact identity 全部换成 v2.2-eager 独立身份。

这些控制只称为 `eager fixed-seed / TF32-off numerical control`，不称为 strict CUDA determinism；
`torch.use_deterministic_algorithms(True)` 保持关闭。除上述 runtime/transport/identity delta 外，不允许根据
spatial audit 的 13 个 mismatch 改 prompt、parser、target、坐标等价规则、阈值或 denominator。

为使上述归因成立，formal runtime 还必须 exact 复现 spatial audit 的执行栈：container image digest
`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`、Python `3.12.3`、
PyTorch `2.11.0+cu130`、CUDA `13.0`、cuDNN `91900`、Transformers `5.6.0`、NVIDIA driver
`570.172.08` 与 `NVIDIA H200`。两个 worker 的这些字段必须完全相同；不能用“同为 eager”掩盖底层栈漂移。

## Fresh 45 与双 H200 拓扑

正式 attempt 必须重新运行全部 30 个 `v2_label_train` 和 15 个 `v2_development` states。v2.1 的 state records、
aggregate、terminal、ledger 或 raw archive 都不能导入、复制或计入 v2.2 分母；pilot 与 spatial audit output
同样不能计数。禁止 filtering、top-up、retry、sample mutation 或跨 attempt 补状态。

两个 worker 固定为一进程一卡，并在同一 Hyper host、同一 container 内运行：

- `even` worker 使用 logical `cuda:0`，处理 index 0、2、…、44，共 23 states；
- `odd` worker 使用 logical `cuda:1`，处理 index 1、3、…、43，共 22 states。

worker 之间禁止 state stealing，两个 output inventory 必须 disjoint，union 必须精确等于 45-state projection；
任一 worker 失败使整个 attempt `INVALID`。两张卡必须使用相同 model snapshot、container image digest 和
scientific runtime metadata；只允许 device、GPU UUID、PCI bus ID、logical device index 与 container-visible
`nvidia-smi` index 不同。logical device 固定为 `0/1`，物理/NVML index 可以是 preflight 选中的任意两个不同
非负编号；GPU UUID 与 PCI bus ID 才是跨映射的物理身份。

两个 worker 都完成 model construction 并 durable 写入 runtime identity 后，coordinator 必须先验证 exact
software stack 相同且 UUID、PCI、NVML index 均不同，再写 barrier release。任何 worker 在 barrier release 前
都不能写 state marker 或调用 generation，避免错误的双卡映射消耗唯一 attempt。

## Global claim 与 non-resumable 边界

coordinator 必须在任一 worker import policy runtime、构造 model 或执行 generation 前，以 exclusive-create
方式 durable claim canonical global sibling ledger 和 root identity。两个 worker 只能在这个 global claim 下写
各自固定 parity 的 marker/terminal records，不能各自创建可独立重跑的 attempt。

每个 state 在首个 generation 前先写 durable attempt marker。v2.2 不提供 `--resume`：worker、coordinator、
model construction 或主机发生任何中断，整个唯一 attempt 都终止为 `INVALID`。已有 terminal record 也不能
作为新 invocation 的跳过前缀；删除 root/ledger、换 output、交换 parity、重生成 completed state，或把旧
v2.1 raw 拷入新 root，都不能恢复或重启该 attempt。

每个 worker 另有 root 外 sibling high-water ledger，更新顺序固定为 sibling → marker/root mirror → policy。
中断造成 root mirror 缺失或落后时，sibling 仍是唯一 attempted/completed truth；stale root 只能是 sibling 的
严格 prefix，并连同 orphan/missing marker/state forensic inventory 封存为 `INVALID`。若 coordinator 本身在
aggregate 前退出，只允许运行 policy-free `seal-interrupted`，它不会 import runtime、retry 或补任何 state。

## Operation schedule 与 gate 不变

每个 state 仍按 v2.1 顺序执行两次 fresh full-history generation；两次都 strict parse、closer/bridge 成功且
canonical action exact agreement 后，才执行 reference 1、reference 2、summary-only 三次 teacher forward 和
repeat/summary 两次 GPU KL。全 attempt 的计划数与硬上限保持：

- generation：90；
- teacher forward：135；
- KL measurement：90。

固定 gate 也不变：45/45 strict parse、45/45 finite logits、45/45 exact canonical repeat agreement，以及至少
8 个 states 满足
`summary_reference_kl > max(1e-4, 10 * mean(repeat_reference_kl))`。合法完成但任一 gate 未通过为
`NO_GO_V2_2_EAGER_FULL_45_SUBSTRATE`；contract/runtime/source/evidence/worker/ledger 不完整为
`INVALID_V2_2_EAGER_FULL_45_SUBSTRATE`。不能使用 spatial audit 结果为本次 gate 预填 PASS。

## 明确禁止的工作

本 source-only freeze 与后续 substrate attempt 中，confirm state/prompt/image/decoder/generation/teacher access、
expert action read、restoration coalition/candidate/label、baseline selection、gate example/model/selection 的计数
都必须为 0。特别是：

- confirm 仍 locked；
- restoration 与 matched-NLL memory pair 尚未开始；
- gate training、checkpoint 和 closed-loop evaluation 尚未开始；
- v2.1 NO-GO 和 spatial audit artifact 均保持 immutable，不因 v2.2 结果回写。

只有 v2.2-eager substrate gate PASS，才允许另行冻结并 push 独立 restoration/confirm source；PASS 本身不自动
授权 confirm execution。若 v2.2-eager 仍不稳定，下一步只能另行冻结 semantic-key / canonical-representative
protocol，不能事后修改本 contract。

## Artifact 与当前状态

正式 terminal 后才可从 canonical root、global ledger 和两个 worker inventory 构建 deterministic USTAR，上传
上述 planned private HF dataset，并在 fresh immutable download、archive byte hash 与 exact inventory 验证后
把 immutable revision 写回 Git 轻量 result。当前阶段只有 source-only freeze：没有 v2.2 policy output、没有
raw archive、没有 HF immutable revision，也没有 substrate verdict。

artifact manifest 必须同时读取 canonical source archive 与不同路径的 fresh immutable download，要求二者
byte-identical，并 exact 绑定 frozen HF repo、path 与 40-hex immutable revision；不能用同一个本地文件自证
fresh download。
