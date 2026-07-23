# Restoration v2.1：official-tool interface rescue

## 结论边界

Restoration v2.1 是新的 policy-interface protocol，不是对 v2 raw output 的 format-only adapter。v2 的
`NO_GO_V2_SUBSTRATE` 与 `NO_GO_ADAPTER_ONLY` 永久保留；v2.1 不重新解析、不重标，也不覆盖旧结果。
它只回答一个更窄的问题：使用 GUI-Owl checkpoint 自带的 `tools=` chat-template 和模型原生
`</tool_call>` 终止后，能否得到足够可靠的单动作接口，供后续 stable-reference substrate 检验使用。

本协议不改变 CausalCache 的 data split、post-state-only high-fidelity block、strong low-fidelity summary、
restricted action inventory、AndroidWorld bridge、视觉预算或 confirm split。`v2_confirm_primary` 始终保持
`CONFIRM_LOCKED`。

这也对应当前路线评审的三项关键建议：reference 是冻结策略的 stable self-behavior，高保真 event 只暴露
post-state image，strong low-fidelity channel 已按冻结八字段实现。v2.1 只替换 output interface，不重新打开
这三项科学设计。

## 冻结接口

- protocol ID：`causalcache_restoration_v2_1_official_tool_interface`；
- policy：`mPLUG/GUI-Owl-1.5-8B-Instruct@06d5faecff74840bab2be2425e9c42667a5d04fc`；
- prompt：通过 pinned processor 的 official `tools=` 参数注入一个 `mobile_use` schema；system/user
  message 不再手写 `# Tools`，assistant 输出不再包含 `Action:` carrier；
- tool schema：只允许 `click`、`long_press`、`swipe`、`type`、`system_button`、`open`、`wait`、
  `answer`、`terminate` 九个 canonical action，并按 action 限定 exact arguments；所有 text argument 的
  schema description 与 system prompt 都要求输入已使用 Unicode NFKC normalization；
- parser：whole-output 必须精确为一组 `<tool_call>\n{single-line JSON}\n</tool_call>`；拒绝前后空白、
  CRLF、多行 JSON、alias、重复 key、非有限数、第二个 JSON、第二个 call、`Action:`、analysis、
  observation、truncated JSON、非 NFKC text 与任何额外文本；
- teacher target：从 `<tool_call>` opener 到 `</tool_call>` closer 全部进入 distance span；JSON 使用
  official assistant `tool_calls` 的 Jinja/tojson bytes，包括 `", "` / `": "` spacing、canonical insertion order 与
  原始 Unicode UTF-8，不加入 `Action:` 或 `<|im_end|>`；解码后的 text argument 仍须严格为 NFKC；
- AndroidWorld bridge：parser 成功后仍必须通过既有 canonical action bridge，不允许新 executor action。

## 冻结 generation boundary

单次 generation 固定为 greedy、batch size 1、`max_new_tokens=256`。运行时验证 pinned tokenizer 的
assistant prefix、tool opener/closer、chat-template SHA、标准 EOS 与 pad token IDs。标准 EOS tokens 在生成
期间被 suppression；singleton `</tool_call>` token 是本轮唯一 generation EOS，且由模型生成的 closer
保留在 decoded output 中。若模型没有生成 closer，就运行到 token cap 并按 strict failure 记录；不得由
软件补 closer、裁剪到第一个 JSON、丢弃 suffix 或 fallback 到 v2 parser。

为了让未来 teacher-forced KL 与 deployed constrained policy 一致，reference 和 candidate logits 都必须
对同一组被 suppression 的标准 EOS token `151645/151643`，在每个 scored position、FP32
`log_softmax` 前应用相同的有限 BF16 `torch.finfo(dtype).min` mask，并在 metadata 中记录 mask。canonical
teacher target 必须与这两个 token disjoint；否则 runtime fail closed。这属于预先冻结的输出格式约束，
不得根据 pilot output 再调整。

`</tool_call>` 若出现在 JSON string 内也会触发终止，随后 strict parser 会把不完整 JSON 判失败。协议不为
该极少见情况加入 syntax repair 或第二条 generation 路径。

## 两阶段 interface gate

### A. Processor-only audit

在任何 v2.1 model forward 或 generation 前，对既有 45 个 screening states 的 full-history 与 summary-only
prompt 各处理一次，共 90 prompts。audit 必须验证 official tools schema 确实进入 chat template、所有 prompt
在 `sequence_length + 256` 下不超过 context、1/3/4/5-image shape 合法、assistant/tool/teacher token boundary
无 merge，并明确记录 `model_weights_materialized_as_tensors=false`、`policy_model_loaded=false`、
`policy_forward_executed=false`、`policy_generation_executed=false`、
`full_artifact_including_confirm_bytes_validated_by_loader=true`、
`confirm_state_prompt_or_image_exposed_to_decoder_or_processor=false` 与
`confirm_processor_prompt_count=0`。这里 loader 会验证并 hash 包含 confirm bytes 的完整 artifact；这不等于把
confirm 内容交给 processor。processor 输入只来自 45 个 screening states，confirm prompt/image 数量严格为 0。
正式入口为 `cd /data/CausalCache/code && python3 -m scripts.audit_gui_owl_v2_1_processor ...`；完整显式参数见
`code/README.md`，output 必须在 repo 外 exclusive-create。evidence 同时绑定 frozen Hyper00
host/container/image identity、Python/platform/package versions、start/end/duration，并明确记录
`gpu_operations_executed=false`、`random_seed=null` 与 `seed_not_applicable=true`。

正式 preflight 已在 source `d0205afd789cda602fbf8964505d9de9a1b1fe53` 上通过：45 states × 2 fidelity
共 90 prompts，official tools 注入 90/90，image-count distribution 为 1-image 45、3/4/5-image 各 15，
context overflow 为 0，input token 长度 3,759--15,013，最长输入加 256 generation budget 后仍低于 32,768。
运行明确记录 policy model 未实例化、weights 未 materialize 为 tensors、forward/generate 未执行、GPU
operation 为 false；这只关闭 processor gate，不是 policy-interface pilot 结果。Git 轻量证据见
`data/results/archive/restoration_v2_1_processor_preflight/`，raw evidence 固定在 private HF tag
`v2.1-processor-preflight-v1` / immutable revision
`85576161b7cb8bbae14e46a482c42b5be5bf1d7e`，并已 fresh-download 逐 byte 复核。

### B. 固定 15-state development pilot

pilot 只使用 selection manifest 已冻结的 `v2_development` 15 states，保持 manifest order。每个 state 只对
full-history prompt 做一次 greedy generation；不 retry、不 top-up、不换 prompt、不做第二次 generation，且
不执行 teacher forward、KL、restoration attribution 或 expert matching。loader 仍可验证/hash 完整 artifact，
但 decoder 不会获得任何 confirm state/prompt/image，confirm generation count 固定为 0。
正式入口为 `cd /data/CausalCache/code && python3 -m scripts.run_restoration_v2_1_interface_pilot ...`；只有独立验证通过的
processor preflight 才能作为输入。
正式 attempt ID 固定为 `restoration-v2-1-interface-pilot-v1`，唯一 persistent output path 固定为
`/data/experiments/causalcache/restoration-v2-1-interface-pilot-v1`。alternate path 与首次 attempt 后删除
官方目录再运行均不允许；这使 no-retry 约束跨 CLI invocation 成立，而不只在调用者选择的单个目录内成立。
attempt 还固定到 `hyper00` / `node-radixark-16-0000`、container
`69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df`、image digest
`sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa` 与 `cuda:0`；跨 host/container/device
运行不属于同一正式 attempt。

正式执行顺序固定为 CPU/data authorization、durable global ledger/root/run-manifest claim、policy runtime
import/model construction、exclusive `runtime_identity.json`、逐 state attempt marker、generation。constructor/OOM
发生在 claim 后时固定写 0/0 `INVALID`；进程中断后 `--resume` 只允许复核已有 terminal evidence，或把
ledger-only/root-partial/marker-only 状态封存成 `INVALID`，绝不继续 generation。成功构造后记录完整 35-key
runtime metadata，并在每个 attempt 与 generation 前复核其 hash 和 run-contract binding。

只有以下条件全部满足才输出 `PASS_V2_1_INTERFACE_PILOT`：

- exact whole-output strict parse：15/15；
- model-emitted canonical closer：15/15；
- canonical action 到 AndroidWorld bridge：15/15；
- max-token truncation：0；
- extra prose、`Action:`、observation、第二 JSON 或第二 tool call：0。

固定分母内任何一个失败都输出 `NO_GO_V2_1_INTERFACE_PILOT`，不得重跑失败 state 或从 label-train 补样本。
通过只证明 interface substrate 可用，不证明 repeat stability、finite KL、memory sensitivity、restoration
oracle 或 paper claim。只有通过后，才允许按已冻结状态顺序进入完整 45-state v2.1 substrate；pilot output
不能被改写成旧 v2 success，confirm 仍锁定。

唯一正式 attempt 已在 source `70724bdb82484cd5645d174765724bd1f83b5ace` 上完成并输出
`PASS_V2_1_INTERFACE_PILOT`：15/15 strict parse、15/15 model-emitted closer、15/15 AndroidWorld bridge，
generation calls 恰为 15；retry、top-up、truncation、extra output、teacher、KL、restoration 与 confirm
generation 均为 0。运行耗时 84.279 秒。raw evidence 见 private HF tag `v2.1-interface-pilot-v1` / immutable
revision `bdff8ca71f150afd80d6291b4ecec76cbf9e7432`，Git 轻量 binding 见
`data/results/archive/restoration_v2_1_interface_pilot/`。这只解除 native output interface gate，不是完整 45-state
stable-reference 或 restoration 结果。

## 审计与完成条件

任何 v2.1 policy output 前，interface/runtime、machine-readable contract、validator、tests、README/docs 都
必须 commit 并 push canonical `main`。正式 run 还必须绑定 clean `HEAD == origin/main == remote main`、Git
blob hashes、model snapshot、processor chat-template、host/GPU/container/runtime identity、完整 argv 与
attempt markers。raw generation trace 只进入 private Hugging Face dataset repo；Git 仅保存轻量 aggregate、
artifact revision、hash 与失败分类。

raw evidence 由 `scripts.manage_restoration_v2_1_pilot_artifact` 打成 normalized deterministic USTAR，包含
canonical output 与 sibling global ledger。validator 对 PASS/NO_GO 重算固定 15-state gate，对 INVALID 复核
严格 partial inventory，并从 source commit 重放 frozen config、selection、policy/source blobs 与 processor
artifact binding。它在独立进程中复用冻结 reducer/schema helper，不声称有第二套 implementation-independent
reducer。

冻结 contract 位于 `code/configs/causalcache_restoration_v2_1_pilot.json`，当前 SHA256 为
`9d51a2ed5d6cc382f297c1b8af3100d784090f72800d637136b88982763fdbf7`，interface source SHA256 为
`90cbefc851bed105de6ea0c8f719aae6313a479ca4589e4160fc4ec3e8de3964`。90-prompt real `AutoProcessor`
audit 与 fixed-15 interface pilot 均已正式通过。下一阶段只允许先冻结、验证并 push unchanged-interface
full-45 substrate source/contract；当前结果仍不授权 confirm，也不证明 repeat stability、finite KL、memory
sensitivity、restoration oracle 或论文主张。
