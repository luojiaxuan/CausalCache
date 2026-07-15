# Restoration v2.1：official-tool interface rescue

## 结论边界

Restoration v2.1 是新的 policy-interface protocol，不是对 v2 raw output 的 format-only adapter。v2 的
`NO_GO_V2_SUBSTRATE` 与 `NO_GO_ADAPTER_ONLY` 永久保留；v2.1 不重新解析、不重标，也不覆盖旧结果。
它只回答一个更窄的问题：使用 GUI-Owl checkpoint 自带的 `tools=` chat-template 和模型原生
`</tool_call>` 终止后，能否得到足够可靠的单动作接口，供后续 stable-reference substrate 检验使用。

本协议不改变 CausalCache 的 data split、post-state-only high-fidelity block、strong low-fidelity summary、
restricted action inventory、AndroidWorld bridge、视觉预算或 confirm split。`v2_confirm_primary` 始终保持
`CONFIRM_LOCKED`。

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
  official template 的 `", "` / `": "` spacing，不加入 `Action:` 或 `<|im_end|>`；
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
无 merge，并明确记录 `model_weights_loaded=false`、`forward_executed=false`、`generate_executed=false`、
`confirm_accessed=false`。

### B. 固定 15-state development pilot

pilot 只使用 selection manifest 已冻结的 `v2_development` 15 states，保持 manifest order。每个 state 只对
full-history prompt 做一次 greedy generation；不 retry、不 top-up、不换 prompt、不做第二次 generation，且
不执行 teacher forward、KL、restoration attribution、expert matching 或 confirm access。

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

## 审计与完成条件

任何 v2.1 policy output 前，interface/runtime、machine-readable contract、validator、tests、README/docs 都
必须 commit 并 push canonical `main`。正式 run 还必须绑定 clean `HEAD == origin/main == remote main`、Git
blob hashes、model snapshot、processor chat-template、host/GPU/container/runtime identity、完整 argv 与
attempt markers。raw generation trace 只进入 private Hugging Face dataset repo；Git 仅保存轻量 aggregate、
artifact revision、hash 与失败分类。
