# Processor-only Execution-CF v1 invalid attempt

## 结论

唯一正式 v1 attempt 的终态是
`INVALID_PROCESSOR_FREEZE_EXECUTION_CF_V1_IMAGE_CONTRACT_DRIFT`。它在任何
AutoProcessor、policy/vision forward、restoration label 或 training 之前 fail closed，不能被当作
processor freeze artifact，也不得上传 Hugging Face。

首个确定性违反点是 worker 0 检查的第 228 个 observation：source bytes 是合法 `PNG/RGB`、
`2208x1840`、无 EXIF；冻结 contract 只接受 `PNG/RGBA` 且 alpha extrema=`[255,255]`。
这否定的是 v1 对全量 GUIOdyssey image mode 的过窄假设，不是 restoration objective 或 selector
效果。

## 运行身份与证据

- producer：`b3472bfbd0a26541a8c6f2e207b88741e256f59a`；
- Execution-CF SHA256：
  `66fd93c64669be734f82830af7d623e9cb97263f90613a5809eb665078b2fba7`；
- Hyper00 CPU-only container：`sglang-omni-jaxan-07181624`；
- start / primary failure / final exit：
  `02:03:17 / 02:14:20 / 02:16:22 UTC`；outer exit code=`1`；
- first failure：`ocr-worker-00`，
  `INVALID_DERIVED_ARTIFACT_BEFORE_POLICY_OUTPUT: image contract drifted`；
- output root 未创建；保留的 sibling staging 为
  `/data/artifacts/.causalcache-set-utility-processor-freeze-v2-b3472bf.incomplete`；
- staging：8 files、`666,629,393` bytes；canonical path/size/file-SHA tree digest=
  `9190b1b140de9b507b4b396694443a5e1293848ae9b02b1309ef81a53bc57e84`；
- completed receipt/artifact shard=`0/0`；worker 1--3 在主失败后收到 SIGTERM，避免继续数小时
  无法发布的 CPU 工作；partial tar 原样保留。

完整结构化身份、首个 violation、每个外置 evidence 文件的 size/SHA256 和科学边界见
[`summary.json`](summary.json)。完整正式 argv 见
[Execution-CF 的正式运行状态](../../../docs/set_utility_processor_freeze_execution_cf.md#正式运行状态)。

## 科学边界

- policy/vision forward：0；
- final candidate state：0；
- restoration label / predictor training：0 / 0；
- matched-NLL / closed-loop：0 / 0；
- Hugging Face mutation：0；
- threshold、候选规则、OCR normalization、worker count、2,400-state denominator 均未修改。

partial OCR 和 partial tar 不是完成结果，不计入正式 denominator，也不能作为 repaired run 的输入。
原计划 tag `phase1-b2-processor-freeze-v1` 未创建 canonical revision，invalid attempt 的 publication
状态为 `INVALID_FAILED_PRESERVED_NO_HF_PUBLICATION`。

## 下一步

先对冻结 1,200 trajectories 的全部 18,792 个 selected source images 做独立只读 format census，
再根据 census 结果另立 versioned image-input contract 和新 output namespace。不得在原 v1 上静默放宽
mode、跳过 observation 或覆盖 `.incomplete`。
