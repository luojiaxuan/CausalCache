# Restoration v2 substrate screening

本目录记录第一次固定 45-state development substrate screening。它是合法完成的 negative result，不是
runtime invalid：90 个 prompt 的 processor-only shape sweep 全部通过，45 个 state 均调用一次冻结
GUI-Owl generation；随后 45/45 都在第一次 reference generation 的严格 parser 处失败，因此正式结论为
`NO_GO_V2_SUBSTRATE`。

## 正式结果

- run commit：`a0001cbc4d4c3e2584ae3bbcff217ce664e5e064`；
- run contract SHA256：`ed1791e148a18f39f3fc04fd0b4aacf66bfc3225e8fc92a38c9c97835e1c61f1`；
- aggregate SHA256：`57a2456c2667380fecf5f9b4a66c6ed60c5572d284fcf62e4f70348f87532df8`；
- Git summary SHA256：`7b8e32be20b758b158293ec8ebb1017b6d68b557341b8826685a9206aa3ecf66`；
- fixed denominator：45 states，其中 `v2_label_train=30`、`v2_development=15`；
- strict parse：0/45；finite-logit coverage：0/45；repeat agreement：0/45；
- failure：45 个 `GUIOwlV2GenerationParseError`，全部位于 `reference_generation_1`；
- teacher forward、KL measurement、restoration label 均为 0；
- 没有 OOM、state retry、resume、top-up、换样本或 confirm access；confirm 保持 `CONFIRM_LOCKED`。

这里的 0 finite logits 和 0 memory-sensitive states 是 parse failure 之后没有进入 teacher-forced distance 的
结果，不能解释为 logits 非 finite 或 memory 不敏感。正式 gate 不会用事后 parser 重新计算或覆盖。

## 运行身份

运行位于 Hyper00 单张 physical GPU 2：`NVIDIA H200`，UUID
`GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81`；container
`69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df` 只看见 `cuda:0`。
runtime 为 Python 3.12.3、PyTorch 2.11.0+cu130、CUDA build 13.0、cuDNN 91900、Transformers 5.6.0，
GUI-Owl 使用 BF16、`do_sample=false`、`max_new_tokens=256`。完整 argv、input hashes、shape sweep、latency/
memory 与 gate fields 在 `summary.json`。

## 原始 artifact

包含 native outputs、state records、attempt markers、run manifest 与 log 的 93 个 raw files 没有进入 Git，
而是打成一个 deterministic tar shard：

- private HF dataset：
  `gavinlaw/causalcache-guiodyssey-restoration-v2-mobile@restoration-v2-substrate-screening-v1.0.0`；
- immutable revision：`c073e143b935a79befd8ab1fd7123796792efad8`；
- prefix：`runs/restoration-v2-substrate-screening-v1/`；
- archive：`restoration-v2-substrate-screening-20260715T182823Z.tar.gz`；
- archive SHA256：`c3619a177f2b3bd958295bf1fb09653333eb6a735004713158a457a45534a45e`；
- manifest SHA256：`ce817c5047484d1f9cecb6b60423e318a6509f4480477c5a1504cff58bc456f2`。

已从该 immutable revision fresh download 两个文件并逐文件复核 SHA。`artifact_manifest.json` 记录 schema、
生成命令、打包命令、source-file manifest hash 与 raw-data boundary。

## 解释与下一步边界

只读 format inventory 显示，45/45 都使用预期 `Action:` + `<tool_call>` prefix；43/45 条 output 的首个
balanced JSON 可以 canonicalize，只作为宽松 format diagnostic，不是 parser coverage。wrapper 和闭合形式与
冻结 parser 不一致；保守的 envelope-only mechanical normalization 上界为 40/45。剩余 5 条包括多 action、
截断 JSON、第二 JSON 或额外 observation，不能通过“静默取第一个动作”恢复。

因此该 run 否定当前 exact v2 parser/policy substrate 合约，却尚未测量 restoration hypothesis。任何救援都必须
保留本结果，先冻结一个 versioned compatibility protocol 和 fail-closed replay tests，再从新的 clean pushed
commit 重新授权整套 45-state run；不能在原 run 上改写 parse status，也不能打开 confirm。
