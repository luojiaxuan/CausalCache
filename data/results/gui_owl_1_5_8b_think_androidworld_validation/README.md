# GUI-Owl-1.5-8B-Think AndroidWorld validation（valid rejection）

## 结论

`mPLUG/GUI-Owl-1.5-8B-Think@afe3707fc84caebc4d7046118b34493ecf8bb060` 未通过预注册的
AndroidWorld frozen-policy gate，不能作为 CausalCache 主实验的 validated teacher。runner 在 40 个
原子 checkpoint 时已确认 success gate 数学上不可达；两个已在途 worker 完成 score/tear-down 后，最终
保存 42/62 条 records。

- official success：9；固定 62 分母下界 9/62（14.52%）；
- unobserved：20；即使全部成功，上界仍只有 29/62（46.77%），低于所需 31/62；
- model steps：513；parse success：512；parse coverage：99.81%，通过 95% parse gate；
- outcomes：9 `official_success`、23 `terminal_failure`、1 `parse_failure`、9
  `infrastructure_failure`；
- termination：18 `policy_terminated`、14 `step_budget_exhausted`、1 `parse_error`、9
  `exception`；
- 运行中未修改 prompt、parser、action equivalence、model、visual preprocessing、validation plan 或
  threshold；final test 保持 sealed。

9 个 exception 包括 6 个 AndroidWorld HTTP 500、2 个 live instance 与 frozen plan 不一致，以及 1 个
任务初始 reward 已为 1.0。它们全部按 fail-closed 规则保留在固定分母中，没有 retry 或删除。唯一 parse
failure 是 `FilesDeleteFile[0]`；其余已观测 model actions 均满足冻结 grammar。

不能把 9/42 报告成 benchmark success rate，因为 early-stop 后的 observed subset 受 round-robin 完成
顺序影响。合法结论只有固定 62 分母下界 9/62、上界 29/62，以及上界低于 50% gate。

## Frozen execution

- run Git：`36526c58997be5409f65c5976fb35e17aa007ad7`，clean detached checkout；
- host：Aries `x86_64`；physical GPU 1，NVIDIA RTX A6000，driver `550.107.02`；
- policy container：`sglang-omni-jaxan-07141905`；image ID
  `sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21`；registry
  repo digest `sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa`；
- runtime：Python 3.12.3、PyTorch 2.11.0+cu130、CUDA 13.0、Transformers 5.6.0、BF16；
- AndroidWorld：4 个独立 executors，ports `5000–5003`，server image
  `sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486`；
- preprocessing：model-default visual resolution；最多 5 张截图；`do_sample=false`；
  `max_new_tokens=256`；
- wall time：6384.83 s（2026-07-14 19:14:07–21:00:32 UTC）；4 environment workers、1 serialized
  policy runtime。

GPU monitor 共记录 86 个 10 秒窗口；window-average utilization 的 mean 为 84.7%、median 为 87%，30 个
窗口至少 90%。forward 窗口峰值反复达到 100%；较低窗口来自 emulator reset/action/score 与收尾阶段。
单 GPU 已是最小配置，未为提高利用率改变冻结 worker topology。

## Artifacts

- 轻量机器可读结果：[`summary.json`](summary.json)；SHA256
  `d12caac81e8d2b0f611fd368a3da5746428cc83da594e8a3bfb1be4157ffc4de`；
- 完整 42 条 episode traces：private HF dataset
  [`gavinlaw/causalcache-androidworld-validation-mobile`](https://huggingface.co/datasets/gavinlaw/causalcache-androidworld-validation-mobile)，
  tag `v0.2.0`，revision `0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`；
- trace shard：`data/gui-owl-1.5-8b-think/validation-00000-of-00001.jsonl.gz`，42 records、92,016
  bytes、SHA256 `a5e9e771fde087f0684b016a884e37ec4300066c4b3a37e42e79303cd3e787fb`；
- HF payload 内另含 frozen summary 与 payload manifest；旧 `v0.1.0` tag 仍指向 Instruct validation 的
  immutable revision `3fcca45fffe9842c9fcebbf5c6c27c9540bb1515`。

[`dataset_manifest.json`](dataset_manifest.json) 是 Git 中对 immutable HF payload 的索引；其中
`hf_payload_files` 的 bytes/SHA256 描述 `0faf767e...` 上下载的五个文件，不是该 Git 索引自身的哈希。

同一输入已独立打包两次并逐字节比较；上传后又从 immutable HF revision 下载 5 个 payload files，
逐个核对 SHA256、gzip 解压 record count 与唯一有序 `plan_index`。

## Reproduction

正式 runner 的 scientific argv 记录在 [`run_manifest.json`](run_manifest.json)。结果完成后使用 Git
`b83c913cf28d46413b74e15e0070f8c0f48a2ee8` 的 packager：

```bash
python3 -m scripts.package_androidworld_validation \
  --validation-run-dir /data/experiments/gui_owl_1_5_8b_think_androidworld_validation \
  --validation-plan code/configs/androidworld_validation_plan.json \
  --output-dir /data/experiments/gui_owl_1_5_8b_think_androidworld_hf_payload \
  --policy-slug gui-owl-1.5-8b-think \
  --source-run-git-commit 36526c58997be5409f65c5976fb35e17aa007ad7 \
  --hf-repo gavinlaw/causalcache-androidworld-validation-mobile \
  --hf-tag v0.2.0
```

该结果只回答 replacement frozen policy 是否足以支持后续实验。它不是 CausalCache attribution、memory
gate、matched-NLL 或方法效果结果。按预注册 change control，本轮在这里停止，不生成 restoration labels。
