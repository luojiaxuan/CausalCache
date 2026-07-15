# Code 目录

本目录是 CausalCache 全部可执行逻辑的 Git source of truth：

```text
code/
├── causalcache/    # Python package：schema、attribution、selection、policy adapter
├── scripts/        # 可复现 CLI；使用 python -m scripts.<name>
├── tests/          # 单元测试与契约回归
├── configs/        # 冻结实验配置、snapshot manifest、task split
└── requirements/   # 特定 backbone 的额外依赖
```

## 本地开发

根目录的 `make` target 不要求 editable install：

```bash
make test validate-contract
make paper
```

若要从仓库根直接运行某个 CLI，先安装 editable package；`scripts` 与 `causalcache` 都由根目录
`pyproject.toml` 从 `code/` 发现：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m scripts.validate_contract \
  --config code/configs/phase0_contract.json \
  --decision data/fixtures/validated_decision.json
```

共享机器中的 virtualenv 必须放在持久挂载 `/data` 下，例如 `/data/.venv/causalcache`；不要依赖
容器层。实验语义参数使用 CLI 或 committed JSON config 显式传入，不使用临时环境变量覆盖 seed、
model revision、dtype、budget、prompt、preprocessing 或 threshold。

## 修改规则

- 改动 frozen contract 时，同时改 `code/configs/`、对应 tests、论文与 `docs/progress.md`；
- 新 policy adapter 先补 parser/prompt/runtime smoke test，再运行 GPU job；
- material GPU run 按 `code/configs/run_manifest.schema.json` 记录跨芯片 provenance；
- `data/results/` 只接收轻量 summary、CSV 与 README；raw traces、datasets、checkpoint 进入 HF；
- 每个可复现实验里程碑在同一小提交中更新 code、config、result summary 和 progress，然后 push
  `main`。

GUI-Owl Think 的冻结输出边界允许开头最多一个小写且闭合的 `<think>...</think>`
block；剔除后仍必须完整匹配单行 `Action:` 和唯一 `mobile_use` `<tool_call>`。未闭合、
多 block、中缀/后缀 thinking 或额外文本全部 fail closed。

AndroidWorld full validation 必须显式传入
`--early-stop-when-success-is-mathematically-impossible`。runner 只在原子 episode checkpoint
写盘后检查固定分母上的 success 上界；达不到 gate 时不再分配新 episode，已在途的
worker 仍完成 score 与 teardown。`--resume` 只接受 instance、plan index 和文件名都与冻结
plan 一致的 checkpoint，并要求 episode 内的 Git commit、model snapshot、runtime、processor、
generation、plan 和 server digest 完全一致；非 resume 运行要求空 output directory。命令必须用
`--run-git-commit` 显式传入 full SHA，runner 会同时校验实际 HEAD 与 clean worktree。

Qwen-family diagnostic 的 teacher-forced 接口由
`QwenPolicyRuntime.teacher_forced_action_log_probs` 提供：调用方显式传入 canonical action token ids，
runtime 只拼接 `action[:-1]`，并取 prompt 最后位置开始的 $L$ 个 full-vocabulary logits。输出是 CPU
float32 log-probabilities；`full_vocab_action_path_kl` 负责逐 token KL 与 mean/sum 聚合。该接口不加入
generated special tokens，也不把 pathwise KL 描述成完整 sequence-action KL。

`causalcache.diagnostic` 是不依赖 GPU 的 result reducer：canonicalize 首个 action JSON，计算 RGB
histogram similarity，在完整 feasible-coalition distance table 上确定 recent/similarity/random/oracle，
并只按 frozen config 输出 `INVALID`、`NO_GO_DIAGNOSTIC`、`INCONCLUSIVE_NEGATIVE` 或
`INCONCLUSIVE_POSITIVE`。random 是 maximal feasible coalitions 的解析期望，不引入隐藏 seed。

冻结 diagnostic 通过 fail-closed CLI 执行：

```bash
python3 -m scripts.run_go_no_go_diagnostic \
  --config code/configs/go_no_go_diagnostic_v1.json \
  --dataset-tar /data/artifacts/guiodyssey-pilot-00000.tar \
  --model-dir /data/artifacts/models/Qwen3-VL-8B-Instruct \
  --device cuda:0 \
  --output-dir /data/experiments/go-no-go-diagnostic-v1 \
  --run-git-commit <FULL_CLEAN_MAIN_SHA> \
  --container-image-digest <SHA256_IMAGE_ID>
```

runner 拒绝 dirty/mismatched Git、dataset SHA、model snapshot、visual accounting 或 full-history
executable mismatch；异常写单个 `failure.json` 并返回 nonzero。成功只写轻量 `summary.json`，不保存
full-vocabulary logits。coalition forwards 固定为 36 个 mixed-fidelity inputs、2 个 full references 和 2 个
repeat-noise probes；selector、random expectation 与 $K$ sweep 都读取同一 distance cache。

正式 validation 结束后只上传聚合 payload，不直接上传逐 episode 小文件：

```bash
python3 -m scripts.package_androidworld_validation \
  --validation-run-dir /data/experiments/gui_owl_1_5_8b_think_androidworld_validation \
  --validation-plan code/configs/androidworld_validation_plan.json \
  --output-dir /data/experiments/gui_owl_1_5_8b_think_androidworld_hf_payload \
  --policy-slug gui-owl-1.5-8b-think \
  --source-run-git-commit <FULL_RUN_GIT_COMMIT> \
  --hf-repo gavinlaw/causalcache-androidworld-validation-mobile \
  --hf-tag v0.2.0
```

packager 会重新验证 summary、冻结 plan、所有 episode instance/index/filename/run contract 与完整或
数学确定 early-stop 计数，然后按 `plan_index` 写一个 `mtime=0` 的 deterministic gzip JSONL。HF 布局
固定为 `data/<policy-slug>/...` 与 `runs/<policy-slug>/...`；输出目录必须为空，payload manifest 只记录
预期 repo/tag 和内容 hash，不记录尚未产生的 HF OID。
