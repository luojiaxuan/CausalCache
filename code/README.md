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
make test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces
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
.venv/bin/python -m scripts.validate_restoration_v2_contract \
  --config code/configs/causalcache_restoration_v2.json
.venv/bin/python -m scripts.validate_restoration_v2_interfaces \
  --contract code/configs/causalcache_restoration_v2.json \
  --action-fixture data/fixtures/gui_owl_v2_action_roundtrip.json \
  --prompt-fixture data/fixtures/restoration_v2_prompt_low_fidelity.json \
  --interface-manifest data/manifests/restoration_v2_interfaces.json
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

当前 v2 scientific contract 由 `causalcache.restoration_v2_contract` fail closed 验证。validator 同时锁定
stable self-behavior reference、post-state-only intervention、八字段 strong summary、restricted action
inventory、exact 8+10+5 历史 exposure 边界、fixed 20-state confirm 与两级 gate。它不替代历史
`phase0_contract.json` validator；两者代表不同版本的 estimand，必须分别通过。

v2 executable interface 使用显式 versioned 模块 `causalcache.policy.gui_owl_v2` 与
`causalcache.low_fidelity_v2`，不修改历史 v1 parser/prompt/schema。CPU validator 对 restricted grammar、
canonical teacher target、AndroidWorld payload、八字段 serialization 和 steps 4/5/6 共 28 个 prompt
coalitions 做 fail-closed 检查，并验证 `data/manifests/restoration_v2_interfaces.json` 中的逐文件
SHA256；其中 step 6 覆盖全部 16 个 coalitions。默认输出的
`androidworld_json_action_constructor_validation.status=not_run` 是刻意的：只有显式传
`--androidworld-source-root`、`--androidworld-source-revision`、`--container-image-digest` 和
`--run-git-commit`，通过 clean-check/module-origin 验证，并让 14 个合法 payload 通过 pinned
`JSONAction(**payload)` 后，才能把 constructor integration 记为 passed。该 constructor 仍不等于真实
device-side executor dispatch；后者已由独立 formal result 闭合，见
`data/results/restoration_v2_executor_dispatch/`。两种检查都不加载 policy、不生成 v2 output。

历史 v1 replacement-teacher GUI-Owl Think 的冻结输出边界允许开头最多一个小写且闭合的 `<think>...</think>`
block；剔除后仍必须完整匹配单行 `Action:` 和唯一 `mobile_use` `<tool_call>`。未闭合、
多 block、中缀/后缀 thinking 或额外文本全部 fail closed。v2 Instruct parser 不继承该例外，任何 thinking
block 都拒绝。

正式 device-side executor 证据使用三个独立入口：

- `scripts.inspect_restoration_v2_executor_container` 在 Aries host 读取 live Docker/container/source identity；
- `scripts.validate_restoration_v2_executor_dispatch` 在绑定的 runtime container 内重新执行
  native-output parser/bridge、14 个 valid requests 和一个必须 HTTP 500 的 negative actuation control；
- `scripts.validate_restoration_v2_executor_evidence` 从 raw HTTP records 独立重算 denominator 与 verdict。

正式 run 使用 unique attempt ID、exclusive output、pre/post live inspection 和 immutable packaging；失败后
不得覆盖同一路径。canonical summary 内嵌两次 inspection 的原始 UTF-8 JSON，action/reset/health 等
compact HTTP records 也保留 raw response bytes 的 UTF-8 表示，使 reducer 能重算 length、SHA256 和
decoded body。大体积 screenshot pixels 只保留 frozen runner 计算的 transport digest 与 shape。

完整参数和 claim 边界见 `docs/restoration_v2_executor_dispatch.md`；canonical result 可用
`make validate-restoration-v2-executor-dispatch` 复核。三者都不加载 policy。

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

Qwen3-VL 的 processor 同时返回 sequence-aligned `attention_mask` 与 `mm_token_type_ids`。teacher
forcing 必须把二者与 action prefix 一起扩展：mask 填 1，新 action tokens 是文本所以 multimodal type
填 0；任何未知的 prompt-length tensor 直接拒绝，避免 3D RoPE 在静默错位的输入上继续运行。

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

独立多轨迹 artifact 用冻结的 16-file manifest 构建。CLI 会先重算全部 2.25 GB source files 的 size/SHA，
再以命名 exclusion rules 扫描 rows、做 salted trajectory 排序和 shortest-prefix split；输出是一个
deterministic tar shard，不允许覆盖非空目录：

```bash
python3 -m scripts.build_guiodyssey_independent \
  --config code/configs/independent_reference_gate_v1.json \
  --source-file-manifest data/manifests/independent_reference_gate_v1_source_files.json \
  --source-root /data/source/guiodyssey-independent-v1 \
  --output-dir /data/artifacts/causalcache-guiodyssey-independent-v1
```

artifact 上传 private HF、将 immutable revision 与 manifest/shard SHA 回写 config 并 push 后，才运行
formal UI-TARS reference gate：

```bash
python3 -m scripts.run_independent_ui_tars_reference_gate \
  --config code/configs/independent_reference_gate_v1.json \
  --dataset-tar /data/artifacts/causalcache-guiodyssey-independent-v1/data/guiodyssey-independent-00000.tar \
  --hardware-anchor-summary data/results/ui_tars_hyper00_hardware_anchor/summary.json \
  --model-dir /data/artifacts/models/UI-TARS-1.5-7B \
  --device cuda:0 \
  --output-dir /data/experiments/independent-reference-gate-v1 \
  --run-git-commit <FULL_CLEAN_MAIN_SHA> \
  --container-image-digest <SHA256_IMAGE_ID>
```

runner 在加载模型前验证 clean Git、frozen interface hashes、anchor、HF revision、tar/manifest SHA 与完整
split denominator。合法 gate failure 输出 `NO_GO_CURRENT_REFERENCE_STACK` 并正常退出；契约/运行错误写
`failure.json` 并返回 nonzero。reference 通过前禁止读取 `oracle_pilot` policy output。

restoration v2 不能直接从只含 8+15 trajectories 的 parent tar 继续选 confirm。CPU materializer 会重扫
pinned 16 个 Parquet、复现完整 111-pool SHA，并在 fixed first-20/no-top-up 规则下写 exact selection 与
append-only exposure ledger：

```bash
python3 -m scripts.materialize_restoration_v2_selection \
  --v2-contract code/configs/causalcache_restoration_v2.json \
  --v1-config code/configs/independent_reference_gate_v1.json \
  --source-file-manifest data/manifests/independent_reference_gate_v1_source_files.json \
  --source-root /data/source/guiodyssey-independent-v1 \
  --parent-manifest /data/staging/causalcache-parent/manifest.json \
  --v1-summary data/results/independent_reference_gate_v1/summary.json \
  --git-revision <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output-selection /data/tmp/restoration-v2-selection.json \
  --output-exposure /data/tmp/restoration-v2-exposure.json
```

输出采用 exclusive-create；selection 保存完整 eligible records 和 65 个 screening/confirm state content
witnesses，exposure 的 negative claim 只是 pre-output process declaration。正式产物进入 Git 后用
`scripts.validate_restoration_v2_selection` 独立复核。

2026-07-15 canonical formal run 已从 pushed `main@30879c0` 在 Hyper00 完成，并在两个空目录中
得到 byte-identical 结果。canonical 产物为
`data/manifests/restoration_v2_selection.json` 和 `data/manifests/restoration_v2_exposure.json`，SHA256 分别为
`292c7e52...` / `bc122482...`；可从仓库根目录运行 `make validate-restoration-v2-selection`
独立复核。

restoration v2 OCR/image implementation 位于
`causalcache.restoration_v2_text_backend`，冻结 config 为
`configs/restoration_v2_ocr_backend.json`，full runtime lock 为
`requirements/restoration_v2_ocr_lock.txt`。本机只做 source/config validation 时不需安装 OCR extra；
Hyper00 end-to-end 用 `scripts.validate_restoration_v2_ocr_backend inspect-golden|validate-golden`。该模块
保存 full uncapped OCR tokens，不用 capped summary delta 反推 OCR+RGB baseline。当前 model HF
revision/golden 尚 pending，详见 `docs/restoration_v2_ocr.md`。

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
