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

## Subset-search v1

`causalcache.subset_search` 实现 black-box set utility 上的 deterministic exact、true conditional greedy、
2x2 bounded exchange 和 beam search；`causalcache.subset_search_ablation` 绑定 controlled synthetic、既有
phase-0 与旧 v1 cached coalition table。equal-cost 时 density greedy 自动省略，所有方法统计全部 evaluated
coalitions。true-$U$ exchange/beam 只是 offline oracle-search diagnostic，不能当作 learned gate latency。

focused tests：

```bash
cd code
python3 -m unittest tests.test_subset_search tests.test_subset_search_ablation -v
```

正式 CPU run 必须先 commit/push source，并从 clean canonical `main` 运行；不需要 GPU preflight：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.run_subset_search_ablation \
  --repository-root /absolute/path/to/CausalCache \
  --config /absolute/path/to/CausalCache/code/configs/subset_search_ablation_v1.json \
  --source-git-commit <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output-dir /absolute/path/to/CausalCache/data/results/subset_search_ablation_v1
```

result commit/push 后，从 clean descendant main 复算并验证 committed summary：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_subset_search_ablation \
  --repository-root /absolute/path/to/CausalCache \
  --config /absolute/path/to/CausalCache/code/configs/subset_search_ablation_v1.json \
  --summary /absolute/path/to/CausalCache/data/results/subset_search_ablation_v1/summary.json
```

canonical result 已从 clean `main@45bcf7e` 运行该 validator 并返回
`VALID_SUBSET_SEARCH_ABLATION_V1`；命令保留为后续 source/result 变更后的强制复验入口。

完整 estimand、query accounting 与 claim boundary 见 `ablations/subset_search.md`。

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

v2.1 interface rescue 使用独立模块 `causalcache.policy.gui_owl_v2_1`、
`causalcache.policy.gui_owl_v2_1_runtime` 与 `causalcache.restoration_v2_1_contract`。它通过 pinned
processor 的 official `tools=` 注入 schema，删除旧 `Action:` carrier，并只接受模型完整生成的一组
`<tool_call>`。generation closer、标准 EOS suppression、teacher EOS finite mask、chat-template 与 token IDs
均被 contract hash 绑定。`scripts.validate_restoration_v2_1_contract` 只验证 immutable source/contract；独立
90-prompt processor evidence 现已正式通过，但只有其 Git manifest commit/push 并从 clean immutable HF
download 复核后，才授权唯一 fixed-15 pilot。

v2.1 的 processor-only 正式入口是 `scripts.audit_gui_owl_v2_1_processor`，独立重算/验证逻辑在
`causalcache.restoration_v2_1_processor_audit`。它用真实 pinned `AutoProcessor` 处理 45 states ×
reference/summary-only 共 90 prompts，验证 official tools、tensor/shape、context 与全部九种 action 的
teacher golden；teacher bytes 与 official assistant `tool_calls` Jinja/tojson 一致，包括 canonical insertion order
与原始 Unicode UTF-8，decoded text 仍须 strict NFKC。不加载 model weights，也不 forward/generate。
loader 会验证/hash 包含 confirm bytes 的完整 artifact，但交给 decoder/processor 的 confirm
state/prompt/image 为 0，
`confirm_processor_prompt_count=0`。正式 output 必须在 Git repo 外 exclusive-create；Hyper00 正式 evidence
已通过并绑定 `data/results/restoration_v2_1_processor_preflight/artifact.json`。

```bash
cd /data/CausalCache/code
python3 -m scripts.audit_gui_owl_v2_1_processor \
  --repository-root /data/CausalCache \
  --derived-artifact-root /data/tmp/causalcache-restoration-v2-derived-1a01f23-hf-redownload \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --snapshot-manifest /data/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json \
  --v2-config /data/CausalCache/code/configs/causalcache_restoration_v2.json \
  --v2-1-config /data/CausalCache/code/configs/causalcache_restoration_v2_1_pilot.json \
  --selection-manifest /data/CausalCache/data/manifests/restoration_v2_selection.json \
  --ocr-backend-config /data/CausalCache/code/configs/restoration_v2_ocr_backend.json \
  --artifact-binding-manifest /data/CausalCache/data/manifests/restoration_v2_derived_artifact.json \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id 69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --run-git-commit <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --output /data/experiments/causalcache/restoration-v2-1-processor-preflight-v1/formal-result.json
```

正式 raw JSON 先上传 private HF dataset
`gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile` 并取得 immutable revision；随后只把轻量
artifact manifest 写入 Git：

```bash
cd /data/CausalCache/code
python3 -m scripts.package_restoration_v2_1_processor_evidence \
  --repository-root /data/CausalCache \
  --raw-evidence /data/experiments/causalcache/restoration-v2-1-processor-preflight-v1/formal-result.json \
  --hf-repo gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile \
  --hf-immutable-revision <HF_COMMIT_OID> \
  --hf-path processor-preflight-v1/formal-result.json \
  --current-git-commit <PROCESSOR_AUDIT_SOURCE_SHA> \
  --output /data/CausalCache/data/results/restoration_v2_1_processor_preflight/artifact.json
```

raw prompt/input-ID evidence 不进入 Git；pilot 从 immutable artifact 的本地下载件复核 SHA、size、source commit
与 compact reduction。packager 本身不联网证明 HF revision 存在；正式流程必须在 upload/tag 后从 40-hex
immutable revision fresh download，并逐 byte 复核 manifest 中的 SHA256/size，之后才允许提交 Git manifest。

完成态为 `PASSED_RESTORATION_V2_1_90_PROMPT_PROCESSOR_PREFLIGHT`。raw JSON 位于 private HF dataset
`gavinlaw/causalcache-restoration-v2-1-processor-preflight-mobile`，tag
`v2.1-processor-preflight-v1`，immutable revision
`85576161b7cb8bbae14e46a482c42b5be5bf1d7e`；Git 结果说明见
`data/results/restoration_v2_1_processor_preflight/README.md`。fixed-15 runner 必须读取上述 immutable revision 的
fresh download，不能读取未绑定的本地 run output。

只有该 evidence 经独立 validator 通过，才能调用 fixed-15 no-retry runner
`scripts.run_restoration_v2_1_interface_pilot`。runner 只读取冻结的 15 个 `v2_development` states，每 state
先写 attempt marker 再作一次 full-history generation；interrupted marker 不得 retry。parse failure 进入科学
`NO_GO`，OOM/runtime/contract failure 进入 invalid。下列是唯一正式 attempt 已使用的 exact argv；canonical
root/ledger 已被永久 claim，不得再次执行或加 `--resume` 继续 generation：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_1_interface_pilot \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_1_pilot.json \
  --processor-preflight /data/tmp/causalcache-restoration-v2-1-processor-preflight-hf-redownload/processor-preflight-v1/formal-result.json \
  --derived-artifact-root /data/tmp/causalcache-restoration-v2-derived-1a01f23-hf-redownload \
  --scientific-config /data/CausalCache/code/configs/causalcache_restoration_v2.json \
  --selection-manifest /data/CausalCache/data/manifests/restoration_v2_selection.json \
  --ocr-backend-config /data/CausalCache/code/configs/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --device cuda:0 \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id 69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --output-dir /data/experiments/causalcache/restoration-v2-1-interface-pilot-v1
```

无论 pilot 终止为 `PASS`、科学 `NO_GO` 还是 runtime `INVALID`，都先在同一 clean source commit 上把
canonical output 与 sibling ledger 打成 deterministic USTAR；raw archive 上传 private Hugging Face dataset 后，
再在 Git 中生成轻量 manifest。正式入口如下：

```bash
cd /data/CausalCache/code
python3 -m scripts.manage_restoration_v2_1_pilot_artifact archive \
  --repository-root /data/CausalCache \
  --raw-output-dir /data/experiments/causalcache/restoration-v2-1-interface-pilot-v1 \
  --global-attempt-ledger /data/experiments/causalcache/.restoration-v2-1-interface-pilot-v1.attempt.json \
  --output /data/experiments/causalcache/restoration-v2-1-interface-pilot-v1.raw.tar \
  --source-git-commit <PILOT_SOURCE_SHA>

python3 -m scripts.manage_restoration_v2_1_pilot_artifact create-manifest \
  --repository-root /data/CausalCache \
  --raw-archive /data/experiments/causalcache/restoration-v2-1-interface-pilot-v1.raw.tar \
  --source-git-commit <PILOT_SOURCE_SHA> \
  --hf-repo gavinlaw/causalcache-restoration-v2-1-interface-pilot-mobile \
  --hf-immutable-revision <HF_COMMIT_OID> \
  --hf-path raw/restoration-v2-1-interface-pilot-v1.tar \
  --output /data/CausalCache/data/results/restoration_v2_1_interface_pilot/artifact.json
```

`validate` 子命令从 immutable HF revision 的 fresh download 或 extracted tree 重算相同证据，并要求当前 clean
`main` 是 source commit 的 descendant。这里是独立进程调用同一冻结 reducer/schema helper 重算，不声称为第二套
implementation-independent reducer。CLI 不联网证明 revision 存在，因此正式回写同样要求 fresh immutable
download 的 archive SHA256/size 与 Git manifest 完全相等。

唯一正式 attempt 已输出 `PASS_V2_1_INTERFACE_PILOT`。raw 34-file USTAR 位于 private HF dataset
`gavinlaw/causalcache-restoration-v2-1-interface-pilot-mobile`，tag `v2.1-interface-pilot-v1`，immutable
revision `bdff8ca71f150afd80d6291b4ecec76cbf9e7432`，SHA256
`f71d5fd575dde48ae8b3e50a19dd2fbecfa02d7d5ae6087f909a47dfd7032064`，133,120 bytes；fresh download
已逐 byte 复核。Git manifest commit/push 后，clean `main@6458111` 的 committed-binding validator 返回
`VALID_RESTORATION_V2_1_INTERFACE_PILOT_ARTIFACT` 与同一 PASS。结果说明见
`data/results/restoration_v2_1_interface_pilot/README.md`。该 runner 不得用于 full-45；后者必须先有新的冻结
contract/source 与独立 canonical attempt identity。

full-45 child contract 现已冻结为
`configs/causalcache_restoration_v2_1_full_45.json`，SHA256
`0924dd66fab9440bed66585765e9b1f5ab6fb80fbdf65a1492efba7ae81e116b`。source-only validation 不读取 policy
或 GPU，也不授权执行：

```bash
cd /data/CausalCache/code
python3 -m scripts.validate_restoration_v2_1_full_45_contract \
  --repository-root /data/CausalCache \
  --config /data/CausalCache/code/configs/causalcache_restoration_v2_1_full_45.json
```

唯一正式 full-45 attempt 已从 clean pushed `main` 使用下面的 absolute argv；以下命令只作为 provenance，
不得再次执行。`--pilot-evidence` 与 `--processor-preflight` 均来自各自 private HF immutable revision 的
fresh download，首次 invocation 未带 `--resume`。canonical root 与 sibling ledger 已永久 claim，不能删除、
换目录或重跑：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_1_full_45_substrate \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_1_full_45.json \
  --pilot-evidence /data/tmp/causalcache-restoration-v2-1-interface-pilot-hf-redownload/raw/restoration-v2-1-interface-pilot-v1.tar \
  --processor-preflight /data/tmp/causalcache-restoration-v2-1-processor-preflight-hf-redownload/processor-preflight-v1/formal-result.json \
  --derived-artifact-root /data/tmp/causalcache-restoration-v2-derived-1a01f23-hf-redownload \
  --scientific-config /data/CausalCache/code/configs/causalcache_restoration_v2.json \
  --selection-manifest /data/CausalCache/data/manifests/restoration_v2_selection.json \
  --ocr-backend-config /data/CausalCache/code/configs/restoration_v2_ocr_backend.json \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --device cuda:0 \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id 69f2b1742e8fd9baac5080b2b97ee1f3c7c1df520f908a4541cadde9d28194df \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --output-dir /data/experiments/causalcache/restoration-v2-1-full-45-substrate-v1
```

每 state 必须完整执行两次 reference generation；两次都 parse/closer/bridge 成功并 canonical agreement 后，
才执行 3 次 teacher forward 与 2 次 GPU KL。parse、agreement 或 non-finite 是科学失败并留在 45 分母；
contract/runtime/OOM/bridge invariant 是 `INVALID`。`--resume` 只跳过 terminal prefix；marker 无 terminal 会
永久封存为 `INVALID`，不会继续该 state。root 外 sibling ledger 的 durable high-water journal 还会让删除
marker+terminal pair 或整个 state/attempt 目录的行为 fail closed，不能通过删文件重生成已尝试 state。

terminal outcome 出现后，已在同一个 source commit 上用下列命令打 deterministic USTAR：

```bash
cd /data/CausalCache/code
python3 -m scripts.manage_restoration_v2_1_full_45_artifact archive \
  --repository-root /data/CausalCache \
  --raw-output-dir /data/experiments/causalcache/restoration-v2-1-full-45-substrate-v1 \
  --global-attempt-ledger /data/experiments/causalcache/.restoration-v2-1-full-45-substrate-v1.attempt.json \
  --output /data/experiments/causalcache/restoration-v2-1-full-45-substrate-v1.raw.tar \
  --source-git-commit <FULL_45_SOURCE_SHA>
```

raw archive 上传 private dataset
`gavinlaw/causalcache-restoration-v2-1-full-45-substrate-mobile` 的
`raw/restoration-v2-1-full-45-substrate-v1.tar`，tag `v2.1-full-45-substrate-v1`。只有取得 immutable revision、
fresh-download 并逐 byte 验证后，才可 `create-manifest` 写入
`data/results/restoration_v2_1_full_45_substrate/artifact.json`。完整 gate、artifact 与 promotion 边界见
`docs/restoration_v2_1_full_45.md`。

唯一正式 attempt 已按上述 argv 在 source
`7a5b6d5710fe4d054936b5aa474648149f725edb` 完成，结果为
`NO_GO_V2_1_FULL_45_SUBSTRATE`：45/45 strict parse，32/45 exact canonical repeat agreement，32 个
memory-sensitive states；restoration、baseline、gate training 与 confirm work 均为 0。raw 94-file USTAR
SHA256 为 `8cd53d6e56d5ad509da2af91d73d9e83b4db989ffc26aa18e1bca84e4c4f4fa4`，private HF immutable revision
为 `814506ef1450838d4bc6ed3d89fe53e0773d92fb`，fresh download 已逐 byte 复核。Git compact evidence 见
`data/results/restoration_v2_1_full_45_substrate/`。该 runner/attempt 不得再次执行；当前 contract 不授权
restoration 或 confirm。manifest push 后，clean
`main@554c51e417d702a6bc759b1f592979a4c38c5283` 的 `validate` 子命令从 fresh archive 返回
`VALID_RESTORATION_V2_1_FULL_45_ARTIFACT`、`archive_hash_verified=true` 与同一 NO-GO。

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

restoration v2 不使用上述把 full-vocabulary tensor 搬到 CPU 的历史 diagnostic 路径。
`causalcache.restoration_v2_gpu_kl` 只接受同一 CUDA device 上的 `[B,T,V]` tensor：reference
是 normalized FP32 log-probabilities，candidate 是 BF16 logits 或已归一化的 FP32 log-probabilities；
FP32 reduction 与 `[B]` 输出都保持 GPU-resident。finite input、normalization、nonnegative KL 与
finite output predicates 也在 device 上计算；invalid example 只变成 `NaN` final distance。primitive
内部不读取 validation scalar，也不传输 full tensor；调用方只读取最终 distance scalar，并在 nonfinite
时 fail closed。Python audit metadata 是静态记录，不是 device tensor value transfer。正式 batch 可用
`reference.expand(B,-1,-1)` 零拷贝复用 reference。独立 float64 CPU oracle 只能用于审计，
等价容差固定为 `atol=1e-6, rtol=1e-5`。

`causalcache.restoration_v2_batching` 使用显式 `microbatch_size=2`，按
`(image_count, sequence_length)` 精确分组、组键升序执行、组内按唯一 `input_index`
排序，并在 JSON-ready audit 中明确记录 `automatic_oom_fallback=false`。这两个模块只冻结
compute/planning semantics。

`causalcache.policy.gui_owl_v2_runtime` 验证完整 pinned model snapshot 与 Transformers source，固定单张
CUDA device、BF16 weights，以及每图 target 2560 effective visual tokens 对应的 exact
`min_pixels=max_pixels`；actual grid/tokens 仍逐图记录。
processor native assistant prefix 后追加 exact fixed carrier；carrier/tool-call boundary 若发生 tokenizer
merge 就 fail closed，distance 只覆盖 canonical `<tool_call>` token span。teacher forcing 仅接受 batch 1/2、
同一 decision state 的同一个 canonical reference action、equal image count 与 exact sequence shape；
已知 prompt-aligned fields 才能被扩展，并通过
`logits_to_keep=distance token count` 返回 GPU BF16 `[B,T,V]`。native generation 固定 batch 1、
`do_sample=false`、`max_new_tokens=256` 与 strict parser。

`scripts.audit_restoration_v2_gpu_compute` 是 policy-blind、synthetic-only CUDA runner。它要求 clean exact
Git commit、显式 CUDA device 及 host/container provenance，审计 batch 1 对独立 float64 CPU oracle、batch 2
对两次 batch 1、logits 对 pre-normalized log-probs、zero-stride reference、invalid-to-NaN，以及
single-decision-state microbatch 2/no-OOM 语义。Hyper00 单张 H200 formal run 已通过；
`scripts.validate_restoration_v2_gpu_compute_audit` 不 import runner/compute modules，而是从 run commit Git
blobs 独立复核 source hashes、standard JSON、provenance 与全部数值/规划字段。轻量 evidence 位于
`data/results/restoration_v2_gpu_compute_audit/`。这不等于真实 model runtime pass，也不关闭 execution
config；必须等 dependency-8 readiness validator 一并闭合后才能生成 v2 policy output。

`causalcache.data.restoration_v2_screening` 是 fail-closed screening view。它先调用现有 derived artifact 与
selection validators，再独立读取 canonical trajectory JSONL 和 regular-file-only USTAR，逐条绑定 manifest/
selection/image SHA；返回值只含 label-train/development 的 15 trajectories、45 states、90 images，confirm
role 无 API 可寻址。`scripts.audit_gui_owl_v2_processor` 只加载真实 pinned `AutoProcessor`，检查 deterministic
portrait/landscape、token boundaries、1/5-image 与 nested batch-2 的 exact tensor/grid/dtype accounting；权重
文件会做 SHA，但不 materialize model tensors，不 forward/generate。

`scripts.validate_restoration_v2_readiness` 是正式 policy import 前唯一授权入口。它固定 dependency-8 evidence
为 GPU summary + GPU independent validation + real processor summary，重算 processor run commit 与最终
implementation commit 的 Git blobs，并要求 clean `HEAD == origin/main`、exact 14-role source inventory、
`SCREENING_ALLOWED` 与 `CONFIRM_LOCKED`。`scripts.run_restoration_v2_substrate_screening` 只能在该授权后 dynamic
import runtime；随后先完成全部 90 prompts 的 processor-only context sweep，再执行固定 45-state denominator。
每 state 写 no-retry attempt marker，保留 parse-failure raw output；OOM、contract/shape/model/kernel error 是
fatal invalid，只有 parse、repeat-action mismatch 与 non-finite distance 进入 scientific substrate gate。

formal processor evidence 位于 `data/results/restoration_v2_processor_audit/`：Hyper00 clean commit 审计冻结
`Qwen3VLProcessor/Qwen2Tokenizer/Qwen2VLImageProcessor`、真实 portrait/landscape grids、每图 2,584 effective
visual tokens、1/5-image 与 nested batch-2 exact shapes，以及全部零 policy-output declarations。该 evidence
已通过当前 readiness parser 的 exact-value validation。完成态
`configs/restoration_v2_execution_hyper00_v1.json` 进一步绑定 8 项 evidence、14 个 source roles、真实
processor geometry、Hyper00 GPU-2 runtime 与 microbatch=2；当前 config SHA256 为
`819cb973...91ca0`。production runner 会在 artifact/model load 前实时复核 GPU UUID、visible count、driver、
compute capability、PyTorch/CUDA/cuDNN 与 Transformers。GPU-0 首次正式验证 evidence 位于
`data/results/restoration_v2_readiness/gpu0-summary.json`，仅作历史记录；GPU-2 readiness manifest 已绑定 clean
implementation commit `14faaa4...452aa`，formal clean-Git validation 已在 `main@caa4f37` 返回
`SCREENING_ALLOWED + CONFIRM_LOCKED`，canonical summary 位于
`data/results/restoration_v2_readiness/summary.json`。

第一次 45-state run 的 format diagnosis 使用独立的
`causalcache.policy.gui_owl_v2_compat` 与
`causalcache.restoration_v2_parser_replay`。compatibility parser 始终先运行原 strict parser，只对白名单中的
历史 `Action:` envelope、四种已观测 JSON wrapper 做 canonicalization，并可丢弃五种精确白名单 suffix；后者是
显式记录的 format recovery，不代表 native output well-formed，也不补全任何缺失语法。多动作、截断、
第二 JSON、observation、duplicate key、非 finite number 与未知 action 均 fail closed。replay 直接读取单个
deterministic tar 的已哈希 bytes，不按路径重开，也不向文件系统展开 raw outputs；它固定检查 96-member inventory、archive/run contract/source
commit、原 `NO_GO_V2_SUBSTRATE` aggregate 和 45 个 no-retry state record，再重跑 parser、strict round-trip 与
AndroidWorld bridge。正式 CLI 是 `scripts.replay_restoration_v2_parser_compat`，要求 clean
`HEAD == origin/main == remote main`，把实际 imported modules 与该 commit 的 Git blobs 做 pre/post 双重绑定，
并使用已提交 golden contract 固定 HF revision、0/43/40 totals、五个 rejection identities 和 45-record
classification hash。result exclusive-write；它只做离线格式复核，不 import/load/forward/generate policy，
也不能改写原 run。

上述 CLI 已从 clean pushed `main@fc3adf13d48bb016014f7efa62bd27c8a4d12f49` 对 HF immutable archive
正式运行并通过全部 pre/post identity 与 golden checks，输出 `NO_GO_ADAPTER_ONLY`。Git evidence 位于
`data/results/restoration_v2_parser_compatibility/`；其 40/45 compatibility acceptance 中只有 25 个 clean
EOF，另外 15 个执行 exact-whitelist suffix recovery，0 个包含 model-emitted canonical closer。

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
保存 full uncapped OCR tokens，不用 capped summary delta 反推 OCR+RGB baseline。private HF model
`gavinlaw/causalcache-rapidocr-ppocrv5-mobile-en@0dbc766a73ee88d10d52285d434dbfec58617835`
已 immutable-verify；`make validate-restoration-v2-ocr-artifact` 可离线 fail closed 复核 14 个 Git source 与
6 个 model files，并复核 real-screen 5-file dataset artifact。6-image real-screen golden 已在 Hyper00
两次独立构建、三次 replay（含 HF immutable re-download）通过，dependency 5 已闭合，详见
`docs/restoration_v2_ocr.md`。

real-screen pre-output 实现位于 `causalcache.data.restoration_v2_real_screen`，正式入口是
`scripts.materialize_restoration_v2_real_screen`，独立复算入口是
`scripts.validate_restoration_v2_real_screen`。运行 OCR 前先验证包含 17 个逐文件 SHA 的 source contract：

```bash
cd code
python3 -m scripts.validate_restoration_v2_real_screen source \
  --source-contract ../data/manifests/restoration_v2_real_screen_source.json \
  --repository-root ..
```

materializer 必须运行在 `HEAD == origin/main == --git-revision` 的 clean checkout，重新验证 2.25 GB 的
16 个 Parquet 后只加载 45 个 screening states 的原始截图。它按 SHA 去重、同 SHA 取最小 member path，
固定 55/20 orientation pool 中各前 3 张；confirm screenshot、policy output 与 restoration output 都不读取。
输出固定为 `.gitattributes`、`README.md`、deterministic USTAR、canonical OCR JSONL 与 provenance manifest
共 5 个文件。artifact validator 重新读取 raw source、重放 6 次 OCR、逐字节重建 USTAR/JSONL/manifest，
并给出覆盖完整 5-file tree 的 hash。正式 Hyper00 argv 见 `docs/execution.md`；canonical tag
`ocr-real-screen-golden-v1.0.0` 已固定到 HF revision
`9ebbbbbc4666e8a065f4ecb5240491c70f05e21b`，tree SHA256 为 `605d6396...7e25`。

restoration v2 non-oracle baseline 的纯公式实现位于
`causalcache.restoration_v2_baselines`：summary-only 取空集、recent 固定 events 3/4、random 是全部六个
2-of-4 subset 的解析期望（没有 seed 或 sampled selector）、OCR+RGB 使用 full uncapped OCR token set 与
256x256 RGB 的 16^3 joint histogram、policy-vision 使用 spatial-merger output 的 mean-pool/L2/cosine。
top-2 tie 按 frozen `isclose` tolerance 后取较小 event step。当前公式和纯 CPU tests 已实现；policy-vision
唯一入口位于 `causalcache.policy.gui_owl_v2_vision`：只调用 final main merger `pooler_output`，排除
pre-merger `last_hidden_state` 与全部 DeepStack features；完整 14-file model snapshot 和 Transformers 5.6.0
三份 source SHA 必须先验证。`make validate-restoration-v2-baselines` 复核 11-file source manifest；dependency 6
已闭合。

完整 GUIOdyssey restoration-v2 derived artifact 的实现位于
`causalcache.data.guiodyssey_restoration_v2`，正式入口为
`scripts.build_guiodyssey_restoration_v2` 与 `scripts.validate_guiodyssey_restoration_v2`。builder 不接受裸
OCR JSONL：它从 exact 35 条 pinned raw trajectory 重建 210 张 `observation-000..005`，验证完整 raw
tool-call/canonical executed-action 一致性，再用 pinned CPU RapidOCR runtime 按 member path 排序生成 OCR。
固定输出是逻辑上的 6-file artifact projection：root control files、deterministic image USTAR、full uncapped
OCR JSONL、35-trajectory JSONL 与 payload manifest。builder 的 post-write validation 和独立 validator 都
强制 35/175/65/210 counts、exact image inventory、runtime/source/HF identity，并重跑 210 条 OCR 逐条
compare；fixture 模式可以显式关闭 formal counts，但正式 CLI 不能。

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

## Spatial reference audit v1

该 audit 只诊断 v2.1 已暴露的 13 个 mismatch。source/config、operation budget、父 raw archive 和 exposure
ledger 见 `docs/spatial_reference_audit_v1.md`。正式 runner 每次只接受一个 frozen profile，要求
`HEAD == origin/main == --source-git-commit`、clean worktree、绝对路径和 canonical output root：

```bash
cd /data/repo/code
python3 -m scripts.run_spatial_reference_audit_v1 \
  --repository-root /data/repo \
  --config /data/repo/code/configs/spatial_reference_audit_v1.json \
  --source-git-commit <FULL_CLEAN_PUSHED_MAIN_SHA> \
  --parent-raw-archive /data/artifacts/spatial-reference-audit-v1/restoration-v2-1-full-45-substrate-v1.tar \
  --derived-artifact-root /data/artifacts/causalcache-restoration-v2-derived-v1 \
  --model-dir /data/artifacts/models/GUI-Owl-1.5-8B-Instruct \
  --device cuda:0 \
  --profile-id bf16_auto \
  --host-alias hyper01 \
  --host-hostname node-radixark-16-0001 \
  --container-id <FULL_CONTAINER_ID> \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa
```

在同一 container/device 中依次运行 `bf16_auto`、`bf16_eager_control`、`fp32_eager_control`。首个命令会
exclusive-create sibling ledger `/data/experiments/causalcache/.spatial-reference-audit-v1.attempt.json`；每个
profile/state 在 forward 前创建 durable no-retry marker，terminal record 也逐 state durable 写入。alternate
root、删除 ledger/root、profile/state retry 和顺序跳跃都必须 fail closed。中断后不得换路径或重新挑选 eager
结果，只能保留为 invalid 并另行冻结 versioned amendment。

在创建 sibling ledger 前，runner 必须核对上述 exact image digest，以及 Python `3.12.3`、PyTorch
`2.11.0+cu130` / CUDA `13.0`、cuDNN `91900`、Transformers `5.6.0`、driver `570.172.08`。固定的
behavior-changing environment-variable 名单必须全部 absent；cache 和 GPU routing variables 不受此规则影响。
auto profile 的 observed attention 必须非 eager，eager profiles 的所有 non-null observed implementations 必须
为 eager。任一项漂移都发生在 durable attempt claim 前并 fail closed。

三个 profile 都完成后，在同一 clean pushed descendant 上独立聚合：

```bash
cd /data/repo/code
python3 -m scripts.validate_spatial_reference_audit_v1 \
  --repository-root /data/repo \
  --config /data/repo/code/configs/spatial_reference_audit_v1.json \
  --bf16-auto /data/experiments/causalcache/spatial-reference-audit-v1/profiles/000-bf16_auto/terminal.json \
  --bf16-eager /data/experiments/causalcache/spatial-reference-audit-v1/profiles/001-bf16_eager_control/terminal.json \
  --fp32-eager /data/experiments/causalcache/spatial-reference-audit-v1/profiles/002-fp32_eager_control/terminal.json \
  --summary-output /data/experiments/causalcache/spatial-reference-audit-v1/summary.json
```

validator 会独立重开父 raw archive 并重建 13-state evidence，不信任 profile 中复制的父字段；它还逐个验证
attempt start、隐藏 profile claims、全部 profile/state starts 与 ledger/terminals 的 exact schema、argv、路径、顺序和
operation counts。shared-prefix 两次 forward 的 image grid/prompt/aligned-input shape 必须 exact 一致，full branches
分别验证 action-token length 且共享同一 base image/prompt shape。runner/validator 的设备名都严格要求
`NVIDIA H200`。

唯一 attempt 已完成，但上面的原 validator 因两个读取契约错误在 summary 前 fail closed；不得重跑 profile。
从新的 clean pushed `main` 只运行纯离线 repair：

```bash
cd /data/CausalCache/code
python3 -m scripts.validate_spatial_reference_audit_v1_validation_repair_v1 \
  --repository-root /data/CausalCache \
  --config /data/CausalCache/code/configs/spatial_reference_audit_v1_validation_repair_v1.json
```

repair 会把旧 config 与 33-file source inventory 逐 blob 绑定回 raw commit
`c093bd8f92ab97427acb427bd2d66fb6b20b556a`，动态按 realized grid 核算 visual tokens，并与 parent
generation witness exact 对齐。它不加载 processor/model、不运行 forward，只能 exclusive-create canonical
`summary.json`，随后复核 audit root 由 70 增至 71 个文件、加入 sibling ledger 后 package inventory 为 72。

validator 完成后，用冻结 packager 将 canonical root 与 root 外 sibling ledger 一并封装；所有路径和 source commit
都显式传入，archive 只能 exclusive-create，不能覆盖或换名重试：

```bash
python3 -m scripts.package_spatial_reference_audit_v1 \
  --repository-root /data/repo \
  --config /data/repo/code/configs/spatial_reference_audit_v1.json \
  --source-git-commit c093bd8f92ab97427acb427bd2d66fb6b20b556a \
  --audit-root /data/experiments/causalcache/spatial-reference-audit-v1 \
  --global-attempt-ledger /data/experiments/causalcache/.spatial-reference-audit-v1.attempt.json \
  --output /data/experiments/causalcache/spatial-reference-audit-v1.tar
```

packager 拒绝 symlink、FIFO/device/socket 等 non-regular member，按路径排序并将 USTAR metadata 统一为
mode `0644`、uid/gid/mtime `0`；写前和写后都会重新读取全部 members 并重建 canonical bytes，要求 byte identity。
raw archive 随后上传 private HF dataset
`gavinlaw/causalcache-spatial-reference-audit-mobile@spatial-reference-audit-v1` 并 fresh immutable download；
canonical revision 为 `d6b2312e458ce3b2b1dc8463a323a8d7dbc945c1`，archive SHA256 为
`d62ad05f6fdef06a3551f2ebe9f83f28327068f0020ff3e61891da4f46ce5ecc`。Git 只回写 compact
summary/artifact binding。严格 CUDA deterministic mode 需要项目禁止的 scientific environment
variable，因此 profile 只声称 eager fixed-seed/TF32-off numerical control，不声称数学确定性。所有命令的
confirm/restoration/gate operation count 必须为 0。

最终判定为三分支：eager 不稳定则进入 semantic reference；只有 eager 13/13 且 auto 非 13/13 才称为
`EAGER_SPECIFIC_RECOVERY_OF_EXACT_STABILITY`；若两者均 13/13，则结论是本次 numerical audit inconclusive，
不得把稳定性归因给 eager。FP32 永不参与 pass/fail。本次 repaired artifact 落在 eager-specific 分支。

## Restoration v2.2-eager source freeze 与正式结果

`configs/causalcache_restoration_v2_2_eager.json` 冻结 fresh-45 child contract。它只继承 v2.1 full-45 的
official-tool interface、prompt/parser/teacher/KL、45-state projection、gate 与 90/135/90 operation ceiling，并将
runtime 固定为 BF16 eager、seed 0、TF32 off、cuDNN deterministic on/benchmark off 和 float32 matmul
`highest`。同时 exact 绑定 spatial audit 的 container image、Python/PyTorch/CUDA/cuDNN/Transformers/driver 与
H200 stack。这不是 strict CUDA determinism，也不能把 spatial audit 的 13 个 mismatch 直接计为本轮 stable。

execution 固定同一 Hyper H200 host/container 内两个 worker：logical `cuda:0` 处理 even indices 0--44 的
23 states，logical `cuda:1` 处理 odd indices 1--43 的 22 states；禁止 state stealing。coordinator 必须在两个
worker import policy runtime 前 exclusive-create 独立于 v2.1 的 global sibling ledger
`/data/experiments/causalcache/.restoration-v2-2-eager-full-45-substrate-v1.attempt.json`，并绑定 canonical root
`/data/experiments/causalcache/restoration-v2-2-eager-full-45-substrate-v1`。旧 v2.1 root、ledger、state records、
terminal、aggregate 与 raw archive 都不能导入、复制或计入新 45-state denominator。

两张物理卡由 preflight 选择后映射为 container logical `cuda:0/1`；container-visible `nvidia-smi` index 不要求
等于 0/1，但必须是两个不同非负整数，且 UUID/PCI bus ID 都不同。两个 worker 写完 runtime identity 后先在
coordinator barrier 中验证 exact stack 与 distinct physical identity，barrier release 前不得写 marker 或执行
generation。generation success 与 strict-parse failure 都只额外记录实际 `device` provenance，不改变 v2.1
output/action/logits。

source commit push 后，唯一可建立 formal source freeze 的 validator 命令必须显式带 strict flag：

```bash
cd /data/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_eager_contract \
  --repository-root /data/CausalCache \
  --config /data/CausalCache/code/configs/causalcache_restoration_v2_2_eager.json \
  --require-clean-pushed-main
```

production runner 的固定参数面如下；尖括号只在 preflight 后替换为 fresh immutable evidence path、实际 model
path 与 64-hex container ID，canonical output/ledger 和 image digest 不能改：

```bash
cd /data/CausalCache/code
python3 -m scripts.run_restoration_v2_2_eager_substrate \
  --repository-root /data/CausalCache \
  --contract /data/CausalCache/code/configs/causalcache_restoration_v2_2_eager.json \
  --spatial-reference-evidence <FRESH_SPATIAL_RAW_TAR> \
  --v2-1-full-45-evidence <FRESH_V2_1_FULL_45_RAW_TAR> \
  --pilot-evidence <FRESH_V2_1_PILOT_RAW_TAR> \
  --processor-evidence <FRESH_PROCESSOR_FORMAL_RESULT_JSON> \
  --derived-artifact-root <VALIDATED_DERIVED_ARTIFACT_ROOT> \
  --scientific-config /data/CausalCache/code/configs/causalcache_restoration_v2.json \
  --selection-manifest /data/CausalCache/data/manifests/restoration_v2_selection.json \
  --ocr-backend-config /data/CausalCache/code/configs/restoration_v2_ocr_backend.json \
  --snapshot-manifest /data/CausalCache/code/configs/gui_owl_1_5_8b_snapshot.json \
  --model-dir <GUI_OWL_1_5_8B_INSTRUCT_MODEL_DIR> \
  --host-alias hyper00 \
  --host-hostname node-radixark-16-0000 \
  --container-id <FULL_64_HEX_CONTAINER_ID> \
  --container-image-digest sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa \
  --output-dir /data/experiments/causalcache/restoration-v2-2-eager-full-45-substrate-v1 \
  --global-ledger /data/experiments/causalcache/.restoration-v2-2-eager-full-45-substrate-v1.attempt.json
```

v2.2 没有 `--resume`。若 coordinator 在 aggregate 前硬中断，只能用下列 policy-free 命令封存 existing
attempt；它读取 root 外 sibling high-water，不 import model、不补 state：

```bash
python3 -m scripts.manage_restoration_v2_2_eager_artifact seal-interrupted \
  --raw-output-dir /data/experiments/causalcache/restoration-v2-2-eager-full-45-substrate-v1 \
  --global-attempt-ledger /data/experiments/causalcache/.restoration-v2-2-eager-full-45-substrate-v1.attempt.json
```

terminal 后先执行 `archive`，上传 private HF，再把 canonical source archive 与不同路径的 fresh immutable
download 同时交给 `create-manifest`；同一路径、byte drift、非 frozen repo/path 或非 40-hex revision 都会拒绝。

source validator 不加载 processor/model、不调用 GPU，也不授权正式 attempt。
formal freeze 已在 clean pushed `main@b3a6303d69b1145fbf195e0bd18b9b3065a6f213` 建立：config SHA256
`f473bb8a1657072235dd73bf78a93aff27b438d7baca65b7ef6096cf985effa7`，50-file formal inventory SHA256
`bb6351e4dd5b4470ed1add86dbcbaa714a56063ba8ecf07268b6f13f630f9e54`。唯一正式 attempt 的 execution source
为 `main@8ae07519f14ac3635f292ee93a7b6d624507427e`，结果是 45/45 parse、45/45 exact repeat、45/45 finite
logits、45 memory-sensitive states，正式 `PASS_V2_2_EAGER_FULL_45_SUBSTRATE`。generation/teacher/KL 为
90/135/90；restoration、gate 与 confirm output/count 仍全为 0。

102-file raw USTAR SHA256 为 `b22827e6e2d8d33b03686fc177dc8f9c55c5133470fbe40f9fb3e33cce809fb5`，
private HF revision 为 `3577099d505b8c652d764f41269df911128ec767`，fresh immutable download 已逐 byte
复核。Git compact result 见 `data/results/restoration_v2_2_eager_full_45_substrate/`，完整边界见
`docs/restoration_v2_2_eager.md`。

## Restoration v2.2 label source freeze

`configs/causalcache_restoration_v2_2_labels.json`、
`scripts.validate_restoration_v2_2_label_contract`、`scripts.run_restoration_v2_2_labels` 与
`scripts.manage_restoration_v2_2_label_artifact` 已在 clean pushed
`main@3942d687d03bf63ea683fe8ad906a161eb10dc27` 冻结。contract SHA256 为
`56b29f6879ef14167b20a39d0d61ebd0e698e3bbf0457062b63453180e71cf87`，29-file source inventory
SHA256 为 `8d0ecf6b0df75f9fa7753631b8fca5efd98c71a7953007e684393e6e8fe52d9b`。source-only validation：

```bash
cd /data/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_label_contract \
  --contract configs/causalcache_restoration_v2_2_labels.json \
  --repository-root ..
```

冻结的正式 schedule 为 45 states、420 条 raw `D(S)`、45 个 primary exact-subset oracle、435 条
deployment conditional-marginal labels、465 次 teacher forward、420 次 GPU KL 与 0 generation。双 H200
worker 固定 23/22 parity，microbatch 固定为 1；confirm、gate training、matched-NLL 与 closed-loop counter
必须为 0。canonical root、sibling ledger 与 raw archive 分别为：

```text
/data/experiments/causalcache/restoration-v2-2-eager-labels-v1
/data/experiments/causalcache/.restoration-v2-2-eager-labels-v1.attempt.json
/data/experiments/causalcache/restoration-v2-2-eager-labels-v1.raw.tar
```

正式 runner 的完整显式 argv、fresh immutable parent/derived inputs 与 lifecycle 见
`docs/restoration_v2_2_labels.md`；不得通过 alternate path、resume、retry 或 top-up 绕过唯一 attempt。terminal
后才可用 artifact manager 生成 deterministic USTAR、上传 planned private HF dataset
`gavinlaw/causalcache-restoration-labels-mobile@v2.2-eager-train-dev-exact-v1` 并从 immutable revision fresh
download 验证。formal v1 attempt 已因指定 model snapshot directory 不存在而 fail closed；两个 worker 都在
state marker、teacher forward 与 KL 前退出，attempted/completed states 为 0/0，原 identity 不得 retry/resume。
没有 raw labels 或 HF repo/revision；compact failure binding 位于
`data/results/restoration_v2_2_eager_labels_v1_attempt/`。replacement 必须使用新 attempt identity，并把 model
snapshot existence/revision/inventory validation 前移到 durable claim 之前。source-freeze 完整回归为 575 tests
OK（skipped 11），`make paper` 通过。

## Restoration v2.2 label v2 pre-claim repair

replacement contract 是 `configs/causalcache_restoration_v2_2_labels_v2_repair.json`，SHA256 为
`7fc96883b905a5f026c18e65c3c7e377972559c775ac7ec95030ea1f04c2f94c`。它保持 v1 的 scientific estimand、
45-state denominator、coalition enumeration、23/22 parity 与 465-forward schedule 不变，只冻结新的 execution
identity：

```text
attempt: restoration-v2-2-eager-labels-v2
revision: v2_preclaim_repair
output: /data/experiments/causalcache/restoration-v2-2-eager-labels-v2
ledger: /data/experiments/causalcache/.restoration-v2-2-eager-labels-v2.attempt.json
archive: /data/experiments/causalcache/restoration-v2-2-eager-labels-v2.raw.tar
HF tag/path: v2.2-eager-train-dev-exact-v2 / raw/v2.2-eager-train-dev-exact-v2.tar
```

v2 authorization 固定 `/data/artifacts/models/GUI-Owl-1.5-8B-Instruct`。在 global ledger exclusive-create 之前，
`validate_model_snapshot_preclaim` 要求 root 为非 symlink real directory、repo slug 与 basename 一致、`.snapshot.json`
与 Git manifest 完全一致，并复用 `verify_frozen_vision_runtime` 逐 SHA 验证 14 files / 17,545,907,171 bytes 与
Transformers source。missing path、wrong basename、symlink、partial shard 均不得创建 output/ledger。runner、spawn
worker rebuild、state/worker/global terminal、aggregate、USTAR prefix 与 artifact manifest 全部按 v1/v2 profile
验证，避免 replacement 结果泄漏 v1 identity。完整回归 580 tests PASS（skipped 11），targeted repair/runner/
artifact tests 21/21 PASS，`compileall` 与 `git diff --check` 通过；formal v2 GPU attempt 尚未运行。

## Restoration v2.2 selector geometry

45-state v2 replacement label run 已完成并上传 private HF immutable artifact。训练 gate 前先运行独立的
policy-free selector geometry。parent v1 contract 是
`configs/causalcache_restoration_v2_2_selector_geometry.json`，SHA256 为
`8022dcdec272916b7975d696a3ce6b54022c7414cd348a715c55b0d3d694dad5`。第一次 replay 的核心 selector
数值正确，但 reporting-completeness audit 发现 interaction joint cells 与 analytic-random 集合几何没有完整
落盘，旧 output 未提交。versioned repair contract 是
`configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json`，SHA256 为
`2d312f54559f67aafe7efec2656d23171000e8b0f41d2d3c923f6a3c8b43be4c`。source-only validator：

```bash
cd /absolute/path/to/CausalCache/code
python3 -m scripts.validate_restoration_v2_2_selector_geometry_v2_contract \
  --contract configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json \
  --repository-root ..
```

本分析固定 45 states / 15 trajectories，逐 `n=2/3/4` 报告 `B=0..n`；primary slice 是
`n=4,B=2`，`n=2,B=2` 只作无压缩 ceiling。selector matrix 包含 exact at-most-B、true conditional greedy、
budget-conditioned independent、full-path Shapley independent、dynamic recent、analytic random，以及
exact-cardinality/forced-fill sensitivity。统计先在 trajectory 内等权，再让 trajectory 等权；paired bootstrap
固定 10,000 次、seed 271828、90% percentile interval。

本阶段只读取 immutable raw $D(S)$ 并做 CPU reduction；policy/generation/teacher/KL、gate training、matched-NLL、
closed-loop、confirm/test access 和 policy-vision feature forward 均固定为 0。OCR/RGB 与 policy-vision baseline
只允许在 primary slice 的后续独立 feature stage 中加入，policy-vision 还必须先做新的 feature-only source freeze。

source commit push 后运行：

```bash
python3 -m scripts.run_restoration_v2_2_selector_geometry_v2 run \
  --repository-root /absolute/path/to/CausalCache \
  --contract /absolute/path/to/CausalCache/code/configs/causalcache_restoration_v2_2_selector_geometry_v2_repair.json \
  --labels-archive /fresh/immutable/raw/v2.2-eager-train-dev-exact-v2.tar \
  --source-git-commit <CLEAN_PUSHED_MAIN_SHA> \
  --output-dir /absolute/path/to/CausalCache/data/results/restoration_v2_2_selector_geometry_v2_repair
```

result commit/push 后把 subcommand 改为 `validate`，会从 raw table 重算并逐 byte 验证三份 Git result files。
canonical v2 repair 已从 clean pushed `main@9a4eca5a53c2a9a3340c6274b9fa5ff9012a5a64` 生成，pre-commit
replay 返回 `VALID_RESTORATION_V2_2_SELECTOR_GEOMETRY_V2_REPAIR`。结果位于
`data/results/restoration_v2_2_selector_geometry_v2_repair/`；180 rows、144 joint cells（83 nonempty / 61
empty）和全部 0-operation declarations 已闭合。提交 result 后仍须在 clean descendant `main` 再运行同一
`validate` 命令。
