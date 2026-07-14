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
