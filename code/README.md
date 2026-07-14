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
