# Data 目录

本目录只保存适合 Git review 的小数据与轻量实验记录，不是大型 artifact 仓库。

```text
data/
├── fixtures/    # 单测和 contract 使用的最小确定性 fixture
└── results/     # README、summary.json、轻量 CSV/manifest
```

以下内容禁止作为唯一副本留在 Git 或共享机器本地盘：raw screenshots、完整 rollout traces、生成
dataset、model weights、checkpoint、adapter。它们以 Hugging Face dataset/model repo 为 canonical
source，并在顶层 `README.md` 和相关 result README 中记录 repo、revision/tag、schema 与生成命令。

`data/cache/`、`data/raw/`、`data/staging/`、`data/local/` 与 `data/tmp/` 已被 Git 忽略，只能用于
本地短期 staging。注意：仓库相对路径 `data/` 与容器持久挂载点绝对路径 `/data` 是两个不同概念。

当前 reusable artifacts：

- GUIOdyssey pilot：private HF dataset
  `gavinlaw/causalcache-guiodyssey-pilot-mobile@1de9c34ff029d4c01665cdaca74436ae24bff276`；
- GUI-Owl AndroidWorld validation traces：private HF dataset
  `gavinlaw/causalcache-androidworld-validation-mobile@v0.2.0`
  (`0faf767e7c1f64b5f39fde1ac6913ca93337d8f2`)；旧 Instruct artifact 保留在 immutable
  `v0.1.0`。

最新轻量运行记录：

- `results/ui_tars_hyper00_hardware_anchor/`：Hyper00 H200 精确复现旧 A6000 UI-TARS 9/9 parsed、4/9
  executable-match vector，允许独立 reference gate 留在 Hyper00；

- `results/gui_owl_1_5_8b_think_smoke_strict/`：Hyper01 首次 1/5-image Think checkpoint smoke；
  finite logits 通过，strict parser 因闭合 thinking prefix 按设计拒绝。
- `results/gui_owl_1_5_8b_think_smoke/`：format-only parser 适配后的原参数重跑；finite logits
  与 parse 均为 2/2，interface smoke 通过。
- `results/gui_owl_1_5_8b_think_androidworld_validation/`：Aries 42-checkpoint 数学 early-stop；
  parse 512/513，但 official-success 上界 29/62，candidate 被有效拒绝。
