# AAAI-27 supplement 与 code package

本次提交包从分支 `jaxan/trusting-zhukovsky-00d2bf` 的 source commit
`1c85026ef66e79925861de6bacc04245376244a0` 生成。

## Arm 标签更正

- CausalCache = HGKV+selector：三轮均为 full=`43/117`、
  memory-critical=`19/62`、control=`24/55`，均值
  `36.8%/30.6%/43.6%`；
- Frozen+selector：三轮 full=`39/117, 42/117, 38/117`，均值
  `33.9%/28.0%/40.6%`；
- 历史 `frozensel` 文件名实际对应第一行，历史
  `mobileworld_hgkv_selected_b4` 目录实际对应第二行。历史名称仅保留 provenance，
  不再作为 arm 语义来源。

## 最终产物

- `output/submission/causalcache_aaai27_supplement.pdf`：5 页 Letter PDF；
- `output/submission/causalcache_aaai27_code.zip`：匿名 code-only archive；
- `output/submission/SHA256SUMS.txt`：两份产物的 SHA-256。

Code ZIP 共 1,023 个成员：1,005 个 Python 源文件、11 个白名单 JSON config、
4 个 requirements/metadata 文本文件，以及 README、`pyproject.toml` 和完整性
manifest。封包器拒绝顶层 `data/`、`docs/`、`paper/`、`output/`、`.git/`，并拒绝
CSV/JSONL/Parquet、图片、PDF、模型权重、checkpoint、压缩包与运行产物。
`code/causalcache/data/*.py` 只包含 dataset interface Python 源码，不包含数据记录。

## 验证记录

- focused tests：`31 passed, 2 skipped`；
- ZIP manifest、source revision、匿名标识与禁止路径/后缀检查：通过；
- supplement LaTeX：无 overfull、undefined reference 或 citation；
- PDF：Letter、5 页、仅 Type 1 字体，无 Type 3；逐页渲染检查无裁切或重叠；
- PDF 文本核验包含更正后的逐轮计数、memory-critical `+11.3pp
  [+2.7,+20.4]`，以及描述性 interaction `+10.1pp [-2.2,+22.9]`,
  `p=0.114`。

重建命令：

```bash
make supplement
make submission-code
```
