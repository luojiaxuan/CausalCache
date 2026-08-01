# AAAI-27 submission packages

Supplement 与 code package 从分支 `jaxan/trusting-zhukovsky-00d2bf` 的
source commit `1c85026ef66e79925861de6bacc04245376244a0` 生成。LaTeX source
package 从同一分支的 commit `3bdbeef33e73024b897f6a7f3d7ea448d72df4bd`
生成。

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
- `output/submission/causalcache_aaai27_latex.zip`：匿名、可独立编译的
  LaTeX source archive；
- `output/submission/SHA256SUMS.txt`：三份产物的 SHA-256。

Code ZIP 共 1,023 个成员：1,005 个 Python 源文件、11 个白名单 JSON config、
4 个 requirements/metadata 文本文件，以及 README、`pyproject.toml` 和完整性
manifest。封包器拒绝顶层 `data/`、`docs/`、`paper/`、`output/`、`.git/`，并拒绝
CSV/JSONL/Parquet、图片、PDF、模型权重、checkpoint、压缩包与运行产物。
`code/causalcache/data/*.py` 只包含 dataset interface Python 源码，不包含数据记录。

LaTeX ZIP 保留一个 `causalcache_aaai27_latex/` 根目录，包含 `main.tex`、
`supplement.tex`、`ReproducibilityChecklist.tex`、`references.bib`、官方
`aaai2027.sty/.bst`，以及正文实际引用的四份 `figures/*.pdf`。它排除归档草稿
`main_v2.tex`、可编辑图源、构建脚本、README、aux/log/fls/bbl、已编译的
main/supplement/checklist PDF、代码和数据。

## 验证记录

- focused tests：`31 passed, 2 skipped`；
- ZIP manifest、source revision、匿名标识与禁止路径/后缀检查：通过；
- supplement LaTeX：无 overfull、undefined reference 或 citation；
- PDF：Letter、5 页、仅 Type 1 字体，无 Type 3；逐页渲染检查无裁切或重叠；
- PDF 文本核验包含更正后的逐轮计数、memory-critical `+11.3pp
  [+2.7,+20.4]`，以及描述性 interaction `+10.1pp [-2.2,+22.9]`,
  `p=0.114`。
- 在全新的临时目录中解压 LaTeX ZIP 后，`main.tex`、`supplement.tex` 和
  `ReproducibilityChecklist.tex` 均通过 `latexmk -pdf`；main 为 9 页、
  supplement 为 5 页，main/supplement 无 overfull、undefined reference 或
  undefined citation，也没有原仓库绝对路径依赖。Checklist 保留官方模板自身的
  incomplete `\\iftrue` warning，仍能正常生成 2 页 PDF。

重建命令：

```bash
make supplement
make submission-code
```

LaTeX ZIP 使用上方列出的精确白名单，从 `paper/` 复制到临时根目录后以
`zip -X -9 -r` 打包；不得直接压缩整个 `paper/` 目录。
