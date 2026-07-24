# AAAI-27 论文骨架

本目录使用 AAAI-27 官方 anonymous submission 模板：

- 官方入口：<https://aaai.org/authorkit27/>
- 下载文件：`AuthorKit27.zip`
- 下载日期：2026-07-14
- Author kit SHA-256：`e28c6ac9bc6eb3b4e2d849547d2cefb5162610ee39d0a12e0dc62d1126b44a7d`
- `AnonymousSubmission2027.tex` SHA-256：`035ebdb17e57885a1fd43a188fd17777bdbf90f1fda1a1e000c49c7f52ce1f9d`
- `aaai2027.sty` SHA-256：`391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623`
- `aaai2027.bst` SHA-256：`5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c`
- 官方 checklist 源 SHA-256：`06a3459158089bf1c64b738986118f1d1566e816da4b710c6397561e33c3d5e6`

编译：

```bash
make paper
```

输出：`output/pdf/causalcache_aaai27.pdf`。

`main.tex` 保持单一正文源文件，以符合 AAAI author kit 的提交要求。当前未完成的段落级结果以
`[Pending: ...]` 标记，表格单元格以 `TBD` 标记；不能在获得可复核结果前替换为经验性结论。

## AAAI-27 submission 约束

- 主稿使用 US Letter、官方 `aaai2027.sty/.bst`、`submission` 选项、匿名作者块和双栏格式；
- 非参考文献内容最多 7 页，主 PDF 最多 9 页，第 8--9 页只能放 references；
- table caption 必须放在表格下方；table body 可用 9pt，caption 保持 10pt；
- acknowledgments 在匿名投稿阶段省略，不在主稿中嵌入作者或 affiliation；
- [`ReproducibilityChecklist.tex`](ReproducibilityChecklist.tex) 是 AuthorKit27 的独立 checklist
  模板。AAAI-27 要求单独上传，因此不 `\input` 到 `main.tex`；当前仍是待填写模板。

## 当前 bibliography 状态

- `references.bib` 使用 AAAI author--year BibTeX 路径：正式会议论文采用
  `@inproceedings`，书籍章节采用 `@incollection`，仅有 arXiv 版本的工作采用
  AuthorKit27 指定的 `@misc`，并记录 `eprint`、`archivePrefix`、`primaryClass` 与
  arXiv URL；
- 当前 15 个 BibTeX entries 均在正文中至少引用一次，正文中的 15 个 cite keys 也均有
  对应条目，无重复、缺失或未使用 entry；
- `aaai2027.sty` 自动选择 `aaai2027.bst`，正文统一使用 `\citep`，无需在
  `main.tex` 中重复声明 `\bibliographystyle`。
