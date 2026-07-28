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
make supplement
# 或同时生成两份 PDF
make paper-all
```

输出：

- 主稿：`output/pdf/causalcache_aaai27.pdf`；
- 独立 supplementary document：
  `output/pdf/causalcache_aaai27_supplement.pdf`。

`main.tex` 保持单一正文源文件，以符合 AAAI author kit 的提交要求。当前未完成的段落级结果以
`[Pending: ...]` 标记，表格单元格以 `TBD` 标记；不能在获得可复核结果前替换为经验性结论。
`supplement.tex` 是单独上传的 supplementary PDF，不被 `main.tex` 引入，也不计入主稿
7 页 technical content。当前 supplementary 为 2 页，包含完整 per-budget gate、
selector ablation、MobileWorld per-round/split 与 OSWorld paired 统计。

## 主图

- Figure 1：`figures/causalcache_overview.{svg,pdf,png}`，7 inch 双栏宽，
  解释 complete summary trace、固定 `B=4` 的 fidelity reallocation、HGKV 和
  CausalCache-P/LA 部署路径；
- Figure 2：`figures/causalcache_qualitative_cart.{svg,pdf,png}`，使用真实
  `CartInfoNotificationTask` r2 轨迹，展示 `[11,12,13,14] → [6,7,8,12]`
  与 realized `k=3`；图按 event trace → 双路径 allocation → event 7 像素恢复
  → 短信 current screen → 真实 action/result 组织，不使用三面板解释框；
- 两图由 `figures/build_causalcache_figures.py` 生成，`make figures` 可独立重建；
- SVG 是可编辑源；投稿 PDF 经 Ghostscript 将文字转成矢量轮廓，以满足 AAAI
  的零 Type 3 字体要求；
- 证据、crop provenance、hash 和灰度/字体检查见
  [`../docs/paper_figures_v1.md`](../docs/paper_figures_v1.md)，原子 claim ledger 见
  [`../docs/paper_claim_evidence.tsv`](../docs/paper_claim_evidence.tsv)。

## 当前论文层级

- 顶层只保留 `Introduction`、`Related Work`、`Method`、`Experiments` 和
  `Limitations and Conclusion`；
- problem formulation 与方法组件归入 `Method`，实验设置、主结果和 ablation 归入
  `Experiments`，避免官方 AAAI 居中 section 标题把论文切得过碎；
- 当前版本仍是随 `main` 与 ablation 结果更新的 working draft，不冻结 claim 或版式。

## 主结果表契约(2026-07-27 起,固定预算 fidelity reallocation)

- 当前 working claim 是条件式的(见 `main.tex` Limitations 引文块与仓库根 README):
  不主张 sparse 总优于 recent,主张条件边际有时更高且方法学会识别;
- Figure 1 定义 complete summary trace 下的 fixed-`B` 比较；Recent-`B` 与
  CausalCache 始终各有 `B` 张 active history images，只改变哪些事件被提升为
  summary-plus-image；`B=4` 是主设置，`B=8` 是未训练预算外推；
- `Table~\ref{tab:per-r}` 按 `B ∈ {1,2,4,8}` 报告 matched-budget allocation，
  行内 selected-vs-recent 是主比较，跨 `B` 只作分析；realized `k` 必须单独报告；
- policy 侧 DiD gate 表(`Table~\ref{tab:did-gate}`)已填真实 r=0 数值
  (HGKV/ungated/Full-LoRA 三行,dev 94 组);per-r 分层与零样本表保持 `TBD`
  直至可复核结果落地;
- selector 行按实际使用的图片数计费,STOP 是合法输出;不得把 singleton gain
  求和冒充 set utility。

## AAAI-27 submission 约束

- 主稿使用 US Letter、官方 `aaai2027.sty/.bst`、`submission` 选项、匿名作者块和双栏格式；
- 非参考文献内容最多 7 页，主 PDF 最多 9 页，第 8--9 页只能放 references；
- AAAI-27 允许单独上传 supplementary document，长度不计入主稿页数，但 reviewer
  无义务阅读，所以核心定义、主比较和 headline evidence 仍留在 `main.tex`；
- AuthorKit27 明确禁止 `hyperref`、`navigator` 以及任何嵌入链接/书签的 package；
  因而正文的 author--year citation（如 `(Srivastava 2026; Liu et al. 2026)`）
  按官方要求不做成到 References 的可点击链接；
- references 必须从正文自然流入，不能用 `\clearpage`、`\newpage` 或其他手工分页命令；
- table caption 必须放在表格下方；table body 可用 9pt，caption 保持 10pt；
- acknowledgments 在匿名投稿阶段省略，不在主稿中嵌入作者或 affiliation；
- [`ReproducibilityChecklist.tex`](ReproducibilityChecklist.tex) 是 AuthorKit27 的独立 checklist
  模板。AAAI-27 要求单独上传，因此不 `\input` 到 `main.tex`；当前仍是待填写模板。

## 当前 bibliography 状态

- `references.bib` 使用 AAAI author--year BibTeX 路径：正式会议论文采用
  `@inproceedings`，书籍章节采用 `@incollection`，仅有 arXiv 版本的工作采用
  AuthorKit27 指定的 `@misc`，并记录 `eprint`、`archivePrefix`、`primaryClass` 与
  arXiv URL；
- 当前 43 个 BibTeX entries 均在正文中至少引用一次，覆盖 web/GUI agent benchmark、
  agent memory、长上下文与视觉 token compression，以及集合效用建模；正文中的
  43 个 cite keys 也均有对应条目，无重复、缺失或未使用 entry；
- `aaai2027.sty` 自动选择 `aaai2027.bst`，正文统一使用 `\citep`，无需在
  `main.tex` 中重复声明 `\bibliographystyle`。
