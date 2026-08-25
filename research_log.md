# CausalCache Research Log

## 2026-08-24：arXiv 2608.22577 replacement

- 假设：用户指定的 `causalcache_main_eq_r13.pdf` 对应历史 HGKV 正文，而不是 2026-08-23
  首次提交所用的后续正文；replacement 必须先恢复精确源，再只改作者、单位和邮箱。
- 定位：PDF 元数据与 Git 时间线锁定 `jaxan/trusting-zhukovsky-00d2bf` 分支的 commit
  `1dfd5b0a6aa0cb7c2ca8576c7d42fd862a6baf88`。从该 commit 冷编译所得 9 页 PDF 与用户 PDF
  文件大小相同，抽取文本逐 token 一致。
- 变更：恢复该 commit 的 `paper/main.tex` 与 4 张图；作者固定为 Jiaxuan Luo、Zhanfeng Liao、
  Jiayao Teng、Yuan Wang，删除 Haojian Huang；Jiaxuan/Jiayao 标注 Johns Hopkins University，
  Zhanfeng/Yuan 标注 Tsinghua University；邮箱使用 JHU `@jhu.edu` 与清华学生邮箱
  `@mails.tsinghua.edu.cn`。同步更新 arXiv 作者与摘要 metadata，使其与恢复正文一致。
- 验证：本地预览、最小源码包冷编译及 arXiv TeX Live 2025 服务器编译均为 9 页 US Letter；
  服务器状态 `SUCCEEDED`，首页 4 位作者、单位和邮箱正确，无 Haojian Huang。公开文章仍为
  [arXiv:2608.22577](https://arxiv.org/abs/2608.22577)。
- 发布：2026-08-24 PDT 正式提交 `submit/7989797`，类型为 `Replacement of 2608.22577`；
  primary `cs.AI`、无 cross-list、arXiv non-exclusive license，当前状态 `processing`。

## 2026-08-23：arXiv 作者版

- 假设：`main` 上的 761 行 `paper/main.tex` 是当前正文，可在不分叉内容的前提下生成实名 preprint。
- 变更：增加条件式 `preprint` 入口、5 位作者、独立预览 target 与最小 arXiv 源码包脚本；AAAI 默认入口继续匿名。
- 验证：预览与源码包冷启动均为 9 页 US Letter、45,146 个抽取文本字符完全一致；5 位作者齐全，
  无 Anonymous、Pending、TBD、缺失引用、缺失交叉引用或 overfull box；19 个字体全部嵌入，
  无 Type 3、annotation、outline，逐页渲染检查无裁切或重叠；默认 `main.tex` 另行编译确认仍为匿名且不泄露作者名。
- 决策：保留单一正文源；单位信息在作者确认前不猜测、不写入。
- 发布：2026-08-23 以 `submit/7983880` 正式提交 arXiv；primary `cs.AI`、无 cross-list、
  arXiv non-exclusive license。arXiv TeX Live 2025 编译 9 页成功，服务器 PDF 首页作者正确；
  当前状态 `submitted`，公开 identifier 待审核/公告后回填。
