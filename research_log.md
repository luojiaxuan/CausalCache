# CausalCache Research Log

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
