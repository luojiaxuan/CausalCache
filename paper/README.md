# AAAI-27 论文骨架

本目录使用 AAAI-27 官方 anonymous submission 模板：

- 官方入口：<https://aaai.org/authorkit27/>
- 下载文件：`AuthorKit27.zip`
- 下载日期：2026-07-14
- Author kit SHA-256：`e28c6ac9bc6eb3b4e2d849547d2cefb5162610ee39d0a12e0dc62d1126b44a7d`
- `aaai2027.sty` SHA-256：`391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623`
- `aaai2027.bst` SHA-256：`5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c`

编译：

```bash
make paper
```

输出：`output/pdf/causalcache_aaai27.pdf`。

`main.tex` 保持单一正文源文件，以符合 AAAI author kit 的提交要求。当前所有未完成实验结果均以 `[Pending: ...]` 显式标记，不能在获得可复核结果前替换为经验性结论。
