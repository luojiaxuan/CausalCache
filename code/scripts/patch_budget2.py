#!/usr/bin/env python3
"""让预算 B 可由环境变量控制(v2,修 v1 的正则错位)。

# note (luojiaxuan): v1 用 `GUIOwl\\([^)]*\\)` 匹配,`[^)]*` 在遇到**内层**
# GUIOwlWrapper(...) 的第一个 ')' 就停了,于是把 last_image 插进了 Wrapper 的
# 参数表 -> TypeError。教训:嵌套括号不能用 [^)]* 匹配,要么括号计数,要么
# 锚定行尾。这里改为**按行处理 + 定位最后一个 ')'**,并断言插入后语法合法。
#
# 背景(v1 注释保留):官方 runner 没有 --last_image 命令行 flag,它是
# GUIOwl.__init__ 的构造参数。此前用该 flag 启动的 B=0 与 B=11 臂全部
# FATAL 退出,统计到的是残留旧数据 —— "recent-B11 33.3%" 已作废。
# 语义:last_image 含当前帧,B = last_image - 1(B=0->1, B=2->3, B=4->5)。
"""
import ast
import pathlib
import sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/run_official.py")
s = p.read_text(encoding="utf-8")

if "CC_LAST_IMAGE" in s:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

lines = s.splitlines(keepends=True)
hit = None
for i, ln in enumerate(lines):
    if "agent = gui_owl.GUIOwl(" in ln:
        hit = i
        break
if hit is None:
    raise SystemExit("ANCHOR_NOT_FOUND")

ln = lines[hit]
close = ln.rstrip().rfind(")")
if close < 0:
    raise SystemExit("NO_CLOSING_PAREN_ON_LINE")
新 = (ln[:close]
      + ", last_image=int(__import__('os').environ.get('CC_LAST_IMAGE', '5'))"
      + ln[close:])
lines[hit] = 新
out = "".join(lines)
ast.parse(out)          # 语法自检:坏补丁绝不落盘
p.write_text(out, encoding="utf-8")
print("BUDGET_PATCH_V2_OK")
print("  ->", 新.strip()[:150])
