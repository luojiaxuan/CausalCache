#!/usr/bin/env python3
"""把官方 GUI-Owl agent 的帧保留逻辑换成可插拔策略(唯一改动面)。

# note (luojiaxuan): 2026-08-23 路线定案。官方 `cut_current_messages` 里
# `non_empty_user_indices[:-last_image]` 就是 recent-B —— 我们整个课题
# ("哪些历史帧该留")精确地就是替换这一处。本补丁:
#   * 只替换该函数体,其余逐字节不动(prompt / 坐标 / 解析 / 判分全不碰);
#   * 策略由环境变量 CC_FRAME_POLICY 选择(recent|random|learned|oracle),
#     缺省 recent —— 即**未设置时与官方行为完全一致**,这是安全默认;
#   * 幂等:已打过补丁则跳过;打补丁前备份原文件。
# 配套 code/tests/test_frame_policy_parity.py 已穷举验证 recent 分支与官方
# 逐字节等价(1-8 帧 × 1-6 预算,53 用例)。
"""

from __future__ import annotations

import argparse
import ast
import shutil
import time
from pathlib import Path

ANCHOR = "  def cut_current_messages(self, messages, last_image=2):"

REPLACEMENT = '''  def cut_current_messages(self, messages, last_image=2):
    # note (luojiaxuan): CausalCache 唯一改动面 —— 保留哪几帧。
    # 官方原逻辑等价于 policy="recent"(取最近 last_image 个),见
    # causalcache/frame_policy.py 与 tests/test_frame_policy_parity.py。
    # CC_FRAME_POLICY 未设置时行为与官方逐字节相同。
    import os as _os
    non_empty_user_indices = []
    for i, msg in enumerate(messages):
      if msg.get('role') == 'user' and msg.get('content') and len(msg['content']) > 0:
        non_empty_user_indices.append(i)

    _name = _os.environ.get('CC_FRAME_POLICY', 'recent')
    if _name == 'recent' or len(non_empty_user_indices) <= last_image:
      keep = non_empty_user_indices[-last_image:] if last_image > 0 else []
    else:
      from causalcache.frame_policy import POLICIES
      _pol = getattr(self, '_cc_policy', None)
      if _pol is None or getattr(_pol, 'name', None) != _name:
        _pol = POLICIES[_name]() if _name != 'learned' else POLICIES[_name](
            getattr(self, '_cc_scorer'))
        self._cc_policy = _pol
      keep = _pol.select(non_empty_user_indices, last_image,
                         oracle_indices=getattr(self, '_cc_oracle', ()))

    indices_to_clear = [i for i in non_empty_user_indices if i not in set(keep)]
    for index in indices_to_clear:
      if index == 1:
        messages[index]['content'] = [messages[index]['content'][0]]
      else:
        messages[index]['content'] = []

    return messages
'''


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-file", required=True,
                   help="官方 gui_owl.py 路径(emulator 容器内 /android_world/agents/)")
    args = p.parse_args()

    path = Path(args.agent_file)
    src = path.read_text(encoding="utf-8")

    if "CC_FRAME_POLICY" in src:
        print("PATCH_ALREADY")
        return
    if ANCHOR not in src:
        raise SystemExit("ANCHOR_NOT_FOUND: 官方文件结构已变,拒绝盲改")

    start = src.index(ANCHOR)
    # note (luojiaxuan): 函数体到下一个同级 def 为止;不用正则贪婪匹配,
    # 避免把后续方法一起吞掉。
    nxt = src.index("\n  def ", start + len(ANCHOR))
    end = nxt + 1

    shutil.copy(path, f"{path}.orig-{int(time.time())}")
    out = src[:start] + REPLACEMENT + src[end:]
    ast.parse(out)          # 语法自检:坏补丁绝不落盘
    path.write_text(out, encoding="utf-8")
    print("PATCH_WRITTEN")
    print(f"  备份 {path}.orig-*")


if __name__ == "__main__":
    main()
