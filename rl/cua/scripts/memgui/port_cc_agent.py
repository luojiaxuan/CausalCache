# note (luojiaxuan): 把 MobileWorld 树上带 CausalCache 选帧钩子的 gui_owl_1_5 agent 移植到
# MemGUI-Bench 的 agent 分支:取 MemGUI 版为底,替换 history 窗口/消息拼装段为钩子版,
# 追加 _cc_* 辅助函数,history_n 改为可由 CC_HISTORY_N 注入。产物放在 MemGUI 树外,
# 通过 `mg eval --agent-type <path>` 使用。
import re, sys
SRC = "/data01/jaxan/mw/MobileWorld/src/mobile_world/agents/implementations/gui_owl_1_5.py"
BASE = "/data01/jaxan/memgui/src/mobile_world/agents/implementations/gui_owl_1_5.py"
OUT = "/data01/jaxan/memgui_gui_owl_cc.py"
src, base = open(SRC).read(), open(BASE).read()

def segment(text):
    a = text.index("# ── History windowing")
    a = text.rfind("\n", 0, a) + 1
    b = text.index('f"Constructed messages')
    b = text.rfind("logger.debug(", 0, b)
    b = text.rfind("\n", 0, b) + 1
    return a, b
sa, sb = segment(src); ba, bb = segment(base)
out = base[:ba] + src[sa:sb] + base[bb:]
old_hn = '        self.history_n = self.runtime_conf.pop("history_n", 1)\n'
new_hn = ('        # note (luojiaxuan): 预算 B = history_n - 1,由 CC_HISTORY_N 注入。\n'
          '        self.history_n = int(\n'
          '            os.environ.get("CC_HISTORY_N", self.runtime_conf.pop("history_n", 1))\n'
          '        )\n')
assert out.count(old_hn) == 1
out = out.replace(old_hn, new_hn)
h = src.index("# note (luojiaxuan): CausalCache 帧选择策略")
out = out.rstrip("\n") + "\n\n\n" + src[h:]
if not re.search(r"^import os\b|^import .*\bos\b", out, re.M):
    out = out.replace("import json", "import json\nimport os", 1)
open(OUT, "w").write(out)
print(f"ported: base={len(base.splitlines())} lines, src_segment={len(src[sa:sb].splitlines())} lines, out={len(out.splitlines())} lines")
