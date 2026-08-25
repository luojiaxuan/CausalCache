import re
from causalcache_cua.gui_owl.protocol import _END_PUNCTUATIONS, add_period_robustly
import mobile_world.agents.utils.helpers as H

src = open("/data01/jaxan/mw/MobileWorld/src/mobile_world/agents/utils/helpers.py").read()
m = re.search(r"END_PUNCTUATIONS = \{(.*?)\n    \}", src, re.S)
official = set(eval("{" + m.group(1) + "}"))
print("sets_equal:", official == _END_PUNCTUATIONS)
print("only_official:", sorted(official - _END_PUNCTUATIONS),
      "only_mine:", sorted(_END_PUNCTUATIONS - official))
for t in ["do x", "中文测试", "ends.", "mixed 中 abc", "", "结尾;", "tail~"]:
    a, b = H.add_period_robustly(t), add_period_robustly(t)
    assert a == b, (t, a, b)
print("add_period parity OK")
