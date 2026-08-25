# note (luojiaxuan): gui_owl family 注册入口。CUA-Lite agents 层没有
# out-of-tree env-var 钩子(CUA_LITE_REGISTRATION_MODULES 只管 gym env),
# 注册依赖本模块被 import 的 key= 副作用 —— ROLLOUT_MODULE shim 与
# 任何用到该 family 的进程都必须先 import 它。顺序同上游 bootstrap 约定:
# action_space → protocol → adapter → agent。
from __future__ import annotations

from sglang_omni_rl.gui_owl import action_space as _action_space  # noqa: F401
from sglang_omni_rl.gui_owl import protocol as _protocol  # noqa: F401
from sglang_omni_rl.gui_owl import adapter as _adapter  # noqa: F401
from sglang_omni_rl.gui_owl import agent as _agent  # noqa: F401
