#!/usr/bin/env python3
# note (luojiaxuan): 给已打补丁的 MobileWorld agent 增加 learned 策略分支(v2)。
# CC_FRAME_POLICY=learned 时,_cc_select_frames 改为 POST selector 服务;
# 服务不可达直接抛异常(fail loud,不回退 recent)。
# 同时开 CC_DUMP_DIR:每步把 (messages 摘要, S, response) 落盘 —— 训练数据面。
import ast
import sys

F = ("/data01/jaxan/mw/MobileWorld/src/mobile_world/agents/"
     "implementations/gui_owl_1_5.py")
s = open(F, encoding="utf-8").read()
if "CC_SELECTOR_URL" in s:
    print("ALREADY_PATCHED")
    sys.exit(0)

ANCHOR = '    raise ValueError("unknown CC_FRAME_POLICY: %r" % policy)'
NEW = '''    if policy == "learned":
        import json as _json
        import urllib.request as _rq

        url = _os.environ["CC_SELECTOR_URL"]
        payload = _json.dumps({
            "frames_b64": _frames_b64 or [], "step": step,
            "budget": budget, "task": _task, "episode": _episode,
        }).encode()
        req = _rq.Request(url + "/select", data=payload,
                          headers={"Content-Type": "application/json"})
        with _rq.urlopen(req, timeout=30) as r:
            idx = _json.loads(r.read())["indices"]
        return sorted(idx) + [total]
    raise ValueError("unknown CC_FRAME_POLICY: %r" % policy)'''

assert ANCHOR in s, "anchor missing"
s = s.replace(ANCHOR, NEW, 1)

# 签名补三个透传参数(帧 b64 列表 / 任务名 / episode 标识)
OLD_SIG = 'def _cc_select_frames(total, budget, policy=None, step=0, salt=""):'
NEW_SIG = ('def _cc_select_frames(total, budget, policy=None, step=0, salt="", '
           '_frames_b64=None, _task="", _episode=""):')
assert OLD_SIG in s, "sig missing"
s = s.replace(OLD_SIG, NEW_SIG, 1)

# 调用处传入帧数据与任务标识
OLD_CALL = '''        _cc_S = _cc_select_frames(
            total_history_count,
            keep_as_messages,
            step=total_history_count,
            salt=str(getattr(self, "instruction", "")),
        )'''
NEW_CALL = '''        _cc_S = _cc_select_frames(
            total_history_count,
            keep_as_messages,
            step=total_history_count,
            salt=str(getattr(self, "instruction", "")),
            _frames_b64=[c[0] for c in self.history_user_content[:total_history_count]],
            _task=str(getattr(self, "instruction", ""))[:80],
            _episode=str(id(self)),
        )'''
assert OLD_CALL in s, "call missing"
s = s.replace(OLD_CALL, NEW_CALL, 1)

# 训练数据面:模型响应落盘(messages 太大,存 S + response + 步号,
# 训练期用 history_user_content 的 b64 重建输入)
HOOK_ANCHOR = "        logger.debug("
# note (luojiaxuan): messages 含 b64 图,体积大但为训练数据面所必需 ——
# executor 的 PG 必须在真实 rollout 输入上重前向。
DUMP = '''        _dump = __import__("os").environ.get("CC_DUMP_DIR")
        if _dump:
            import json as _j
            with open(f"{_dump}/{id(self)}.jsonl", "a") as _f:
                _f.write(_j.dumps({
                    "step": total_history_count, "S": _cc_S,
                    "task": str(getattr(self, "instruction", ""))[:80],
                    "messages": messages,
                }) + "\\n")
        logger.debug('''
assert HOOK_ANCHOR in s
s = s.replace(HOOK_ANCHOR, DUMP, 1)

ast.parse(s)
open(F, "w", encoding="utf-8").write(s)
print("LEARNED_PATCH_OK")
