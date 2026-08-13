"""任务 DSL、workflow 模板库、实例生成器与 template 级划分(Phase 1 之三)。

# note (luojiaxuan): 契约见 contract.py(冻结,只 import 不改);窗口内容来自
# apps.py 的六个构造器。核心设计:
#   * **每个 template 恒有两个必需变量 v1/v2 与一条固定动作骨架**,memory regime
#     只改「证据在哪一帧可见」—— instruction 逐字相同、专家动作逐条相同、步数相同,
#     模型无法从任务类型/长度/措辞反推 regime;
#   * 证据「不可见」一律靠**真实遮挡**(全屏窗口被切走、set_visible 覆盖),不存在
#     「我们不给这一帧」式的假不可见 —— selector 的动作本身就是选帧;
#   * 答案写进**单一** Record 输入框(只有一次 type 决策),recent_sufficient 才
#     在决策步上成立;多输入框会让第二次输入天然掉出 recent-2;
#   * 任务自有控件集中在窗口底部保留带(BAND_TOP 以下)且 style["scroll"]=False,
#     不随 apps.py 滚动区偏移/裁剪;命中测试按 widgets 逆序,后追加者优先。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from causalcache_agentic import apps
from causalcache_agentic.contract import (
    MEMORY_REGIMES, SCREEN_H, SCREEN_W, TASKBAR_H, Assertion, Effect, Rect,
    ScreenState, TaskSpec, Widget, Window, make_taskbar, pixel_to_norm,
)
from causalcache_agentic.executor import is_scroll_content

# ---------------------------------------------------------------- 版面与词表
WIN_H = SCREEN_H - TASKBAR_H
BAND_TOP = WIN_H - 196                      # 任务自有控件保留带顶边
ROW1, ROW2, ROW3 = BAND_TOP + 20, BAND_TOP + 68, BAND_TOP + 120

APP_TITLES = {k: k.capitalize() for k in
              ("files", "writer", "calc", "browser", "mail", "settings")}
VIEWS = {"files": ("Recent", "Archive"), "writer": ("Draft", "Notes"),
         "calc": ("Sheet 1", "Sheet 2"), "browser": ("Summary", "Details"),
         "mail": ("Inbox", "Updates"), "settings": ("General", "Advanced")}
COMMIT_TEXT = {"save": "Save", "send": "Send", "apply": "Apply"}
_ORGS = ("northwind", "vertex", "lumen", "kestrel", "arbor", "solace")
_SUBJ_WORD = ("batch", "record", "entry", "docket", "lot")
_REGIONS = ("north", "south", "east", "west", "central", "harbor", "ridge")
_REF_PFX = ("ORD", "INV", "TKT", "CSE", "SHP")
_PFX_POOL = ("QA", "FY26", "AUD", "REV")
_ALPHA, _DIGIT = "ABCDEFGHJKLMNPQRSTUVWXYZ", "23456789"


# ------------------------------------------------------------- 动作构造
# note (luojiaxuan): 动作口径 = policy 输出 [0,999] 归一 computer_use 调用,
# 与 executor.apply_action 逐字同源(左键/键入/组合键/滚动/终止)。

def _norm(rect: Rect, dy: int = 0) -> list[int]:
    cx, cy = rect.center
    return list(pixel_to_norm(cx, cy - dy))


def act_click(rect: Rect, dy: int = 0) -> dict[str, Any]:
    return {"action": "left_click", "coordinate": _norm(rect, dy)}


def act_type(text: str) -> dict[str, Any]:
    return {"action": "type", "text": text}


def act_key(*keys: str) -> dict[str, Any]:
    return {"action": "key", "keys": list(keys)}


def act_scroll(rect: Rect, pixels: int) -> dict[str, Any]:
    """pixels 按**冻结工具口径**给:正数向上滚、负数向下滚(executor.SCROLL_SIGN)。"""
    return {"action": "scroll", "coordinate": _norm(rect), "pixels": int(pixels)}


def act_terminate(status: str = "success") -> dict[str, Any]:
    # note (luojiaxuan): 每次返回新 dict —— 动作会被 env 记进轨迹并可能就地加注,
    # 共享同一个字面量会让所有任务的最后一步互相串改。
    return {"action": "terminate", "status": status}


# ------------------------------------------------------------- 控件工具

def _mkw(wid: str, kind: str, rect: Rect, text: str = "", value: str = "",
         visible: bool = True, on_click: Sequence[Effect] = ()) -> Widget:
    # note (luojiaxuan): style["scroll"]=False 让 executor 与 render 都把任务控件
    # 当作非滚动内容,不受 apps.py 滚动区的偏移与裁剪影响。
    return Widget(wid=wid, kind=kind, rect=rect, text=text, value=value,
                  visible=visible, on_click=tuple(on_click), style={"scroll": False})


def _find(win: Window, wid: str) -> Widget:
    for w in win.widgets:
        if w.wid == wid:
            return w
    raise KeyError(wid)


def _add_card(win: Window, cid: str, header: str,
              lines: Sequence[tuple[str, str]]) -> list[str]:
    """在保留带加一张证据卡片(初始隐藏),返回其控件 wid 列表。"""
    wids = [f"task::{win.app_id}::{cid}::head"]
    win.widgets.append(_mkw(wids[0], "label", Rect(40, ROW2, 1400, 34), header,
                            visible=False))
    for i, (cap, val) in enumerate(lines):
        wids.append(f"task::{win.app_id}::{cid}::line{i}")
        win.widgets.append(_mkw(wids[-1], "label", Rect(40 + i * 900, ROW3, 860, 36),
                                f"{cap}: {val}", value=val, visible=False))
    return wids


def _add_tabs(win: Window, tabs: Sequence[tuple[str, list[str]]]) -> list[str]:
    """页签按钮:点一个 = 显示该页卡片、隐藏同窗其他页卡片(真实遮挡)。"""
    allw = [w for _, ws in tabs for w in ws]
    out: list[str] = []
    for i, (label, ws) in enumerate(tabs):
        out.append(f"task::{win.app_id}::tab{i}")
        win.widgets.append(_mkw(
            out[-1], "button", Rect(40 + i * 300, ROW1, 280, 40), label,
            on_click=tuple(Effect("set_visible", target=w,
                                  value=("1" if w in ws else "0")) for w in allw)))
    return out


def _reserve_band(win: Window) -> None:
    """把窗口滚动区下沿抬到保留带之上,滚动量按真实内容底重算。

    # note (luojiaxuan): 必须无条件做(哪怕 scroll_max=0)—— render 会把滚动层整块
    # paste 回去、**覆盖**区内的任务控件,executor 也会判区外的点不可点。
    """
    sa = win.scroll_area
    if sa is None or sa.y + sa.h <= BAND_TOP - 12:
        return
    new_h = max(120, BAND_TOP - 12 - sa.y)
    bottom = max((w.rect.y + w.rect.h for w in win.widgets
                  if is_scroll_content(win, w)), default=sa.y)
    win.scroll_area = Rect(sa.x, sa.y, sa.w, new_h)
    win.scroll_max = max(0, bottom - (sa.y + new_h))


_LOCATE_KINDS = ("list_item", "cell", "menu_item", "label", "button", "icon")


def _locate(win: Window, key: str) -> Widget | None:
    """按文本在 apps.py 的内容里定位一个可点控件(供 SEARCH 用)。"""
    # note (luojiaxuan): 滚动内容的 rect.y 是布局坐标不是屏幕坐标,不能按 y 排除。
    cands = [w for w in win.widgets
             if not w.wid.startswith("task::") and w.kind in _LOCATE_KINDS
             and w.style.get("role") != "taskbar"
             and (is_scroll_content(win, w) or w.rect.y < BAND_TOP)
             and (key in (w.text or "") or key in (w.value or ""))]
    return min(cands, key=lambda w: _LOCATE_KINDS.index(w.kind), default=None)


def _click_actions(win: Window, w: Widget) -> list[dict[str, Any]] | None:
    """点到某控件所需的动作(必要时先滚动);滚动后仍够不到 → None,改用兜底按钮。"""
    if not (is_scroll_content(win, w) and win.scroll_area is not None
            and win.scroll_max > 0):
        return [act_click(w.rect)]
    area = win.scroll_area
    need = max(0, min(win.scroll_max, w.rect.y + w.rect.h - (area.y + area.h)))
    cx, cy = w.rect.center
    if not area.contains(cx, cy - need):
        return None
    # note (luojiaxuan): 要把 need 像素的内容翻上来 = **向下滚**,冻结工具口径下
    # 写成负 pixels(executor 再按 SCROLL_SIGN 变回 scroll_y += need)。
    return ([act_scroll(area, -need)] if need > 0 else []) + [act_click(w.rect, need)]


# ------------------------------------------------------------- DSL primitive

@dataclass(frozen=True)
class PrimitiveResult:
    """一个 DSL primitive 的贡献:状态声明 / 专家动作 / 断言 / 探针 / 叙述。"""

    state_mutations: tuple[Effect, ...] = ()
    expert_actions: tuple[dict[str, Any], ...] = ()
    assertions: tuple[Assertion, ...] = ()
    memory_probe: dict[str, Any] = field(default_factory=dict)
    narration: str = ""


@dataclass
class _Ctx:
    """构建期上下文:动作序列 + 「每一帧屏幕上可见哪些变量」的账本。"""

    state: ScreenState
    actions: list[dict[str, Any]] = field(default_factory=list)
    frames: list[tuple[str, ...]] = field(default_factory=list)
    cur: tuple[str, ...] = ()
    assertions: list[Assertion] = field(default_factory=list)

    def emit(self, acts: Sequence[dict[str, Any]],
             vars_after: tuple[str, ...] | None = None) -> None:
        for a in acts:
            self.frames.append(self.cur)      # 该动作决策前观测到的那一帧
            self.actions.append(a)
        if vars_after is not None:
            self.cur = vars_after

    def win(self, app_id: str) -> Window:
        return self.state.windows[app_id]


def prim_switch_app(ctx: _Ctx, app_id: str,
                    vars_after: tuple[str, ...] = ()) -> PrimitiveResult:
    """SWITCH_APP:点任务栏切 app(全屏窗口 ⇒ 上一个 app 的内容真实消失)。"""
    btn = _find(ctx.win("shell"), f"taskbar::{app_id}")
    ctx.emit([act_click(btn.rect)], vars_after)
    return PrimitiveResult(btn.on_click, (act_click(btn.rect),),
                           narration=f"switch to {app_id}")


def prim_read(ctx: _Ctx, app_id: str, tab_wid: str,
              vars_after: tuple[str, ...]) -> PrimitiveResult:
    """READ:点开某页显示证据卡片;此后这些变量只存在于该帧的像素里。"""
    btn, frame = _find(ctx.win(app_id), tab_wid), len(ctx.frames)
    ctx.emit([act_click(btn.rect)], vars_after)
    return PrimitiveResult(btn.on_click, (act_click(btn.rect),),
                           memory_probe={"reveals": list(vars_after), "frame": frame},
                           narration=f"read {list(vars_after)} in {app_id}")


def prim_search(ctx: _Ctx, app_id: str, key: str, reveal: Sequence[Effect],
                fallback: str) -> PrimitiveResult:
    """SEARCH:在长列表里定位目标项并点开(够不到先滚动);定位失败则兜底按钮。"""
    win = ctx.win(app_id)
    hit = _locate(win, key)
    acts = _click_actions(win, hit) if hit is not None else None
    if hit is None or acts is None:
        hit = _mkw(f"task::{app_id}::hit", "button", Rect(940, ROW1, 380, 40),
                   fallback, on_click=reveal)
        win.widgets.append(hit)
        acts = [act_click(hit.rect)]
    else:
        hit.on_click = tuple(reveal)          # 由任务接管点击语义,行为可预测
    ctx.emit(acts)
    return PrimitiveResult(tuple(reveal), tuple(acts), narration=f"search {key!r}")


def prim_write(ctx: _Ctx, app_id: str, field_wid: str, text: str,
               vars_after: tuple[str, ...] | None = None) -> PrimitiveResult:
    """WRITE:点进输入框并键入(输入框初始为空,type 后的值精确等于该文本)。"""
    w = _find(ctx.win(app_id), field_wid)
    ctx.emit([act_click(w.rect)], vars_after)
    ctx.emit([act_type(text)])
    return PrimitiveResult(expert_actions=(act_click(w.rect), act_type(text)),
                           narration=f"write {text!r} into {field_wid}")


def prim_copy(ctx: _Ctx, src_app: str, src_wid: str, dst_app: str, dst_wid: str,
              vars_on_dst: tuple[str, ...] = ()) -> PrimitiveResult:
    """COPY:ctrl+c 取源控件的值,切到目标窗口后 ctrl+v 粘进输入框。"""
    src = _find(ctx.win(src_app), src_wid)
    ctx.emit([act_click(src.rect), act_key("ctrl", "c")])
    prim_switch_app(ctx, dst_app, vars_on_dst)
    dst = _find(ctx.win(dst_app), dst_wid)
    ctx.emit([act_click(dst.rect), act_key("ctrl", "v")])
    return PrimitiveResult(expert_actions=tuple(ctx.actions[-5:]),
                           narration=f"copy {src_wid} -> {dst_wid}")


def prim_compare(values: Sequence[tuple[str, int]], mode: str) -> PrimitiveResult:
    """COMPARE:对比两处数值并据此选分支(返回胜出项的名字)。"""
    pick = (max if mode == "max" else min)(values, key=lambda kv: kv[1])
    return PrimitiveResult(memory_probe={"result": pick[0]},
                           narration=f"compare({mode}) -> {pick[0]}")


def prim_transform(raw: Sequence[str], mode: str, prefix: str) -> PrimitiveResult:
    """TRANSFORM:对读到的值做确定性变换(大小写 / 加前缀 / 求和)。"""
    a, b = raw[0], raw[1]
    out = (str(int(a) + int(b)) if mode == "sum" else
           {"upper": f"{a.upper()} {b}",
            "prefix": f"{prefix}-{a} {b}"}.get(mode, f"{a} {b}"))
    return PrimitiveResult(memory_probe={"result": out}, narration=f"transform {out}")


def prim_distract(win: Window, cid: str, header: str,
                  lines: Sequence[tuple[str, str]]) -> PrimitiveResult:
    """DISTRACT:引入一屏格式相同但错误的内容(当前屏无从分辨,选错帧必写错值)。"""
    wids = _add_card(win, cid, header, lines)
    return PrimitiveResult(
        tuple(Effect("set_visible", target=w, value="1") for w in wids),
        memory_probe={"distractor": [v for _, v in lines], "wids": wids},
        narration=f"distractor card on {win.app_id}")


def prim_save(ctx: _Ctx, app_id: str, wid: str) -> PrimitiveResult:
    """SAVE:触发 save/apply 按钮并结束回合。"""
    w = _find(ctx.win(app_id), wid)
    ctx.emit([act_click(w.rect)])
    ctx.emit([act_terminate()])
    return PrimitiveResult(w.on_click, (act_click(w.rect), act_terminate()),
                           narration=f"commit on {app_id}")


def prim_send(ctx: _Ctx, app_id: str, wid: str) -> PrimitiveResult:
    """SEND:触发 mail 发送并结束回合(机制同 SAVE,断言用 commit_kind 区分)。"""
    return prim_save(ctx, app_id, wid)


def prim_verify(ctx: _Ctx, assertions: Sequence[Assertion]) -> PrimitiveResult:
    """VERIFY:产出终局断言(声明式,无自然语言、无模型参与)。"""
    ctx.assertions.extend(assertions)
    return PrimitiveResult(assertions=tuple(assertions), narration="verify")


PRIMITIVES: dict[str, Callable[..., PrimitiveResult]] = {
    "READ": prim_read, "WRITE": prim_write, "COPY": prim_copy,
    "COMPARE": prim_compare, "TRANSFORM": prim_transform, "SEARCH": prim_search,
    "SAVE": prim_save, "SEND": prim_send, "SWITCH_APP": prim_switch_app,
    "DISTRACT": prim_distract, "VERIFY": prim_verify,
}


# ------------------------------------------------------------- 模板表

@dataclass(frozen=True)
class TemplateDef:
    template_id: str
    family: str
    tier: int                       # curriculum 档位:2 / 3 / 4-5 stage
    sources: tuple[str, ...]        # 1 个 → 同 app 双页签;2 个 → 跨 app
    sink: str
    commit: str                     # save | send | apply
    op: str                         # copy | upper | prefix | sum | max | min
    subject: str
    caps: tuple[str, str]           # 两个变量的抬头(sum/max/min 改用地名)
    search_app: str = ""            # 非空 → 额外的 SEARCH + 访问码阶段
    clip: bool = False              # 访问码走 ctrl+c/ctrl+v 而非键入


_T = TemplateDef
TEMPLATES: tuple[TemplateDef, ...] = (
    _T("f1_files_writer_copy", "invoice_handoff", 2, ("files",), "writer", "save",
       "copy", "invoice", ("Reference", "Amount")),
    _T("f1_files_writer_upper", "invoice_handoff", 2, ("files",), "writer", "save",
       "upper", "invoice", ("Reference", "Amount")),
    _T("f2_mail_settings_copy", "inbox_config", 2, ("mail",), "settings", "apply",
       "copy", "policy", ("Profile", "Retention")),
    _T("f2_mail_settings_prefix", "inbox_config", 2, ("mail",), "settings", "apply",
       "prefix", "policy", ("Profile", "Retention")),
    _T("f3_browser_calc_writer", "web_ledger", 3, ("browser", "calc"), "writer",
       "save", "copy", "shipment", ("Tracking", "Weight")),
    _T("f3_browser_files_writer", "web_ledger", 3, ("browser", "files"), "writer",
       "save", "prefix", "shipment", ("Tracking", "Weight")),
    _T("f4_calc_files_mail", "report_dispatch", 3, ("calc", "files"), "mail",
       "send", "copy", "report", ("Ticket", "Total")),
    _T("f4_settings_calc_mail", "report_dispatch", 3, ("settings", "calc"), "mail",
       "send", "upper", "report", ("Ticket", "Total")),
    _T("f5_calc_browser_max", "budget_compare", 4, ("calc", "browser"), "writer",
       "save", "max", "budget", ("Region", "Spend"), search_app="files"),
    _T("f5_browser_calc_sum", "budget_compare", 4, ("browser", "calc"), "writer",
       "save", "sum", "budget", ("Region", "Spend"), search_app="files"),
    _T("f6_files_browser_mail", "audit_composite", 5, ("files", "browser"), "mail",
       "send", "copy", "audit", ("Case", "Balance"), search_app="calc", clip=True),
    _T("f6_mail_settings_writer", "audit_composite", 5, ("mail", "settings"),
       "writer", "save", "prefix", "audit", ("Case", "Balance"),
       search_app="browser"),
)
BY_ID = {t.template_id: t for t in TEMPLATES}
FAMILIES = tuple(dict.fromkeys(t.family for t in TEMPLATES))


# ------------------------------------------------------------- 随机内容

def _mix(*parts: int) -> int:
    v = 1469598103934665603
    for p in parts:
        v = (v * 1099511628211 + int(p)) & 0xFFFFFFFFFFFF
    return v


def _ref(rng: random.Random) -> str:
    return (f"{rng.choice(_REF_PFX)}-{''.join(rng.choice(_ALPHA) for _ in range(3))}"
            f"-{''.join(rng.choice(_DIGIT) for _ in range(4))}")


def _amount(rng: random.Random) -> str:
    return f"{rng.randrange(120, 9800)}.{rng.randrange(10, 99)}"


def _code(rng: random.Random) -> str:
    return "".join(rng.choice(_ALPHA + _DIGIT) for _ in range(6))


def _app_spec(kind: str, org: str, subject: str, long_key: str) -> dict[str, Any]:
    """给 apps.py 构造器的内容 spec。内容只作背景,证据一律走任务卡片。"""
    tags = [f"{subject}-{i:03d}" for i in range(1, (26 if long_key else 5) + 1)]
    if long_key:
        tags[-3] = long_key                   # 埋在列表深处,SEARCH 需滚动才够得到
    body: dict[str, dict[str, Any]] = {
        "files": {"path": f"/home/user/{org}", "files": [
            {"name": f"{t}.csv", "size": f"{40 + i * 7} KB",
             "modified": f"2026-05-{(i % 27) + 1:02d}",
             "content": f"{org} {subject} sheet {t}"} for i, t in enumerate(tags)]},
        "writer": {"title": f"{org} {subject} notes", "editable_id": f"task::{kind}::d",
                   "paragraphs": [f"Working notes for the {org} {subject} desk.",
                                  "Fill the form below before closing this file."]},
        "calc": {"title": f"{org}_{subject}.csv", "headers": ["Item", "Owner", "Units"],
                 "rows": [[t, org, str(100 + i * 3)] for i, t in enumerate(tags)]},
        "browser": {"url": f"https://{org}.example/{subject}", "title": f"{org} desk",
                    "lines": [f"{t} logged by the {org} desk" for t in tags],
                    "links": [{"text": t, "url": f"https://{org}.example/{t}"}
                              for t in tags[:2]]},
        "mail": {"folder": "Inbox", "compose": False, "messages": [
            {"from": f"desk@{org}.example", "subject": f"{subject} {t}",
             "body": f"Routine note about {subject} {t}."} for t in tags[:3]]},
        "settings": {"sections": [
            {"name": "General", "items": [{"label": f"{subject} {t}", "value": org}
                                          for t in tags[:3]]},
            {"name": "Advanced", "items": [{"label": "Telemetry", "value": "off"}]}]},
    }[kind]
    return {"app_id": kind, **body}


_BUILDERS = {k: getattr(apps, f"build_{k}_app") for k in APP_TITLES}


# ------------------------------------------------------------- 证据摆放
# note (luojiaxuan): regime → (卡片 A 放什么, 卡片 B 放什么, sink staged 面板)。
# 两种摆法由 prng 二选一,使「哪张卡片有货」无法靠位置记忆。banner:none=不放;
# hide=点进 Record 框时收起(⇒ 最近两帧看得见、当前帧看不见);keep=一直留着
# (⇒ 当前屏已足够)。
_SLOTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "recent_sufficient": (("none", "none", "hide"),) * 2,
    "history_irrelevant": (("none", "none", "keep"),) * 2,
    "one_old_frame": (("both", "none", "none"), ("none", "both", "none")),
    "two_frame_complementary": (("v1", "v2", "none"), ("v2", "v1", "none")),
    "distractor_heavy": (("both", "foil", "none"), ("foil", "both", "none")),
}
# note (luojiaxuan): slot → (卡片抬头, 该卡片承载的变量, 展示哪几列)
_CARD = {"both": ("confirmed", ("v1", "v2"), (0, 1)),
         "v1": ("confirmed", ("v1",), (0,)), "v2": ("confirmed", ("v2",), (1,)),
         "foil": ("cancelled", (), (0, 1)), "none": ("draft", (), ())}


def _card_lines(slot: str, caps: tuple[str, str], vals: tuple[str, str],
                foil: tuple[str, str]) -> list[tuple[str, str]]:
    if slot == "none":
        return [("note", "no confirmed entry in this view")]
    src = foil if slot == "foil" else vals
    return [(caps[i], src[i]) for i in _CARD[slot][2]]


def _foil(prng: random.Random, numeric: bool,
          vals: tuple[str, str]) -> tuple[str, str]:
    """干扰值:格式与正确值相同,但**结论必然不同**(比较翻转、求和错开、串不等)。"""
    if numeric:
        a, b = prng.randrange(120, 4000), prng.randrange(4200, 9800)
        f0, f1 = (str(a), str(b)) if int(vals[0]) > int(vals[1]) else (str(b), str(a))
        if int(f0) + int(f1) == int(vals[0]) + int(vals[1]):
            f0 = str(int(f0) + 1)
        return f0, f1
    out = (_ref(prng), _amount(prng))
    while out[0] == vals[0] or out[1] == vals[1]:
        out = (_ref(prng), _amount(prng))
    return out


# ------------------------------------------------------------- 实例构建

_PHRASES = (
    "Open {srcs} and find the {subj} entry whose status is confirmed. Then go to "
    "{sink}, press Edit record, type {ans} into the Record box and press {commit}.",
    "There is exactly one confirmed {subj} entry across {srcs}. Take it to {sink}: "
    "press Edit record, put {ans} in the Record box, then press {commit}.",
    "Check {srcs} for the confirmed {subj} entry. In {sink}, press Edit record, "
    "enter {ans} in the Record box and finish with {commit}.",
)
_ANS_TEXT = {"copy": "the {c0} then a space then the {c1}",
             "upper": "the {c0} in upper case then a space then the {c1}",
             "prefix": "the {c0} prefixed with '{p}-' then a space then the {c1}",
             "sum": "the sum of the two figures",
             "max": "the name of the line with the larger figure",
             "min": "the name of the line with the smaller figure"}


def _build(tpl: TemplateDef, seed: int, regime: str) -> TaskSpec:
    base = _mix(seed, TEMPLATES.index(tpl))
    rng = random.Random(base)                                    # 内容:与 regime 无关
    prng = random.Random(base + 1 + MEMORY_REGIMES.index(regime))   # 摆放:与之有关

    # ---- 内容(先抽完再分支,保证 instruction / 专家动作跨 regime 逐字相同)
    org, subj = rng.choice(_ORGS), f"{tpl.subject} {rng.choice(_SUBJ_WORD)}"
    numeric = tpl.op in ("sum", "max", "min")
    if numeric:
        caps = tuple(rng.sample(_REGIONS, 2))
        lo, hi = rng.randrange(120, 4000), rng.randrange(4200, 9800)
        vals = (str(lo), str(hi)) if rng.random() < 0.5 else (str(hi), str(lo))
    else:
        caps, vals = tpl.caps, (_ref(rng), _amount(rng))
    pfx = rng.choice(_PFX_POOL)
    answer = (prim_compare([(caps[0], int(vals[0])), (caps[1], int(vals[1]))], tpl.op)
              if tpl.op in ("max", "min")
              else prim_transform(vals, tpl.op, pfx)).memory_probe["result"]
    code = _code(rng) if tpl.search_app else ""
    key = f"{tpl.subject}-{rng.randrange(300, 980)}" if tpl.search_app else ""
    phrase = rng.choice(_PHRASES)

    # ---- 窗口(sink 置顶;任务栏统一放在 shell 窗口里)
    order = list(tpl.sources) + ([tpl.search_app] if tpl.search_app else [])
    order.append(tpl.sink)
    wins: dict[str, Window] = {}
    for kind in order:
        win = _BUILDERS[kind](_app_spec(kind, org, subj,
                                        key if kind == tpl.search_app else ""))
        # note (luojiaxuan): apps.py 若自带任务栏,槽位会与 shell 的重叠却指向别的
        # app(它无从知道全部 app id),必须去掉,否则切应用会点错。
        win.widgets = [w for w in win.widgets if w.style.get("role") != "taskbar"]
        _reserve_band(win)
        wins[kind] = win
    state = ScreenState(
        windows={"shell": Window("shell", "", Rect(0, 0, SCREEN_W, SCREEN_H),
                                 make_taskbar(order, [APP_TITLES[k] for k in order])),
                 **wins},
        z_order=["shell"] + [k for k in order if k != tpl.sink] + [tpl.sink])

    # ---- 证据卡片与页签
    slots = _SLOTS[regime][0 if prng.random() < 0.5 else 1]
    foil = _foil(prng, numeric, vals)
    hosts = (tpl.sources[0], tpl.sources[-1])
    tabs: dict[str, list[tuple[str, list[str]]]] = {h: [] for h in hosts}
    for i, (slot, host) in enumerate(zip(slots[:2], hosts)):
        head, lines = f"status: {_CARD[slot][0]}", _card_lines(slot, caps, vals, foil)
        cw = (prim_distract(wins[host], f"c{i}", head, lines).memory_probe["wids"]
              if slot == "foil" else _add_card(wins[host], f"c{i}", head, lines))
        tabs[host].append((VIEWS[host][len(tabs[host])], cw))
    tab_wids = {h: _add_tabs(wins[h], tabs[h]) for h in dict.fromkeys(hosts)}
    code_wids = (_add_card(wins[tpl.search_app], "code", f"{key} — access record",
                           [("Access code", code)]) if tpl.search_app else [])

    # ---- sink 表单:banner + Edit record 解锁 + 访问码 + Record 框 + 提交/丢弃
    sink, tid = wins[tpl.sink], f"task::{tpl.sink}"
    ban, fw, cw_ = f"{tid}::banner", f"{tid}::answer", f"{tid}::code"
    sw, dw, alb = f"{tid}::{tpl.commit}", f"{tid}::discard", f"{tid}::alabel"
    commit_eff = [Effect("flag_from", target="answer", payload={"widget": fw}),
                  Effect("set_flag", target="committed", value="1"),
                  Effect("set_flag", target="commit_kind", value=tpl.commit)]
    if tpl.search_app:
        commit_eff.insert(1, Effect("flag_from", target="access_code",
                                    payload={"widget": cw_}))
    parts = [
        (ban, "label", Rect(40, ROW1, 1380, 40), f"staged {subj} · status: confirmed"
         f" · {caps[0]}: {vals[0]} · {caps[1]}: {vals[1]}", slots[2] != "none", ()),
        (f"{tid}::unlock", "button", Rect(1480, ROW1, 360, 40), "Edit record", True,
         tuple(Effect("set_visible", target=x, value="1") for x in (fw, sw, dw, alb))),
        (alb, "label", Rect(40, ROW3 + 8, 200, 30), "Record", False, ()),
        # note (luojiaxuan): 点进 Record 框收起 banner —— recent_sufficient 就靠它
        # 做到「最近两帧看得见、当前帧看不见」;其余 regime 该 on_click 为空。
        (fw, "text_field", Rect(250, ROW3, 620, 44), "", False,
         (Effect("set_visible", target=ban, value="0"),) if slots[2] == "hide" else ()),
        (sw, "button", Rect(940, ROW3, 260, 44), COMMIT_TEXT[tpl.commit], False,
         tuple(commit_eff)),
        (dw, "button", Rect(1240, ROW3, 260, 44), "Discard", False,
         (Effect("set_flag", target="discarded", value="1"),
          Effect("set_flag", target="committed", value="0")))]
    if tpl.search_app:
        parts[2:2] = [(f"{tid}::clabel", "label", Rect(40, ROW2 + 6, 240, 30),
                       "Access code", True, ()),
                      (cw_, "text_field", Rect(300, ROW2, 380, 40), "", True, ())]
    for wid, kind, rect, text, vis, eff in parts:
        sink.widgets.append(_mkw(wid, kind, rect, text, visible=vis, on_click=eff))

    # ---- 专家脚本(骨架与 regime 无关,只有屏幕内容随 regime 变)
    staged = ("v1", "v2") if slots[2] != "none" else ()
    ctx = _Ctx(state=state, cur=staged)
    dis_steps: list[int] = []
    for i, host in enumerate(hosts):
        first = i == 0 or host != hosts[0]
        if first:
            prim_switch_app(ctx, host, ())
        res = prim_read(ctx, host, tab_wids[host][0 if first else 1],
                        _CARD[slots[i]][1])
        if slots[i] == "foil":
            # note (luojiaxuan): 干扰卡片点开后的**下一帧**就是"格式相同但值是错的"
            # 那张屏 —— 选错帧必写错值。只记这一帧:它正是 selector 最可能误选的那张,
            # 也是 expert.py 的 oracle_plus_distractor 子集要注入的干扰。
            dis_steps.append(int(res.memory_probe["frame"]) + 1)
    if tpl.search_app:
        prim_switch_app(ctx, tpl.search_app, ())
        prim_search(ctx, tpl.search_app, key,
                    [Effect("set_visible", target=x, value="1") for x in code_wids],
                    f"Open {key}")
        ctx.cur = ("code",)
        if tpl.clip:
            prim_copy(ctx, tpl.search_app, code_wids[1], tpl.sink, cw_, staged)
        else:
            prim_switch_app(ctx, tpl.sink, staged)
            prim_write(ctx, tpl.sink, cw_, code)
    else:
        prim_switch_app(ctx, tpl.sink, staged)
    ctx.emit([act_click(_find(sink, f"{tid}::unlock").rect)])
    decision = len(ctx.actions) + 1               # 唯一一次「必须已经知道答案」的步
    prim_write(ctx, tpl.sink, fw, answer,
               vars_after=staged if slots[2] == "keep" else ())
    (prim_send if tpl.commit == "send" else prim_save)(ctx, tpl.sink, sw)

    asserts = [Assertion("flag_equals", "answer", answer),
               Assertion("flag_equals", "committed", "1"),
               Assertion("flag_equals", "commit_kind", tpl.commit),
               Assertion("flag_absent", "discarded")]
    if tpl.search_app:
        asserts.insert(1, Assertion("flag_equals", "access_code", code))
    prim_verify(ctx, asserts)
    _check_regime(regime, ctx.frames, decision, tpl.template_id)

    instr = phrase.format(srcs=" and ".join(APP_TITLES[s] for s in tpl.sources),
                          subj=subj, sink=APP_TITLES[tpl.sink],
                          ans=_ANS_TEXT[tpl.op].format(c0=caps[0], c1=caps[1], p=pfx),
                          commit=COMMIT_TEXT[tpl.commit])
    if tpl.search_app:
        instr += (f" The form also needs the access code printed on the {key} page "
                  f"in {APP_TITLES[tpl.search_app]}.")
    need = {"v1", "v2"}
    return TaskSpec(
        task_id=f"{tpl.template_id}::{regime}::{seed}", template_id=tpl.template_id,
        family=tpl.family, regime=regime, seed=seed, instruction=instr,
        initial_state=state, assertions=tuple(ctx.assertions),
        expert_actions=tuple(ctx.actions), max_steps=len(ctx.actions) + 8,
        # note (luojiaxuan): memory_probe 仅诊断 —— 禁止进 policy 输入、禁止作 reward。
        memory_probe={
            "required_steps": [i for i, f in enumerate(ctx.frames) if need & set(f)],
            "distractor_steps": [i for i in dis_steps if i < len(ctx.frames)],
            "regime": regime, "decision_step": decision,
            "recent2": [decision - 2, decision - 1],
            "frame_vars": {i: list(f) for i, f in enumerate(ctx.frames) if f},
            "variables": {"v1": vals[0], "v2": vals[1], "answer": answer,
                          "code": code, "captions": list(caps),
                          "distractor": list(foil) if "foil" in slots else []}},
        params={"tier": tpl.tier, "op": tpl.op, "org": org, "subject": subj,
                "apps": order, "placement": list(slots), "answer": answer,
                "access_code": code, "search_key": key})


def _check_regime(regime: str, frames: list[tuple[str, ...]], d: int,
                  tid: str) -> None:
    """构造期自证:regime 语义确实由「像素上何时可见」实现,不达标即 fail fast。"""
    need = {"v1", "v2"}
    cur, old = set(frames[d]), [set(f) for f in frames[:d - 2]]
    seen = set(frames[d - 1]) | set(frames[d - 2]) | cur
    cover = [f for f in old if need & f]
    if regime == "history_irrelevant":
        ok = need <= cur
    elif regime == "recent_sufficient":
        ok = need <= seen and not need <= cur
    elif regime == "two_frame_complementary":
        ok = (not need <= seen and len(cover) >= 2
              and not any(need <= f for f in old) and need <= set().union(*cover))
    else:
        ok = not need <= seen and any(need <= f for f in old)
    if not ok:
        raise ValueError(f"{tid}/{regime}: regime 约束不成立 (d={d}, frames={frames})")


# ------------------------------------------------------------- 对外 API

def generate_task(template_id: str, seed: int, regime: str) -> TaskSpec:
    """按 (template, seed, regime) 确定性地实例化一个任务。"""
    if template_id not in BY_ID:
        raise KeyError(f"未知 template_id: {template_id!r}")
    if regime not in MEMORY_REGIMES:
        raise ValueError(f"未知 regime: {regime!r}")
    return _build(BY_ID[template_id], seed, regime)


def generate_batch(n: int, seed: int, families: list[str] | None = None,
                   regime_weights: dict[str, float] | None = None) -> list[TaskSpec]:
    """生成 n 个实例;families 可给 family 名或 template_id(两者都接受)。

    # note (luojiaxuan): 各 regime 比例**每批随机抖动**(不固定 50/50),抖动量由
    # 传入 seed 派生 —— 免得 selector 学到一个跨批不变的先验。
    """
    pool = [t for t in TEMPLATES if families is None
            or t.family in families or t.template_id in families]
    if not pool:
        raise ValueError(f"families 过滤后模板为空: {families!r}")
    rng = random.Random(_mix(seed, 977))
    jit = [max(1e-6, (regime_weights or {}).get(r, 1.0))
           * 2.718281828 ** rng.uniform(-0.7, 0.7) for r in MEMORY_REGIMES]
    return [_build(pool[i % len(pool)], _mix(seed, i + 1) % 10_000_019,
                   rng.choices(list(MEMORY_REGIMES), weights=jit, k=1)[0])
            for i in range(n)]


def split_families(seed: int) -> dict[str, list[str]]:
    """按 **family/template** 划分(不是按实例):整族留出,同族实例不散落两侧。"""
    rng = random.Random(_mix(seed, 31))
    simple = [f for f in FAMILIES
              if max(t.tier for t in TEMPLATES if t.family == f) <= 3]
    ood = {rng.choice(simple), rng.choice([f for f in FAMILIES if f not in simple])}
    return {"train": [t.template_id for t in TEMPLATES if t.family not in ood],
            "syn_ood": [t.template_id for t in TEMPLATES if t.family in ood]}


def make_dataset(split: str, n: int, seed: int) -> list[TaskSpec]:
    """split ∈ {train, syn_iid, syn_ood};syn_iid = train 的模板 + 另一批实例 seed。"""
    sp = split_families(seed)
    if split == "syn_iid":
        return generate_batch(n, _mix(seed, 7717), families=sp["train"])
    if split not in sp:
        raise ValueError(f"未知 split: {split!r}")
    return generate_batch(n, _mix(seed, 1 if split == "train" else 2),
                          families=sp[split])


__all__ = ["PrimitiveResult", "PRIMITIVES", "TemplateDef", "TEMPLATES", "BY_ID",
           "FAMILIES", "generate_task", "generate_batch", "split_families",
           "make_dataset", "act_click", "act_type", "act_key", "act_scroll",
           "act_terminate"]
