"""Phase 1 渲染器:纯函数 `ScreenState -> PIL.Image`(RGB 1920x1080)。

# note (luojiaxuan): 路线决策 E1 —— PIL 直绘桌面 GUI,不起浏览器 / X server。三条硬约束:
#   * **纯函数且确定性**:同 state 渲染多次必须逐字节相同 —— 禁止 random / 时间 /
#     进程级哈希序(遍历只走 z_order 与 widget 列表这类显式顺序);
#   * **信息只走像素**:任务要模型记住的东西必须画出来;state.clipboard 这类不可见内部量
#     **不画**,否则记忆任务被短路;
#   * **高吞吐**:整帧只建一张 Image(可滚动窗口多一张裁剪层),文本掩膜跨帧缓存。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from causalcache_agentic.contract import (SCREEN_H, SCREEN_W, TASKBAR_H, Rect,
                                          ScreenState, Widget, Window)

__all__ = ["render", "render_to_file", "PALETTE", "APP_ACCENTS", "METRICS",
           "KIND_FONT_SIZE", "STYLE_KEYS"]

RGB = tuple[int, int, int]


def _hex(s: str) -> RGB:
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# ------------------------------------------------------------------ 常量表(换肤只改这里)
PALETTE: dict[str, RGB] = {k: _hex(v) for k, v in {
    "desktop_bg": "1E2835", "window_bg": "F6F7FA", "window_border": "8A939F",
    "titlebar_text": "FFFFFF", "titlebar_inactive_mix": "8C94A0", "text": "1A1F26",
    "text_muted": "5C6673", "text_disabled": "A2AAB4", "field_bg": "FFFFFF",
    "field_border": "969FAB", "field_sel": "BBD6FB", "button_bg": "E6EAF0",
    "button_hi": "FAFBFD", "button_border": "949DA9", "button_shadow": "BFC6D0",
    "row_even": "FFFFFF", "row_odd": "EFF2F6", "row_sel": "C9DDF7",
    "row_line": "DCE1E8", "cell_bg": "FFFFFF", "cell_header_bg": "E2E8F0",
    "grid": "B2BAC5", "tab_bg": "DDE2EA", "tab_active_bg": "FFFFFF",
    "tab_border": "969FAB", "focus": "1A73E8", "cursor": "14181E",
    "badge_bg": "C43D3D", "badge_text": "FFFFFF", "divider": "C4CBD4",
    "taskbar_bg": "252F3D", "taskbar_line": "151C26", "taskbar_btn": "384557",
    "taskbar_btn_active": "5976A6", "taskbar_text": "EEF1F5",
    "scroll_track": "E3E7ED", "scroll_thumb": "8F99A6", "scroll_border": "C4CBD4",
}.items()}
_P = PALETTE  # note (luojiaxuan): 模块内简写,外部一律用 PALETTE

# note (luojiaxuan): 按 app 家族给标题栏 accent —— 同屏多窗口时一眼可分。
APP_ACCENTS: dict[str, RGB] = {k: _hex(v) for k, v in {
    "files": "2E6CB8", "writer": "4A3E9E", "calc": "1E7A4B", "browser": "0E6E7C",
    "mail": "B24A2C", "settings": "4A5563", "notes": "8A6D1E", "chat": "8E2F6E",
}.items()}
ACCENT_CYCLE: tuple[RGB, ...] = tuple(APP_ACCENTS.values())

METRICS: dict[str, int] = {"titlebar_h": 40, "win_border": 2, "pad_x": 10, "radius": 6,
                           "focus_w": 3, "scrollbar_w": 14, "line_gap": 9}

KIND_FONT_SIZE: dict[str, int] = {
    "title": 30, "label": 20, "button": 20, "text_field": 20, "text_area": 19, "tab": 20,
    "list_item": 20, "cell": 19, "menu_item": 19, "icon": 17, "checkbox": 20, "badge": 15,
}

# note (luojiaxuan): renderer 认识的 widget.style 键(其余键一律忽略,不报错):
#   accent RGB 或 "#RRGGBB" | role taskbar|primary|danger|muted | active tab/任务栏当前项
#   selected 选中 | checked 勾选(缺省看 value) | header cell 表头 | row 行号(缺省按 y 排)
#   align left|center|right | mono 等宽 | bold 粗体 | size 覆盖字号 | sub 第二行小字
#   scroll 强制属于/不属于滚动层 | placeholder text_field/text_area 空值提示
STYLE_KEYS: tuple[str, ...] = ("accent", "role", "active", "selected", "checked",
    "header", "row", "align", "mono", "bold", "size", "sub", "scroll", "placeholder")

_CHECKED = ("1", "true", "yes", "on", "checked")
_LX, _MAC ="/usr/share/fonts/truetype/", "/System/Library/Fonts/"
_FONT_FILES: dict[str, tuple[str, ...]] = {
    "regular": (_LX + "dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                _LX + "liberation/LiberationSans-Regular.ttf",
                _MAC + "Supplemental/Arial.ttf", _MAC + "Helvetica.ttc",
                "/Library/Fonts/Arial.ttf", "/Library/Fonts/Roboto-Regular.ttf"),
    "bold": (_LX + "dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
             _LX + "liberation/LiberationSans-Bold.ttf",
             _MAC + "Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
             "/Library/Fonts/Roboto-Bold.ttf"),
    "mono": (_LX + "dejavu/DejaVuSansMono.ttf",
             "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
             _LX + "liberation/LiberationMono-Regular.ttf", _MAC + "Menlo.ttc",
             _MAC + "Supplemental/Courier New.ttf"),
}


# ------------------------------------------------------------------ 字体与文本
@lru_cache(maxsize=256)
def _font(size: int, weight: str = "regular") -> tuple[Any, int]:
    """按 (size, weight) 缓存字体,返回 (字体, faux-bold 描边宽度)。

    候选路径按序尝试(Linux DejaVu/Liberation → macOS Arial/Helvetica),全失败才退化
    load_default();没有真 bold 时用 regular + 1px 描边模拟,保证粗细层次处处都在。
    """
    for path in _FONT_FILES.get(weight, ()):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size), 0
            except OSError:
                continue
    if weight == "bold":
        return _font(size, "regular")[0], 1
    if weight == "mono":
        return _font(size, "regular")
    return ImageFont.load_default(), 0


@lru_cache(maxsize=256)
def _vm(font: Any) -> tuple[int, int]:
    """(把一行文字视觉中心对到给定 y 所需的绘制偏移, 行高)。"""
    try:
        _, y0, _, y1 = font.getbbox("Ahgq")
    except Exception:  # pragma: no cover
        return -8, 14
    return -((y0 + y1) // 2), max(1, y1 - y0)


def _pick_font(w: Widget) -> tuple[Any, int]:
    """控件字体:style 的 size/mono/bold 覆盖 KIND_FONT_SIZE 的默认。"""
    size = max(6, int(w.style.get("size", KIND_FONT_SIZE.get(w.kind, 20))))
    weight = "mono" if w.style.get("mono") else (
        "bold" if (w.style.get("bold") or w.kind == "title") else "regular")
    return _font(size, weight)


@lru_cache(maxsize=3072)
def _text_mask(font: Any, stroke: int, s: str) -> tuple[Image.Image, int, int]:
    """文字灰度掩膜 + 相对绘制点的偏移,带缓存。

    # note (luojiaxuan): FreeType 逐字形栅格化约 30µs/字,整屏上千字形就是几十 ms,
    # 而相邻帧文本绝大多数完全相同。缓存掩膜后重复文本只剩一次 `ImageDraw.bitmap`
    # (实测快 ~80 倍),且与直接 `draw.text` 输出**逐字节相同**。
    """
    try:
        bb = font.getbbox(s, stroke_width=stroke)
    except TypeError:  # note (luojiaxuan): 位图兜底字体不认 stroke_width
        bb = font.getbbox(s)
    m = Image.new("L", (max(1, bb[2] - bb[0] + 1), max(1, bb[3] - bb[1] + 1)), 0)
    ImageDraw.Draw(m).text((-bb[0], -bb[1]), s, font=font, fill=255,
                           stroke_width=stroke, stroke_fill=255)
    return m, bb[0], bb[1]


def _text(d: ImageDraw.ImageDraw, x: int, cy: int, s: str, fs: tuple[Any, int],
          fill: RGB) -> None:
    """在 (x, 视觉垂直中心 cy) 画一行文字;越界部分由底层自动裁剪。"""
    if not s:
        return
    m, ox, oy = _text_mask(fs[0], fs[1], s)
    d.bitmap((x + ox, cy + _vm(fs[0])[0] + oy), m, fill=fill)


def _tw(s: str, fs: tuple[Any, int]) -> float:
    try:
        return fs[0].getlength(s)
    except (AttributeError, TypeError):  # pragma: no cover
        bb = fs[0].getbbox(s)
        return float(bb[2] - bb[0])


def _fit(s: str, fs: tuple[Any, int], max_w: int) -> str:
    """按像素宽度截断,超出加省略号。"""
    if max_w <= 0 or not s:
        return ""
    if _tw(s, fs) <= max_w:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _tw(s[:mid] + "…", fs) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return (s[:lo] + "…") if lo else ""


def _wrap(s: str, fs: tuple[Any, int], max_w: int) -> list[str]:
    """按像素宽度贪心折行,保留显式换行,超长单词硬切。"""
    out: list[str] = []
    if max_w <= 0:
        return out
    for para in s.split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = word if not cur else cur + " " + word
            if _tw(cand, fs) <= max_w:
                cur = cand
                continue
            if cur:
                out.append(cur)
            while _tw(word, fs) > max_w and len(word) > 1:
                cut = 1
                while cut < len(word) and _tw(word[:cut + 1], fs) <= max_w:
                    cut += 1
                out.append(word[:cut])
                word = word[cut:]
            cur = word
        out.append(cur)
    return out


def _align_x(s: str, fs: tuple[Any, int], r: Rect, pad: int, align: str) -> int:
    if align == "center":
        return r.x + max(0, int((r.w - _tw(s, fs)) // 2))
    if align == "right":
        return r.x + max(pad, int(r.w - pad - _tw(s, fs)))
    return r.x + pad


# ------------------------------------------------------------------ 形状小工具
def _color(spec: Any, default: RGB) -> RGB:
    """接受 (r,g,b) / [r,g,b] / "#RRGGBB",其余一律回落 default。"""
    if isinstance(spec, (tuple, list)) and len(spec) == 3:
        return (int(spec[0]) & 255, int(spec[1]) & 255, int(spec[2]) & 255)
    if isinstance(spec, str) and len(spec) == 7 and spec.startswith("#"):
        try:
            return _hex(spec[1:])
        except ValueError:
            return default
    return default


def _blend(a: RGB, b: RGB, t: float) -> RGB:
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def _box(r: Rect, dy: int = 0) -> tuple[int, int, int, int]:
    return (r.x, r.y + dy, r.x + max(0, r.w - 1), r.y + dy + max(0, r.h - 1))


def _rrect(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int,
           fill: RGB | None, outline: RGB | None = None, width: int = 1) -> None:
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    rad = max(0, min(radius, (box[2] - box[0]) // 2, (box[3] - box[1]) // 2))
    d.rounded_rectangle(box, radius=rad, fill=fill, outline=outline, width=width)


def _caret(d: ImageDraw.ImageDraw, x: int, cy: int, h: int) -> None:
    """光标:一条 2px 竖线。**常亮不闪**,闪烁会让渲染不再是 state 的函数。"""
    d.rectangle((x, cy - h // 2, x + 1, cy + h // 2), fill=_P["cursor"])


@dataclass
class _Ctx:
    """一次窗口绘制期间共享的只读上下文(不含任何随机 / 时间来源)。"""

    accent: RGB
    focus_wid: str | None
    active_app: str | None
    select_all: bool
    row_of: dict[str, int] = field(default_factory=dict)


# ------------------------------------------------------------------ 控件绘制
def _draw_widget(d: ImageDraw.ImageDraw, w: Widget, ctx: _Ctx,
                 off: tuple[int, int]) -> None:
    r = Rect(w.rect.x + off[0], w.rect.y + off[1], w.rect.w, w.rect.h)
    kind = w.kind
    if kind == "divider":  # note (luojiaxuan): 无文字,先返回免得白取一次字体
        if r.h <= r.w:
            cy = r.y + r.h // 2
            d.line((r.x, cy, r.x + r.w - 1, cy), fill=_P["divider"], width=1)
        else:
            cx = r.x + r.w // 2
            d.line((cx, r.y, cx, r.y + r.h - 1), fill=_P["divider"], width=1)
        return

    accent, fs = _color(w.style.get("accent"), ctx.accent), _pick_font(w)
    fg = _P["text"] if w.enabled else _P["text_disabled"]
    pad, align = METRICS["pad_x"], str(w.style.get("align", "left"))

    if kind == "button":
        _draw_button(d, w, r, ctx, fs, accent)
    elif kind in ("text_field", "text_area"):
        _draw_input(d, w, r, ctx, fs, accent, kind == "text_area")
    elif kind == "list_item":
        _draw_list_item(d, w, r, ctx, fs, accent, fg, align)
    elif kind == "cell":
        _draw_cell(d, w, r, ctx, fs, fg, align)
    elif kind == "tab":
        _draw_tab(d, w, r, fs, accent)
    elif kind == "checkbox":
        _draw_checkbox(d, w, r, fs, fg, accent)
    elif kind == "icon":
        _draw_icon(d, w, r, fs, fg, accent)
    elif kind == "menu_item":
        if w.style.get("selected"):
            d.rectangle(_box(r), fill=_P["row_sel"])
        d.line((r.x + pad, r.y + r.h - 1, r.x + r.w - pad, r.y + r.h - 1),
               fill=_P["row_line"], width=1)
        _text(d, r.x + pad, r.y + r.h // 2, _fit(w.text, fs, r.w - 2 * pad), fs, fg)
    elif kind == "badge":
        _rrect(d, _box(r), r.h // 2, fill=_color(w.style.get("accent"), _P["badge_bg"]))
        lab = _fit(w.text or w.value, fs, r.w - 12)
        _text(d, _align_x(lab, fs, r, 6, "center"), r.y + r.h // 2, lab, fs,
              _P["badge_text"])
    else:  # note (luojiaxuan): title / label / 未知 kind 一律当文本画
        role = str(w.style.get("role", ""))
        col = {"muted": _P["text_muted"], "danger": _P["badge_bg"]}.get(
            role, fg if kind in ("title", "label") else _P["text_muted"])
        shown = _fit(w.text or w.value, fs, r.w)
        _text(d, _align_x(shown, fs, r, 0, align), r.y + r.h // 2, shown, fs, col)

    if ctx.focus_wid == w.wid:  # note (luojiaxuan): 焦点控件画一圈明显的外框
        g = METRICS["focus_w"]
        _rrect(d, (r.x - g, r.y - g, r.x + r.w - 1 + g, r.y + r.h - 1 + g),
               METRICS["radius"] + g, None, _P["focus"], g)


def _draw_button(d: ImageDraw.ImageDraw, w: Widget, r: Rect, ctx: _Ctx,
                 fs: tuple[Any, int], accent: RGB) -> None:
    role = str(w.style.get("role", ""))
    if role == "taskbar":
        # note (luojiaxuan): 高亮当前 app —— 目标从 on_click 的 focus/open_app 读。
        tgt = next((e.target for e in w.on_click
                    if e.kind in ("focus_app", "open_app") and e.target), None)
        act = (tgt or w.wid.split("::")[-1]) == ctx.active_app or bool(
            w.style.get("active"))
        bg = _P["taskbar_btn_active"] if act else _P["taskbar_btn"]
        _rrect(d, _box(r), METRICS["radius"], bg, _blend(bg, (255, 255, 255), 0.18), 1)
        if act:
            d.rectangle((r.x + 6, r.y + r.h - 4, r.x + r.w - 7, r.y + r.h - 2),
                        fill=_hex("E8EFFA"))
        lab = _fit(w.text, fs, r.w - 20)
        _text(d, _align_x(lab, fs, r, 10, "center"), r.y + r.h // 2, lab, fs,
              _P["taskbar_text"])
        return

    if not w.enabled:
        base, border, txt = _P["row_odd"], _P["button_shadow"], _P["text_disabled"]
    elif role in ("primary", "danger"):
        base = accent if role == "primary" else _P["badge_bg"]
        border, txt = _blend(base, (0, 0, 0), 0.25), (255, 255, 255)
    else:
        base, border, txt = _P["button_bg"], _P["button_border"], _P["text"]
    # note (luojiaxuan): 底部一条暗边 + 顶部一条亮边 = 廉价但有效的"悬浮感"。
    _rrect(d, _box(r, 2), METRICS["radius"], _P["button_shadow"])
    _rrect(d, _box(r), METRICS["radius"], base, border, 1)
    if w.enabled and role not in ("primary", "danger"):
        d.line((r.x + 3, r.y + 1, r.x + r.w - 4, r.y + 1), fill=_P["button_hi"])
    lab = _fit(w.text or w.value, fs, r.w - 16)
    _text(d, _align_x(lab, fs, r, 8, "center"), r.y + r.h // 2, lab, fs, txt)


def _draw_input(d: ImageDraw.ImageDraw, w: Widget, r: Rect, ctx: _Ctx,
                fs: tuple[Any, int], accent: RGB, multiline: bool) -> None:
    focused = ctx.focus_wid == w.wid
    d.rectangle(_box(r), fill=_P["field_bg"] if w.enabled else _P["row_odd"],
                outline=accent if focused else _P["field_border"],
                width=2 if focused else 1)
    pad, inner = 8, r.w - 16
    col = _P["text"] if w.enabled else _P["text_disabled"]
    line_h, cy0 = _vm(fs[0])[1] + 6, (r.y + r.h // 2 if not multiline else r.y + 20)
    if not w.value:
        ph = _fit(str(w.style.get("placeholder", w.text)), fs, inner)
        _text(d, r.x + pad, cy0, ph, fs, _P["text_disabled"])
        if focused:
            _caret(d, r.x + pad, cy0, line_h)
        return

    if not multiline:
        shown, tx, cy = _fit(w.value, fs, inner), r.x + pad, r.y + r.h // 2
        if focused and ctx.select_all:  # note (luojiaxuan): ctrl+a 后的选中高亮
            d.rectangle((tx, cy - line_h // 2, tx + int(_tw(shown, fs)),
                         cy + line_h // 2), fill=_P["field_sel"])
        _text(d, tx, cy, shown, fs, col)
        if focused:
            _caret(d, tx + int(_tw(shown, fs)) + 1, cy, line_h)
        return

    step, cx, cy = _vm(fs[0])[1] + METRICS["line_gap"], r.x + pad, cy0
    for ln in _wrap(w.value, fs, inner):
        if cy + step // 2 > r.y + r.h - 2:
            break
        _text(d, r.x + pad, cy, ln, fs, col)
        cx, cy = r.x + pad + int(_tw(ln, fs)) + 1, cy + step
    if focused:
        _caret(d, cx, cy - step, line_h)


def _draw_list_item(d: ImageDraw.ImageDraw, w: Widget, r: Rect, ctx: _Ctx,
                    fs: tuple[Any, int], accent: RGB, fg: RGB, align: str) -> None:
    idx = int(w.style.get("row", ctx.row_of.get(w.wid, 0)))
    sel = bool(w.style.get("selected"))
    d.rectangle(_box(r), fill=_P["row_sel"] if sel else (
        _P["row_odd"] if idx % 2 else _P["row_even"]))
    d.line((r.x, r.y + r.h - 1, r.x + r.w - 1, r.y + r.h - 1), fill=_P["row_line"])
    if sel:
        d.rectangle((r.x, r.y, r.x + 3, r.y + r.h - 1), fill=accent)
    pad, sub = METRICS["pad_x"] + 6, str(w.style.get("sub", ""))
    body = w.text or w.value
    if sub and r.h >= 44:
        sfs = _font(max(12, int(_vm(fs[0])[1] * 0.85)), "regular")
        _text(d, r.x + pad, r.y + r.h // 2 - 11, _fit(body, fs, r.w - 2 * pad), fs, fg)
        _text(d, r.x + pad, r.y + r.h // 2 + 12, _fit(sub, sfs, r.w - 2 * pad), sfs,
              _P["text_muted"])
        return
    shown = _fit(body, fs, r.w - 2 * pad)
    _text(d, _align_x(shown, fs, r, pad, align), r.y + r.h // 2, shown, fs, fg)


def _draw_cell(d: ImageDraw.ImageDraw, w: Widget, r: Rect, ctx: _Ctx,
               fs: tuple[Any, int], fg: RGB, align: str) -> None:
    header = bool(w.style.get("header"))
    idx = int(w.style.get("row", ctx.row_of.get(w.wid, 0)))
    if header:
        bg = _P["cell_header_bg"]
    elif w.style.get("selected"):
        bg = _P["row_sel"]
    else:
        bg = _P["cell_bg"] if idx % 2 == 0 else _P["row_odd"]
    d.rectangle(_box(r), fill=bg, outline=_P["grid"], width=1)
    use = _font(int(w.style.get("size", KIND_FONT_SIZE["cell"])), "bold") if header \
        else fs
    shown = _fit(w.text if header else (w.value or w.text), use, r.w - 12)
    _text(d, _align_x(shown, use, r, 6, align), r.y + r.h // 2, shown, use,
          _P["text"] if header else fg)


def _draw_tab(d: ImageDraw.ImageDraw, w: Widget, r: Rect, fs: tuple[Any, int],
              accent: RGB) -> None:
    active = bool(w.style.get("active"))
    bg = _P["tab_active_bg"] if active else _P["tab_bg"]
    box, rad = _box(r), METRICS["radius"]
    _rrect(d, box, rad, bg, _P["tab_border"], 1)
    # note (luojiaxuan): 抹掉下缘圆角 → 视觉上与内容区连成一体,像真的选项卡。
    d.rectangle((box[0] + 1, box[3] - rad, box[2] - 1, box[3]), fill=bg)
    d.line((box[0], box[3] - rad, box[0], box[3]), fill=_P["tab_border"])
    d.line((box[2], box[3] - rad, box[2], box[3]), fill=_P["tab_border"])
    if active:
        d.rectangle((box[0] + 1, box[1] + 1, box[2] - 1, box[1] + 3), fill=accent)
    lab = _fit(w.text or w.value, fs, r.w - 16)
    _text(d, _align_x(lab, fs, r, 8, "center"), r.y + r.h // 2, lab, fs,
          _P["text"] if active else _P["text_muted"])


def _draw_checkbox(d: ImageDraw.ImageDraw, w: Widget, r: Rect, fs: tuple[Any, int],
                   fg: RGB, accent: RGB) -> None:
    side = max(14, min(22, r.h - 4))
    bx, by = r.x + 2, r.y + (r.h - side) // 2
    on = bool(w.style.get("checked")) or w.value.strip().lower() in _CHECKED
    d.rectangle((bx, by, bx + side, by + side), fill=accent if on else _P["field_bg"],
                outline=_P["field_border"], width=1)
    if on:
        d.line((bx + side * .22, by + side * .52, bx + side * .44, by + side * .74),
               fill=(255, 255, 255), width=3)
        d.line((bx + side * .44, by + side * .74, bx + side * .79, by + side * .26),
               fill=(255, 255, 255), width=3)
    tx = bx + side + 10
    _text(d, tx, r.y + r.h // 2, _fit(w.text, fs, r.x + r.w - tx), fs, fg)


def _draw_icon(d: ImageDraw.ImageDraw, w: Widget, r: Rect, fs: tuple[Any, int],
               fg: RGB, accent: RGB) -> None:
    """图标 = accent 圆角方块 + 首字母;矩形够高时下方再加一行说明文字。

    # note (luojiaxuan): 说明文字必须先确认放得下 —— 文件列表里的行内图标只有
    # 28px 高,硬画说明文字会压到下一行上去(踩过:整列 "DOC" 骑在相邻行上)。
    """
    caption = r.h >= 44
    side = max(16, min(48, r.w - 8, r.h - 22 if caption else r.h - 4))
    bx = r.x + (r.w - side) // 2
    by = r.y + (6 if caption else max(0, (r.h - side) // 2))
    _rrect(d, (bx, by, bx + side, by + side), 8, accent)
    gfs, glyph = _font(22, "bold"), (w.text or w.value or "?").strip()[:1].upper()
    _text(d, bx + max(2, (side - int(_tw(glyph, gfs))) // 2), by + side // 2, glyph,
          gfs, (255, 255, 255))
    if caption:
        cap = _fit(w.text, fs, r.w - 4)
        _text(d, r.x + max(0, (r.w - int(_tw(cap, fs))) // 2), by + side + 14, cap,
              fs, fg)


# ------------------------------------------------------------------ 窗口绘制
def _accent_for(win: Window) -> RGB:
    """窗口 accent:widget style 显式指定 > app 家族表 > 确定性回落。"""
    for w in win.widgets:
        if "accent" in w.style and w.kind in ("title", "tab", "badge"):
            return _color(w.style["accent"], _P["taskbar_btn_active"])
    for key, col in APP_ACCENTS.items():
        if win.app_id == key or win.app_id.startswith(key):
            return col
    for key, col in APP_ACCENTS.items():
        if key in win.app_id:
            return col
    # note (luojiaxuan): 用 ord 求和做索引,**不用 hash()**(hash 每进程加盐 → 非确定)。
    return ACCENT_CYCLE[sum(ord(c) for c in win.app_id) % len(ACCENT_CYCLE)]


def _in_scroll(w: Widget, area: Rect) -> bool:
    """滚动层归属:style["scroll"] 显式优先,否则"水平重叠且不完全在区域上方"。"""
    if "scroll" in w.style:
        return bool(w.style["scroll"])
    return (str(w.style.get("role", "")) != "taskbar"  # note (luojiaxuan): 任务栏不滚
            and w.rect.x < area.x + area.w and w.rect.x + w.rect.w > area.x
            and w.rect.y + w.rect.h > area.y)


def _row_index_map(widgets: list[Widget]) -> dict[str, int]:
    """未给 style["row"] 时按**纵坐标**给 list_item / cell 编行号(同 y = 同一行)。"""
    out: dict[str, int] = {}
    for kind in ("list_item", "cell"):
        pos = {y: i for i, y in
               enumerate(sorted({w.rect.y for w in widgets if w.kind == kind}))}
        out.update({w.wid: pos[w.rect.y] for w in widgets if w.kind == kind})
    return out


def _draw_titlebar(d: ImageDraw.ImageDraw, win: Window, accent: RGB, act: bool) -> None:
    r, th = win.rect, METRICS["titlebar_h"]
    bar = accent if act else _blend(accent, _P["titlebar_inactive_mix"], .55)
    d.rectangle((r.x, r.y, r.x + r.w - 1, r.y + th - 1), fill=bar)
    d.line((r.x, r.y + th - 1, r.x + r.w - 1, r.y + th - 1),
           fill=_blend(bar, (0, 0, 0), 0.25))
    tfs, col = _font(21, "bold"), _P["titlebar_text"]
    _text(d, r.x + 16, r.y + th // 2, _fit(win.title, tfs, r.w - 160), tfs, col)
    # note (luojiaxuan): 最小化/最大化/关闭用线条画,不依赖字体是否有对应字形。
    cy, bx = r.y + th // 2, r.x + r.w - 30
    d.line((bx - 6, cy + 6, bx + 6, cy - 6), fill=col, width=2)
    d.line((bx - 6, cy - 6, bx + 6, cy + 6), fill=col, width=2)
    d.rectangle((bx - 46, cy - 6, bx - 34, cy + 6), outline=col, width=2)
    d.line((bx - 86, cy + 5, bx - 74, cy + 5), fill=col, width=2)


def _draw_window(img: Image.Image, d: ImageDraw.ImageDraw, win: Window,
                 state: ScreenState) -> None:
    accent = _accent_for(win)
    d.rectangle(_box(win.rect), fill=_P["window_bg"], outline=_P["window_border"],
                width=METRICS["win_border"])
    _draw_titlebar(d, win, accent, state.active_app == win.app_id)

    groups = set(win.active_groups)
    # note (luojiaxuan): 两类控件不在这里画:
    #   * role="taskbar" —— 任务栏恒在最上层,由 _draw_taskbar 在所有窗口之后统一画,
    #     与 executor.hit_test 的 _taskbar_hit(任务栏优先命中)口径一致;
    #   * role="titlebar" —— 窗口标题由 _draw_titlebar 从 win.title 画成真实窗口装饰,
    #     app 构造器那份同文本控件再画一遍就会与它错位重影(踩过)。
    vis = [w for w in win.widgets
           if w.visible and (w.group is None or w.group in groups)
           and str(w.style.get("role", "")) not in ("taskbar", "titlebar")]
    ctx = _Ctx(accent=accent,
               focus_wid=(state.focus[1] if state.focus
                          and state.focus[0] == win.app_id else None),
               active_app=state.active_app, select_all=state.select_all,
               row_of=_row_index_map(vis))

    area = win.scroll_area
    scrolled: list[Widget] = []
    for w in vis:
        if area is not None and _in_scroll(w, area):
            scrolled.append(w)  # note (luojiaxuan): 留到裁剪层再画
        else:
            _draw_widget(d, w, ctx, (0, 0))
    if area is None:
        return

    # note (luojiaxuan): 滚动区单独画一层再贴回来 —— 超出区域的控件自然被裁掉。
    layer = Image.new("RGB", (max(1, area.w), max(1, area.h)), _P["window_bg"])
    ld = ImageDraw.Draw(layer)
    sy = max(0, min(int(win.scroll_y), max(0, int(win.scroll_max))))
    for w in scrolled:
        _draw_widget(ld, w, ctx, (-area.x, -(area.y + sy)))
    img.paste(layer, (area.x, area.y))
    d.rectangle(_box(area), outline=_P["scroll_border"], width=1)

    x0, track = area.x + area.w - METRICS["scrollbar_w"], area.h - 4
    d.rectangle((x0, area.y, area.x + area.w - 1, area.y + area.h - 1),
                fill=_P["scroll_track"], outline=_P["scroll_border"], width=1)
    if win.scroll_max <= 0:
        th, ty = track, area.y + 2
    else:
        th = max(36, int(track * area.h / float(area.h + win.scroll_max)))
        ty = area.y + 2 + int((track - th) * min(1.0, sy / float(win.scroll_max)))
    _rrect(d, (x0 + 2, ty, area.x + area.w - 3, ty + th - 1), 5, _P["scroll_thumb"])


# ------------------------------------------------------------------ 任务栏
def _draw_taskbar(d: ImageDraw.ImageDraw, state: ScreenState) -> None:
    """底板 + step 计数 + 任务栏按钮。**在所有窗口之后画**,恒在最上层。

    # note (luojiaxuan): 顺序与 executor.hit_test 一致(它先扫任务栏再扫窗口)——
    # 若先画任务栏再画窗口,一个覆盖到底部的窗口就会把按钮盖住却仍然点得到,
    # "看得见的"和"点得到的"分叉是最难查的一类环境 bug。
    # 右下角 step 计数当"时钟":它来自 state,确定性;真实时间会破坏逐位复现。
    """
    y0 = SCREEN_H - TASKBAR_H
    d.rectangle((0, y0, SCREEN_W - 1, SCREEN_H - 1), fill=_P["taskbar_bg"])
    d.line((0, y0, SCREEN_W - 1, y0), fill=_P["taskbar_line"], width=2)
    sfs, lab = _font(18, "regular"), f"step {state.step:02d}"
    _text(d, SCREEN_W - 20 - int(_tw(lab, sfs)), y0 + TASKBAR_H // 2, lab, sfs,
          _P["taskbar_text"])

    # note (luojiaxuan): 任务栏按钮可能在每个窗口里各有一份副本(apps.compose_screen)
    # 或只存在于一个 shell 窗口里(tasks.py)。取 z_order 从顶往下**第一个**带任务栏
    # 按钮的开着的窗口画一遍即可,两种摆法都对,且不会重复绘制。
    for app_id in reversed(state.z_order):
        win = state.windows.get(app_id)
        if win is None or not win.open:
            continue
        btns = [w for w in win.widgets
                if str(w.style.get("role", "")) == "taskbar" and w.visible]
        if not btns:
            continue
        ctx = _Ctx(accent=_accent_for(win),
                   # note (luojiaxuan): 任务栏 wid 全局唯一,焦点按 wid 比即可 ——
                   # 焦点可能落在另一个窗口里的同名副本上。
                   focus_wid=(state.focus[1] if state.focus else None),
                   active_app=state.active_app, select_all=state.select_all)
        for w in btns:
            _draw_widget(d, w, ctx, (0, 0))
        return


# ------------------------------------------------------------------ 顶层 API
def render(state: ScreenState) -> Image.Image:
    """把 ScreenState 画成 1920x1080 RGB 截图。纯函数:不改入参、不含随机。"""
    img = Image.new("RGB", (SCREEN_W, SCREEN_H), _P["desktop_bg"])
    d = ImageDraw.Draw(img)
    for app_id in state.z_order:  # note (luojiaxuan): 底 → 顶,末位是活动窗口
        win = state.windows.get(app_id)
        if win is not None and win.open:
            _draw_window(img, d, win, state)
    _draw_taskbar(d, state)
    return img


def render_to_file(state: ScreenState, path: str, *, compress_level: int = 6) -> str:
    """渲染并存 PNG,返回该路径。默认 level 6:实测与 level 1 同速但小 25%。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    render(state).save(path, format="PNG", compress_level=compress_level)
    return path
