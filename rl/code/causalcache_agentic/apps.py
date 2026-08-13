"""六个 mock 应用的窗口构造器(Phase 1 程序化 GUI 环境)。

# note (luojiaxuan): 本文件只做**构造期布局** —— widget.rect 全是屏幕绝对像素,
# 渲染期不再算布局、不再换行;全流程零随机。业务文本一律来自 spec,代码里只留
# 表头/字段名这类排版用静态标签。与 render/executor/tasks 的配套约定:
#   * 可编辑控件(text_field/text_area/可编辑 cell)以 **value 为准**、构造时 text="";
#     只读控件 text 与 value 同写(便于 widget_value_* 断言);渲染器请优先画 value,
#     value 为空再回退 text。
#   * 长列表:window.scroll_area 给出可滚区域,超出的控件正常摆在区域下方
#     (y 会超出屏幕),由渲染器按 scroll_y 平移并裁剪。
#   * 每个构造器支持 spec["app_id"];所有 wid 以 f"{app_id}::" 开头(多实例不撞)。
#   * 通用注入:spec["extra_effects"]={wid_suffix: [Effect,...] 或 {"click"/
#     "double_click"/"enter": [...]}}、spec["meta"]={wid_suffix: dict};suffix 可写全 wid
#     或省略前缀,写错的键报错;要加新控件 tasks.py 直接往 Window.widgets 追加。
"""

from __future__ import annotations

from typing import Any, Sequence

from .contract import (SCREEN_H, SCREEN_W, TASKBAR_H, Effect, Rect, ScreenState,
                       Widget, Window, make_taskbar)

# ---- 布局常量 --------------------------------------------------------------
# note (luojiaxuan): 依次为 窗口矩形(1920x1032)、标题栏高、工具栏高、窗口内边距、
# 列表行高、正文行高、表格行高、保守字宽(偏大,构造期换行/截断用,保证渲染不溢出)、
# 内容区上下边界(104 / 1008)。
WINDOW_RECT = Rect(0, 0, SCREEN_W, SCREEN_H - TASKBAR_H)
TITLE_H, TOOLBAR_H, PAD = 56, 48, 24
ROW_H, LINE_H, CELL_H, CHAR_W = 44, 34, 40, 10
CONTENT_TOP, CONTENT_BOTTOM = TITLE_H + TOOLBAR_H, WINDOW_RECT.h - PAD


# ---- 通用工具 --------------------------------------------------------------
def _fit(text: Any, width_px: int) -> str:
    """把文本截断到给定像素宽度内(构造期完成,渲染期不再重排)。"""
    n = max(4, width_px // CHAR_W)
    t = str(text)
    return t if len(t) <= n else t[: n - 3] + "..."


def _wrap(text: Any, width_px: int) -> list[str]:
    """确定性词换行:按保守字宽把一段文本切成若干行(超长单词硬切)。"""
    limit = max(8, width_px // CHAR_W)
    lines: list[str] = []
    cur = ""
    for word in str(text).split():
        while len(word) > limit:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:limit])
            word = word[limit:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= limit:
            cur = cur + " " + word
        else:
            lines.append(cur)
            cur = word
    if cur or not lines:
        lines.append(cur)
    return lines


def _eff(items: Any) -> tuple[Effect, ...]:
    """把 spec 给的 effect 列表(Effect 或 dict)规范成 tuple[Effect, ...]。"""
    return tuple(e if isinstance(e, Effect) else Effect(**dict(e))
                 for e in (items or ()))


def _w(wid: str, kind: str, rect: Rect, text: Any = "", value: Any = None,
       role: str = "", style: dict | None = None, **kw: Any) -> Widget:
    """建控件:text 按 rect 宽度截断;value 缺省等于未截断的原始 text。"""
    st = dict(style or {})
    if role:
        st["role"] = role
    return Widget(wid=wid, kind=kind, rect=rect, text=_fit(text, rect.w),
                  value=str(text) if value is None else str(value), style=st, **kw)


def _titlebar(app_id: str, title: str) -> Widget:
    return _w(f"{app_id}::titlebar", "title", Rect(0, 0, SCREEN_W, TITLE_H),
              title, role="titlebar")


def _scroll_max(area: Rect, content_bottom: int) -> int:
    return max(0, content_bottom + PAD - (area.y + area.h))


def _window(app_id: str, title: str, widgets: list[Widget], spec: dict,
            area: Rect, content_bottom: int,
            default_groups: Sequence[str] = ()) -> Window:
    """注入 spec 的 extra_effects / meta 后打包成 Window(键写错就报错)。"""
    index = {w.wid: w for w in widgets}

    def pick(key: str) -> Widget:
        w = index.get(str(key)) or index.get(f"{app_id}::{key}")
        if w is None:
            raise ValueError(f"{app_id}: 注入键 {key!r} 不存在;可用后缀示例 "
                             f"{[k.split('::', 1)[-1] for k in list(index)[:8]]}")
        return w

    for key, val in (spec.get("extra_effects") or {}).items():
        w = pick(key)
        if isinstance(val, dict):
            w.on_click += _eff(val.get("click"))
            w.on_double_click += _eff(val.get("double_click"))
            w.on_enter += _eff(val.get("enter"))
        else:
            w.on_click += _eff(val)
    for key, val in (spec.get("meta") or {}).items():
        pick(key).meta.update(dict(val))
    return Window(app_id=app_id, title=title, rect=WINDOW_RECT, widgets=widgets,
                  scroll_area=area, scroll_max=_scroll_max(area, content_bottom),
                  active_groups=tuple(spec.get("active_groups", default_groups)))


# ---- Files -----------------------------------------------------------------
FILES_ICON_X, FILES_NAME_X, FILES_SIZE_X, FILES_TIME_X = 32, 88, 1180, 1420
FILES_RIGHT = SCREEN_W - PAD


def build_files_app(spec: dict) -> Window:
    """文件管理器:路径面包屑 + 文件列表(图标/名称/大小/修改时间),行是 list_item。

    spec 键:app_id="files"、window_title、path、open_flag、select_flag、
    files=[{name,size,modified,content,id,icon,on_click,on_double_click,opens_app,meta}]。
    默认交互:单击 set_flag(select_flag ← name);双击 set_flag(open_flag ← name),
    该条目给了 opens_app 时再追加 open_app。行 widget 的 value = 该文件 content。
    """
    app_id = str(spec.get("app_id", "files"))
    path = str(spec.get("path", ""))
    title = str(spec.get("window_title", f"Files — {path}" if path else "Files"))
    open_flag = str(spec.get("open_flag", f"{app_id}_opened"))
    select_flag = spec.get("select_flag", f"{app_id}_selected")

    ws = [_titlebar(app_id, title),
          _w(f"{app_id}::path", "label", Rect(PAD, TITLE_H + 6, 1400, 36), path,
             role="breadcrumb")]
    for sfx, x, wpx, lab in (("name", FILES_NAME_X, 900, "Name"),
                             ("size", FILES_SIZE_X, 200, "Size"),
                             ("modified", FILES_TIME_X, 440, "Modified")):
        ws.append(_w(f"{app_id}::head::{sfx}", "label",
                     Rect(x, CONTENT_TOP, wpx, 36), lab, role="colhead"))
    top = CONTENT_TOP + 40
    area = Rect(PAD, top, SCREEN_W - 2 * PAD, CONTENT_BOTTOM - top)
    bottom = top
    for i, ent in enumerate(spec.get("files") or ()):
        fid, name = str(ent.get("id", i)), str(ent.get("name", ""))
        y = top + i * ROW_H
        bottom = y + ROW_H
        if "on_click" in ent:
            click = _eff(ent["on_click"])
        else:
            click = ((Effect("set_flag", target=str(select_flag), value=name),)
                     if select_flag else ())
        if "on_double_click" in ent:
            dbl = _eff(ent["on_double_click"])
        else:
            dbl = (Effect("set_flag", target=open_flag, value=name),)
            if ent.get("opens_app"):
                dbl += (Effect("open_app", target=str(ent["opens_app"])),)
        sty, base = {"alt": i % 2}, f"{app_id}::file::{fid}"
        # note (luojiaxuan): 行本体是 list_item(带 name);图标/大小/时间是同一行的
        # 旁挂控件并共用同一组 effect —— executor 命中其中任何一个,行为都一致。
        ws.append(_w(base, "list_item",
                     Rect(FILES_NAME_X, y, FILES_RIGHT - FILES_NAME_X, ROW_H - 2),
                     _fit(name, FILES_SIZE_X - FILES_NAME_X - 16),
                     value=ent.get("content", ""), role="row", style=sty,
                     on_click=click, on_double_click=dbl,
                     meta=dict(ent.get("meta") or {})))
        for sfx, kind, rect, txt in (
                ("::icon", "icon", Rect(FILES_ICON_X, y + 8, 48, ROW_H - 16),
                 ent.get("icon", "DOC")),
                ("::size", "label", Rect(FILES_SIZE_X, y, 200, ROW_H - 2),
                 ent.get("size", "")),
                ("::modified", "label",
                 Rect(FILES_TIME_X, y, FILES_RIGHT - FILES_TIME_X, ROW_H - 2),
                 ent.get("modified", ""))):
            ws.append(_w(base + sfx, kind, rect, txt, role="row", style=sty,
                         on_click=click, on_double_click=dbl))
    return _window(app_id, title, ws, spec, area, bottom)


# ---- Writer ----------------------------------------------------------------
WRITER_X, WRITER_W = 320, 1280


def build_writer_app(spec: dict) -> Window:
    """文档编辑器:文档标题 + 多段正文(每视觉行一个 label)+ 可编辑区 + 工具栏。

    spec 键:app_id="writer"、window_title、title、paragraphs=[str]、
    editable_id="body"(含 "::" 时视作完整 wid)、editable_text、editable_label、
    editable_height、save_flag、save_effects。
    "<app_id>::save" 默认:flag_from(save_flag ← 编辑区 value)+
    set_flag(f"{app_id}_save_clicked"="1")。
    """
    app_id = str(spec.get("app_id", "writer"))
    doc_title = str(spec.get("title", ""))
    title = str(spec.get("window_title", doc_title or "Writer"))
    eid = str(spec.get("editable_id", "body"))
    edit_wid = eid if "::" in eid else f"{app_id}::{eid}"
    save_flag = str(spec.get("save_flag", f"{app_id}_saved"))

    ws = [_titlebar(app_id, title),
          _w(f"{app_id}::save", "button",
             Rect(SCREEN_W - PAD - 140, TITLE_H + 6, 140, 36), "Save",
             role="primary",
             on_click=_eff(spec["save_effects"]) if "save_effects" in spec else (
                 Effect("flag_from", target=save_flag,
                        payload={"widget": edit_wid}),
                 Effect("set_flag", target=f"{app_id}_save_clicked", value="1")))]
    top = CONTENT_TOP + 16
    area = Rect(WRITER_X - 16, top, WRITER_W + 32, CONTENT_BOTTOM - top)
    y = top
    ws.append(_w(f"{app_id}::doctitle", "label", Rect(WRITER_X, y, WRITER_W, 52),
                 doc_title, role="heading"))
    y += 68
    for i, para in enumerate(spec.get("paragraphs") or ()):
        for j, line in enumerate(_wrap(para, WRITER_W)):
            ws.append(_w(f"{app_id}::para::{i}::L{j}", "label",
                         Rect(WRITER_X, y, WRITER_W, LINE_H), line, role="body"))
            y += LINE_H
        y += 16
    if spec.get("editable_label"):
        ws.append(_w(f"{app_id}::editable_label", "label",
                     Rect(WRITER_X, y, WRITER_W, LINE_H), spec["editable_label"],
                     role="fieldlabel"))
        y += LINE_H + 4
    edit_h = int(spec.get("editable_height", 140))
    ws.append(_w(edit_wid, "text_area", Rect(WRITER_X, y, WRITER_W, edit_h),
                 value=spec.get("editable_text", ""), role="editable"))
    return _window(app_id, title, ws, spec, area, y + edit_h)


# ---- Calc ------------------------------------------------------------------
CALC_ROWHDR_W = 80


def build_calc_app(spec: dict) -> Window:
    """表格:表头 + 网格 cell,cell wid 形如 "<app_id>::cell::R{r}C{c}"。

    spec 键:app_id="calc"、window_title、title、headers=[str]、rows=[[str]]、
    editable_cells=[(r,c) 或 "R{r}C{c}"]、col_width、
    cell_effects={"R0C1": [Effect,...]}。
    可编辑 cell:value=初值、text=""(以 value 为准);只读 cell 两者同写。
    """
    app_id = str(spec.get("app_id", "calc"))
    sheet = str(spec.get("title", ""))
    title = str(spec.get("window_title", sheet or "Calc"))
    headers = [str(h) for h in (spec.get("headers") or ())]
    rows = [[str(c) for c in r] for r in (spec.get("rows") or ())]
    ncols = max([len(headers)] + [len(r) for r in rows] + [1])
    avail = SCREEN_W - 2 * PAD - CALC_ROWHDR_W
    col_w = int(spec.get("col_width", 0)) or min(260, max(90, avail // ncols))
    if col_w * ncols > avail:            # note (luojiaxuan): 无横向滚动,必须夹紧
        col_w = avail // ncols
    editable = {e if isinstance(e, str) else f"R{int(e[0])}C{int(e[1])}"
                for e in (spec.get("editable_cells") or ())}
    cell_effects = spec.get("cell_effects") or {}

    ws = [_titlebar(app_id, title),
          _w(f"{app_id}::sheet", "label", Rect(PAD, TITLE_H + 6, 900, 36), sheet,
             role="toolbar"),
          _w(f"{app_id}::head::corner", "cell",
             Rect(PAD, CONTENT_TOP, CALC_ROWHDR_W, CELL_H), "", role="colhead")]
    for c in range(ncols):
        ws.append(_w(f"{app_id}::head::C{c}", "cell",
                     Rect(PAD + CALC_ROWHDR_W + c * col_w, CONTENT_TOP, col_w,
                          CELL_H), headers[c] if c < len(headers) else "",
                     role="colhead"))
    top = CONTENT_TOP + CELL_H
    area = Rect(PAD, top, CALC_ROWHDR_W + ncols * col_w, CONTENT_BOTTOM - top)
    bottom = top
    for r, row in enumerate(rows):
        y = top + r * CELL_H
        bottom = y + CELL_H
        ws.append(_w(f"{app_id}::rowhdr::R{r}", "cell",
                     Rect(PAD, y, CALC_ROWHDR_W, CELL_H), r + 1, role="rowhead"))
        for c in range(ncols):
            key, val = f"R{r}C{c}", row[c] if c < len(row) else ""
            ed = key in editable
            ws.append(_w(f"{app_id}::cell::{key}", "cell",
                         Rect(PAD + CALC_ROWHDR_W + c * col_w, y, col_w, CELL_H),
                         "" if ed else val, value=val, role="cell",
                         style={"editable": int(ed), "alt": r % 2},
                         on_click=_eff(cell_effects.get(key))))
    return _window(app_id, title, ws, spec, area, bottom)


# ---- Browser ---------------------------------------------------------------
BROWSER_X, BROWSER_W = 200, 1520


def build_browser_app(spec: dict) -> Window:
    """浏览器:地址栏(text_field)+ go 按钮 + 可选标签页 + 正文行 + 链接。

    spec 键:app_id="browser"、window_title、url、title、lines=[str 或 {text,group}]、
    links=[{text,url,id,tab,group,effects}]、tabs=[{id,text}]、go_effects、visit_flag。
    链接默认:set_value(地址栏 ← url)+ set_flag(visit_flag ← url 或 text),
    给了 tab 再追加 switch_tab。"<app_id>::go" 默认 flag_from 地址栏 + set_flag。
    """
    app_id = str(spec.get("app_id", "browser"))
    url, page_title = str(spec.get("url", "")), str(spec.get("title", ""))
    title = str(spec.get("window_title", page_title or "Browser"))
    url_wid = f"{app_id}::url"
    visit_flag = str(spec.get("visit_flag", f"{app_id}_visited"))

    ws = [_titlebar(app_id, title),
          _w(url_wid, "text_field", Rect(100, TITLE_H + 4, 1560, 40), value=url,
             role="address",
             on_enter=(Effect("set_flag", target=f"{app_id}_go", value="1"),)),
          _w(f"{app_id}::go", "button", Rect(1680, TITLE_H + 4, 120, 40), "Go",
             role="primary",
             on_click=_eff(spec["go_effects"]) if "go_effects" in spec else (
                 Effect("flag_from", target=f"{app_id}_url",
                        payload={"widget": url_wid}),
                 Effect("set_flag", target=f"{app_id}_go", value="1")))]
    top = CONTENT_TOP
    tabs = list(spec.get("tabs") or ())
    for i, tb in enumerate(tabs):
        tid = str(tb.get("id", i))
        ws.append(_w(f"{app_id}::tab::{tid}", "tab",
                     Rect(PAD + i * 240, top, 232, 36), tb.get("text", ""),
                     role="tab", on_click=(Effect("switch_tab", target=app_id,
                                                  payload={"tab": tid}),)))
    if tabs:
        top += 44
    area = Rect(BROWSER_X - 16, top, BROWSER_W + 32, CONTENT_BOTTOM - top)
    y = top + 8
    ws.append(_w(f"{app_id}::pagetitle", "label",
                 Rect(BROWSER_X, y, BROWSER_W, 52), page_title, role="heading"))
    y += 68
    for i, line in enumerate(spec.get("lines") or ()):
        grp = line.get("group") if isinstance(line, dict) else None
        text = line.get("text", "") if isinstance(line, dict) else line
        for j, seg in enumerate(_wrap(text, BROWSER_W)):
            ws.append(_w(f"{app_id}::line::{i}::L{j}", "label",
                         Rect(BROWSER_X, y, BROWSER_W, LINE_H), seg, role="body",
                         group=grp))
            y += LINE_H
        y += 8
    for i, ln in enumerate(spec.get("links") or ()):
        href, text = str(ln.get("url", "")), str(ln.get("text", ""))
        if "effects" in ln:
            eff = _eff(ln["effects"])
        else:
            eff = (Effect("set_value", target=url_wid, value=href),
                   Effect("set_flag", target=visit_flag, value=href or text))
            if ln.get("tab"):
                eff += (Effect("switch_tab", target=app_id,
                               payload={"tab": str(ln["tab"])}),)
        ws.append(_w(f"{app_id}::link::{ln.get('id', i)}", "button",
                     Rect(BROWSER_X, y, BROWSER_W, LINE_H), text, value=href,
                     role="link", group=ln.get("group"), on_click=eff))
        y += LINE_H + 6
    return _window(app_id, title, ws, spec, area, y)


# ---- Mail ------------------------------------------------------------------
MAIL_LIST_X, MAIL_LIST_W, MAIL_BODY_X = 316, 568, 916
MAIL_BODY_W, MAIL_MSG_H = SCREEN_W - MAIL_BODY_X - PAD, 72


def build_mail_app(spec: dict) -> Window:
    """邮件:左侧文件夹 + 中间邮件列表(list_item)+ 右侧正文;compose 时右下三栏。

    spec 键:app_id="mail"、window_title、folder、folders=[str]、compose=bool、
    compose_top、compose_to / compose_subject / compose_body(初值)、read_flag、
    send_flag_prefix、send_effects、messages=[{from,subject,body,id,on_click,meta}]。
    列表项默认:switch_tab(tab=f"msg::{id}")+ set_flag(read_flag ← subject);
    正文按 group=f"msg::{id}" 分组,默认只显示第一封(active_groups 可覆盖)。
    "<app_id>::send" 默认:flag_from 三个字段 → f"{p}_to"/f"{p}_subject"/f"{p}_body",
    再 set_flag(f"{p}"="1")(p = send_flag_prefix,默认 f"{app_id}_sent")。
    """
    app_id = str(spec.get("app_id", "mail"))
    folder = str(spec.get("folder", ""))
    title = str(spec.get("window_title", f"Mail — {folder}" if folder else "Mail"))
    compose = bool(spec.get("compose", False))
    read_flag = str(spec.get("read_flag", f"{app_id}_read"))
    prefix = str(spec.get("send_flag_prefix", f"{app_id}_sent"))
    body_bottom = int(spec.get("compose_top", 620)) if compose else CONTENT_BOTTOM

    ws = [_titlebar(app_id, title),
          _w(f"{app_id}::folderbar", "label", Rect(PAD, TITLE_H + 6, 900, 36),
             folder, role="toolbar")]
    top = CONTENT_TOP + 8
    for i, name in enumerate(spec.get("folders") or ([folder] if folder else ())):
        ws.append(_w(f"{app_id}::folder::{i}", "list_item",
                     Rect(16, top + i * ROW_H, 268, ROW_H - 2), name, role="nav",
                     on_click=(Effect("set_flag", target=f"{app_id}_folder",
                                      value=str(name)),)))
    area = Rect(MAIL_LIST_X - 8, top, MAIL_LIST_W + 16, CONTENT_BOTTOM - top)
    bottom, groups = top, []
    for i, m in enumerate(spec.get("messages") or ()):
        mid = str(m.get("id", i))
        grp = f"msg::{mid}"
        groups.append(grp)
        sender, subject = str(m.get("from", "")), str(m.get("subject", ""))
        y = top + i * MAIL_MSG_H
        bottom = y + MAIL_MSG_H
        eff = _eff(m["on_click"]) if "on_click" in m else (
            Effect("switch_tab", target=app_id, payload={"tab": grp}),
            Effect("set_flag", target=read_flag, value=subject))
        sty = {"alt": i % 2}
        # note (luojiaxuan): 列表项上半是发件人 label、下半是 list_item(主题),
        # 二者共用同一组 effect,点整行任意位置行为一致。
        ws.append(_w(f"{app_id}::msg::{mid}::from", "label",
                     Rect(MAIL_LIST_X, y + 4, MAIL_LIST_W, 30), sender,
                     role="row", style=sty, on_click=eff))
        ws.append(_w(f"{app_id}::msg::{mid}", "list_item",
                     Rect(MAIL_LIST_X, y + 34, MAIL_LIST_W, 32), subject,
                     role="row", style=sty, on_click=eff,
                     meta=dict(m.get("meta") or {})))
        # note (luojiaxuan): 正文按 group 分页,只有 active_groups 里的那封可见。
        ws.append(_w(f"{app_id}::body::{mid}::subject", "label",
                     Rect(MAIL_BODY_X, top, MAIL_BODY_W, 44), subject,
                     role="heading", group=grp))
        ws.append(_w(f"{app_id}::body::{mid}::from", "label",
                     Rect(MAIL_BODY_X, top + 48, MAIL_BODY_W, 34), sender,
                     role="meta", group=grp))
        by = top + 96
        room = max(1, (body_bottom - by) // LINE_H)
        for j, seg in enumerate(_wrap(m.get("body", ""), MAIL_BODY_W)[:room]):
            ws.append(_w(f"{app_id}::body::{mid}::L{j}", "label",
                         Rect(MAIL_BODY_X, by + j * LINE_H, MAIL_BODY_W, LINE_H),
                         seg, role="body", group=grp))
    if compose:
        cy = body_bottom + 12
        for key, lab, h in (("to", "To", 40), ("subject", "Subject", 40),
                            ("body", "Body", 160)):
            ws.append(_w(f"{app_id}::composelabel::{key}", "label",
                         Rect(MAIL_BODY_X, cy, 96, 34), lab, role="fieldlabel"))
            ws.append(_w(f"{app_id}::{key}", "text_field",
                         Rect(MAIL_BODY_X + 100, cy, MAIL_BODY_W - 100, h),
                         value=spec.get(f"compose_{key}", ""), role="editable"))
            cy += h + 12
        ws.append(_w(f"{app_id}::send", "button",
                     Rect(SCREEN_W - PAD - 140, min(cy, CONTENT_BOTTOM - 40), 140,
                          36), "Send", role="primary",
                     on_click=_eff(spec["send_effects"]) if "send_effects" in spec
                     else (Effect("flag_from", target=f"{prefix}_to",
                                  payload={"widget": f"{app_id}::to"}),
                           Effect("flag_from", target=f"{prefix}_subject",
                                  payload={"widget": f"{app_id}::subject"}),
                           Effect("flag_from", target=f"{prefix}_body",
                                  payload={"widget": f"{app_id}::body"}),
                           Effect("set_flag", target=prefix, value="1"))))
    return _window(app_id, title, ws, spec, area, bottom, groups[:1])


# ---- Settings --------------------------------------------------------------
SET_X, SET_LABEL_W, SET_VALUE_X, SET_VALUE_W, SET_ITEM_H = 200, 560, 800, 720, 52


def build_settings_app(spec: dict) -> Window:
    """设置:分节 + 每项 label/value,部分项可编辑(text_field)+ apply 按钮。

    spec 键:app_id="settings"、window_title、applied_flag、apply_effects、
    sections=[{name, items=[{label,value,id,editable,flag,meta}]}]。
    项 wid = f"{app_id}::item::{id}"(id 缺省 f"s{节序}i{项序}");"<app_id>::apply"
    默认:逐个可编辑项 flag_from(flag ← 该项 value,flag 缺省 f"{app_id}_{id}"),
    最后 set_flag(applied_flag="1")。
    """
    app_id = str(spec.get("app_id", "settings"))
    title = str(spec.get("window_title", "Settings"))
    applied_flag = str(spec.get("applied_flag", f"{app_id}_applied"))
    apply_x = SCREEN_W - PAD - 160

    ws, top = [_titlebar(app_id, title)], CONTENT_TOP
    area = Rect(SET_X - 16, top, SET_VALUE_X + SET_VALUE_W - SET_X + 32,
                CONTENT_BOTTOM - top)
    y, apply_eff = top, ()
    for si, sec in enumerate(spec.get("sections") or ()):
        ws.append(_w(f"{app_id}::section::{si}", "label",
                     Rect(SET_X, y, SET_LABEL_W + SET_VALUE_W, 44),
                     sec.get("name", ""), role="section"))
        y += 52
        for ii, item in enumerate(sec.get("items") or ()):
            iid = str(item.get("id", f"s{si}i{ii}"))
            wid, val = f"{app_id}::item::{iid}", str(item.get("value", ""))
            ed = bool(item.get("editable"))
            ws.append(_w(f"{app_id}::itemlabel::{iid}", "label",
                         Rect(SET_X, y, SET_LABEL_W, SET_ITEM_H - 8),
                         item.get("label", ""), role="fieldlabel"))
            ws.append(_w(wid, "text_field" if ed else "label",
                         Rect(SET_VALUE_X, y, SET_VALUE_W, SET_ITEM_H - 8),
                         "" if ed else val, value=val,
                         role="editable" if ed else "value",
                         meta=dict(item.get("meta") or {})))
            if ed:
                apply_eff += (Effect(
                    "flag_from", target=str(item.get("flag", f"{app_id}_{iid}")),
                    payload={"widget": wid}),)
            y += SET_ITEM_H
        y += 12
    ws.append(_w(f"{app_id}::apply", "button", Rect(apply_x, TITLE_H + 6, 160, 36),
                 "Apply", role="primary",
                 on_click=_eff(spec["apply_effects"]) if "apply_effects" in spec
                 else apply_eff + (Effect("set_flag", target=applied_flag,
                                          value="1"),)))
    return _window(app_id, title, ws, spec, area, y)


# ---- 组屏 ------------------------------------------------------------------
def compose_screen(windows: list[Window], active: str,
                   flags: dict | None = None) -> ScreenState:
    """把若干窗口组装成 ScreenState(z_order 里 active 在末位)。

    taskbar 策略:**把 taskbar widget 追加进每个窗口的 widgets**(每窗一份独立
    副本,不共享对象),而不是另起一个常驻任务栏窗口 —— 这样 z_order[-1] 永远是
    真正的活动应用(ScreenState.active_app 与 focus_app 语义不被任务栏窗口污染),
    且无论哪个全屏窗口在最上层任务栏都可见可点。重复调用幂等(先剔除已有
    "taskbar::" 前缀控件再追加);会就地改写传入的 Window 对象。
    make_taskbar 每格 240px,故最多 8 个窗口,超出直接报错而不是画到屏幕外。
    """
    ids = [w.app_id for w in windows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"app_id 重复:{ids}")
    if active not in ids:
        raise ValueError(f"active={active!r} 不在窗口列表 {ids} 中")
    if len(ids) > SCREEN_W // 240:
        raise ValueError(f"任务栏最多 {SCREEN_W // 240} 个应用,收到 {len(ids)}")
    titles = [w.title for w in windows]
    for win in windows:
        win.widgets = [w for w in win.widgets if not w.wid.startswith("taskbar::")]
        win.widgets.extend(make_taskbar(ids, titles))
    return ScreenState(windows={w.app_id: w for w in windows},
                       z_order=[a for a in ids if a != active] + [active],
                       flags=dict(flags or {}))


# ---- 自检 ------------------------------------------------------------------
def _self_check() -> None:
    """六个 app 各建一实例(含超屏长列表),断言不变量后逐个组屏打印。"""
    demos = [
        (build_files_app({
            "path": "/home/user/reports",
            "files": [{"name": f"report_{i}.txt", "size": f"{i + 1} KB",
                       "modified": "2026-08-01 09:12", "content": f"body {i}"}
                      for i in range(28)],
            "meta": {"file::3": {"carries": "order_id"}}}), []),
        (build_writer_app({
            "title": "Quarterly note", "paragraphs": ["alpha beta " * 30] * 3,
            "editable_id": "body", "editable_label": "Summary",
            "extra_effects": {"save": [Effect("set_flag", target="w_done",
                                              value="1")]}}), ["writer::body"]),
        (build_calc_app({
            "title": "Orders", "headers": ["id", "qty", "price", "total"],
            "rows": [[f"A{r}", str(r), "9.5", ""] for r in range(30)],
            "editable_cells": [(0, 3), "R1C3"]}),
         ["calc::cell::R0C3", "calc::cell::R1C3"]),
        (build_browser_app({
            "url": "http://intra/orders", "title": "Order 8891",
            "lines": ["long body line " * 12] * 4,
            "links": [{"text": "details", "url": "http://intra/orders/8891"}],
            "tabs": [{"id": "t0", "text": "Orders"}, {"id": "t1", "text": "S"}]}),
         ["browser::url"]),
        (build_mail_app({
            "folder": "Inbox", "folders": ["Inbox", "Sent"],
            "messages": [{"from": f"a{i}@x.io", "subject": f"Order {i}",
                          "body": "please confirm " * 20} for i in range(20)],
            "compose": True}), ["mail::to", "mail::subject", "mail::body"]),
        (build_settings_app({
            "sections": [{"name": f"Section {s}",
                          "items": [{"label": f"Item {s}{i}", "value": f"v{i}",
                                     "editable": i == 0} for i in range(6)]}
                         for s in range(4)]}), ["settings::item::s0i0"]),
    ]
    for win, editables in demos:
        wids = [w.wid for w in win.widgets]
        assert len(wids) == len(set(wids)), f"{win.app_id}: wid 冲突"
        assert all(w.startswith(f"{win.app_id}::") for w in wids)
        area = win.scroll_area
        for w in win.widgets:
            r = w.rect
            assert 0 <= r.x and r.x + r.w <= SCREEN_W, f"{w.wid} 横向越界"
            assert r.y >= 0 and r.h > 0 and r.w > 0, f"{w.wid} 尺寸非法"
            # note (luojiaxuan): 滚动区内的控件允许 y 超屏(渲染期按 scroll_y 裁剪)
            if not (area is not None and r.y >= area.y and r.x >= area.x - 1):
                assert r.y + r.h <= SCREEN_H, f"{w.wid} 竖直越界(非滚动区)"
        by_id = {w.wid: w for w in win.widgets}
        for e in editables:
            assert by_id[e].kind in ("text_field", "text_area", "cell"), f"{e} 类型错"
            assert by_id[e].text == "", f"{e} 应以 value 为准(text 必须为空)"
        st = compose_screen([win], win.app_id)
        top = st.windows[win.app_id]
        print(f"{win.app_id:9s} widgets={len(top.widgets):4d}(含 taskbar) "
              f"scroll_area={top.scroll_area} scroll_max={top.scroll_max:4d} "
              f"groups={top.active_groups} active={st.active_app}")
    st = compose_screen([w for w, _ in demos], "calc")
    assert st.z_order[-1] == "calc" and len(st.windows) == 6
    assert all(any(w.wid == "taskbar::mail" for w in win.widgets)
               for win in st.windows.values())
    print(f"multi     z_order={st.z_order};taskbar 每窗各一份")


if __name__ == "__main__":
    _self_check()
