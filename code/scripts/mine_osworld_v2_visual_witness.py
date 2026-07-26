#!/usr/bin/env python3
"""OSWorld 2.0 官方轨迹的 visual-history witness 第一轮挖掘。

# note (luojiaxuan): 数据 = xlangai/osworld2.0-trajectory 里 score>=0.95 的 13 条
# model-task 轨迹(官方 evaluator 通过,trace action 当 gold)。三种 action log:
#   * claude 系:``action.input`` 是 anthropic computer-use 语义动作(模型坐标系),
#     ``command`` 是原生像素 pyautogui —— 用 input 的语义 + 每轨迹对 click 对拟合
#     x/y 线性 scale 把模型坐标换算到原生像素;
#   * qwen:``action`` 是单语句原生 pyautogui(像素坐标)+ WAIT/DONE 哨兵;
#   * gpt:``action.action`` 是多语句 pyautogui 程序(还有无截图的 batch 步),
#     宏步只进文字历史,只有能约化成单原语的步才能作 target。
# 帧语义:step k 的 png 是该步动作执行后的观测(step_1 的动作就是 screenshot),
# 故决策点 t 的当前截图 = t-1 的 png,gold = t 的动作。
# 决策点 = "离开 source/reference view、进入 target edit" 的第一/第三步(ref 段
# 由每任务关键词表对 response 文本判定)。四臂:B(r=0)/ B+selected(段末 ref 帧)
# / B+nearby-irrelevant / B+next-recent,另加 ref 池与 random-old 候选做全表挖掘。
# 正例判据(放宽):修复 base-wrong,或显著抬升 gold 的 logprob/margin。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

MINE_SCHEMA = "causalcache.osworld_v2_visual_witness_mine.v1"

SELECTED_PAIRS = [
    ("claude-opus-4-7", "003"), ("claude-opus-4-7", "074"),
    ("claude-sonnet-4-6-max", "003"), ("claude-sonnet-4-6-max", "074"),
    ("claude-sonnet-4-6-medium", "074"),
    ("gpt-5.5", "003"), ("gpt-5.5", "105"), ("gpt-5.5", "107"),
    ("gpt-5.6", "003"), ("gpt-5.6", "032"), ("gpt-5.6", "105"), ("gpt-5.6", "107"),
    ("qwen3.7", "074"),
]

# note (luojiaxuan): reference-view 判定关键词(对每步 response 文本,casefold 后
# 子串匹配)。逐任务人工定,宁可偏召回 —— 决策点在 ref 段结束的下一步,多标一个
# ref 步只是挪动段边界,不会伪造证据。
REFERENCE_KEYWORDS = {
    "003": ["picture", "filter", "hong kong", "image viewer", "eog", "photo",
            "city.zip", "filter.zip", "unzip", "thumbnail"],
    "032": ["blog_homepage", "screenshot", "theme", "homepage image",
            "reference image", "saved image"],
    "074": ["toolathlon", "requirement", "reference", "os-world.github.io",
            "original site", "target design", "requirement.txt", "document"],
    "105": ["flair", "t1", "t2", "segmentation", "slice", "volume", "axial",
            "coronal", "sagittal", "overlay", "label"],
    "107": ["brief", "datasheet", "design brief", "pdf", "ad8232", "pinout",
            "reference"],
}
EDIT_HINTS = {
    "003": ["gimp", "layer", "blend", "composite", "export", "pptx", "slide"],
    "032": ["hexo", "terminal", "config", "install", "serve", "_config"],
    "074": ["html", "css", "editor", "code", "vscode", "index", "edit"],
    "105": ["paint", "erase", "correct", "fix", "adjust", "edit", "brush"],
    "107": ["kicad", "schematic", "wire", "place", "footprint", "symbol"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "score", "reduce"], required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="osworld2_traj/website_demo 的上级目录")
    parser.add_argument("--instructions", type=Path, required=True,
                        help="task_id -> instruction 的 JSON")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-dp-per-traj", type=int, default=8)
    parser.add_argument("--max-ref-pool", type=int, default=12)
    # score/reduce 共用
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--snapshot-manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-cache", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--visual-tokens", type=int, default=480)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--coord-tol-loose", type=int, default=50)
    parser.add_argument("--coord-tol-strict", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--per-unit-output", type=Path, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 逐模型 action 解析
# ---------------------------------------------------------------------------


def _norm999_px(px: float, size: int) -> int:
    value = round(px * 999 / (size - 1))
    return max(0, min(999, value))


class StepParse:
    """一步的解析产物:history mapping(必有)+ 可选 target tool_call。"""

    def __init__(self, history: dict[str, Any], target: dict[str, Any] | None,
                 reason: str | None) -> None:
        self.history = history
        self.target = target
        self.reason = reason


def _target(args: dict[str, Any]) -> dict[str, Any]:
    return {"name": "computer_use", "arguments": args}


def _statements(code: str) -> list[tuple[str, list[Any], dict[str, Any]]]:
    """把 pyautogui 程序拆成 (qualified, args, kwargs) 列表;忽略 import/sleep/
    PAUSE 包裹等已知噪声语句;遇到未知结构抛 ValueError。"""
    module = ast.parse(code)
    calls: list[tuple[str, list[Any], dict[str, Any]]] = []

    def literal(node: ast.expr) -> Any:
        return ast.literal_eval(node)

    def walk_body(body: list[ast.stmt]) -> None:
        for statement in body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(statement, ast.Assign):
                continue  # _osworld_original_pyautogui_pause = ... / PAUSE = ...
            if isinstance(statement, ast.Try):
                walk_body(statement.body)
                for handler in statement.handlers:
                    walk_body(handler.body)
                walk_body(statement.finalbody)
                continue
            if not isinstance(statement, ast.Expr) or not isinstance(
                statement.value, ast.Call
            ):
                raise ValueError("unsupported_statement")
            call = statement.value
            if not isinstance(call.func, ast.Attribute) or not isinstance(
                call.func.value, ast.Name
            ):
                raise ValueError("unsupported_call")
            qualified = f"{call.func.value.id}.{call.func.attr}"
            if qualified in {"time.sleep", "pyautogui.sleep", "pyautogui.screenshot",
                             "pyautogui.FAILSAFE"}:
                calls.append((qualified, [], {}))
                continue
            calls.append((
                qualified,
                [literal(arg) for arg in call.args],
                {kw.arg: literal(kw.value) for kw in call.keywords if kw.arg},
            ))

    walk_body(module.body)
    return calls


_PY_NOOPS = {"time.sleep", "pyautogui.sleep", "pyautogui.screenshot"}


def _map_pyautogui_call(
    qualified: str, args: list[Any], kwargs: dict[str, Any],
    *, screen: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """单条原生像素 pyautogui 调用 -> (history mapping, target args, 不可 target 理由)。"""
    width, height = screen

    def xy() -> tuple[float, float]:
        if "x" in kwargs and "y" in kwargs:
            return float(kwargs["x"]), float(kwargs["y"])
        if len(args) >= 2 and all(type(v) in (int, float) for v in args[:2]):
            return float(args[0]), float(args[1])
        raise ValueError("missing_coordinate")

    def coord999() -> list[int]:
        x, y = xy()
        return [_norm999_px(x, width), _norm999_px(y, height)]

    if qualified in _PY_NOOPS:
        return {"type": "screenshot"}, None, "noop_step"
    if qualified == "pyautogui.click":
        button = kwargs.get("button", "left")
        clicks = kwargs.get("clicks", 1)
        name = {"left": "left_click", "right": "right_click",
                "middle": "middle_click"}.get(button)
        if name is None:
            raise ValueError("unknown_button")
        if clicks == 2:
            name = "double_click"
        elif clicks not in (1, 2):
            return ({"type": "click", "raw": f"clicks={clicks}"}, None,
                    "multi_click_not_in_computer_use")
        x, y = xy()
        history = {"type": {"left_click": "click", "right_click": "right_click",
                            "middle_click": "click", "double_click": "double_click"}[name],
                   "x": round(x), "y": round(y)}
        if name == "middle_click":
            history["button"] = "middle"
        return history, {"action": name, "coordinate": coord999()}, None
    if qualified == "pyautogui.doubleClick":
        x, y = xy()
        return ({"type": "double_click", "x": round(x), "y": round(y)},
                {"action": "double_click", "coordinate": coord999()}, None)
    if qualified == "pyautogui.rightClick":
        x, y = xy()
        return ({"type": "right_click", "x": round(x), "y": round(y)},
                {"action": "right_click", "coordinate": coord999()}, None)
    if qualified == "pyautogui.middleClick":
        x, y = xy()
        return ({"type": "click", "x": round(x), "y": round(y), "button": "middle"},
                {"action": "middle_click", "coordinate": coord999()}, None)
    if qualified == "pyautogui.tripleClick":
        x, y = xy()
        return ({"type": "click", "x": round(x), "y": round(y), "clicks": 3},
                None, "triple_click_not_in_computer_use")
    if qualified == "pyautogui.moveTo":
        x, y = xy()
        return ({"type": "move", "x": round(x), "y": round(y)},
                {"action": "mouse_move", "coordinate": coord999()}, None)
    if qualified == "pyautogui.dragTo":
        x, y = xy()
        return ({"type": "drag", "x": round(x), "y": round(y)},
                {"action": "left_click_drag", "coordinate2": coord999()}, None)
    if qualified in {"pyautogui.write", "pyautogui.typewrite"}:
        text = kwargs.get("message", kwargs.get("text"))
        if text is None and args:
            text = args[0]
        if not isinstance(text, str):
            raise ValueError("write_text_invalid")
        return ({"type": "type_text", "text": text},
                {"action": "type", "text": text}, None)
    if qualified == "pyautogui.press":
        key = args[0] if args else kwargs.get("keys")
        if isinstance(key, list):
            if len(key) != 1:
                raise ValueError("press_key_list")
            key = key[0]
        if not isinstance(key, str) or not key:
            raise ValueError("press_key_invalid")
        return ({"type": "press", "key": key.casefold()},
                {"action": "key", "keys": [key.casefold()]}, None)
    if qualified == "pyautogui.hotkey":
        keys = list(args[0]) if len(args) == 1 and isinstance(args[0], list) else list(args)
        if not keys or not all(isinstance(k, str) and k for k in keys):
            raise ValueError("hotkey_keys_invalid")
        keys = [k.casefold() for k in keys]
        if len(keys) == 1:
            return ({"type": "press", "key": keys[0]},
                    {"action": "key", "keys": keys}, None)
        if len(keys) > 5:
            raise ValueError("hotkey_too_many_keys")
        return ({"type": "hotkey", "keys": keys}, {"action": "key", "keys": keys}, None)
    if qualified in {"pyautogui.scroll", "pyautogui.hscroll"}:
        amount = args[0] if args else kwargs.get("clicks")
        if type(amount) is not int or amount == 0:
            raise ValueError("scroll_amount_invalid")
        if qualified == "pyautogui.scroll":
            return ({"type": "scroll", "dy": amount},
                    {"action": "scroll", "pixels": amount}, None)
        return ({"type": "scroll", "dy": 0, "dx": amount},
                {"action": "hscroll", "pixels": amount}, None)
    if qualified in {"pyautogui.keyDown", "pyautogui.keyUp"}:
        key = args[0] if args else ""
        return ({"type": "press", "key": str(key).casefold()}, None, "key_half_stroke")
    raise ValueError(f"unknown_primitive:{qualified}")


def parse_native_pyautogui(code: str, *, screen: tuple[int, int]) -> StepParse:
    """qwen / gpt 的原生像素 pyautogui 程序。单原语步才有 target。"""
    stripped = code.strip()
    if stripped in {"WAIT", "wait"}:
        return StepParse({"type": "wait"}, _target({"action": "wait"}), None)
    if stripped in {"DONE", "done"}:
        return StepParse({"type": "done"},
                         _target({"action": "terminate", "status": "success"}), None)
    if stripped in {"FAIL", "fail"}:
        return StepParse({"type": "fail"},
                         _target({"action": "terminate", "status": "failure"}), None)
    try:
        calls = _statements(stripped)
    except (ValueError, SyntaxError) as error:
        return StepParse({"type": "unparsed", "raw": stripped[:120]}, None,
                         f"unparseable:{error}")
    semantic = [c for c in calls if c[0] not in _PY_NOOPS]
    if not semantic:
        return StepParse({"type": "screenshot"}, None, "noop_step")
    if len(semantic) == 1:
        qualified, args, kwargs = semantic[0]
        try:
            history, target_args, reason = _map_pyautogui_call(
                qualified, args, kwargs, screen=screen)
        except ValueError as error:
            return StepParse({"type": "unparsed", "raw": stripped[:120]}, None,
                             f"unparseable:{error}")
        return StepParse(history, None if target_args is None else _target(target_args),
                         reason)
    if (len(semantic) == 2 and semantic[0][0] == "pyautogui.moveTo"
            and semantic[1][0].startswith("pyautogui.click")):
        # moveTo + click 惯用组合 -> click
        return parse_native_pyautogui(
            "pyautogui.click(%r, %r)" % tuple(semantic[1][1][:2])
            if len(semantic[1][1]) >= 2 else "pyautogui.click()", screen=screen)
    if (len(semantic) == 2 and semantic[0][0] == "pyautogui.moveTo"
            and semantic[1][0] == "pyautogui.dragTo"):
        width, height = screen
        sx, sy = float(semantic[0][1][0]), float(semantic[0][1][1])
        ex, ey = float(semantic[1][1][0]), float(semantic[1][1][1])
        history = {"type": "drag", "x": round(ex), "y": round(ey)}
        target = _target({
            "action": "left_click_drag",
            "coordinate": [_norm999_px(sx, width), _norm999_px(sy, height)],
            "coordinate2": [_norm999_px(ex, width), _norm999_px(ey, height)],
        })
        return StepParse(history, target, None)
    # 宏步:全部映射进一个 macro 历史条目;不可作 target。
    parts = []
    for qualified, args, kwargs in semantic[:6]:
        try:
            history, _, _ = _map_pyautogui_call(qualified, args, kwargs, screen=screen)
            parts.append(history)
        except ValueError:
            parts.append({"type": "unparsed", "raw": qualified})
    return StepParse({"type": "macro", "calls": parts}, None, "macro_step")


_KEY_SPLIT = re.compile(r"[+\-]")


def parse_anthropic_input(
    record: dict[str, Any], *, scale: tuple[float, float], screen: tuple[int, int],
) -> StepParse:
    """claude 系 action.input(anthropic computer-use 语义) -> 双形态。

    ``scale`` 把模型坐标系换算到原生像素(每轨迹从 (input, command) click 对拟合)。
    """
    payload = record.get("input") or {}
    action = payload.get("action")
    width, height = screen
    sx, sy = scale

    def to_px(coordinate: Any) -> tuple[int, int]:
        if (not isinstance(coordinate, list) or len(coordinate) != 2
                or any(type(v) not in (int, float) for v in coordinate)):
            raise ValueError("bad_coordinate")
        return round(float(coordinate[0]) * sx), round(float(coordinate[1]) * sy)

    def c999(coordinate: Any) -> list[int]:
        x, y = to_px(coordinate)
        return [_norm999_px(x, width), _norm999_px(y, height)]

    try:
        if action == "screenshot":
            return StepParse({"type": "screenshot"}, None, "noop_step")
        if action in {"left_click", "right_click", "middle_click", "double_click"}:
            x, y = to_px(payload.get("coordinate"))
            history = {"type": {"left_click": "click", "right_click": "right_click",
                                "middle_click": "click", "double_click": "double_click"}[action],
                       "x": x, "y": y}
            if action == "middle_click":
                history["button"] = "middle"
            return StepParse(history, _target({"action": action,
                                               "coordinate": c999(payload.get("coordinate"))}),
                             None)
        if action == "triple_click":
            x, y = to_px(payload.get("coordinate"))
            return StepParse({"type": "click", "x": x, "y": y, "clicks": 3}, None,
                             "triple_click_not_in_computer_use")
        if action == "mouse_move":
            x, y = to_px(payload.get("coordinate"))
            return StepParse({"type": "move", "x": x, "y": y},
                             _target({"action": "mouse_move",
                                      "coordinate": c999(payload.get("coordinate"))}), None)
        if action == "left_click_drag":
            start = payload.get("start_coordinate")
            end = payload.get("coordinate")
            ex, ey = to_px(end)
            arguments = {"action": "left_click_drag", "coordinate2": c999(end)}
            if start is not None:
                arguments["coordinate"] = c999(start)
            return StepParse({"type": "drag", "x": ex, "y": ey},
                             _target(arguments), None)
        if action == "type":
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("type_text_invalid")
            return StepParse({"type": "type_text", "text": text},
                             _target({"action": "type", "text": text}), None)
        if action == "key":
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("key_text_invalid")
            keys = [part.casefold() for part in _KEY_SPLIT.split(text) if part]
            if not keys or len(keys) > 5:
                raise ValueError("key_combo_invalid")
            history = ({"type": "press", "key": keys[0]} if len(keys) == 1
                       else {"type": "hotkey", "keys": keys})
            return StepParse(history, _target({"action": "key", "keys": keys}), None)
        if action == "scroll":
            direction = payload.get("scroll_direction")
            amount = payload.get("scroll_amount")
            if direction not in {"up", "down", "left", "right"} or \
                    type(amount) is not int or amount <= 0:
                raise ValueError("scroll_invalid")
            if direction in {"up", "down"}:
                pixels = amount if direction == "up" else -amount
                return StepParse({"type": "scroll", "dy": pixels},
                                 _target({"action": "scroll", "pixels": pixels}), None)
            pixels = amount if direction == "right" else -amount
            return StepParse({"type": "scroll", "dy": 0, "dx": pixels},
                             _target({"action": "hscroll", "pixels": pixels}), None)
        if action == "wait":
            return StepParse({"type": "wait"}, _target({"action": "wait"}), None)
        if action in {"hold_key", "cursor_position"}:
            return StepParse({"type": "unparsed", "raw": action}, None,
                             f"unsupported:{action}")
        raise ValueError(f"unknown_action:{action}")
    except ValueError as error:
        return StepParse({"type": "unparsed", "raw": str(payload)[:120]}, None,
                         f"unparseable:{error}")


def fit_claude_scale(entries: list[dict[str, Any]]) -> tuple[float, float]:
    """从 (input.coordinate, command 像素) click 对拟合模型坐标 -> 原生像素比例。"""
    xs: list[tuple[float, float]] = []
    ys: list[tuple[float, float]] = []
    pattern = re.compile(
        r"pyautogui\.(?:click|doubleClick|rightClick|middleClick|tripleClick|moveTo)"
        r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)")
    for entry in entries:
        action = entry.get("action") or {}
        payload = action.get("input") or {}
        coordinate = payload.get("coordinate")
        command = action.get("command") or ""
        if (isinstance(coordinate, list) and len(coordinate) == 2
                and all(type(v) in (int, float) for v in coordinate)):
            match = pattern.search(command)
            if match and float(coordinate[0]) > 0 and float(coordinate[1]) > 0:
                xs.append((float(coordinate[0]), float(match.group(1))))
                ys.append((float(coordinate[1]), float(match.group(2))))
    if len(xs) < 3:
        raise ValueError("not enough click pairs to fit claude coordinate scale")
    sx = statistics.median(px / mx for mx, px in xs if mx)
    sy = statistics.median(py / my for my, py in ys if my)
    return sx, sy


# ---------------------------------------------------------------------------
# build 模式
# ---------------------------------------------------------------------------


def load_traj(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_manifest(args: argparse.Namespace) -> None:
    from PIL import Image

    instructions = json.loads(args.instructions.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    counters: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    records = []
    per_traj_stats = []

    for model, task in SELECTED_PAIRS:
        directory = args.data_root / "website_demo" / model / "tasks" / task
        traj_path = directory / "traj.jsonl"
        if not traj_path.is_file():
            counters["traj_missing"] += 1
            continue
        raw_entries = load_traj(traj_path)
        # 只保留带 png 的条目为"帧步";无 png 的动作(gpt batch)并进下一帧步。
        entries: list[dict[str, Any]] = []
        pending_actions: list[Any] = []
        for entry in raw_entries:
            name = entry.get("screenshot_file")
            if not name or name == "None" or not (directory / name).is_file():
                pending_actions.append(entry)
                continue
            entry["_carried"] = pending_actions
            pending_actions = []
            entries.append(entry)
        if len(entries) < 12:
            counters["traj_too_few_frames"] += 1
            continue
        with Image.open(directory / entries[0]["screenshot_file"]) as probe:
            screen = probe.size
        family = ("anthropic" if model.startswith("claude")
                  else "qwen" if model.startswith("qwen") else "gpt")
        scale = (1.0, 1.0)
        if family == "anthropic":
            scale = fit_claude_scale(raw_entries)

        parsed: list[StepParse] = []
        responses: list[str] = []
        for entry in entries:
            action = entry.get("action")
            carried = entry.get("_carried") or []
            if family == "anthropic":
                step = parse_anthropic_input(action or {}, scale=scale, screen=screen)
            elif family == "qwen":
                code = action if isinstance(action, str) else ""
                step = (parse_native_pyautogui(code, screen=screen) if code
                        else StepParse({"type": "screenshot"}, None, "noop_step"))
            else:
                code = (action or {}).get("action") if isinstance(action, dict) else ""
                blob = "\n".join(
                    [((e.get("action") or {}).get("action") or "")
                     for e in carried if isinstance(e.get("action"), dict)]
                    + [code or ""])
                step = parse_native_pyautogui(blob.strip() or "pyautogui.sleep(0)",
                                              screen=screen)
            parsed.append(step)
            counters[f"steps_{family}"] += 1
            if step.reason:
                reasons[step.reason.split(":")[0]] += 1
            text = str(entry.get("response") or "")
            if family == "anthropic":
                raw = ((entry.get("action") or {}).get("raw_response") or "")
                text = f"{text} {raw}"
            responses.append(text.casefold())

        keywords = REFERENCE_KEYWORDS[task]
        is_ref = [any(k in text for k in keywords) for text in responses]
        # ref 段(1-based 帧步号)
        segments = []
        start = None
        for index, flag in enumerate(is_ref):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                segments.append((start, index - 1))
                start = None
        if start is not None:
            segments.append((start, len(is_ref) - 1))
        segments = [(a, b) for a, b in segments if b - a >= 0]

        decision_points = []
        for a, b in segments:
            for offset in (1, 3):
                t = b + offset
                if t >= len(entries):
                    continue
                if is_ref[t]:
                    continue  # 还在看 reference,不算"已进入编辑"
                step = parsed[t]
                if step.target is None:
                    counters["dp_skipped_target_unavailable"] += 1
                    continue
                if t < 6:
                    counters["dp_skipped_too_early"] += 1
                    continue
                decision_points.append((t, (a, b)))
        # 去重 + 每轨迹上限
        seen = set()
        unique_dps = []
        for t, seg in decision_points:
            if t not in seen:
                seen.add(t)
                unique_dps.append((t, seg))
        if len(unique_dps) > args.max_dp_per_traj:
            unique_dps = [unique_dps[i] for i in sorted(
                rng.sample(range(len(unique_dps)), args.max_dp_per_traj))]

        ref_steps_all = [i for i, flag in enumerate(is_ref) if flag]
        traj_id = f"{model}|{task}"
        for t, (a, b) in unique_dps:
            # 候选池(全部是"事件 post 帧"步号,事件 j 的 post 帧 = entries[j] png,
            # 决策点 t 的事件集是 1..t-1,post 帧可恢复的事件是 1..t-1)
            def usable(event: int) -> bool:
                return 1 <= event <= t - 1
            selected_event = b if usable(b) else None
            ref_pool = [event for event in ref_steps_all
                        if usable(event) and event != selected_event]
            if len(ref_pool) > args.max_ref_pool:
                stride = len(ref_pool) / args.max_ref_pool
                ref_pool = [ref_pool[int(i * stride)] for i in range(args.max_ref_pool)]
            nearby = None
            for cand in (a - 1, b + 1):
                if usable(cand) and not is_ref[cand] and cand != t - 1:
                    nearby = cand
                    break
            next_recent = t - 1 if usable(t - 1) else None
            old_pool = [event for event in range(1, max(1, t - 10))
                        if not is_ref[event]]
            random_old = rng.choice(old_pool) if old_pool else None

            arms = [{"arm_id": "B", "extra": []}]
            if selected_event is not None:
                arms.append({"arm_id": "B_selected", "extra": [selected_event]})
            if nearby is not None:
                arms.append({"arm_id": "B_nearby_irrelevant", "extra": [nearby]})
            if next_recent is not None:
                arms.append({"arm_id": "B_next_recent", "extra": [next_recent]})
            for index, event in enumerate(ref_pool):
                arms.append({"arm_id": f"B_ref_{index:02d}_e{event}",
                             "extra": [event]})
            if random_old is not None:
                arms.append({"arm_id": "B_random_old", "extra": [random_old]})

            history = []
            history_ok = True
            for j in range(1, t):
                step = parsed[j]
                if step.history is None:
                    history_ok = False
                    break
                history.append({
                    "step_id": j,
                    "action": step.history,
                    "osworld_action": str(
                        (entries[j].get("action") or {}) if family != "qwen"
                        else entries[j].get("action"))[:160],
                })
            if not history_ok:
                counters["dp_dropped_history_unparseable"] += 1
                continue
            image_relpaths = [
                f"website_demo/{model}/tasks/{task}/{entries[i]['screenshot_file']}"
                for i in range(t)
            ]
            dp_id = hashlib.sha256(f"{traj_id}|{t}".encode()).hexdigest()[:20]
            records.append({
                "schema_version": MINE_SCHEMA,
                "dp_id": dp_id,
                "traj_id": traj_id,
                "model": model,
                "task": task,
                "os": "ubuntu",
                "stratum": f"osworld|{task}",
                "instruction": instructions[task],
                "step": t,
                "traj_len": len(entries),
                "screen_size": list(screen),
                "ref_segment": [a, b],
                "arms": arms,
                "r_values": [0],
                "image_relpaths": image_relpaths,
                "history": history,
                "target_tool_call": parsed[t].target,
                "target_text": render_target_text_local(parsed[t].target),
                "target_action": parsed[t].target["arguments"]["action"],
            })
        per_traj_stats.append({
            "traj": traj_id, "frames": len(entries), "screen": list(screen),
            "ref_steps": len(ref_steps_all), "ref_segments": len(segments),
            "decision_points": sum(1 for r in records if r["traj_id"] == traj_id),
            "parse_reasons": dict(Counter(
                (p.reason or "ok").split(":")[0] for p in parsed)),
        })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    total_units = sum(len(r["arms"]) for r in records)
    summary = {
        "schema_version": MINE_SCHEMA + ".summary",
        "seed": args.seed,
        "decision_points": len(records),
        "units_expected": total_units,
        "counters": dict(counters),
        "step_reasons": dict(reasons),
        "per_traj": per_traj_stats,
    }
    if args.summary_output:
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render_target_text_local(tool_call: dict[str, Any]) -> str:
    payload = json.dumps(tool_call, ensure_ascii=False)
    return f"<tool_call>\n{payload}\n</tool_call>"


# ---------------------------------------------------------------------------
# score / reduce 模式(复用 AgentNet screening 的打分件)
# ---------------------------------------------------------------------------


def score_units(args: argparse.Namespace) -> None:
    from scripts.screen_agentnet_decision_points import (
        ScoreCache, DecisionPointScorer, cache_fingerprint,
    )
    from causalcache.osworld_gui_owl import GUIOwlOSWorldRuntime

    records = [json.loads(line) for line in
               args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    fingerprint = cache_fingerprint(args, manifest_sha)
    cache = ScoreCache(args.score_cache, shard_index=args.shard_index,
                       shard_count=args.shard_count, fingerprint=fingerprint,
                       readonly=False)
    units = []
    for record in records:
        for arm in record["arms"]:
            units.append((record["dp_id"], arm["arm_id"]))
    units = [u for i, u in enumerate(units)
             if i % args.shard_count == args.shard_index]
    if args.limit:
        units = units[: args.limit]
    by_id = {record["dp_id"]: record for record in records}
    pending = [u for u in units if cache.get(f"{u[0]}|{u[1]}") is None]
    print(f"shard {args.shard_index}/{args.shard_count}: {len(units)} units, "
          f"{len(units) - len(pending)} cached, {len(pending)} to score")
    if not pending:
        return
    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    scorer = DecisionPointScorer(runtime, image_root=args.data_root,
                                 tol_loose=args.coord_tol_loose,
                                 tol_strict=args.coord_tol_strict)
    done = 0
    for dp_id, arm_id in pending:
        record = by_id[dp_id]
        arm = next(a for a in record["arms"] if a["arm_id"] == arm_id)
        entry = scorer.score(record, 0, extra_restored=arm["extra"])
        entry["arm_id"] = arm_id
        entry["extra_restored"] = arm["extra"]
        cache.put(f"{dp_id}|{arm_id}", entry)
        done += 1
        if args.heartbeat is not None:
            args.heartbeat.write_text(json.dumps(
                {"done": done, "total": len(pending), "time": time.time()}) + "\n")
        if done % 20 == 0:
            print(f"shard {args.shard_index}: {done}/{len(pending)} scored")
    print(f"shard {args.shard_index}: complete ({done} scored)")


def reduce_units(args: argparse.Namespace) -> None:
    from scripts.screen_agentnet_decision_points import ScoreCache, cache_fingerprint

    records = [json.loads(line) for line in
               args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    cache = ScoreCache(args.score_cache, shard_index=0, shard_count=1,
                       fingerprint=cache_fingerprint(args, manifest_sha),
                       readonly=True)
    missing = []
    rows = []
    for record in records:
        base = cache.get(f"{record['dp_id']}|B")
        if base is None:
            missing.append(f"{record['dp_id']}|B")
            continue
        for arm in record["arms"]:
            entry = cache.get(f"{record['dp_id']}|{arm['arm_id']}")
            if entry is None:
                missing.append(f"{record['dp_id']}|{arm['arm_id']}")
                continue
            rows.append({
                "dp_id": record["dp_id"],
                "traj_id": record["traj_id"],
                "task": record["task"],
                "step": record["step"],
                "target_action": record["target_action"],
                "arm_id": arm["arm_id"],
                "extra_restored": arm["extra"],
                "type_match": entry["type_match"],
                "loose": entry["target_match_loose"],
                "strict": entry["target_match_strict"],
                "coordinate_distance": entry["coordinate_distance"],
                "mean_target_logprob": entry["mean_target_logprob"],
                "min_token_margin": entry["min_token_margin"],
                "delta_mean_target_logprob":
                    entry["mean_target_logprob"] - base["mean_target_logprob"],
                "delta_min_token_margin":
                    entry["min_token_margin"] - base["min_token_margin"],
                "base_loose": base["target_match_loose"],
                "fixes_base_wrong":
                    (not base["target_match_loose"]) and entry["target_match_loose"],
            })
    if missing:
        raise SystemExit(f"缺 {len(missing)} 个单元(例:{missing[:5]})")
    report = {
        "schema_version": MINE_SCHEMA + ".report",
        "rows": len(rows),
        "by_arm_family": {},
    }
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = (row["arm_id"] if not row["arm_id"].startswith("B_ref_")
                  else "B_ref_pool")
        families.setdefault(family, []).append(row)
    for family, members in sorted(families.items()):
        report["by_arm_family"][family] = {
            "n": len(members),
            "loose_rate": sum(1 for m in members if m["loose"]) / len(members),
            "strict_rate": sum(1 for m in members if m["strict"]) / len(members),
            "mean_delta_logprob": statistics.fmean(
                m["delta_mean_target_logprob"] for m in members),
            "fixes_base_wrong": sum(1 for m in members if m["fixes_base_wrong"]),
        }
    if args.report_output:
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.per_unit_output:
        with args.per_unit_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "build":
        build_manifest(args)
    elif args.mode == "score":
        score_units(args)
    else:
        reduce_units(args)


if __name__ == "__main__":
    main()
