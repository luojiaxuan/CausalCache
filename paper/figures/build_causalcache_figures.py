#!/usr/bin/env python3
"""Build the editable CausalCache overview and audited qualitative figures."""

from __future__ import annotations

import base64
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

INK = "#18212B"
MUTED = "#5B6773"
LIGHT = "#F5F7F9"
GRAY = "#D9DEE3"
GRAY_DARK = "#66727D"
BLUE = "#1769AA"
BLUE_LIGHT = "#DCEAF5"
ORANGE = "#C85A00"
ORANGE_LIGHT = "#FCE8D5"
PURPLE = "#6B46A1"
PURPLE_LIGHT = "#E9DDF3"
RED = "#B83232"
RED_LIGHT = "#F8DEDE"
GREEN = "#287A3E"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    out: list[str],
    x: float,
    y: float,
    value: str,
    *,
    size: float = 18,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    italic: bool = False,
) -> None:
    size = max(size, 18)
    style = "italic" if italic else "normal"
    escaped_value = esc(value)
    out.append(
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" font-style="{style}" '
        f'xml:space="preserve">'
        f"{escaped_value}</text>"
    )


def multiline(
    out: list[str],
    x: float,
    y: float,
    lines: list[str],
    *,
    size: float = 18,
    line_height: float = 21,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
) -> None:
    for index, value in enumerate(lines):
        text(
            out,
            x,
            y + index * line_height,
            value,
            size=size,
            weight=weight,
            fill=fill,
            anchor=anchor,
        )


def rect(
    out: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "white",
    stroke: str = INK,
    stroke_width: float = 1.5,
    radius: float = 8,
    dash: str | None = None,
) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr}/>'
    )


def line(
    out: list[str],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = INK,
    stroke_width: float = 2,
    dash: str | None = None,
    arrow: bool = False,
) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    out.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"{dash_attr}{marker}/>'
    )


def badge(
    out: list[str],
    x: float,
    y: float,
    width: float,
    value: str,
    *,
    fill: str,
    stroke: str,
    text_fill: str = "white",
) -> None:
    rect(
        out,
        x,
        y,
        width,
        25,
        fill=fill,
        stroke=stroke,
        stroke_width=1.5,
        radius=12.5,
    )
    text(
        out,
        x + width / 2,
        y + 18,
        value,
        size=17,
        weight=700,
        fill=text_fill,
        anchor="middle",
    )


def panel(out: list[str], x: float, y: float, width: float, height: float) -> None:
    rect(
        out,
        x,
        y,
        width,
        height,
        fill="white",
        stroke="#B8C0C8",
        stroke_width=1.4,
        radius=12,
    )


def event_card(
    out: list[str],
    x: float,
    y: float,
    width: float,
    *,
    mode: str,
    label: str,
    compact: bool = False,
) -> None:
    height = 48 if compact else 66
    summary_h = 13 if compact else 17
    palette = {
        "low": (LIGHT, GRAY_DARK, "5 4"),
        "recent": (BLUE_LIGHT, BLUE, None),
        "distant": (ORANGE_LIGHT, ORANGE, None),
        "wrong": ("url(#redHatch)", RED, None),
        "current": ("white", INK, None),
    }
    fill, stroke, dash = palette[mode]
    rect(
        out,
        x,
        y,
        width,
        height,
        fill=fill,
        stroke=stroke,
        stroke_width=3 if mode == "current" else 1.8,
        radius=6,
        dash=dash,
    )
    if mode != "low":
        out.append(
            f'<rect x="{x + 6}" y="{y + 6}" width="{width - 12}" '
            f'height="{height - summary_h - 15}" rx="3" fill="white" '
            f'fill-opacity="0.88" stroke="{stroke}" stroke-width="1"/>'
        )
        line(
            out,
            x + 11,
            y + 15,
            x + width - 11,
            y + 15,
            stroke=stroke,
            stroke_width=1.4,
        )
        line(
            out,
            x + 11,
            y + 22,
            x + width - 18,
            y + 22,
            stroke=stroke,
            stroke_width=1.1,
        )
    out.append(
        f'<rect x="{x + 4}" y="{y + height - summary_h - 4}" '
        f'width="{width - 8}" height="{summary_h}" rx="3" '
        f'fill="{GRAY}" stroke="{GRAY_DARK}" stroke-width="0.9"/>'
    )
    text(
        out,
        x + width / 2,
        y + height + (17 if compact else 18),
        label,
        size=16 if compact else 17,
        weight=700,
        fill=stroke,
        anchor="middle",
    )


def flow_box(
    out: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    *,
    fill: str = LIGHT,
    stroke: str = GRAY_DARK,
    title_color: str = INK,
    dashed: bool = False,
) -> None:
    rect(
        out,
        x,
        y,
        width,
        height,
        fill=fill,
        stroke=stroke,
        stroke_width=1.6,
        radius=7,
        dash="5 4" if dashed else None,
    )
    base = y + height / 2 - (len(lines) - 1) * 9 + 6
    for index, value in enumerate(lines):
        text(
            out,
            x + width / 2,
            base + index * 18,
            value,
            size=18,
            weight=700 if index == 0 else 400,
            fill=title_color,
            anchor="middle",
        )


def svg_start(
    width: int,
    height: int,
    *,
    physical_height_in: float,
    title_value: str,
    desc_value: str,
) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="7in" height="{physical_height_in}in" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title desc">'
        ),
        f"<title id=\"title\">{esc(title_value)}</title>",
        f"<desc id=\"desc\">{esc(desc_value)}</desc>",
        "<defs>",
        (
            '<marker id="arrow" markerWidth="9" markerHeight="7" refX="8" '
            'refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" '
            f'fill="{INK}"/></marker>'
        ),
        (
            '<pattern id="redHatch" width="8" height="8" '
            'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="8" height="8" fill="{RED_LIGHT}"/>'
            f'<line x1="0" y1="0" x2="0" y2="8" stroke="{RED}" '
            'stroke-width="2"/></pattern>'
        ),
        "</defs>",
        (
            '<style>text{font-family:Arial,Helvetica,sans-serif;'
            'letter-spacing:0;} .hair{shape-rendering:geometricPrecision;}</style>'
        ),
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]


def embed_png(
    out: list[str],
    path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    stroke: str = INK,
    stroke_width: float = 2,
) -> None:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    rect(
        out,
        x,
        y,
        width,
        height,
        fill="white",
        stroke=stroke,
        stroke_width=stroke_width,
        radius=5,
    )
    out.append(
        f'<image x="{x + 3}" y="{y + 3}" width="{width - 6}" '
        f'height="{height - 6}" preserveAspectRatio="xMidYMid meet" '
        f'href="data:image/png;base64,{encoded}"/>'
    )


def build_overview() -> str:
    out = svg_start(
        1008,
        560,
        physical_height_in=3.89,
        title_value="CausalCache conditional high-fidelity restoration overview",
        desc_value=(
            "Three panels show two-fidelity event memory, matched-budget "
            "reallocation, and HGKV training plus CausalCache-P deployment."
        ),
    )
    panel(out, 10, 10, 245, 540)
    panel(out, 263, 10, 374, 540)
    panel(out, 645, 10, 353, 540)

    text(out, 24, 39, "(a) Event fidelity", size=22, weight=700)
    text(out, 24, 69, "Complete event trace", size=18, weight=700, fill=MUTED)
    for index, x in enumerate((25, 69, 113, 157, 201), start=1):
        event_card(
            out,
            x,
            84,
            36,
            mode="distant" if index == 3 else "low",
            label=f"e{index}",
            compact=True,
        )
    multiline(
        out,
        24,
        163,
        ["Every event keeps", "its summary strip."],
        size=18,
        line_height=21,
        fill=MUTED,
    )
    rect(
        out,
        24,
        185,
        217,
        96,
        fill=LIGHT,
        stroke="#B8C0C8",
        stroke_width=1.2,
        radius=7,
    )
    multiline(
        out,
        36,
        209,
        [
            "low = summary",
            "high = summary",
            "       + archived image",
        ],
        size=18,
        line_height=27,
        weight=700,
    )
    flow_box(
        out,
        24,
        307,
        103,
        70,
        ["PROMOTE", "attach", "image"],
        fill=ORANGE_LIGHT,
        stroke=ORANGE,
        title_color=ORANGE,
    )
    flow_box(
        out,
        138,
        307,
        103,
        70,
        ["DEMOTE", "keep", "summary"],
        fill=LIGHT,
        stroke=GRAY_DARK,
        title_color=GRAY_DARK,
    )
    line(out, 128, 342, 136, 342, stroke=INK, stroke_width=2, arrow=True)
    text(out, 24, 398, "Cold visual archive", size=18, weight=700)
    for index, x in enumerate((26, 68, 110, 152, 194), start=1):
        rect(
            out,
            x,
            412,
            31,
            42,
            fill=GRAY,
            stroke=GRAY_DARK,
            stroke_width=1.2,
            radius=3,
            dash="4 3",
        )
        text(out, x + 15.5, 438, f"v{index}", size=18, fill=MUTED, anchor="middle")
    rect(
        out,
        24,
        478,
        217,
        72,
        fill=ORANGE_LIGHT,
        stroke=ORANGE,
        stroke_width=1.5,
        radius=7,
    )
    multiline(
        out,
        132.5,
        499,
        ["Archive pixels", "are reattached.", "Never text-generated."],
        size=18,
        line_height=21,
        weight=700,
        fill=ORANGE,
        anchor="middle",
    )

    text(out, 277, 39, "(b) Fixed-B reallocation", size=22, weight=700)
    badge(
        out,
        535,
        51,
        84,
        "B = 4",
        fill=INK,
        stroke=INK,
    )
    text(out, 278, 76, "Recent-4", size=18, weight=700, fill=BLUE)
    for index in range(8):
        x = 278 + index * 43
        event_card(
            out,
            x,
            88,
            34,
            mode="recent" if index >= 4 else "low",
            label=f"e{index + 1}",
            compact=True,
        )
    text(
        out,
        449,
        164,
        "Promote the latest B events",
        size=17,
        weight=700,
        fill=BLUE,
        anchor="middle",
    )
    text(out, 278, 198, "CausalCache", size=18, weight=700, fill=ORANGE)
    text(
        out,
        620,
        198,
        "e3 for e5; still B=4",
        size=17,
        weight=700,
        fill=ORANGE,
        anchor="end",
    )
    for index in range(8):
        x = 278 + index * 43
        if index == 2:
            mode = "distant"
        elif index >= 5:
            mode = "recent"
        else:
            mode = "low"
        event_card(out, x, 210, 34, mode=mode, label=f"e{index + 1}", compact=True)
    rect(
        out,
        277,
        326,
        346,
        110,
        fill=LIGHT,
        stroke="#AAB4BE",
        stroke_width=1.3,
        radius=7,
    )
    text(out, 450, 348, "LOCKED INVARIANTS", size=18, weight=700, anchor="middle")
    multiline(
        out,
        450,
        371,
        [
            "same goal + complete summaries",
            "same current screen",
            "same resolution + B active images",
            "only promoted events change",
        ],
        size=18,
        line_height=20,
        fill=MUTED,
        anchor="middle",
    )
    rect(
        out,
        277,
        450,
        346,
        84,
        fill=ORANGE_LIGHT,
        stroke=ORANGE,
        stroke_width=1.5,
        radius=7,
    )
    multiline(
        out,
        290,
        470,
        [
            "Promote e3 only if Delta_t(e3 | S)",
            "> utility of displaced e5",
            "k is realized, not preset",
            "B is not archival capacity",
        ],
        size=18,
        line_height=19,
        weight=700,
        fill=INK,
    )

    text(out, 659, 39, "(c) Learning & deployment", size=22, weight=700)
    multiline(
        out,
        659,
        65,
        ["Training: matched B;", "only restoration identity changes"],
        size=18,
        line_height=21,
        fill=MUTED,
    )
    flow_box(
        out,
        659,
        105,
        97,
        60,
        ["Recent", "restoration"],
        fill=BLUE_LIGHT,
        stroke=BLUE,
        title_color=BLUE,
    )
    flow_box(
        out,
        766,
        105,
        101,
        60,
        ["Relevant", "restoration"],
        fill=ORANGE_LIGHT,
        stroke=ORANGE,
        title_color=ORANGE,
    )
    flow_box(
        out,
        877,
        105,
        101,
        60,
        ["Wrong", "restoration"],
        fill="url(#redHatch)",
        stroke=RED,
        title_color=RED,
    )
    for x in (707.5, 816.5, 927.5):
        line(out, x, 166, x, 183, stroke=INK, stroke_width=1.5, arrow=True)
    rect(out, 659, 188, 319, 39, fill="white", stroke=GRAY_DARK, stroke_width=1.4, radius=5)
    segments = [
        (659, 55, GRAY, "goal"),
        (714, 87, GRAY, "summary"),
        (801, 88, PURPLE_LIGHT, "pixels"),
        (889, 53, "white", "now"),
        (942, 36, GRAY, "act"),
    ]
    for x, width, fill, value in segments:
        segment_stroke = (
            PURPLE if fill == PURPLE_LIGHT else INK if value == "now" else GRAY_DARK
        )
        segment_stroke_width = 3 if value == "now" else 1
        out.append(
            f'<rect x="{x + 1}" y="189" width="{width - 2}" height="37" '
            f'fill="{fill}" stroke="{segment_stroke}" '
            f'stroke-width="{segment_stroke_width}"/>'
        )
        text(
            out,
            x + width / 2,
            213,
            value,
            size=18,
            weight=700 if fill == PURPLE_LIGHT else 400,
            fill=PURPLE if fill == PURPLE_LIGHT else INK,
            anchor="middle",
        )
    line(out, 845, 227, 845, 245, stroke=PURPLE, stroke_width=2.3, arrow=True)
    flow_box(
        out,
        835,
        248,
        135,
        52,
        ["HGKV", "history pixels"],
        fill=PURPLE_LIGHT,
        stroke=PURPLE,
        title_color=PURPLE,
    )
    multiline(
        out,
        659,
        255,
        ["Other tokens", "bypass HGKV", "B=0: exact bypass"],
        size=18,
        line_height=20,
        fill=MUTED,
    )
    text(out, 659, 332, "Deployment", size=18, weight=700)
    badge(
        out,
        838,
        312,
        140,
        "CausalCache-P",
        fill=ORANGE,
        stroke=ORANGE,
    )
    p_boxes = [
        (659, 348, 75, ["Recent", "B prompt"]),
        (740, 348, 75, ["proposal", "action"]),
        (821, 348, 75, ["select", "+ swap"]),
        (902, 348, 76, ["decide", "again*"]),
    ]
    for index, (x, y, width, lines_) in enumerate(p_boxes):
        flow_box(
            out,
            x,
            y,
            width,
            62,
            lines_,
            fill=ORANGE_LIGHT if index >= 2 else LIGHT,
            stroke=ORANGE if index >= 2 else GRAY_DARK,
            title_color=ORANGE if index >= 2 else INK,
        )
        if index < len(p_boxes) - 1:
            line(out, x + width + 2, y + 28, p_boxes[index + 1][0] - 4, y + 28, arrow=True)
    text(out, 978, 430, "P: proposal-conditioned default", size=18, fill=MUTED, anchor="end")
    badge(
        out,
        659,
        451,
        141,
        "CausalCache-LA",
        fill="white",
        stroke=GRAY_DARK,
        text_fill=GRAY_DARK,
    )
    flow_box(
        out,
        804,
        448,
        84,
        54,
        ["last", "action", "ref."],
        fill="white",
        stroke=GRAY_DARK,
        title_color=GRAY_DARK,
        dashed=True,
    )
    flow_box(
        out,
        900,
        448,
        78,
        54,
        ["single", "policy", "pass"],
        fill="white",
        stroke=GRAY_DARK,
        title_color=GRAY_DARK,
        dashed=True,
    )
    line(out, 890, 475, 898, 475, stroke=GRAY_DARK, dash="5 4", arrow=True)
    text(out, 659, 529, "LA: efficiency branch, not default", size=18, fill=MUTED)

    out.append("</svg>")
    return "\n".join(out)


def build_qualitative() -> str:
    out = svg_start(
        1008,
        410,
        physical_height_in=2.85,
        title_value="A real CartInfoNotificationTask restoration case",
        desc_value=(
            "A completed event trace branches into Recent-4 and CausalCache "
            "allocations. CausalCache restores an enlarged event-7 order "
            "screenshot before both paths reach the message composer, where "
            "Recent-4 types the wrong action and CausalCache types the correct action."
        ),
    )

    def event_chip(
        x: float,
        y: float,
        value: str,
        *,
        fill: str = LIGHT,
        stroke: str = GRAY_DARK,
        width: float = 34,
    ) -> None:
        rect(
            out,
            x,
            y,
            width,
            30,
            fill=fill,
            stroke=stroke,
            stroke_width=1.8,
            radius=15,
        )
        text(
            out,
            x + width / 2,
            y + 21,
            value,
            size=18,
            weight=700,
            fill=stroke,
            anchor="middle",
        )

    text(out, 18, 25, "Complete event trace", size=20, weight=700)
    trace_values = ("e1", "...", "e6", "e7", "e8", "...", "e11", "e12", "e13", "e14")
    trace_x = 16
    for value in trace_values:
        width = 28 if value != "..." else 24
        if value == "e7":
            fill, stroke = ORANGE_LIGHT, ORANGE
        elif value in {"e11", "e12", "e13", "e14"}:
            fill, stroke = BLUE_LIGHT, BLUE
        else:
            fill, stroke = LIGHT, GRAY_DARK
        event_chip(trace_x, 42, value, fill=fill, stroke=stroke, width=width)
        trace_x += width + 4
    line(out, 18, 79, 306, 79, stroke=INK, stroke_width=2.2, arrow=True)

    out.append(
        f'<path d="M307 79 L307 147 L319 147" fill="none" '
        f'stroke="{INK}" stroke-width="2.4" marker-end="url(#arrow)"/>'
    )
    out.append(
        f'<path d="M307 79 L307 278 L319 278" fill="none" '
        f'stroke="{INK}" stroke-width="2.4" marker-end="url(#arrow)"/>'
    )

    text(out, 324, 125, "Recent-4", size=22, weight=700, fill=BLUE)
    for index, value in enumerate(("e11", "e12", "e13", "e14")):
        event_chip(
            324 + index * 40,
            136,
            value,
            fill=BLUE_LIGHT,
            stroke=BLUE,
            width=36,
        )

    text(out, 324, 256, "CausalCache", size=22, weight=700, fill=ORANGE)
    for index, value in enumerate(("e6", "e7", "e8", "e12")):
        event_chip(
            324 + index * 40,
            267,
            value,
            fill=ORANGE_LIGHT if value != "e12" else BLUE_LIGHT,
            stroke=ORANGE if value != "e12" else BLUE,
            width=36,
        )

    out.append(
        f'<path d="M484 151 L610 151 L653 198" fill="none" '
        f'stroke="{BLUE}" stroke-width="3" marker-end="url(#arrow)"/>'
    )
    line(out, 484, 282, 496, 282, stroke=ORANGE, stroke_width=3, arrow=True)

    text(out, 581, 201, "restored e7", size=20, weight=700, fill=ORANGE, anchor="middle")
    embed_png(
        out,
        ASSETS / "cart_order_evidence.png",
        496,
        211,
        170,
        181,
        stroke=ORANGE,
        stroke_width=3,
    )
    out.append(
        f'<path d="M666 302 L684 302 L684 226" fill="none" '
        f'stroke="{ORANGE}" stroke-width="3" marker-end="url(#arrow)"/>'
    )

    text(out, 735, 171, "Current screen", size=20, weight=700, anchor="middle")
    embed_png(
        out,
        ASSETS / "cart_message_input.png",
        676,
        181,
        118,
        49,
        stroke=INK,
        stroke_width=2.5,
    )

    out.append(
        f'<path d="M794 205 L803 205 L803 148 L807 148" fill="none" '
        f'stroke="{RED}" stroke-width="2.6" marker-end="url(#arrow)"/>'
    )
    out.append(
        f'<path d="M794 205 L803 205 L803 323 L807 323" fill="none" '
        f'stroke="{GREEN}" stroke-width="2.6" marker-end="url(#arrow)"/>'
    )

    text(out, 811, 83, "Recent-4 action", size=20, weight=700, fill=BLUE)
    rect(
        out,
        808,
        92,
        190,
        112,
        fill="white",
        stroke="#AAB4BE",
        stroke_width=1.5,
        radius=7,
    )
    multiline(
        out,
        820,
        113,
        [
            'type("Women\'s',
            "Cotton Short Sleeve",
            "T-shirt,",
            '202306010001")',
        ],
        size=18,
        line_height=22,
        weight=700,
    )
    line(out, 966, 178, 989, 197, stroke=RED, stroke_width=4)
    line(out, 989, 178, 966, 197, stroke=RED, stroke_width=4)

    text(out, 811, 258, "CausalCache action", size=20, weight=700, fill=ORANGE)
    rect(
        out,
        808,
        267,
        190,
        122,
        fill="white",
        stroke="#AAB4BE",
        stroke_width=1.5,
        radius=7,
    )
    multiline(
        out,
        820,
        291,
        [
            'type("经典白色T恤,',
            "保湿面霜套装,",
            '639281475036294")',
        ],
        size=18,
        line_height=26,
        weight=700,
    )
    out.append(
        f'<path d="M964 362 L974 374 L992 349" fill="none" '
        f'stroke="{GREEN}" stroke-width="4" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
    )

    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "causalcache_overview.svg").write_text(
        build_overview(),
        encoding="utf-8",
    )
    (ROOT / "causalcache_qualitative_cart.svg").write_text(
        build_qualitative(),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
