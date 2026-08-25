# note (luojiaxuan): gui_owl adapter ↔ 打补丁官方 agent 的字节级 parity 测试。
# fixture 由 scripts/gen_parity_fixtures.py 在 hyper00 官方 venv 生成
# (monkeypatch LLM client 捕获真实 messages,图像哨兵化为 md5)。
# 本测试在 CUA-Lite 环境重放同一轨迹,比较 (role, parts) 归一化序列:
# 相邻 text part 合并、image part 以 md5 身份对齐 —— 即 chat template
# 消费面的全部信息。askuser 案例跳过(env note 文本格式偏差,已记录)。
# 运行(hyper00 cua-lite checkout 根):
#   PYTHONPATH=/data01/jaxan/cua/cc_recipe uv run python -m pytest \
#     /data01/jaxan/cua/cc_recipe/tests/test_parity_gui_owl.py -q
import hashlib
import io
import json
import os
import random as _random
import sys

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                        "parity_fixtures_v1.jsonl")


def synth_png(seed: int):
    from PIL import Image

    rng = _random.Random(seed)
    return Image.new(
        "RGB", (64, 64),
        (rng.randrange(256), rng.randrange(256), rng.randrange(256)),
    )


def _img_md5(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def normalize(messages, index_to_md5=None):
    """(role, [("text", s) | ("image", md5)]) 序列;相邻 text 合并。"""
    out = []
    for m in messages:
        c = m.get("content")
        parts = []
        if isinstance(c, str):
            parts = [("text", c)]
        else:
            for p in c or []:
                if p.get("type") == "text":
                    parts.append(("text", p.get("text", "")))
                elif p.get("type") == "image_sentinel":
                    parts.append(("image", p["md5"]))
                elif p.get("type") == "image":
                    parts.append(("image", index_to_md5[p["index"]]))
        merged = []
        for kind, val in parts:
            if kind == "text" and merged and merged[-1][0] == "text":
                merged[-1] = ("text", merged[-1][1] + val)
            else:
                merged.append((kind, val))
        # 尾空 text 不影响 chat template 消费面?—— 不做删剪,严格保序比较。
        out.append((m.get("role"), merged))
    return out


def load_cases():
    recs = [json.loads(line) for line in open(FIXTURES)]
    return [r for r in recs if "_askuser" not in r["case"]]


@pytest.mark.parametrize("rec", load_cases(),
                         ids=lambda r: f"{r['case']}_s{r['step']}")
def test_render_parity(rec):
    from causalcache_cua.registration import _adapter  # noqa: F401 注册副作用
    from causalcache_cua.gui_owl.adapter import GuiOwlMobileUseAdapter
    from lite.core.metadata import LiteCUAMetadata
    from lite.core import LiteSample

    case = rec["case"]  # e.g. recent_h3_T5_tool
    policy, h, T, tool_mode = case.split("_")
    hist_n = int(h[1:])
    T = int(T[1:])
    step = rec["step"]

    # ── 重建合成图像(与生成器同种子)并校验 md5 对齐 ──
    imgs = [synth_png(1000 * hist_n + 10 * T + t) for t in range(step + 1)]
    md5s = [_img_md5(im) for im in imgs]
    assert md5s == rec["history_imgs_md5"], "本地合成图与 fixture 不一致"

    # ── 重建 Lite 轨迹:turn0 = 指令 + 帧0;turn t = assistant + obs(帧t) ──
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": rec["instruction"]},
            {"type": "image", "index": 0},
        ],
    }]
    for t in range(step):
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": rec["responses"][t]}],
        })
        obs_parts = []
        tool_text = rec["tool_calls"][t + 1] if t + 1 < len(rec["tool_calls"]) else None
        if tool_text:
            obs_parts.append({"type": "text", "text": str(tool_text)})
        obs_parts.append({"type": "image", "index": t + 1})
        messages.append({"role": "user", "content": obs_parts})

    metadata = LiteCUAMetadata(dims=("mobile", "use"))
    sample = LiteSample(messages=messages, images=imgs, metadata=metadata)

    os.environ["CC_SEED"] = "0"
    adapter = GuiOwlMobileUseAdapter(
        metadata=metadata, history_n=hist_n, frame_policy=policy,
        selector_url="", random_seed="0",
    )
    rendered = adapter.render_step(sample, step + 1, imgs)

    got = normalize(rendered, index_to_md5=md5s)
    want = normalize(rec["messages"])
    assert got == want, (
        f"\n--- want ---\n{json.dumps(want, ensure_ascii=False, indent=1)[:4000]}"
        f"\n--- got ---\n{json.dumps(got, ensure_ascii=False, indent=1)[:4000]}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--no-header"]))
