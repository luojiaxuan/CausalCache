#!/usr/bin/env python3
# note (luojiaxuan): parity fixture 生成器 —— 在 hyper00 的 MobileWorld venv 里
# 驱动**打过补丁的**官方 agent(gui_owl_1_5),monkeypatch LLM client 捕获
# 每步实际发送的 messages(比 CC_DUMP 可靠:后者引用未定义变量,从未跑通)。
# 产出 JSONL fixture,图像 b64 替换为 md5 哨兵以便本地 CUA-Lite adapter
# 逐字节对拍(文本全量保留,图像按哨兵身份对齐)。
# 用法(hyper00,MobileWorld checkout 根目录):
#   CC_HISTORY_N=3 CC_FRAME_POLICY=recent uv run python gen_parity_fixtures.py \
#     --out fixtures_recent_h3.jsonl
# 遍历矩阵见 main;总 case 数按 (T × policy × history_n × tool_call 变体) 组合。
import argparse
import base64
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, "src")


def synth_png(seed: int) -> "PIL.Image.Image":
    from PIL import Image

    import random as _r
    rng = _r.Random(seed)
    im = Image.new("RGB", (64, 64),
                   (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    return im


def sentinelize(messages):
    """data:image/png;base64,... -> md5 哨兵;其余内容原样。"""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            items = []
            for it in c:
                if it.get("type") == "image_url":
                    b64 = it["image_url"]["url"].split(",", 1)[-1]
                    h = hashlib.md5(base64.b64decode(b64)).hexdigest()
                    items.append({"type": "image_sentinel", "md5": h})
                else:
                    items.append(it)
            out.append({"role": m["role"], "content": items})
        else:
            out.append(m)
    return out


def canned_response(i: int) -> str:
    return (
        f"Action: do synthetic step {i}.\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action": "click", '
        f'"coordinate": [{100 + i}, {200 + i}]}}}}\n'
        "</tool_call>"
    )


def run_case(policy, hist_n, T, tool_mode, out_f):
    os.environ["CC_FRAME_POLICY"] = policy
    os.environ["CC_HISTORY_N"] = str(hist_n)
    os.environ.setdefault("CC_SEED", "0")
    from mobile_world.agents.implementations.gui_owl_1_5 import GUIOWL15AgentMCP

    agent = GUIOWL15AgentMCP(
        model_name="gui-owl", llm_base_url="http://127.0.0.1:9/v1", api_key="EMPTY",
        runtime_conf={"history_n": hist_n, "max_tokens": 64,
                      "temperature": 0.0, "top_p": 1.0},
        tools=[],  # GUI-only:无 MCP 附加工具(MCPAgent 首位参数)
    )
    agent.instruction = f"synthetic task {policy}-{hist_n}-{T}-{tool_mode}"
    captured = {}

    def fake_llm(**kw):
        captured["messages"] = kw["messages"]
        return canned_response(len(agent.history_responses))

    agent.openai_chat_completions_create = fake_llm

    img_hashes = []
    for t in range(T):
        im = synth_png(1000 * hist_n + 10 * T + t)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        img_hashes.append(hashlib.md5(buf.getvalue()).hexdigest())
        tool_call = None
        ask_user = None
        # note (luojiaxuan): obs 0(首观察)物理上不可能有 tool 结果——
        # 之前没有动作;t=0 挂 tool 会造出官方代码路径可达但真实 episode
        # 不存在的案例(首泡 fold 读 obs0 tool),故只在 t>=1 挂。
        if tool_mode == "tool" and t % 2 == 1:
            tool_call = f"synthetic tool result {t}"
        elif tool_mode == "askuser" and t % 3 == 1:
            ask_user = f"synthetic user reply {t}"
        obs = {"screenshot": im, "tool_call": tool_call,
               "ask_user_response": ask_user}
        agent.predict(obs)
        rec = {
            "case": f"{policy}_h{hist_n}_T{T}_{tool_mode}", "step": t,
            "instruction": agent.instruction,
            "history_imgs_md5": list(img_hashes),
            "tool_calls": [c[1] for c in agent.history_user_content],
            "ask_user": [c[2] for c in agent.history_user_content],
            "responses": list(agent.history_responses[:-1] if agent.history_responses else []),
            "conclusions": list(agent.conclusions[:-1] if agent.conclusions else []),
            "messages": sentinelize(captured["messages"]),
        }
        # note (luojiaxuan): predict 之后 history_responses/conclusions 已含本步,
        # 但 messages 是本步发送前的视图 —— 记录截到上一步的历史供重放。
        rec["responses"] = list(agent.history_responses[:-1])
        rec["conclusions"] = list(agent.conclusions[:-1])
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 0
    with open(args.out, "w") as f:
        for policy in ("recent", "random"):
            for hist_n in (1, 3, 12):
                for T in (1, 2, 5, 9):
                    for tool_mode in ("none", "tool", "askuser"):
                        run_case(policy, hist_n, T, tool_mode, f)
                        n += T
    print(f"fixtures written: {args.out} ({n} step-cases)")


if __name__ == "__main__":
    main()
