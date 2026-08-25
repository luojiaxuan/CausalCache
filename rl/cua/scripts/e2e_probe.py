#!/usr/bin/env python3
# note (luojiaxuan): e2e 单 episode 探针 —— gui_owl adapter + 真 MobileWorld env
# + 真 GUI-Owl 模型(OpenAI 兼容端点)。验证渲染→生成→解析→动作转换→env
# 步进→终局判分全链。不依赖 slime;generate_fn 手写为 OpenAI chat.completions
# 调用(图像转 data URL)。
# 用法(hyper00,cua-lite checkout 根):
#   PYTHONPATH=/data01/jaxan/sglang-omni-rl/cc_recipe CC_HISTORY_N=3 CC_FRAME_POLICY=recent \
#   uv run python /data01/jaxan/sglang-omni-rl/cc_recipe/scripts/e2e_probe.py \
#     --task AcceptMeetingTask --llm http://127.0.0.1:41002/v1 --max-steps 8
import argparse
import asyncio
import base64
import io
import json
import sys

sys.path.insert(0, ".")


def make_generate_fn(base_url: str, model: str):
    import urllib.request

    async def generate_fn(**kw):
        messages, images = kw["messages"], kw["images"]
        oai_msgs = []
        img_iter = iter(images)
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                oai_msgs.append({"role": m["role"], "content": c})
                continue
            parts = []
            for p in c or []:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image":
                    img = next(img_iter)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    parts.append({"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"}})
            oai_msgs.append({"role": m["role"], "content": parts})
        body = json.dumps({
            "model": model, "messages": oai_msgs,
            "temperature": 0.0, "max_tokens": 512,
        }).encode()

        def _post():
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer EMPTY"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())

        resp = await asyncio.get_event_loop().run_in_executor(None, _post)
        return {"response": resp["choices"][0]["message"]["content"]}

    return generate_fn


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="AcceptMeetingTask")
    ap.add_argument("--llm", default="http://127.0.0.1:41002/v1")
    ap.add_argument("--model", default="gui-owl")
    ap.add_argument("--processor-path",
                    default="/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    from sglang_omni_rl import registration  # noqa: F401
    import lite.gym as gym
    from lite.agents.core.agent.base import AgentRegistry
    from transformers import AutoProcessor

    ap_model = args.processor_path or args.model
    processor = AutoProcessor.from_pretrained(ap_model)

    env = gym.make(f"mobileworld@{args.task}", max_steps=args.max_steps,
                   extra_tools=["terminate", "response"])
    try:
        obs = await env.reset()
        print("RESET_OK goal=", (obs.text or "")[:100], flush=True)
        agent = AgentRegistry.get(
            "gui_owl@mobile@use",
            processor=processor,
            generate_fn=make_generate_fn(args.llm, args.model),
            metadata=env.metadata,
        )
        sample = await agent.sample(env, max_steps=args.max_steps)
        steps = (getattr(sample, "rl_steps", None)
                 or getattr(sample, "steps", None) or [])
        print("SAMPLE_TYPE=", type(sample).__name__,
              "attrs=", [a for a in dir(sample) if not a.startswith("_")][:15])
        print("N_STEPS=", len(steps))
        last_reward = None
        for i, st in enumerate(steps):
            resp = (getattr(st, "response", "") or "")[:120].replace("\n", " | ")
            print(f"  step{i}: status={getattr(st, 'status', '?')} "
                  f"reward={getattr(st, 'reward', None)} resp={resp}")
            last_reward = getattr(st, "reward", None)
        print("FINAL_REWARD=", last_reward)
        md = getattr(sample, "metadata", None)
        others = (md.get("others", {}) if isinstance(md, dict)
                  else getattr(md, "others", {}) or {})
        print("EPISODE_RETURN=", others.get("episode_return", "n/a"),
              "CC_EPISODE=", others.get("cc_episode", "n/a"))
    finally:
        await env.close()
        print("CLOSED")
    print("E2E_PROBE_DONE")


if __name__ == "__main__":
    asyncio.run(main())
