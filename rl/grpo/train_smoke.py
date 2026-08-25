#!/usr/bin/env python3
# note (luojiaxuan): joint GRPO smoke 训练步(v2 recipe)。每轮调用一次:
# 读本轮 G 条 selector 臂 rollout 的判分与决策日志,RLOO advantage,
# 更新 selector(PL 联合比率裁剪)与 executor(LoRA, token 级 PG),
# 落 metrics.jsonl 与验收指标。审计要点:episode 级共享 credit(合法但
# 高方差,全量阶段再上 per-step critic 消融);对照臂不进梯度。
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data01/jaxan/mw/MobileWorld/src")
from selector import FrameSelector  # noqa: E402

CLIP = 0.2
ENT_COEF = 0.01
DEV = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 -> 物理 GPU1


def load_groups(root, it, G):
    """返回 groups[task_key][g]=reward 与 episode_map[episode_id]=(task_key,g)。"""
    from mobile_world.runtime.client import scan_finished_tasks

    groups, episode_map, goal_of = {}, {}, {}
    for g in range(1, G + 1):
        d = f"{root}/traj/it{it}_g{g}"
        dirs, scores = scan_finished_tasks(d)
        for name, sc in zip(dirs, scores):
            tj = f"{d}/{name}/traj.json"
            goal = ""
            try:
                dd = json.load(open(tj))
                first = next(iter(dd.values()))
                goal = first["traj"][0]["task_goal"][:80]
            except Exception:  # noqa: BLE001
                pass
            goal_of[(name, g)] = goal
            groups.setdefault(goal, {})[g] = 1.0 if (sc and sc > 0) else 0.0
        for ep_file in glob.glob(f"{d}/../it{it}_g{g}/*.jsonl"):
            ep = os.path.basename(ep_file)[:-6]
            try:
                task_key = json.loads(open(ep_file).readline()).get("task", "")
            except Exception:  # noqa: BLE001
                task_key = ""
            episode_map[ep] = (task_key, g)
    return groups, episode_map


def rloo_adv(groups):
    adv = {}
    for task, rs in groups.items():
        for g, r in rs.items():
            others = [v for k, v in rs.items() if k != g]
            adv[(task, g)] = r - (sum(others) / len(others) if others else 0.0)
    return adv


def train_selector(root, it, adv, episode_map):
    sel = FrameSelector().to(DEV)
    ck = f"{root}/ckpt/selector_latest.pt"
    if os.path.exists(ck):
        sel.load_state_dict(torch.load(ck, map_location=DEV))
    opt = torch.optim.AdamW(sel.parameters(), lr=1e-5, weight_decay=0.01)
    losses, ages, n_used = [], [], 0
    for f in glob.glob(f"{root}/decisions/*.jsonl"):
        for line in open(f):
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            key = episode_map.get(str(rec.get("episode")))
            if not key:
                continue
            a = adv.get(key, 0.0)
            if a == 0.0 or rec["n_frames"] == 0:
                continue
            feats = torch.tensor(rec["feats"], device=DEV)
            pos = torch.arange(rec["n_frames"], device=DEV)
            new_lp = sel.slate_logprob(feats, pos, rec["step"], rec["chosen"])
            ratio = torch.exp(new_lp - torch.tensor(rec["logp"], device=DEV))
            surr = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
            # note (luojiaxuan): 首选分布熵/logT 作可微归一化熵近似
            lg = sel.logits(feats, pos, rec["step"])
            p = torch.softmax(lg, 0)
            ent = -(p * torch.log(p + 1e-9)).sum() / max(1.0, torch.log(torch.tensor(float(rec["n_frames"]))))
            loss = -surr - ENT_COEF * ent
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(sel.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
            ages.append(rec["step"] - (sum(rec["chosen"]) / max(1, len(rec["chosen"]))))
            n_used += 1
    os.makedirs(f"{root}/ckpt", exist_ok=True)
    torch.save(sel.state_dict(), ck)
    return {"sel_loss_mean": sum(losses) / len(losses) if losses else None,
            "sel_updates": n_used,
            "mean_frame_age": sum(ages) / len(ages) if ages else None,
            "sel_grad_last": float(gn) if n_used else None}


def train_executor(root, it, G, adv, episode_map, goal_preds):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as M
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
    mp = "/data04/jaxan/models/GUI-Owl-1.5-8B-Instruct"
    if not os.path.isdir(mp):
        mp = "/data/models/GUI-Owl-1.5-8B-Instruct"
    proc = AutoProcessor.from_pretrained(mp)
    model = M.from_pretrained(mp, torch_dtype=torch.bfloat16).to(DEV)
    model.gradient_checkpointing_enable()
    prev = f"{root}/ckpt/lora_it{it - 1}"
    if os.path.isdir(prev):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, prev, is_trainable=True)
    else:
        cfg = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    import base64
    import io

    from PIL import Image

    losses, n = [], 0
    for ep_file in glob.glob(f"{root}/traj/it{it}_g*/*.jsonl"):
        ep = os.path.basename(ep_file)[:-6]
        key = episode_map.get(ep)
        if not key or adv.get(key, 0.0) == 0.0:
            continue
        a = adv[key]
        steps = [json.loads(x) for x in open(ep_file)]
        for rec in steps[-2:]:  # note (luojiaxuan): smoke 只训每 episode 末 2 步
            msgs, resp = rec.get("messages"), goal_preds.get((key[0], key[1], rec["step"]))
            if not msgs or not resp:
                continue
            imgs = []
            conv = []
            for m in msgs:
                if isinstance(m.get("content"), list):
                    items = []
                    for c in m["content"]:
                        if c.get("type") == "image_url":
                            b64 = c["image_url"]["url"].split(",", 1)[-1]
                            imgs.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
                            items.append({"type": "image"})
                        else:
                            items.append(c)
                    conv.append({"role": m["role"], "content": items})
                else:
                    conv.append(m)
            text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=imgs or None, return_tensors="pt").to(DEV)
            r_ids = proc.tokenizer(resp, return_tensors="pt").input_ids.to(DEV)
            ids = torch.cat([inp.input_ids, r_ids], 1)
            out = model(input_ids=ids, pixel_values=inp.get("pixel_values"),
                        image_grid_thw=inp.get("image_grid_thw"))
            lp = torch.log_softmax(out.logits[:, inp.input_ids.shape[1] - 1:-1], -1)
            tok_lp = lp.gather(-1, r_ids.unsqueeze(-1)).squeeze(-1).mean()
            loss = -a * tok_lp
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(float(loss))
            n += 1
    model.save_pretrained(f"{root}/ckpt/lora_it{it}")
    return {"exec_loss_mean": sum(losses) / len(losses) if losses else None, "exec_updates": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--root", default="/data01/jaxan/grpo")
    args = ap.parse_args()

    groups, episode_map = load_groups(args.root, args.iter, args.g)
    adv = rloo_adv(groups)
    rewards = [r for rs in groups.values() for r in rs.values()]
    mixed = sum(1 for rs in groups.values() if 0 < sum(rs.values()) < len(rs)) / max(1, len(groups))

    # 响应文本索引:goal_preds[(task_key,g,step)] = prediction
    goal_preds = {}
    for g in range(1, args.g + 1):
        for tj in glob.glob(f"{args.root}/traj/it{args.iter}_g{g}/*/traj.json"):
            try:
                dd = json.load(open(tj))
                first = next(iter(dd.values()))
                goal = first["traj"][0]["task_goal"][:80]
                for i, st in enumerate(first["traj"]):
                    goal_preds[(goal, g, i)] = st.get("prediction", "")
            except Exception:  # noqa: BLE001
                continue

    m_sel = train_selector(args.root, args.iter, adv, episode_map)
    m_exec = train_executor(args.root, args.iter, args.g, adv, episode_map, goal_preds)
    metrics = {"iter": args.iter, "n_tasks": len(groups),
               "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
               "reward_nonzero": sum(rewards), "mixed_group_rate": mixed,
               **m_sel, **m_exec}
    with open(f"{args.root}/metrics.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")
    print("METRICS", json.dumps(metrics))
    # 验收即时判读
    print("ACCEPT reward_nonzero:", "PASS" if metrics["reward_nonzero"] > 0 else "FAIL")
    print("ACCEPT finite_losses:",
          "PASS" if all(v is None or v == v for v in (m_sel["sel_loss_mean"], m_exec["exec_loss_mean"])) else "FAIL")


if __name__ == "__main__":
    main()
