#!/usr/bin/env python3
"""Build the sparse-history HGKV training corpus (V3) from the GUI-Odyssey pool.

# note (luojiaxuan): 与 V2 的三处关键差异——
#   1. 直接从 full-pool 取长轨迹(V2 只冻结 1,200 条,>=32 步仅 29 条);
#   2. 每条轨迹抽多个中后段决策点(V2 只取最后一个决策,导致 >=24 历史 97% 是 click);
#   3. 每组五变体,reference 从 B0 换成官方连续 Recent-4。
# 过滤:action_description 连续重复 >=5 的轨迹剔除(字幕退化,>=32 步中占 35%)。
# 目标动作沿用坐标修复逻辑([0,999] 直映射),输出为官方原生格式 Action + tool_call。
"""
from __future__ import annotations
import argparse, hashlib, json, random, unicodedata

from causalcache.policy.gui_owl_official import (
    NO_PREVIOUS_ACTION, OFFICIAL_SYSTEM_PROMPT, build_official_messages,
)
from causalcache.policy.gui_owl_sparse_history import build_sparse_history_messages
from pathlib import Path
from typing import Any

SYSTEM_BUTTONS = {"HOME": "Home", "BACK": "Back", "KEY_HOME": "Home", "KEY_BACK": "Back"}
# note (luojiaxuan): 主结果是 Select-4,但 adapter 必须在 B=1/2/3/4 都校准,
# 故放平分布;每组的 reference 臂用**同预算**的 native Recent-K,不是固定 Recent-4,
# 否则 K<4 时等于逼模型"用更少的图打赢更多的图",会复现旧 adapter 的过度放大。
K_DISTRIBUTION = ((1, 0.15), (2, 0.25), (3, 0.25), (4, 0.35))


def scale(point: Any) -> list[int]:
    # GUIOdyssey 注释坐标已归一化到 [0,1000),直接映射 [0,999],绝不除分辨率
    x, y = point
    return [max(0, min(999, round(float(x) * 999 / 1000))),
            max(0, min(999, round(float(y) * 999 / 1000)))]


def target_arguments(step: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(step.get("action", "")).upper()
    info = step.get("info")
    if kind == "CLICK":
        if isinstance(info, str):
            b = SYSTEM_BUTTONS.get(info)
            return {"action": "system_button", "button": b} if b else None
        if isinstance(info, (list, tuple)) and info and isinstance(info[0], (list, tuple)):
            return {"action": "click", "coordinate": scale(info[0])}
        return None
    if kind == "LONG_PRESS" and isinstance(info, (list, tuple)) and info and isinstance(info[0], (list, tuple)):
        return {"action": "long_press", "coordinate": scale(info[0])}
    if kind == "SCROLL" and isinstance(info, (list, tuple)) and len(info) == 2 \
            and all(isinstance(p, (list, tuple)) for p in info):
        return {"action": "swipe", "coordinate": scale(info[0]), "coordinate2": scale(info[1])}
    if kind == "TEXT":
        t = info if isinstance(info, str) else step.get("text")
        return {"action": "type", "text": unicodedata.normalize("NFKC", t)} if isinstance(t, str) and t else None
    if kind in SYSTEM_BUTTONS:
        return {"action": "system_button", "button": SYSTEM_BUTTONS[kind]}
    if kind == "COMPLETE":
        return {"action": "terminate", "status": "success"}
    return None


def max_run(seq: list[str]) -> int:
    run = best = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if seq else 0


def choose_sparse(rng: random.Random, cur: int, k: int) -> list[int]:
    """Non-contiguous selection with at least one old (age>4) event."""
    pool = list(range(1, cur))
    if len(pool) < k:
        return pool
    old = [s for s in pool if cur - s > 4]
    picked: list[int] = []
    if old:
        picked.append(rng.choice(old))
    while len(picked) < k:
        cand = rng.choice(pool)
        if cand in picked or any(abs(cand - p) == 1 for p in picked):
            if len(pool) < 3 * k:
                if cand in picked:
                    continue
            else:
                continue
        picked.append(cand)
    return sorted(set(picked))


def render_messages(
    *, instruction: str, action_texts: list[str], steps: list[int],
    image_paths: list[str], current_step: int, current_path: str, sparse: bool,
) -> list[dict[str, Any]]:
    """Render a variant into serializable messages (images carried as paths).

    # note (luojiaxuan): encode_sample 期望 messages[*].content 里 type=="image"
    # 的部分带 ``path``(相对 dataset_root),训练时才去开图。native_recent 变体走
    # 官方连续 builder,sparse 变体走 sparse builder,两者输出协议一致。
    """
    if sparse:
        msgs = build_sparse_history_messages(
            instruction=instruction, action_texts=action_texts,
            selected_steps=steps, selected_images=[{"__path__": p} for p in image_paths],
            current_step=current_step, current_image={"__path__": current_path},
        )
    else:
        msgs = build_official_messages(
            goal=instruction, past_action_texts=action_texts,
            recent_images=[{"__path__": p} for p in image_paths],
            current_image={"__path__": current_path},
        )
    out = []
    for m in msgs:
        content = []
        for part in m["content"]:
            if part.get("type") == "image":
                content.append({"type": "image", "path": part["image"]["__path__"]})
            else:
                content.append(dict(part))
        out.append({"role": m["role"], "content": content})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--pool-root", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--max-trajectories", type=int, default=100)
    ap.add_argument("--decisions-per-trajectory", type=int, default=3)
    ap.add_argument("--max-consecutive-repeat", type=int, default=5)
    ap.add_argument("--heldout-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=271828)
    # note (luojiaxuan): 构建是 CPU 密集(parquet 解码 + PNG 落盘),单进程只用 1 核。
    # 用 --shard-index/--shard-count 起多个独立进程并行(每进程各写各的输出目录),
# 比单进程内做池化更简单也更稳;合并时把 samples.jsonl 拼起来、images 目录共存即可。
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    from pyarrow import parquet as pq

    rng = random.Random(args.seed)
    sel = json.loads(args.selection.read_text(encoding="utf-8"))["selected"]
    rng.shuffle(sel)
    by_shard: dict[str, list[dict]] = {}
    for s in sel:
        by_shard.setdefault(s["shard"], []).append(s)

    out = args.output_root
    (out / "images").mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    kept = filtered = 0
    stats = {"decisions": 0, "actions": {}, "K": {}, "history_len": []}

    shard_names = sorted(by_shard)
    if args.shard_count > 1:
        shard_names = shard_names[args.shard_index :: args.shard_count]
    for shard in shard_names:
        items = by_shard[shard]
        if kept >= args.max_trajectories:
            break
        table = pq.read_table(args.pool_root / shard, columns=["messages", "images"])
        msgs_all = table.column("messages").to_pylist()
        imgs_all = table.column("images").to_pylist()
        for it in items:
            if kept >= args.max_trajectories:
                break
            sid = it["source_id"]
            m = msgs_all[it["row"]]
            if isinstance(m, str):
                m = json.loads(m)
            acts = [p.get("text", "") for x in m if x.get("role") == "assistant"
                    for p in (x.get("content") or []) if p.get("type") == "action_description"]
            instr = next((p["text"] for x in m if x.get("role") == "user"
                          for p in (x.get("content") or []) if p.get("type") == "text" and p.get("text")), "")
            if not acts or not instr:
                continue
            if max_run(acts) >= args.max_consecutive_repeat:
                filtered += 1
                continue
            ann_path = args.annotations / f"{sid}.json"
            if not ann_path.exists():
                continue
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            steps_by_idx = {int(s["step"]): s for s in ann.get("steps", [])}
            raw_imgs = imgs_all[it["row"]]
            imgs = [bytes(x["bytes"]) if isinstance(x, dict) else bytes(x) for x in raw_imgs]
            if len(imgs) < 8:
                continue
            idir = out / "images" / sid
            idir.mkdir(parents=True, exist_ok=True)
            for i, payload in enumerate(imgs):
                p = idir / f"obs-{i:03d}.png"
                if not p.exists():
                    p.write_bytes(payload)
            # note (luojiaxuan): 三者对齐关系(全部 0 基)——注释 step[d] 的动作,
            # 由 acts[d] 描述,执行前的屏幕是 imgs[d]。展示用的 Step 编号 = d+1。
            usable = min(len(acts), len(imgs))
            fracs = [0.5, 0.7, 0.85][: args.decisions_per_trajectory]
            decisions = sorted({int(usable * f) for f in fracs if int(usable * f) >= 6})
            decisions = [d for d in decisions if d < usable][: args.decisions_per_trajectory]
            if not decisions:
                continue
            heldout = int.from_bytes(hashlib.sha256(f"sparse_v3:{sid}".encode()).digest()[:4], "big") / 2**32 < args.heldout_fraction
            made = 0
            for d in decisions:
                tgt = target_arguments(steps_by_idx.get(d, {}))
                if tgt is None:
                    continue
                desc = acts[d]
                cur = d + 1  # 展示用的 1 基 Step 编号
                target_text = ('Action: ' + desc + '\n<tool_call>\n{"name": "mobile_use", "arguments": '
                               + json.dumps(tgt, ensure_ascii=False, separators=(", ", ": ")) + '}\n</tool_call>')
                r = rng.random(); acc = 0.0; K = 4
                for k, w in K_DISTRIBUTION:
                    acc += w
                    if r <= acc:
                        K = k; break
                sparse_steps = choose_sparse(rng, cur, K)
                K_eff = len(sparse_steps)
                # 预算对齐:reference 是同样 K_eff 张的连续最近窗口
                recent = list(range(max(1, cur - K_eff), cur))
                donor = rng.choice([x for x in sel if x["source_id"] != sid])["source_id"]
                def path(step: int) -> str:      # 展示 Step s -> imgs[s-1]
                    return f"images/{sid}/obs-{step - 1:03d}.png"
                # note (luojiaxuan): 两条 recent 基线各司其职——
                #   native_recent{K}     官方多轮格式,部署基线(论文主表用);
                #   sameformat_recent{K} 同一 sparse builder 渲染的连续最近 K 步,
                #                        训练 reference 用它,才能把 prompt 格式变量摁住。
                # 实测格式效应占 native 与 sparse 差值的 83%(+0.133/+0.160),
                # 用 native 当训练 reference 会让 adapter 靠格式假象轻松过关。
                variants = {
                    f"native_recent{K_eff}": {"steps": recent, "images": [path(s) for s in recent], "sparse": False},
                    f"sameformat_recent{K_eff}": {"steps": recent, "images": [path(s) for s in recent], "sparse": True},
                    "sparse_correct": {"steps": sparse_steps, "images": [path(s) for s in sparse_steps], "sparse": True},
                    "sparse_step_shuffled": {"steps": sparse_steps,
                        "images": [path(s) for s in (sparse_steps[::-1] if K_eff > 1 else sparse_steps)], "sparse": True},
                    "sparse_irrelevant": {"steps": sparse_steps, "images": [f"DONOR:{donor}:{s}" for s in sparse_steps], "sparse": True},
                    "sparse_duplicate": {"steps": sparse_steps,
                        "images": [path(sparse_steps[0])] * K_eff, "sparse": True},
                }
                pg = f"{sid}:{cur}"
                for name, v in variants.items():
                    samples.append({
                        "pair_group": pg, "variant": name, "episode": sid, "decision_step": cur,
                        "reference_variant": f"native_recent{K_eff}", "budget": K_eff,
                        "instruction": instr, "action_texts": acts[:d],
                        "selected_steps": v["steps"], "selected_images": v["images"],
                        "current_image": f"images/{sid}/obs-{d:03d}.png",
                        "target_text": target_text, "split": "heldout" if heldout else "train",
                        "sparse": v["sparse"], "_needs_donor": name == "sparse_irrelevant",
                    })
                stats["decisions"] += 1
                stats["actions"][tgt["action"]] = stats["actions"].get(tgt["action"], 0) + 1
                stats["K"][str(K_eff)] = stats["K"].get(str(K_eff), 0) + 1
                stats["history_len"].append(cur)
                made += 1
            if made:
                kept += 1

    # note (luojiaxuan): donor 必须来自**已落盘**的轨迹,否则 irrelevant 变体会
    # 指向不存在的图。此处统一在全部图片写完后再解析占位符。
    rendered_sids = sorted({s["episode"] for s in samples})
    rng2 = random.Random(args.seed + 1)
    written = 0
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for s in samples:
            if s.pop("_needs_donor", False):
                pool_ids = [x for x in rendered_sids if x != s["episode"]]
                if not pool_ids:
                    continue
                donor = rng2.choice(pool_ids)
                donor_imgs = sorted((out / "images" / donor).glob("obs-*.png"))
                if len(donor_imgs) < len(s["selected_steps"]):
                    continue
                s["selected_images"] = [
                    f"images/{donor}/{donor_imgs[i % len(donor_imgs)].name}"
                    for i in range(len(s["selected_steps"]))
                ]
            try:
                s["messages"] = render_messages(
                    instruction=s["instruction"], action_texts=s["action_texts"],
                    steps=s["selected_steps"], image_paths=s["selected_images"],
                    current_step=s["decision_step"], current_path=s["current_image"],
                    sparse=s["sparse"],
                )
            except ValueError:
                continue
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            written += 1
    hl = sorted(stats["history_len"])
    manifest = {
        "trajectories": kept, "filtered_degenerate": filtered,
        "decision_groups": stats["decisions"], "samples": written,
        "target_action_mix": stats["actions"], "K_mix": stats["K"],
        "history_len_median": hl[len(hl) // 2] if hl else 0,
        "history_len_p90": hl[int(0.9 * len(hl))] if hl else 0,
        "history_len_max": hl[-1] if hl else 0,
        "heldout_groups": len({s["pair_group"] for s in samples if s["split"] == "heldout"}),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
