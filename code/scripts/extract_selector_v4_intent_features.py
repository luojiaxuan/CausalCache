#!/usr/bin/env python3
"""Intent 特征提取(selector v2 单遍化:意图从 instruction/轨迹学,不再依赖拟议动作)。

# note (luojiaxuan): witness 消融判定主设置 B=4 的显著性依赖意图信号后的路径 2。
# 每个 (决策点, 候选事件) 输出 8 维工程化意图特征,全部**目标无关**、单遍可算、
# 平台可迁移(mobile 侧动作行 = 模型自写 action_text,同一冻结嵌入):
#   0 cos(instr, cand_action)          指令与候选动作行的语义对齐
#   1 cos(cand_action, mean(last3))    候选动作与近期行为的相似度
#   2 cos(cand_action, mean(all))      与全轨迹行为均值
#   3 cos(instr, mean(last3))          状态级:近期行为是否还在贴指令
#   4 max_{j∈last3} cos(cand, act_j)   峰值相似
#   5 progress = min(step/30, 1)       轨迹进度(粗粒度)
#   6 cand_pos = event/decision_step   候选相对位置(与 cheap 重叠,交互用)
#   7 len_norm = min(#tokens/24, 1)    动作行长度
# 嵌入 = 冻结 GUI-Owl 输入嵌入层 mean-pool + L2 归一(不跑 transformer 层,
# 单卡分钟级);只落 8 维,原始嵌入不落盘。输出 jsonl(dp|event 键,断点续跑)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalcache.agentnet_desktop_official import build_official_forms_for_record
from scripts.build_desktop_hgkv_corpus import _split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--singletons-root", type=Path, required=True,
                        help="候选池权威来源(b0 行),与标签同池")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(str(args.model_dir))
    tokenizer = processor.tokenizer
    # 只要输入嵌入层:加载到 CPU 取权重再搬 GPU,避免整模型驻留
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16, device_map="cpu"
    )
    embed = model.get_input_embeddings().to(args.device)
    del model

    def text_embed(texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=64)
            ids = enc["input_ids"].to(args.device)
            mask = enc["attention_mask"].to(args.device).unsqueeze(-1)
            with torch.no_grad():
                vec = (embed(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(vec.float(), dim=-1))
        return torch.cat(out) if out else torch.zeros(0)

    records = {
        str(json.loads(line)["dp_id"]): json.loads(line)
        for line in args.screening_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    b0_pools: dict[str, list[int]] = {}
    for path in sorted(args.singletons_root.glob("singletons.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "b0":
                b0_pools[row["dp_id"]] = [int(e) for e in row["candidate_pool"]]

    ordered = [
        dp for dp in b0_pools
        if dp in records
        and _split(str(records[dp]["task_id"]), seed=args.seed) in set(args.splits)
    ]
    states = [dp for i, dp in enumerate(ordered)
              if i % args.shard_count == args.shard_index]

    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / (
        f"intent.shard{args.shard_index:03d}-of-{args.shard_count:03d}.jsonl"
    )
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["key"])
            except Exception:  # noqa: BLE001
                continue
    handle = out_path.open("a", encoding="utf-8")
    if not done:
        handle.write(json.dumps({
            "key": "__fingerprint__",
            "schema_version": "causalcache.selector_v4_intent_features.v1",
            "dims": 8,
            "embed_source": "gui_owl_input_embeddings_meanpool_l2",
        }) + "\n")
        handle.flush()
        done.add("__fingerprint__")

    import torch.nn.functional as F
    for dp in states:
        if f"{dp}|done" in done:
            continue
        record = records[dp]
        forms = build_official_forms_for_record(record)
        action_lines = [f.action_line for f in forms]
        decision_step = int(record["step"])
        texts = [str(record["instruction"])] + action_lines
        vecs = text_embed(texts)
        instr = vecs[0]
        acts = vecs[1:]
        last3 = acts[-3:] if len(acts) >= 1 else acts
        last3_mean = F.normalize(last3.mean(0, keepdim=True), dim=-1)[0] \
            if len(last3) else instr * 0
        all_mean = F.normalize(acts.mean(0, keepdim=True), dim=-1)[0] \
            if len(acts) else instr * 0
        cos_instr_last3 = float(instr @ last3_mean) if len(last3) else 0.0
        rows_out = []
        for event in b0_pools[dp]:
            # 候选事件 j 的展示动作 = 步骤 j+1(desktop post-frame 语义)
            idx = event  # forms[event] = 步骤 event+1
            if idx >= len(acts):
                continue
            cand = acts[idx]
            tok_count = len(tokenizer(action_lines[idx])["input_ids"])
            peak = float(max((cand @ v for v in last3), default=0.0))
            rows_out.append({
                "key": f"{dp}|{event}", "kind": "intent",
                "dp_id": dp, "event": event,
                "vector": [
                    round(float(instr @ cand), 5),
                    round(float(cand @ last3_mean), 5) if len(last3) else 0.0,
                    round(float(cand @ all_mean), 5) if len(acts) else 0.0,
                    round(cos_instr_last3, 5),
                    round(peak, 5),
                    round(min(decision_step / 30.0, 1.0), 5),
                    round(event / max(decision_step - 1, 1), 5),
                    round(min(tok_count / 24.0, 1.0), 5),
                ],
            })
        for row in rows_out:
            handle.write(json.dumps(row) + "\n")
        handle.write(json.dumps({"key": f"{dp}|done"}) + "\n")
        handle.flush()
        done.add(f"{dp}|done")
    handle.close()
    print(json.dumps({"shard": args.shard_index, "states": len(states)}))


if __name__ == "__main__":
    main()
