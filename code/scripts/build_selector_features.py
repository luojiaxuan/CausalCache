#!/usr/bin/env python3
"""Extract platform-generic selector features and join with U_act labels.

# note (luojiaxuan): selector 第一版是 per-event singleton gain 回归。特征只用
# 平台通用信号(事件年龄/位置、动作类型 one-hot、事件 OCR 与当前指令的词重合、
# 摘要长度、候选数),绝不含 benchmark task id 或应用专属标签,保证零样本可迁移。
# 标签 U_act(j) 来自 ody-selector-labels/uact_labels.jsonl,特征从 variable-history
# 源表反查事件摘要与 OCR。轨迹级哈希切 train/heldout(盐 odyssey_selector_v1)。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ACTION_TYPES = (
    "click",
    "swipe",
    "type",
    "long_press",
    "system_button",
    "open",
    "answer",
    "wait",
    "terminate",
)
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _heldout(source_id: str, salt: str, fraction: float) -> bool:
    digest = hashlib.sha256(f"{salt}:{source_id}".encode()).digest()[:4]
    return int.from_bytes(digest, "big") / 2**32 < fraction


def event_features(
    *,
    event_summary: dict[str, Any],
    event_ocr_tokens: list[str],
    instruction_words: set[str],
    decision_step: int,
    candidate_step: int,
    candidate_count: int,
) -> dict[str, float]:
    age = decision_step - candidate_step
    action_type = str(event_summary.get("action_type", "")).lower()
    feats: dict[str, float] = {
        "age": float(age),
        "age_norm": float(age) / max(decision_step, 1),
        "event_position": float(candidate_step),
        "position_norm": float(candidate_step) / max(decision_step, 1),
        "candidate_count": float(candidate_count),
        "is_most_recent": 1.0 if age == 1 else 0.0,
        "is_oldest": 1.0 if candidate_step == 1 else 0.0,
        "ocr_token_count": float(len(event_ocr_tokens)),
        "summary_arg_len": float(len(str(event_summary.get("action_argument", "")))),
    }
    for a in ACTION_TYPES:
        feats[f"action_{a}"] = 1.0 if action_type == a else 0.0
    event_words = set()
    for tok in event_ocr_tokens:
        event_words |= _words(str(tok))
    overlap = event_words & instruction_words
    feats["ocr_instr_overlap"] = float(len(overlap))
    feats["ocr_instr_overlap_frac"] = float(len(overlap)) / max(
        len(instruction_words), 1
    )
    return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-salt", default="odyssey_selector_v1")
    parser.add_argument("--heldout-fraction", type=float, default=0.15)
    args = parser.parse_args()

    from pyarrow import parquet as pq

    labels: dict[str, dict[str, float]] = {}
    for line in args.labels.open(encoding="utf-8"):
        row = json.loads(line)
        labels[row["pair_group"]] = {
            k: v["u_act"] for k, v in row["candidates"].items()
        }
    wanted_sources = {pg.split(":")[0] for pg in labels}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for shard in sorted(args.source_root.glob("shard-*.parquet")):
            table = pq.read_table(shard).to_pylist()
            for record in table:
                source_id = record["source_id"]
                if source_id not in wanted_sources:
                    continue
                instruction_words = _words(record["task_instruction"])
                events = json.loads(record["history_events_json"])
                summary_by_step = {
                    e["event_step_id"]: e["low_fidelity_summary"] for e in events
                }
                ocr_records = json.loads(record["ocr_records_json"])

                def ocr_for(step_id: int) -> list[str]:
                    rec = None
                    if isinstance(ocr_records, list):
                        rec = ocr_records[step_id] if step_id < len(ocr_records) else None
                    elif isinstance(ocr_records, dict):
                        for key in (str(step_id), f"observation-{step_id:03d}"):
                            if key in ocr_records:
                                rec = ocr_records[key]
                                break
                    return list(rec.get("full_spatial_tokens", [])) if isinstance(
                        rec, dict
                    ) else []

                for pair_group, cand_labels in labels.items():
                    if pair_group.split(":")[0] != source_id:
                        continue
                    decision_step = int(pair_group.split(":")[1])
                    n_cand = len(cand_labels)
                    for cand_str, u_act in cand_labels.items():
                        cand = int(cand_str)
                        summary = summary_by_step.get(cand)
                        if summary is None:
                            continue
                        feats = event_features(
                            event_summary=summary,
                            event_ocr_tokens=ocr_for(cand),
                            instruction_words=instruction_words,
                            decision_step=decision_step,
                            candidate_step=cand,
                            candidate_count=n_cand,
                        )
                        handle.write(
                            json.dumps(
                                {
                                    "source_id": source_id,
                                    "pair_group": pair_group,
                                    "candidate_step": cand,
                                    "u_act": u_act,
                                    "features": feats,
                                    "split": "heldout"
                                    if _heldout(
                                        source_id,
                                        args.heldout_salt,
                                        args.heldout_fraction,
                                    )
                                    else "train",
                                }
                            )
                            + "\n"
                        )
                        written += 1
    print(json.dumps({"feature_rows": written, "sources": len(wanted_sources)}))


if __name__ == "__main__":
    main()
