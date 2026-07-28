#!/usr/bin/env python3
"""Build the selector-allocation DiD corpus v5(多 k policy 训练分布).

# note (luojiaxuan): 动机 = 2026-07-29 两个闭环反转:k≤1 上限把 memory-critical
# 增益整体抹掉(多槽重组才是载体),而 HGKV 只在 k=1 单替换分布上训过——分布
# 漂移。v5 把 S 臂从"最老槽位单替换"换成部署版 selector 实际组合的 S*
# (cheap 特征 + beam exact-B,与 serve 同构),k 随状态自然分布 1..B;
# W 臂 = 同基数替换:S* 的每个促升事件换成 age 最近邻的 wrong 事件
# (following action 与 target 不等价)。R/RA、S0/SA、WA 槽位契约与 v4 逐字
# 相同,trainer 零改动直接吃。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from causalcache.agentnet_desktop_official import (
    DESKTOP_OFFICIAL_PROTOCOL_ID,
    build_official_forms_for_record,
    render_official_target_text,
)
from causalcache.selector_v4_features import (
    candidate_features,
    set_context_features,
)
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_did_corpus_v4 import (
    DESKTOP_ARM_CONTRACT_V2,
    SAMPLE_SCHEMA_V2,
    render_official_selection,
)
from scripts.build_desktop_hgkv_corpus import (
    NEGATIVE_ARM_SLOT,
    NEGATIVE_KIND,
    NEGATIVE_SCALE,
    _split,
    history_action_matches_target,
    messages_sha256,
    sha256_file,
)

CORPUS_SCHEMA = "causalcache.desktop_did_corpus_selected.v5"
PROMPT_FORMAT_V2 = "desktop_official_multiturn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--selector-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--b-values", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--coordinate-tolerance", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--max-groups-per-b", type=int, default=0)
    return parser.parse_args()


def load_scorer(bundle_path: Path):
    import torch

    bundle = torch.load(bundle_path, map_location="cpu")
    dim = len(bundle["feature_names"]) + len(bundle["set_feature_names"])
    hidden = int(bundle["hidden"])

    class _TwoTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cheap = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, 1),
            )
            ro = int(bundle.get("readout_dim") or 0)
            if ro:
                self.readout = torch.nn.Sequential(
                    torch.nn.Linear(ro, 64), torch.nn.GELU(),
                    torch.nn.Dropout(0.0), torch.nn.Linear(64, 1),
                )

        def forward(self, x):
            return self.cheap(x).squeeze(-1)

    model = _TwoTower()
    model.load_state_dict(bundle["model_state"])
    model.eval()
    mean, std = bundle["mean"], bundle["std"]

    def score(feature_rows):
        normed = [
            [(v - m) / s for v, m, s in zip(row, mean, std)]
            for row in feature_rows
        ]
        with torch.no_grad():
            return model(torch.tensor(normed, dtype=torch.float32)).tolist()

    return score, bundle


def candidate_pool(record, forms, image_root: Path):
    """与 score_selector_v4_singletons 相同的池规则:可渲染 + byte 去重。"""
    import hashlib as _h

    current_step = int(record["step"])
    relpaths = record["image_relpaths"]
    seen_hash: dict[str, int] = {}
    pool: list[int] = []
    duplicates: dict[int, int] = {}
    for event in range(current_step - 2, 0, -1):
        if not forms[event].full_response:
            continue
        frame_hash = _h.sha256((image_root / relpaths[event]).read_bytes()).hexdigest()
        if frame_hash in seen_hash:
            duplicates[event] = seen_hash[frame_hash]
            continue
        seen_hash[frame_hash] = event
        pool.append(event)
    pool.sort()
    return pool, duplicates


def compose_selected(record, pool, duplicates, score, *, budget: int, beam: int,
                     coordinate_tolerance: int) -> list[int] | None:
    """beam exact-B 组合(cheap 特征,与 serve 部署路径同构)。"""
    if len(pool) < budget:
        return None
    level: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(budget):
        expanded: list[tuple[tuple[int, ...], float]] = []
        seen: set[tuple[int, ...]] = set()
        for sel, acc in level:
            remaining = [s for s in pool if s not in sel]
            if not remaining:
                continue
            rows = [
                candidate_features(
                    record, candidate_pool=pool, duplicates=duplicates,
                    event=s, coordinate_tolerance=coordinate_tolerance,
                ) + set_context_features(
                    record, selected=list(sel), event=s,
                    duplicates=duplicates,
                    coordinate_tolerance=coordinate_tolerance,
                )
                for s in remaining
            ]
            margins = score(rows)
            order = sorted(zip(margins, remaining), reverse=True)[:beam]
            for margin, s in order:
                child = tuple(sorted([*sel, s]))
                if child in seen:
                    continue
                seen.add(child)
                expanded.append((child, acc + margin))
        if not expanded:
            return None
        expanded.sort(key=lambda item: item[1], reverse=True)
        level = expanded[:beam]
    return list(level[0][0])


def wrong_pool(record, *, coordinate_tolerance: int) -> list[int]:
    """following action 与 target 不等价的事件(select_contrast_events 同规则)。"""
    current_step = int(record["step"])
    screen_size = tuple(record["screen_size"])
    target = record["target_tool_call"]
    history_by_step = {int(e["step_id"]): e for e in record["history"]}
    out = []
    for event_step in range(1, current_step - 1):
        following = history_by_step.get(event_step + 1)
        if following is None:
            continue
        if history_action_matches_target(
            following["action"], target, screen_size=screen_size,
            coordinate_tolerance=coordinate_tolerance,
        ):
            continue
        out.append(event_step)
    return out


def main() -> None:
    args = parse_args()
    score, bundle = load_scorer(args.selector_bundle)
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    states = [
        r for r in records
        if _split(str(r["task_id"]), seed=args.seed) in set(args.splits)
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)

    for b in args.b_values:
        rows: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        k_hist: Counter[int] = Counter()
        referenced: set[str] = set()
        for record in states:
            if args.max_groups_per_b and counters["groups"] >= args.max_groups_per_b:
                break
            current_step = int(record["step"])
            full_window = recent_window(current_step, b)
            if not full_window or min(full_window) < 1:
                counters["rejected_too_short"] += 1
                continue
            dp = str(record["dp_id"])
            try:
                forms = build_official_forms_for_record(record)
            except Exception:
                counters["rejected_forms"] += 1
                continue
            pool, duplicates = candidate_pool(record, forms, args.image_root)
            selected = compose_selected(
                record, pool, duplicates, score, budget=b, beam=args.beam,
                coordinate_tolerance=args.coordinate_tolerance,
            )
            if selected is None:
                counters["rejected_pool_small"] += 1
                continue
            canon = {duplicates.get(s, s) for s in full_window}
            promoted = sorted(s for s in selected if s not in canon
                              and s not in full_window)
            k = len(promoted)
            k_hist[k] += 1
            if k == 0:
                counters["skipped_k0_recent_kept"] += 1
                continue
            # W 臂:每个促升事件换成 age 最近邻 wrong 事件(不可与 S* 重叠)
            negatives = [
                w for w in wrong_pool(
                    record, coordinate_tolerance=args.coordinate_tolerance)
                if w not in selected
            ]
            wrong_set = [s for s in selected if s not in promoted]
            used: set[int] = set()
            feasible = True
            for event in promoted:
                ranked = sorted(
                    (abs(w - event), w) for w in negatives if w not in used
                )
                if not ranked:
                    feasible = False
                    break
                used.add(ranked[0][1])
                wrong_set.append(ranked[0][1])
            if not feasible:
                counters["rejected_no_wrong_match"] += 1
                continue
            wrong_set.sort()

            specs = {
                "recent": list(full_window),
                "recurrence": sorted(selected),
                "wrong": wrong_set,
            }
            try:
                rendered = {}
                for mode, steps in specs.items():
                    messages = render_official_selection(record, forms, steps=steps)
                    rendered[mode] = {
                        "selected_steps": list(steps),
                        "selected_images": [
                            record["image_relpaths"][step] for step in steps],
                        "messages": messages,
                        "messages_sha256": messages_sha256(messages),
                    }
            except ValueError:
                counters["dropped_retained_unrenderable"] += 1
                continue

            target_text = render_official_target_text(record["target_tool_call"])
            pair_group = f"agentnet:{dp}:sel-b{b}"
            episode = str(record["task_id"])
            split = _split(episode, seed=args.seed)
            common = {
                "pair_group": pair_group,
                "episode": episode,
                "decision_step": current_step,
                "budget": b,
                "kept_recent_steps": [s for s in selected if s in full_window
                                      or s in canon],
                "replaced_slot_event": None,
                "k_replaced": k,
                "split": split,
                "instruction": record["instruction"],
                "current_image": record["image_relpaths"][current_step - 1],
                "target_text": target_text,
                "reference_arm_id": "R0",
                "deployment_baseline_arm_id": "R0",
                "source": {
                    "dataset": "AgentNet/OpenCUA",
                    "os": record["os"],
                    "dp_id": dp,
                    "positive_rule": "selector_v4_beam_composition",
                    "positive_action_step": None,
                    "positive_event_step": promoted[0],
                    "wrong_event_step": wrong_set[-1],
                    "promoted_events": promoted,
                    "positive_age": current_step - promoted[0],
                    "wrong_age": current_step - wrong_set[-1],
                    "recent_definition": "distinct_frames_v2",
                    "budget_semantics": "selector_allocation_v5",
                    "renderer": DESKTOP_OFFICIAL_PROTOCOL_ID,
                    "selector_bundle_sha256": sha256_file(args.selector_bundle),
                },
            }
            for slot, (arm_id, role, prompt_format, mode, adapter_mode) in (
                DESKTOP_ARM_CONTRACT_V2.items()
            ):
                row = {
                    "schema_version": SAMPLE_SCHEMA_V2,
                    "sample_id": f"{pair_group}|{slot}",
                    **common,
                    "arm_slot": slot,
                    "arm_id": arm_id,
                    "role": role,
                    "prompt_format": prompt_format,
                    "selection_mode": mode,
                    "adapter_mode": adapter_mode,
                    "variant": f"{slot}_{mode}{b}",
                    "recent_frames_kept": sum(
                        1 for step in rendered[mode]["selected_steps"]
                        if step in full_window
                    ),
                    "memory_config": {
                        "restored_event_step_ids":
                            rendered[mode]["selected_steps"]
                    },
                    **{
                        key: rendered[mode][key]
                        for key in ("selected_steps", "selected_images",
                                    "messages", "messages_sha256")
                    },
                }
                if slot == NEGATIVE_ARM_SLOT:
                    row["negative_kind"] = NEGATIVE_KIND
                    row["negative_scale"] = NEGATIVE_SCALE
                    row["donor_episode"] = episode
                    row["distractor_source_step"] = wrong_set[-1]
                    row["oracle_source_step"] = promoted[0]
                rows.append(row)
            for mode in specs:
                referenced.update(rendered[mode]["selected_images"])
            referenced.add(common["current_image"])
            counters["groups"] += 1
            counters[f"groups_{split}"] += 1

        out = args.output_root / f"samples-selb{b}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "schema_version": CORPUS_SCHEMA,
            "budget": b,
            "counters": dict(sorted(counters.items())),
            "k_histogram": {str(k): v for k, v in sorted(k_hist.items())},
            "selector_bundle_sha256": sha256_file(args.selector_bundle),
            "samples_sha256": sha256_file(out),
            "referenced_paths": len(referenced),
        }
        (args.output_root / f"manifest-selb{b}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True))
        print(json.dumps({"budget": b, "groups": counters["groups"],
                          "k_histogram": manifest["k_histogram"],
                          "rows": len(rows)}))


if __name__ == "__main__":
    main()
