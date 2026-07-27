#!/usr/bin/env python3
"""Fill-to-B 端到端离线验收:selector 实选分配 vs Recent-B vs oracle(同图数真 U)。

# note (luojiaxuan): paper 主表 selector 行的测量脚本。部署语义(2026-07-27 冻结):
# 恒用满 B、recent 兜底、beam 组合、无 STOP 阈值。流程:
#   1. 每个 dev 状态:候选池 = 单例表 b0 行(与标签同池);特征 = cheap+set-context
#      (与训练同函数)+ 读出(候选池内 z 归一,**逐字复刻训练端**);
#   2. selector 推理:beam(宽 3)从 size-1 逐级组到 size-B,边际由训练好的
#      Δ̂(j|S) 打分;recent 帧是普通候选,全拒绝时 beam 自然收敛到 Recent-B;
#   3. 实选分配的**真 U**:先查集合表/锚(teacher 已打分即复用),缺失才用 s300
#      active 整集重打分(共享 true-score cache,断点续跑);
#   4. 汇总:selector−Recent-B 的 episode-cluster bootstrap CI、oracle gap、
#      k = |S∖Recent-B| 分布。B ∈ {2,4}。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from causalcache.agentnet_desktop_official import (
    build_desktop_official_messages,
    build_official_forms_for_record,
    render_official_target_text,
)
from causalcache.selector_v4_features import (
    FEATURE_NAMES,
    SET_FEATURE_NAMES,
    candidate_features,
    set_context_features,
)
from scripts.build_desktop_did_corpus_v2 import recent_window
from scripts.build_desktop_hgkv_corpus import _split

PROMPT_FORMAT = "desktop_official_multiturn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", type=Path, required=True,
                        help="marginal_scorer.pt(two_tower 或 concat)")
    parser.add_argument("--arch", choices=("two_tower", "concat"), required=True)
    parser.add_argument("--singletons-root", type=Path, required=True)
    parser.add_argument("--sets-root", type=Path, required=True)
    parser.add_argument("--readout-root", type=Path, required=True)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--true-score-cache", type=Path, required=True,
                        help="dp|set → 真 U 的共享 jsonl cache(跨 arm 复用)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--b-values", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-states", type=int, default=0)
    return parser.parse_args()


def load_rows(root: Path, pattern: str):
    for path in sorted(root.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def cluster_ci(values_by_episode, *, iterations, seed):
    episodes = sorted(values_by_episode)
    flat = [v for ep in episodes for v in values_by_episode[ep]]
    if not flat:
        return None
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [
            v for ep in (rng.choice(episodes) for _ in episodes)
            for v in values_by_episode[ep]
        ]
        if sample:
            means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "point": point,
        "ci_low": means[int(0.025 * len(means))],
        "ci_high": means[min(int(0.975 * len(means)), len(means) - 1)],
        "n": len(flat),
    }


def main() -> None:
    args = parse_args()
    import torch

    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256:
        raise SystemExit("checkpoint SHA drifted")

    bundle = torch.load(args.scorer, map_location="cpu")
    dim = len(FEATURE_NAMES) + len(SET_FEATURE_NAMES)
    readout_dim = int(bundle.get("readout_dim") or 0)
    hidden = int(bundle["hidden"])
    mean, std = bundle["mean"], bundle["std"]

    class TwoTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cheap = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, 1),
            )
            if readout_dim:
                self.readout = torch.nn.Sequential(
                    torch.nn.Linear(readout_dim, 64), torch.nn.GELU(),
                    torch.nn.Dropout(0.0),
                    torch.nn.Linear(64, 1),
                )

        def forward(self, x, r, rm):
            score = self.cheap(x).squeeze(-1)
            if readout_dim:
                score = score + self.readout(r).squeeze(-1) * rm
            return score

    class Concat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = torch.nn.Dropout(0.0)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim + readout_dim, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, 1),
            )

        def forward(self, x, r, rm):
            r = self.drop(r) * rm.unsqueeze(-1)
            return self.net(torch.cat([x, r], dim=-1)).squeeze(-1)

    model = TwoTower() if args.arch == "two_tower" else Concat()
    model.load_state_dict(bundle["model_state"])
    model.eval()
    norm = lambda f: [(v - m) / s for v, m, s in zip(f, mean, std)]

    records = {
        str(json.loads(line)["dp_id"]): json.loads(line)
        for line in args.screening_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    b0_rows, singles = {}, defaultdict(dict)
    for row in load_rows(args.singletons_root, "singletons.shard*.jsonl"):
        if row.get("kind") == "b0":
            b0_rows[row["dp_id"]] = row
        elif row.get("kind") == "singleton":
            singles[row["dp_id"]][int(row["event"])] = float(row["score"])
    true_sets: dict[str, dict[tuple, float]] = defaultdict(dict)
    anchors: dict[str, dict[int, float]] = defaultdict(dict)
    for row in load_rows(args.sets_root, "sets.shard*.jsonl"):
        if row.get("kind") == "set":
            true_sets[row["dp_id"]][tuple(row["events"])] = float(row["score"])
        elif row.get("kind") == "recent_anchor":
            anchors[row["dp_id"]][int(row["size"])] = float(row["score"])
    readouts: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for row in load_rows(args.readout_root, "readouts.shard*.jsonl"):
        if row.get("kind") == "readout":
            readouts[row["dp_id"]][int(row["event"])] = row["vector"]

    # true-score cache(跨 arm 共享)
    cache: dict[str, float] = {}
    if args.true_score_cache.exists():
        for line in args.true_score_cache.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                cache[row["key"]] = float(row["score"])
            except (json.JSONDecodeError, KeyError):
                continue
    cache_handle = args.true_score_cache.open("a", encoding="utf-8")

    from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
    from causalcache.policy.history_gated_lora import (
        inject_history_gated_kv,
        load_history_gated_state_dict,
    )
    from scripts.run_exploratory_closed_loop_episode import (
        EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from scripts.train_success_sft_lora import (
        adapter_scope_for_sample,
        encode_sample,
        mean_target_logprob,
    )

    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    policy = runtime.model
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.config.use_cache = False
    merge_size = int(runtime.processor.image_processor.merge_size)
    wrapped = inject_history_gated_kv(policy, layer_count=8, rank=8, alpha=16)
    load_history_gated_state_dict(
        wrapped, torch.load(args.checkpoint, map_location="cpu")
    )

    def true_score(record, forms, events) -> float:
        dp = str(record["dp_id"])
        key = f"{dp}|set:{','.join(map(str, sorted(events)))}"
        if key in cache:
            return cache[key]
        hit = true_sets.get(dp, {}).get(tuple(sorted(events)))
        if hit is not None:
            return hit
        if len(events) == 1 and events[0] in singles.get(dp, {}):
            return singles[dp][events[0]]
        relpaths = record["image_relpaths"]
        current_step = int(record["step"])
        messages = build_desktop_official_messages(
            goal=str(record["instruction"]),
            steps=forms,
            shown_events=sorted(events),
            event_images={e: relpaths[e] for e in events},
            current_image=relpaths[current_step - 1],
        )
        serialized = []
        for message in messages:
            content = []
            for part in message["content"]:
                if part.get("type") == "image":
                    content.append({"type": "image", "path": part["image"]})
                else:
                    content.append(dict(part))
            serialized.append({"role": message["role"], "content": content})
        sample = {
            "sample_id": key,
            "prompt_format": PROMPT_FORMAT,
            "messages": serialized,
            "target_text": render_official_target_text(record["target_tool_call"]),
            "memory_config": {"restored_event_step_ids": sorted(events)},
            "adapter_mode": "active",
        }
        encoded = encode_sample(
            runtime, sample, dataset_root=args.image_root, torch=torch
        )
        scope = adapter_scope_for_sample(
            "history_gated_kv", encoded, sample, merge_size=merge_size
        )
        with scope, torch.no_grad():
            value = float(mean_target_logprob(policy, encoded, torch=torch).detach())
        cache[key] = value
        cache_handle.write(json.dumps({"key": key, "score": value}) + "\n")
        cache_handle.flush()
        return value

    dev_states = [
        (dp, base) for dp, base in b0_rows.items()
        if dp in records
        and _split(str(records[dp]["task_id"]), seed=args.seed) == "dev"
    ]
    if args.limit_states:
        dev_states = dev_states[: args.limit_states]

    per_b = {b: defaultdict(list) for b in args.b_values}
    per_b_oracle = {b: defaultdict(list) for b in args.b_values}
    k_dist = {b: defaultdict(int) for b in args.b_values}
    skipped = defaultdict(int)
    scored_fresh = 0

    for dp, base in dev_states:
        record = records[dp]
        current_step = int(record["step"])
        pool = [int(e) for e in base["candidate_pool"]]
        have = singles.get(dp, {})
        if not pool or any(e not in have for e in pool):
            skipped["incomplete_singletons"] += 1
            continue
        forms = None
        dup = base.get("duplicates", {})
        # 读出池内 z 归一(逐字复刻训练端)
        ro = readouts.get(dp, {})
        normed_ro: dict[int, list[float]] = {}
        if ro and readout_dim:
            items = [(e, v) for e, v in ro.items()]
            cols = list(zip(*[v for _, v in items]))
            mus = [sum(c) / len(c) for c in cols]
            sds = [
                max(math.sqrt(sum((x - m) ** 2 for x in c) / len(c)), 1e-6)
                for c, m in zip(cols, mus)
            ]
            for e, v in items:
                normed_ro[e] = [(x - m) / s for x, m, s in zip(v, mus, sds)]

        import torch as _t

        def score_marginals(selected: tuple, candidates: list[int]) -> list[float]:
            xs, rs, rms = [], [], []
            for e in candidates:
                f = candidate_features(
                    record, candidate_pool=pool, duplicates=dup, event=e,
                ) + set_context_features(
                    record, selected=list(selected), event=e, duplicates=dup,
                )
                xs.append(norm(f))
                vec = normed_ro.get(e)
                rs.append(vec if vec is not None else [0.0] * readout_dim)
                rms.append(1.0 if vec is not None else 0.0)
            with _t.no_grad():
                return model(
                    _t.tensor(xs, dtype=_t.float32),
                    _t.tensor(rs, dtype=_t.float32) if readout_dim else None,
                    _t.tensor(rms, dtype=_t.float32) if readout_dim else None,
                ).tolist()

        for b in args.b_values:
            window = recent_window(current_step, b)
            if not window or window[0] < 1:
                skipped[f"b{b}_short_history"] += 1
                continue
            anchor_u = anchors.get(dp, {}).get(b)
            if anchor_u is None:
                skipped[f"b{b}_no_anchor"] += 1
                continue
            if len(pool) < b:
                skipped[f"b{b}_pool_too_small"] += 1
                continue
            # beam 组合(exact-B;recent 是普通候选,全拒绝时收敛 Recent-B)
            level = [((), 0.0)]
            for _size in range(b):
                expanded = []
                seen = set()
                for sel, acc in level:
                    remaining = [e for e in pool if e not in sel]
                    if not remaining:
                        continue
                    margs = score_marginals(sel, remaining)
                    for e, m in sorted(
                        zip(remaining, margs), key=lambda t: -t[1]
                    )[: args.beam]:
                        child = tuple(sorted((*sel, e)))
                        if child in seen:
                            continue
                        seen.add(child)
                        expanded.append((child, acc + m))
                if not expanded:
                    break
                expanded.sort(key=lambda t: -t[1])
                level = expanded[: args.beam]
            chosen = list(level[0][0])
            if len(chosen) < b:
                skipped[f"b{b}_beam_short"] += 1
                continue
            if forms is None:
                forms = build_official_forms_for_record(record)
            try:
                chosen_u = true_score(record, forms, chosen)
            except ValueError:
                skipped[f"b{b}_unrenderable"] += 1
                continue
            if f"{dp}|set:{','.join(map(str, sorted(chosen)))}" not in true_sets.get(dp, {}):
                scored_fresh += 1
            episode = str(record["task_id"])
            per_b[b][episode].append(chosen_u - anchor_u)
            candidates_u = [
                u for s, u in true_sets.get(dp, {}).items() if len(s) == b
            ] + [anchor_u, chosen_u]
            per_b_oracle[b][episode].append(max(candidates_u) - max(chosen_u, anchor_u))
            k = len([e for e in chosen if e not in window])
            k_dist[b][k] += 1

    result = {
        "schema_version": "causalcache.selector_v4_fill_to_b_eval.v1",
        "arch": args.arch,
        "scorer": str(args.scorer),
        "policy_checkpoint_sha256": digest,
        "beam": args.beam,
        "skipped": dict(sorted(skipped.items())),
        "fresh_true_scores": scored_fresh,
    }
    for b in args.b_values:
        result[f"b{b}"] = {
            "selector_minus_recent": cluster_ci(
                per_b[b], iterations=args.bootstrap, seed=args.seed + b),
            "oracle_gap": cluster_ci(
                per_b_oracle[b], iterations=args.bootstrap, seed=args.seed + 50 + b),
            "k_distribution": dict(sorted(k_dist[b].items())),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
