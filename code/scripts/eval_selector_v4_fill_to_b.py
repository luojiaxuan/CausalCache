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
import functools as _functools
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

# note (luojiaxuan): 单遍部署消融 —— 置 1 时把目标依赖的 witness 家族特征清零
# (cheap: witness_match/count/rank/coord_distance;set: n_witness_in_set/
# witness_redundancy),量化 propose-then-select 第二遍买到的排序增益。
import os as _os

ZERO_WITNESS = _os.environ.get("CAUSALCACHE_ZERO_WITNESS_FEATURES") == "1"
_WITNESS_CHEAP_IDX = (10, 11, 12, 18)
_WITNESS_SET_IDX = (24, 25)


def _maybe_zero_witness(features: list[float]) -> list[float]:
    if not ZERO_WITNESS:
        return features
    out = list(features)
    for i in _WITNESS_CHEAP_IDX + _WITNESS_SET_IDX:
        out[i] = 0.0
    return out


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
    parser.add_argument("--intent-root", type=Path, default=None,
                        help="intent 特征目录;bundle 带 intent_dims 时必给")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--b-values", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-states", type=int, default=0)
    # note (luojiaxuan): GPU 并发分片。dev 态按 index 取模切分,每片独立 true-score
    # cache 与输出;shard_count>1 时输出附带 raw 明细,由 merge 脚本汇总后再算
    # episode-cluster CI(单片内的 CI 只是参考值,不作判定)。
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    # note (luojiaxuan): 选集器消融——beam(默认,学习打分器)、random(随机 B 集,
    # per-(dp,b) 固定种子)、bottomk(打分取负,反向选择特异性控制)、
    # rgb_sim(候选帧与当前帧 32×32 RGB 距离最小者,承诺过的相似度基线)。
    parser.add_argument("--chooser",
                        choices=("beam", "random", "bottomk", "rgb_sim"),
                        default="beam")
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


@_functools.lru_cache(maxsize=8192)
def _rgb_thumb(path: str):
    """32×32 RGB 缩略图(float32),供 rgb_sim 基线;坏图返回 None。"""
    try:
        import numpy as _np
        from PIL import Image as _Image

        with _Image.open(path) as im:
            return _np.asarray(
                im.convert("RGB").resize((32, 32)), dtype=_np.float32)
    except Exception:  # noqa: BLE001
        return None


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
    intent_dims = int(bundle.get("intent_dims") or 0)
    intent_map: dict[tuple[str, int], list[float]] = {}
    if intent_dims:
        if args.intent_root is None:
            raise SystemExit("scorer 带 intent_dims,必须提供 --intent-root")
        for row in load_rows(args.intent_root, "intent.shard*.jsonl"):
            if row.get("kind") == "intent":
                intent_map[(row["dp_id"], int(row["event"]))] = row["vector"]
    dim = len(FEATURE_NAMES) + len(SET_FEATURE_NAMES) + intent_dims
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
    if args.shard_count > 1:
        dev_states = dev_states[args.shard_index::args.shard_count]

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
                f = _maybe_zero_witness(candidate_features(
                    record, candidate_pool=pool, duplicates=dup, event=e,
                ) + set_context_features(
                    record, selected=list(selected), event=e, duplicates=dup,
                ))
                if intent_dims:
                    f = f + (intent_map.get((dp, e)) or [0.0] * intent_dims)
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
                # note (luojiaxuan): B∈{1,8} 的 recent 锚不在离线集合表中(表只有
                # Recent-2/4)——现场按同一 true_score 口径补打,进共享缓存。
                if forms is None:
                    forms = build_official_forms_for_record(record)
                try:
                    anchor_u = true_score(record, forms, list(window))
                except ValueError:
                    skipped[f"b{b}_anchor_unrenderable"] += 1
                    continue
            if len(pool) < b:
                skipped[f"b{b}_pool_too_small"] += 1
                continue
            if args.chooser == "random":
                rng_c = random.Random(f"{args.seed}:{dp}:{b}")
                chosen = sorted(rng_c.sample(list(pool), b))
            elif args.chooser == "rgb_sim":
                rels = record.get("image_relpaths") or []
                cur = _rgb_thumb(str(args.image_root / rels[-1]))
                sims = []
                for e in pool:
                    if e - 1 >= len(rels):
                        continue
                    cand = _rgb_thumb(str(args.image_root / rels[e - 1]))
                    if cur is None or cand is None:
                        continue
                    sims.append((e, -float(abs(cur - cand).mean())))
                if len(sims) < b:
                    skipped[f"b{b}_rgb_short"] += 1
                    continue
                sims.sort(key=lambda t: -t[1])
                chosen = sorted(e for e, _ in sims[:b])
            else:
                # beam 组合(exact-B;recent 是普通候选,全拒绝时收敛 Recent-B);
                # bottomk 取负分走同一 beam(反向选择特异性控制)。
                level = [((), 0.0)]
                for _size in range(b):
                    expanded = []
                    seen = set()
                    for sel, acc in level:
                        remaining = [e for e in pool if e not in sel]
                        if not remaining:
                            continue
                        margs = score_marginals(sel, remaining)
                        if args.chooser == "bottomk":
                            margs = [-m for m in margs]
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
    if args.shard_count > 1:
        result["shard_index"] = args.shard_index
        result["shard_count"] = args.shard_count
        result["raw"] = {
            "per_b": {str(b): {ep: vals for ep, vals in per_b[b].items()}
                      for b in args.b_values},
            "per_b_oracle": {str(b): {ep: vals for ep, vals in per_b_oracle[b].items()}
                             for b in args.b_values},
            "k_dist": {str(b): dict(k_dist[b]) for b in args.b_values},
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
