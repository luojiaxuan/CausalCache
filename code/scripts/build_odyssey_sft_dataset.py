#!/usr/bin/env python3
"""Render GUI-Odyssey trajectories into mixed-fidelity margin-SFT samples.

# note (luojiaxuan): 事件 schema 与闭环 low_fidelity_v2 同构,prompt 直接走冻结
# mixed-fidelity builder,与 AndroidWorld 评测逐字节同构;目标动作从原始
# GUIOdyssey 注释按 episode_id+step JOIN 精确坐标(像素 → [0,999] 规范空间),
# 经 canonical GUIOwlV2Action 序列化为官方 teacher target。四变体逻辑与
# build_success_sft_dataset 一致(hash 混采、路径级重排、donor 跨轨迹)。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

from causalcache.exploratory_closed_loop_memory import (
    build_live_gui_owl_v2_1_mixed_fidelity_messages,
    candidate_event_step_ids_from_history,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.policy.gui_owl_v2_1 import serialize_gui_owl_v2_1_teacher_target
from causalcache.set_utility_androidworld import build_shared_early_step_messages
from causalcache.set_utility_live_controller import LiveRichEvent
from scripts.build_success_sft_dataset import (
    _sample_memory_config,
    _serialize_messages,
)

SYSTEM_BUTTONS = {"HOME": "Home", "BACK": "Back", "KEY_HOME": "Home", "KEY_BACK": "Back"}


def _scale(point: Any, resolution: tuple[int, int]) -> tuple[int, int]:
    # note (luojiaxuan): GUIOdyssey 官方注释坐标已归一化到 [0,1000),直接映射到
    # [0,999],绝不除设备分辨率(v1 曾二次归一化导致点击整体拽向左上,0/90 主因)。
    x, y = point
    return (
        max(0, min(999, round(float(x) * 999 / 1000))),
        max(0, min(999, round(float(y) * 999 / 1000))),
    )


def action_from_annotation(
    step: dict[str, Any], resolution: tuple[int, int]
) -> GUIOwlV2Action | None:
    kind = step["action"].upper()
    info = step.get("info")
    if kind == "CLICK":
        if isinstance(info, str):
            if info in ("KEY_HOME",):
                return GUIOwlV2Action(action="system_button", button="Home")
            if info in ("KEY_BACK",):
                return GUIOwlV2Action(action="system_button", button="Back")
            return None
        if isinstance(info, (list, tuple)) and info and isinstance(info[0], (list, tuple)):
            return GUIOwlV2Action(action="click", coordinate=_scale(info[0], resolution))
        return None
    if kind == "LONG_PRESS":
        if isinstance(info, (list, tuple)) and info and isinstance(info[0], (list, tuple)):
            return GUIOwlV2Action(
                action="long_press", coordinate=_scale(info[0], resolution)
            )
        return None
    if kind == "SCROLL":
        if (
            isinstance(info, (list, tuple))
            and len(info) == 2
            and all(isinstance(p, (list, tuple)) for p in info)
        ):
            return GUIOwlV2Action(
                action="swipe",
                coordinate=_scale(info[0], resolution),
                coordinate2=_scale(info[1], resolution),
            )
        return None
    if kind == "TEXT":
        text = info if isinstance(info, str) else step.get("text")
        if isinstance(text, str) and text:
            import unicodedata

            return GUIOwlV2Action(
                action="type", text=unicodedata.normalize("NFKC", text)
            )
        return None
    if kind in SYSTEM_BUTTONS:
        return GUIOwlV2Action(action="system_button", button=SYSTEM_BUTTONS[kind])
    if kind == "COMPLETE":
        return GUIOwlV2Action(action="terminate", status="success")
    return None


def load_heldout_episodes(path: Path) -> set[str]:
    # note (luojiaxuan): heldout.txt 与 scorer --episodes-filter 同格式(空白分隔
    # 轨迹 id);正典 165 条 = trainer heldout_episode_set(salt odyssey_margin_v1,
    # fraction 0.15)的哈希切分。渲染侧只读文件,不自算哈希,防两处规则漂移。
    return set(path.read_text(encoding="utf-8").split())


def split_allows(
    source_id: str, *, split: str, heldout_episodes: set[str] | None
) -> bool:
    if split == "all":
        return True
    if heldout_episodes is None:
        raise ValueError("--split train/heldout requires --heldout-episodes")
    held = source_id in heldout_episodes
    return held if split == "heldout" else not held


def load_state_inventory(path: Path) -> dict[str, list[int]]:
    inventory: dict[str, list[int]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if not row.get("fits") or row.get("role") != "train":
            continue
        source_id, kind, index = row["state_id"].split(":")
        if kind != "decision":
            continue
        inventory.setdefault(source_id, []).append(int(index))
    for decisions in inventory.values():
        decisions.sort()
    return inventory


def load_singleton_scores(path: Path) -> dict[str, dict[int, float]]:
    # note (luojiaxuan): 读 hg-s100 singleton 打分(score_success_action_recovery
    # 输出),按 pair_group 收成 {cid: target_logprob_mean}。只取单事件恢复行
    # (memory_config.restored_event_step_ids 恰含一个 cid;缺省回落
    # singleton_event_step_id),b0(空 restored)与多事件行天然跳过。
    scores: dict[str, dict[int, float]] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        pair_group = row.get("pair_group")
        if pair_group is None:
            continue
        restored = (row.get("memory_config") or {}).get("restored_event_step_ids")
        cid: int | None = None
        if isinstance(restored, list) and len(restored) == 1:
            cid = int(restored[0])
        elif row.get("singleton_event_step_id") is not None:
            cid = int(row["singleton_event_step_id"])
        if cid is None:
            continue
        logprob = row.get("target_logprob_mean")
        if logprob is None:
            continue
        scores.setdefault(pair_group, {})[cid] = float(logprob)
    return scores


def _stable_seed(text: str) -> int:
    # note (luojiaxuan): 稳定种子——用 sha256 而非内置 hash(字符串 hash 每进程
    # 随机化),保证同一 pair_group 跨进程/跨机器/跨重跑选出同一 random anchor;
    # 严禁 Math.random/Date 之类非确定源(渲染必须可复现)。
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def restored_set_key(restored_event_step_ids: Any) -> str:
    # note (luojiaxuan): 稳定集合键——scorer 幂等键与下游 join 都用它;
    # 与协议一致:"-".join(str(x) for x in sorted(ids))。空集(b0)得空串。
    return "-".join(str(x) for x in sorted(int(v) for v in restored_event_step_ids))


def conditional_marginal_plan(
    pair_group: str,
    candidate_scores: Mapping[int, float],
    *,
    shortlist_k: int = 6,
    num_anchors: int = 4,
    second_layer: bool = True,
) -> list[tuple[str, tuple[int, ...], tuple[int, ...], int | None]]:
    """Near-linear conditional-marginal render plan for one decision state.

    # note (luojiaxuan): 协议「Conditional marginal 标签生成(近线性)」的纯函数
    # 实现,返回每行 (variant, restored_ids_sorted, anchor_set_sorted, candidate)。
    # 下游据此打分 U(restored) 后用 restored_set_key join 出
    # Δ(j|S) = U(S∪{j}) − U(S)。此处不做任何图像/模型 I/O,便于单测。
    #
    # 排序基准:shortlist / oracle / diverse 全部用 singleton target_logprob_mean
    # 直接排序——同一决策态内 U_act(j) = logp_hg(singleton_j) − logp_frozen(b0),
    # b0 是同一常数,故按 singleton logprob 排名与按 U_act 排名 **逐态单调等价**,
    # shortlist 无需先减 b0(见协议 §V2 shortlist 说明)。
    """
    candidates = list(candidate_scores)
    if len(candidates) < 2:
        return []
    # note (luojiaxuan): 按 singleton logprob 降序;并列时用 event_step_id 升序
    # 作确定性 tie-break,保证 oracle/diverse/shortlist 全程可复现。
    ranked = sorted(candidates, key=lambda c: (-candidate_scores[c], c))
    shortlist = ranked[: max(1, shortlist_k)]
    id_sorted = sorted(candidates)

    anchors: list[int] = []

    def _add(anchor: int | None) -> None:
        if anchor is not None and anchor not in anchors:
            anchors.append(anchor)

    # note (luojiaxuan): 四类 anchor first-event(协议钦定,不做架构探索):
    #   (a) singleton-oracle top-1 = ranked[0](最高 singleton logprob);
    #   (b) Recent-1 = 候选中最大 event step id;
    #   (c) 一个确定性 random = id 升序表在 sha256(pair_group) 种子下取模位;
    #   (d) diverse = 中位排名 singleton 候选 = ranked[len(ranked)//2](0-based)。
    # 重合即 de-dup(不回填),故实际 distinct anchor 可能少于 num_anchors。
    if num_anchors >= 1:
        _add(ranked[0])
    if num_anchors >= 2:
        _add(max(candidates))
    if num_anchors >= 3:
        _add(id_sorted[_stable_seed(pair_group) % len(id_sorted)])
    if num_anchors >= 4:
        _add(ranked[len(ranked) // 2])

    rows: list[tuple[str, tuple[int, ...], tuple[int, ...], int | None]] = []
    # 1. anchor base rows S={i}(edge1 的减数;S=∅ 已由 b0/singleton 覆盖,不再发)
    for anchor in anchors:
        rows.append(("cond_base", (anchor,), (anchor,), None))
    # 2. first-layer edges Δ(j|{i}):anchor i × shortlist j(j≠i)
    edge1_pairs: dict[frozenset[int], tuple[int, int]] = {}
    for anchor in anchors:
        for cand in shortlist:
            if cand == anchor:
                continue
            restored = tuple(sorted((anchor, cand)))
            rows.append(("cond_edge1", restored, (anchor,), cand))
            edge1_pairs.setdefault(frozenset(restored), restored)
    # 3. second-layer paths:仅沿 beam-2(singleton logprob 之和最高的两条
    #    长度-2 前缀,取自已展开的 first-layer 集合)。为每个 {i,j} 发 base(减数)
    #    与 Δ(k|{i,j})(k∈shortlist\{i,j})。渲染期尚无 conditional 模型,故用
    #    singleton proxy 之和近似 greedy/beam 排序(协议钦定)。
    if second_layer and edge1_pairs:
        beam_pairs = sorted(
            edge1_pairs.values(),
            key=lambda pair: (
                -(candidate_scores[pair[0]] + candidate_scores[pair[1]]),
                pair,
            ),
        )[:2]
        for pair in beam_pairs:
            rows.append(("cond_base", pair, pair, None))
            for cand in shortlist:
                if cand in pair:
                    continue
                restored = tuple(sorted(pair + (cand,)))
                rows.append(("cond_edge2", restored, pair, cand))
    return rows


def render_trajectory(
    row: dict[str, Any],
    *,
    annotations_root: Path,
    decisions: list[int],
    output_root: Path,
    shared_early_decisions: int,
    contrast_variants: bool,
    donor_paths: list[str] | None,
    terminal_states_only: bool = False,
    selector_singletons: bool = False,
    singleton_max_candidates: int = 8,
    conditional_marginals: bool = False,
    conditional_scores: Mapping[str, Mapping[int, float]] | None = None,
    shortlist_k: int = 6,
    num_anchors: int = 4,
    second_layer: bool = True,
) -> list[dict[str, Any]]:
    source_id = row["source_id"]
    annotation_path = annotations_root / f"{source_id}.json"
    if not annotation_path.exists():
        return []
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    resolution = tuple(annotation["device_info"]["device_resolution"]) if isinstance(
        annotation.get("device_info"), dict
    ) and "device_resolution" in annotation["device_info"] else None
    if resolution is None:
        others = {}
        try:
            others = json.loads(row["raw_metadata"]).get("others", {})
        except Exception:  # noqa: BLE001
            pass
        resolution = tuple(others.get("resolution", ())) or None
    if not resolution or len(resolution) != 2:
        return []
    if str(annotation["steps"][-1].get("action", "")).upper() != "COMPLETE":
        return []
    steps_by_index = {int(s["step"]): s for s in annotation["steps"]}

    events_payload = json.loads(row["history_events_json"])
    images = [
        bytes(item["bytes"]) if isinstance(item, dict) else bytes(item)
        for item in row["images"]
    ]
    image_dir = output_root / "images" / source_id
    image_dir.mkdir(parents=True, exist_ok=True)
    for index, payload in enumerate(images):
        target = image_dir / f"observation-{index:03d}.png"
        if not target.exists():
            target.write_bytes(payload)
    ocr_records = json.loads(row["ocr_records_json"])

    from PIL import Image

    def open_image(index: int) -> Any:
        with Image.open(io.BytesIO(images[index])) as raw:
            return raw.convert("RGB")

    def ocr_tokens_for(step_id: int) -> list[str]:
        record = None
        if isinstance(ocr_records, list):
            record = ocr_records[step_id] if step_id < len(ocr_records) else None
        elif isinstance(ocr_records, dict):
            for key in (
                str(step_id),
                f"observation-{step_id:03d}",
                f"observation-{step_id:03d}.png",
                f"images/{source_id}/observation-{step_id:03d}.png",
            ):
                if key in ocr_records:
                    record = ocr_records[key]
                    break
        if isinstance(record, dict):
            return list(record.get("full_spatial_tokens", []))
        return []

    events: list[LiveRichEvent] = []
    for payload in events_payload:
        step_id = payload["event_step_id"]
        summary = dict(payload["low_fidelity_summary"])
        summary.setdefault("step_id", step_id)
        ocr_tokens = ocr_tokens_for(step_id)
        events.append(
            LiveRichEvent.build(
                event_step_id=step_id,
                low_fidelity_v2=summary,
                post_image_png=bytes(images[step_id]),
                post_ocr_tokens=tuple(ocr_tokens),
            )
        )

    # note (luojiaxuan): state 母表只含中段决策点,终止步(COMPLETE)从未入册,
    # v1 训练目标 0 个 terminate 导致闭环永不终止。终止态合成:decision =
    # 事件数+1,全历史 + 末观测,目标 terminate(success)。
    if terminal_states_only:
        last = annotation["steps"][-1]
        if str(last.get("action", "")).upper() != "COMPLETE":
            return []
        terminal_decision = len(events) + 1
        steps_by_index[terminal_decision - 1] = {
            "step": terminal_decision - 1,
            "action": "COMPLETE",
            "info": None,
        }
        decisions = [terminal_decision]
    instruction = row["task_instruction"]
    samples: list[dict[str, Any]] = []
    for decision in decisions:
        step = steps_by_index.get(decision - 1)
        if step is None:
            continue
        action = action_from_annotation(step, resolution)
        if action is None:
            continue
        try:
            target_text = serialize_gui_owl_v2_1_teacher_target(action)
        except (TypeError, ValueError):
            continue
        history = events[: decision - 1]
        current_rel = f"images/{source_id}/observation-{decision - 1:03d}.png"
        if len(history) < shared_early_decisions:
            mode, budget, selected = "shared_early", 0, ()
        else:
            candidate_ids = candidate_event_step_ids_from_history(
                [event.to_mapping() for event in history]
            )
            mode, budget, selected = _sample_memory_config(
                episode_name=source_id,
                step_index=decision,
                candidate_ids=tuple(candidate_ids),
            )
        own_path = {
            step_id: f"images/{source_id}/observation-{step_id:03d}.png"
            for step_id in selected
        }
        # note (luojiaxuan): variant 元组第 5 位 extra 为透传字段字典(默认 {}),
        # conditional 模式借它带 restored_set_key / conditional_anchor_set /
        # conditional_candidate;其余模式保持 {} 不改行为。
        variants: list[
            tuple[str, tuple[int, ...], dict[int, str], int | None, dict[str, Any]]
        ] = []
        if selector_singletons:
            # note (luojiaxuan): selector 标签模式(合同第 13 步)——b0 参考 +
            # 逐候选单图恢复(variant="singleton" + singleton_event_step_id),
            # 同 pair_group 自洽成组:U_act(e) = logp(gated, singleton_e) −
            # logp(frozen, b0)。门槛:候选 ≥2;候选取最近 singleton_max_candidates
            # 个封顶成本;shared-early 段不出标签。
            if len(history) < shared_early_decisions:
                continue
            all_candidates = tuple(
                candidate_event_step_ids_from_history(
                    [event.to_mapping() for event in history]
                )
            )
            if len(all_candidates) < 2:
                continue
            label_candidates = all_candidates[-singleton_max_candidates:]
            variants.append(("b0", (), {}, None, {}))
            for cand in label_candidates:
                variants.append(
                    (
                        "singleton",
                        (cand,),
                        {cand: f"images/{source_id}/observation-{cand:03d}.png"},
                        cand,
                        {},
                    )
                )
        elif conditional_marginals:
            # note (luojiaxuan): conditional-marginal 模式(协议 §V2 近线性计划)——
            # 用 hg-s100 singleton 分做 shortlist/anchor/beam,发 cond_base/
            # cond_edge1/cond_edge2 三族;同 pair_group 成组,靠
            # (variant, restored_set_key, conditional_candidate) 三元组区分行。
            # shared-early 段与无 singleton 分/候选<2 的态均跳过。
            if len(history) < shared_early_decisions:
                continue
            state_scores = (
                (conditional_scores or {}).get(f"{source_id}:{decision}") or {}
            )
            if len(state_scores) < 2:
                continue
            valid_candidates = set(
                candidate_event_step_ids_from_history(
                    [event.to_mapping() for event in history]
                )
            )
            candidate_scores = {
                cid: logprob
                for cid, logprob in state_scores.items()
                if cid in valid_candidates
            }
            if len(candidate_scores) < 2:
                continue
            plan = conditional_marginal_plan(
                f"{source_id}:{decision}",
                candidate_scores,
                shortlist_k=shortlist_k,
                num_anchors=num_anchors,
                second_layer=second_layer,
            )
            for variant_name, restored_ids, anchor_set, candidate in plan:
                variants.append(
                    (
                        variant_name,
                        restored_ids,
                        {
                            step_id: (
                                f"images/{source_id}/observation-{step_id:03d}.png"
                            )
                            for step_id in restored_ids
                        },
                        None,
                        {
                            "restored_set_key": restored_set_key(restored_ids),
                            "conditional_anchor_set": list(anchor_set),
                            "conditional_candidate": candidate,
                        },
                    )
                )
        elif len(history) < shared_early_decisions:
            variants.append(("correct", (), {}, None, {}))
        else:
            variants.append(("correct", selected, dict(own_path), None, {}))
            if contrast_variants and selected:
                variants.append(("b0", (), {}, None, {}))
                if len(selected) >= 2:
                    rotated = dict(
                        zip(
                            selected,
                            [own_path[s] for s in selected[1:]]
                            + [own_path[selected[0]]],
                        )
                    )
                    variants.append(("shuffled", selected, rotated, None, {}))
                if donor_paths is not None and len(donor_paths) >= len(selected):
                    variants.append(
                        (
                            "irrelevant",
                            selected,
                            dict(zip(selected, donor_paths[: len(selected)])),
                            None,
                            {},
                        )
                    )

        def open_rel(rel: str) -> Any:
            with Image.open(output_root / rel) as raw:
                return raw.convert("RGB")

        for (
            variant_name,
            variant_selected,
            variant_paths,
            singleton_id,
            variant_extra,
        ) in variants:
            if len(history) < shared_early_decisions:
                messages = build_shared_early_step_messages(
                    instruction=instruction,
                    history_events=tuple(history),
                    current_image=open_image(decision - 1),
                )
                image_rels = [current_rel]
            else:
                variant_images = {
                    step_id: (
                        open_image(step_id)
                        if variant_paths[step_id] == own_path.get(step_id)
                        else open_rel(variant_paths[step_id])
                    )
                    for step_id in variant_selected
                }
                messages = build_live_gui_owl_v2_1_mixed_fidelity_messages(
                    instruction=instruction,
                    history_events=[event.to_mapping() for event in history],
                    restored_event_step_ids=variant_selected,
                    selected_post_images_by_event_step=variant_images,
                    current_image=open_image(decision - 1),
                )
                image_rels = [
                    variant_paths[step_id] for step_id in variant_selected
                ] + [current_rel]
            sample = {
                "schema_version": "causalcache.odyssey_sft_sample.v1",
                "episode": source_id,
                "task_type": "guiodyssey",
                "task_index": 0,
                "sample_seed": 0,
                "decision_step_id": decision,
                "step_index": decision,
                "variant": variant_name,
                "pair_group": f"{source_id}:{decision}",
                "memory_config": {
                    "mode": (
                        "conditional_marginal"
                        if conditional_marginals
                        else "selector_singleton"
                        if selector_singletons
                        else mode
                    ),
                    "budget": (
                        len(variant_selected)
                        if (conditional_marginals or selector_singletons)
                        else budget
                    ),
                    "restored_event_step_ids": list(variant_selected),
                },
                "messages": _serialize_messages(messages, image_paths=image_rels),
                "target_text": target_text,
                "official_terminal_success": 1.0,
            }
            if selector_singletons:
                sample["singleton_event_step_id"] = singleton_id
            if variant_extra:
                sample.update(variant_extra)
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--state-context", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shared-early-decisions", type=int, default=2)
    parser.add_argument("--contrast-variants", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--trajectory-limit", type=int, default=0)
    parser.add_argument("--terminal-states-only", action="store_true")
    parser.add_argument("--selector-singletons", action="store_true")
    parser.add_argument("--singleton-max-candidates", type=int, default=8)
    parser.add_argument("--conditional-marginals", action="store_true")
    parser.add_argument("--singleton-scores", type=Path, default=None)
    parser.add_argument("--shortlist-k", type=int, default=6)
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--no-second-layer", action="store_true")
    parser.add_argument("--split", choices=("all", "train", "heldout"), default="all")
    parser.add_argument("--heldout-episodes", type=Path, default=None)
    args = parser.parse_args()

    from pyarrow import parquet as pq

    heldout_episodes = (
        load_heldout_episodes(args.heldout_episodes)
        if args.heldout_episodes is not None
        else None
    )
    if args.split != "all" and heldout_episodes is None:
        parser.error("--split train/heldout requires --heldout-episodes")
    conditional_scores: dict[str, dict[int, float]] | None = None
    if args.conditional_marginals:
        if args.singleton_scores is None:
            parser.error("--conditional-marginals requires --singleton-scores")
        conditional_scores = load_singleton_scores(args.singleton_scores)
    inventory = load_state_inventory(args.state_context)
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples_path = (
        args.output_root / f"samples-shard{args.shard_index:03d}.jsonl"
    )
    shards = sorted(args.source_root.glob("shard-*.parquet"))
    donor_paths: list[str] | None = None
    rendered = 0
    total_samples = 0
    with samples_path.open("w", encoding="utf-8") as handle:
        for shard_number, shard_path in enumerate(shards):
            if shard_number % args.shard_count != args.shard_index:
                continue
            for row in pq.read_table(shard_path).to_pylist():
                source_id = row["source_id"]
                if source_id not in inventory:
                    continue
                if not split_allows(
                    source_id, split=args.split, heldout_episodes=heldout_episodes
                ):
                    continue
                if args.trajectory_limit and rendered >= args.trajectory_limit:
                    break
                samples = render_trajectory(
                    row,
                    annotations_root=args.annotations_root,
                    decisions=inventory[source_id],
                    output_root=args.output_root,
                    shared_early_decisions=args.shared_early_decisions,
                    contrast_variants=args.contrast_variants,
                    donor_paths=donor_paths,
                    terminal_states_only=args.terminal_states_only,
                    selector_singletons=args.selector_singletons,
                    singleton_max_candidates=args.singleton_max_candidates,
                    conditional_marginals=args.conditional_marginals,
                    conditional_scores=conditional_scores,
                    shortlist_k=args.shortlist_k,
                    num_anchors=args.num_anchors,
                    second_layer=not args.no_second_layer,
                )
                for sample in samples:
                    handle.write(
                        json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                if samples:
                    rendered += 1
                    total_samples += len(samples)
                    donor_paths = [
                        f"images/{source_id}/observation-{k:03d}.png"
                        for k in range(min(8, len(row["images"])))
                    ]
                    print(
                        json.dumps({"trajectory": source_id, "samples": len(samples)}),
                        flush=True,
                    )
    print(json.dumps({"trajectories": rendered, "samples": total_samples}))


if __name__ == "__main__":
    main()
