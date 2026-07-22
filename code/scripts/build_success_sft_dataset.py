#!/usr/bin/env python3
"""Re-render collected success trajectories into mixed-fidelity SFT samples.

# note (luojiaxuan): 事件摘要在采集时未落盘,这里用冻结的 pinned OCR 对已保存
# 的逐步 PNG 离线重建 LiveRichEvent,与 runtime 管线逐字节同源(同一 PNG 编码、
# 同一 OCR 配置、同一 builder),保证 SFT prompt 与闭环 prompt 精确同构。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from causalcache.exploratory_closed_loop_memory import (
    build_live_gui_owl_v2_1_mixed_fidelity_messages,
    candidate_event_step_ids_from_history,
)
from causalcache.independent_closed_loop_features import (
    FROZEN_GATE_V1_OCR_BACKEND,
    live_history_event_from_transition,
)
from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.set_utility_androidworld import (
    PinnedOnlineOCRProvider,
    build_shared_early_step_messages,
)
from causalcache.set_utility_live_controller import LiveRichEvent

_WORKER_PROVIDER = None
_WORKER_ARGS: dict[str, Any] = {}


def _pool_init(repository_root: str, ocr_model_dir: str) -> None:
    global _WORKER_PROVIDER
    from pathlib import Path as _Path

    _WORKER_PROVIDER = PinnedOnlineOCRProvider.load(
        backend_config_path=_Path(repository_root)
        / "code/configs/restoration_v2_ocr_backend.json",
        backend_manifest_path=_Path(repository_root)
        / "data/manifests/restoration_v2_ocr_backend.json",
        model_dir=_Path(ocr_model_dir),
    )


def _pool_render(job: tuple[str, str, list[str] | None, int, bool]) -> list[dict[str, Any]]:
    episode_path, run_root, donors, shared_early, contrast = job
    samples, _ = render_episode(
        episode_path=Path(episode_path),
        run_root=Path(run_root),
        ocr_provider=_WORKER_PROVIDER,
        shared_early_decisions=shared_early,
        contrast_variants=contrast,
        donor_images=donors,
    )
    return samples


MEMORY_BUDGETS = (0, 2, 4, 8)
MEMORY_MODES = ("recent", "random")


def _action_from_arguments(arguments: dict[str, Any]) -> GUIOwlV2Action:
    values = {key: value for key, value in arguments.items() if value is not None}
    return GUIOwlV2Action(
        action=values["action"],
        coordinate=tuple(values["coordinate"]) if "coordinate" in values else None,
        coordinate2=tuple(values["coordinate2"]) if "coordinate2" in values else None,
        text=values.get("text"),
        button=values.get("button"),
        status=values.get("status"),
    )


def _sample_memory_config(
    *, episode_name: str, step_index: int, candidate_ids: tuple[int, ...]
) -> tuple[str, int, tuple[int, ...]]:
    digest = hashlib.sha256(
        f"success_sft_v1:{episode_name}:{step_index}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    budget = rng.choice(MEMORY_BUDGETS)
    mode = rng.choice(MEMORY_MODES)
    if budget == 0 or not candidate_ids:
        return mode, budget, ()
    take = min(budget, len(candidate_ids))
    if mode == "recent":
        selected = candidate_ids[-take:]
    else:
        selected = tuple(sorted(rng.sample(candidate_ids, take)))
    return mode, budget, selected


def _serialize_messages(
    messages: Any, *, image_paths: list[str]
) -> list[dict[str, Any]]:
    slots = iter(image_paths)
    serialized: list[dict[str, Any]] = []
    for message in messages:
        content = []
        for part in message["content"]:
            if part.get("type") == "image":
                content.append({"type": "image", "path": next(slots)})
            else:
                content.append(dict(part))
        serialized.append({"role": message["role"], "content": content})
    remaining = sum(1 for _ in slots)
    if remaining:
        raise RuntimeError("image slot count drifted during serialization")
    return serialized


def render_episode(
    *,
    episode_path: Path,
    run_root: Path,
    ocr_provider: PinnedOnlineOCRProvider,
    shared_early_decisions: int,
    contrast_variants: bool = False,
    donor_images: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    record = json.loads(episode_path.read_text(encoding="utf-8"))
    episode_name = episode_path.stem
    image_dir = run_root / "images" / episode_name
    steps = record["steps"]

    observations: dict[int, Any] = {}

    def observe(step_index: int) -> Any:
        if step_index not in observations:
            path = image_dir / f"step{step_index:03d}.png"
            with Image.open(path) as image:
                loaded = image.convert("RGB")
            observations[step_index] = ocr_provider.observe(
                image=loaded,
                image_member_path=(
                    f"sft/{episode_name}/step{step_index:03d}.png"
                ),
                metadata={"width": loaded.width, "height": loaded.height},
            )
        return observations[step_index]

    instruction = record["instance"]["goal"]
    samples: list[dict[str, Any]] = []
    history: list[LiveRichEvent] = []
    for step in steps:
        step_index = step["step_index"]
        if "native_output" not in step or "canonical_action" not in step:
            break
        current = observe(step_index)
        image_rel = f"images/{episode_name}/step{step_index:03d}.png"
        decision_step_id = len(history) + 1
        if len(history) < shared_early_decisions:
            mode, budget, selected = "shared_early", 0, ()
        else:
            candidate_ids = candidate_event_step_ids_from_history(
                [event.to_mapping() for event in history]
            )
            mode, budget, selected = _sample_memory_config(
                episode_name=episode_name,
                step_index=step_index,
                candidate_ids=tuple(candidate_ids),
            )
        # note (luojiaxuan): variant 家族共享同一 target;corrupted 变体只供
        # margin 对照与 history-use 评估,trainer 的 CE 必须过滤到 correct/b0。
        # 数据集按路径序列化,因此每个变体必须携带自己的 (事件 id -> 图路径)
        # 映射,shuffled/irrelevant 的路径重排才能在重载时保真。
        own_path = {
            step_id: f"images/{episode_name}/step{step_id:03d}.png"
            for step_id in selected
        }
        variants: list[tuple[str, tuple[int, ...], dict[int, str]]] = []
        if len(history) < shared_early_decisions:
            variants.append(("correct", (), {}))
        else:
            variants.append(("correct", selected, dict(own_path)))
            if contrast_variants and selected:
                variants.append(("b0", (), {}))
                if len(selected) >= 2:
                    rotated_paths = dict(
                        zip(
                            selected,
                            [own_path[s] for s in selected[1:]]
                            + [own_path[selected[0]]],
                        )
                    )
                    variants.append(("shuffled", selected, rotated_paths))
                if donor_images is not None and len(donor_images) >= len(selected):
                    variants.append(
                        (
                            "irrelevant",
                            selected,
                            dict(zip(selected, donor_images[: len(selected)])),
                        )
                    )
        def open_rel(rel: str) -> Any:
            with Image.open(run_root / rel) as raw:
                return raw.convert("RGB")

        for variant_name, variant_selected, variant_paths in variants:
            if len(history) < shared_early_decisions:
                messages = build_shared_early_step_messages(
                    instruction=instruction,
                    history_events=tuple(history),
                    current_image=current.image,
                )
                image_rels = [image_rel]
            else:
                variant_images = {
                    step_id: (
                        observe(step_id).image
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
                    current_image=current.image,
                )
                image_rels = [
                    variant_paths[step_id] for step_id in variant_selected
                ] + [image_rel]
            samples.append(
                {
                    "schema_version": "causalcache.success_sft_sample.v1",
                    "episode": episode_name,
                    "task_type": record["task_type"],
                    "task_index": record["task_index"],
                    "sample_seed": record["sample_seed"],
                    "decision_step_id": decision_step_id,
                    "step_index": step_index,
                    "variant": variant_name,
                    "pair_group": f"{episode_name}:{step_index}",
                    "memory_config": {
                        "mode": mode,
                        "budget": budget,
                        "restored_event_step_ids": list(variant_selected),
                    },
                    "messages": _serialize_messages(
                        messages, image_paths=image_rels
                    ),
                    "target_text": step["native_output"],
                    "official_terminal_success": record[
                        "official_terminal_success"
                    ],
                }
            )
        if "appended_event" not in step:
            continue
        after_path = image_dir / f"step{step_index + 1:03d}.png"
        if not after_path.exists():
            # note (luojiaxuan): 只有非成功局(step 预算耗尽)会缺末步 after 图,
            # 成功局以 terminate 收尾不会走到这里。
            break
        after = observe(step_index + 1)
        action = _action_from_arguments(step["canonical_action"])
        projected = live_history_event_from_transition(
            step_id=len(history) + 1,
            action=action,
            before_image_bytes=current.image_png,
            after_image_bytes=after.image_png,
            backend_config=ocr_provider.backend_config,
            backend_config_sha256=ocr_provider.backend_config_sha256,
            before_ocr_record=current.ocr_record,
            after_ocr_record=after.ocr_record,
            ocr_backend_binding=FROZEN_GATE_V1_OCR_BACKEND,
        )
        event = LiveRichEvent.build(
            event_step_id=len(history) + 1,
            low_fidelity_v2=projected["low_fidelity_v2"],
            post_image_png=after.image_png,
            post_ocr_tokens=projected["post_ocr_spatial_tokens"],
        )
        expected_sha = step["appended_event"]["post_image_sha256"]
        if event.post_image.sha256 != expected_sha:
            raise ValueError(
                f"{episode_name} step {step_index}: rebuilt post image SHA drifted"
            )
        history.append(event)
    next_donors = [
        f"images/{episode_name}/step{index:03d}.png"
        for index in sorted(observations)[:8]
    ]
    return samples, next_donors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--ocr-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shared-early-decisions", type=int, default=2)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--contrast-variants", action="store_true")
    parser.add_argument("--episode-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    ocr_provider = PinnedOnlineOCRProvider.load(
        backend_config_path=args.repository_root
        / "code/configs/restoration_v2_ocr_backend.json",
        backend_manifest_path=args.repository_root
        / "data/manifests/restoration_v2_ocr_backend.json",
        model_dir=args.ocr_model_dir,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_root / "samples.jsonl"
    episode_summaries: list[dict[str, Any]] = []
    rendered = 0
    donor_chain: list[Any] | None = None
    if args.workers > 1:
        # note (luojiaxuan): donor 链只依赖上一成功局的图路径,可从文件系统
        # 预计算,episode 级完全并行;输出按提交序写回保持确定性。
        from concurrent.futures import ProcessPoolExecutor

        jobs = []
        job_records = []
        for run_root in args.run_root:
            eligible = []
            for episode_path in sorted((run_root / "episodes").glob("*.json")):
                record = json.loads(episode_path.read_text(encoding="utf-8"))
                if (
                    record["official_terminal_success"] != 1.0
                    and not args.include_failures
                ):
                    continue
                eligible.append((episode_path, record))
            donors = None
            for episode_path, record in eligible:
                jobs.append(
                    (
                        str(episode_path),
                        str(run_root),
                        donors,
                        args.shared_early_decisions,
                        args.contrast_variants,
                    )
                )
                job_records.append((episode_path, run_root, record))
                image_dir = run_root / "images" / episode_path.stem
                donors = [
                    f"images/{episode_path.stem}/{name}"
                    for name in sorted(p.name for p in image_dir.glob("step*.png"))[:8]
                ]
        with samples_path.open("w", encoding="utf-8") as handle:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_pool_init,
                initargs=(
                    str(args.repository_root),
                    str(args.ocr_model_dir),
                ),
            ) as pool:
                for (episode_path, run_root, record), samples in zip(
                    job_records, pool.map(_pool_render, jobs)
                ):
                    for sample in samples:
                        handle.write(
                            json.dumps(sample, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                    episode_summaries.append(
                        {
                            "episode": episode_path.stem,
                            "run_root": str(run_root),
                            "samples": len(samples),
                            "official_terminal_success": record[
                                "official_terminal_success"
                            ],
                        }
                    )
                    rendered += 1
                    print(
                        json.dumps(
                            {"episode": episode_path.stem, "samples": len(samples)}
                        ),
                        flush=True,
                    )
    else:
      with samples_path.open("w", encoding="utf-8") as handle:
        for run_root in args.run_root:
            donor_chain = None
            for episode_path in sorted((run_root / "episodes").glob("*.json")):
                record = json.loads(episode_path.read_text(encoding="utf-8"))
                if (
                    record["official_terminal_success"] != 1.0
                    and not args.include_failures
                ):
                    continue
                if args.episode_limit and rendered >= args.episode_limit:
                    break
                samples, donor_chain = render_episode(
                    episode_path=episode_path,
                    run_root=run_root,
                    ocr_provider=ocr_provider,
                    shared_early_decisions=args.shared_early_decisions,
                    contrast_variants=args.contrast_variants,
                    donor_images=donor_chain,
                )
                for sample in samples:
                    handle.write(
                        json.dumps(sample, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                episode_summaries.append(
                    {
                        "episode": episode_path.stem,
                        "run_root": str(run_root),
                        "samples": len(samples),
                        "official_terminal_success": record[
                            "official_terminal_success"
                        ],
                    }
                )
                rendered += 1
                print(
                    json.dumps(
                        {"episode": episode_path.stem, "samples": len(samples)}
                    ),
                    flush=True,
                )
    manifest = {
        "schema_version": "causalcache.success_sft_dataset.v1",
        "episodes": episode_summaries,
        "episode_count": rendered,
        "sample_count": sum(entry["samples"] for entry in episode_summaries),
        "shared_early_decisions": args.shared_early_decisions,
        "include_failures": args.include_failures,
        "contrast_variants": args.contrast_variants,
        "memory_budgets": list(MEMORY_BUDGETS),
        "memory_modes": list(MEMORY_MODES),
        "ocr_backend_config_sha256": ocr_provider.backend_config_sha256,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"episodes": rendered, "samples": manifest["sample_count"]}))


if __name__ == "__main__":
    main()
