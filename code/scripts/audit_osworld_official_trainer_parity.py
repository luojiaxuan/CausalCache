#!/usr/bin/env python3
"""Token-by-token parity: trainer 的 osworld_official 编码 vs 在线桌面 runtime。

# note (luojiaxuan): 交接 §8.2 的验收件。对 Desktop DiD 语料抽样若干组,每臂做两条
# 独立管线并逐 token 对账:
#   trainer 侧:corpus 行(序列化 messages,图片是相对路径)经
#     train_success_sft_lora.encode_sample 的 osworld_chat_template 分支编码,
#     取 labels==-100 的 prompt 段;
#   在线侧:从 screening manifest 的原始记录重建同一决策点的 policy request
#     (build_agentnet_cr_request,按该臂的 selection_mode 还原 recent/extras),
#     render 后按 GUIOwlOSWorldRuntime.generate_raw 的编码调用逐实参复刻
#     (processor.apply_chat_template + tools=[_TOOL_SPEC])。
# 比较 input_ids 逐 token、image_grid_thw 逐元素、pixel_values 逐字节。任何一臂
# 不一致即非零退出 —— 这锁死"离线语料行 ≡ 在线部署 prompt"整条链,包括图片路径
# 序列化、renderer、chat template 与视觉预处理。
# 运行处:模型所在 GPU 主机(如 Hyper01);processor 依附完整 runtime 构造,
# 不另抄一份像素参数,避免与在线构造漂移。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from causalcache.agentnet_desktop_cr import (
    build_agentnet_cr_request,
    render_agentnet_cr_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True,
                        help="build_desktop_hgkv_corpus 的 output-root")
    parser.add_argument("--screening-manifest", type=Path, required=True,
                        help="AgentNet screening manifest(原始决策点记录)")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visual-tokens", type=int, default=2560)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--groups", type=int, default=4,
                        help="抽样多少个 pair-group(每组全部臂 + B0 都对账)")
    parser.add_argument("--report-output", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def online_selection(row: dict) -> tuple[int, tuple[int, ...]]:
    """还原该臂在线构造时的 (recent_budget, extra_restored_step_ids)。"""
    mode = row["selection_mode"]
    if mode == "recent":
        return int(row["budget"]), ()
    if mode in ("recurrence", "wrong"):
        return 0, tuple(row["selected_steps"])
    if mode == "none":
        return 0, ()
    raise SystemExit(f"unknown selection_mode {mode!r} on {row['sample_id']}")


def main() -> None:
    args = parse_args()
    import torch
    from causalcache.osworld_gui_owl import _TOOL_SPEC, GUIOwlOSWorldRuntime
    from scripts.train_success_sft_lora import encode_sample

    runtime = GUIOwlOSWorldRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        effective_visual_tokens_per_image=args.visual_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    records = {
        record["dp_id"]: record
        for record in load_jsonl(args.screening_manifest)
    }
    rows = load_jsonl(args.corpus_root / "samples.jsonl")
    b0_path = args.corpus_root / "parity_b0.jsonl"
    if b0_path.exists():
        rows += load_jsonl(b0_path)
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["pair_group"], []).append(row)
    selected_groups = sorted(by_group)[: args.groups]
    if not selected_groups:
        raise SystemExit("corpus has no pair-groups")

    results = []
    failures = 0
    for pair_group in selected_groups:
        for row in sorted(by_group[pair_group], key=lambda r: r["arm_slot"]):
            dp_id = row["source"]["dp_id"]
            record = records.get(dp_id)
            if record is None:
                raise SystemExit(f"screening manifest lacks dp_id {dp_id}")
            # trainer 侧
            encoded = encode_sample(
                runtime, row, dataset_root=args.corpus_root, torch=torch
            )
            if encoded is None:
                raise SystemExit(f"{row['sample_id']}: trainer encoding failed")
            target_length = int((encoded["labels"] != -100).sum())
            trainer_ids = encoded["input_ids"][0, :-target_length].tolist()
            # 在线侧 —— 与 GUIOwlOSWorldRuntime.generate_raw 逐实参相同
            recent_budget, extras = online_selection(row)
            screenshots = [
                (args.corpus_root / relpath).read_bytes()
                for relpath in record["image_relpaths"]
            ]
            request = build_agentnet_cr_request(
                instruction=record["instruction"],
                screenshots=screenshots,
                history_actions=record["history"],
                current_step=int(record["step"]),
                recent_budget=recent_budget,
                extra_restored_step_ids=extras,
                screen_size=tuple(record["screen_size"]),
            )
            online = runtime.processor.apply_chat_template(
                render_agentnet_cr_messages(request),
                tools=[_TOOL_SPEC],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            online_ids = online["input_ids"][0].tolist()
            token_parity = trainer_ids == online_ids
            grid_parity = (
                "image_grid_thw" in encoded
                and encoded["image_grid_thw"].cpu().tolist()
                == online["image_grid_thw"].cpu().tolist()
            )
            pixel_parity = (
                "pixel_values" in encoded
                and encoded["pixel_values"].shape == online["pixel_values"].shape
                and bool(
                    torch.equal(
                        encoded["pixel_values"].cpu(), online["pixel_values"].cpu()
                    )
                )
            )
            entry = {
                "sample_id": row["sample_id"],
                "arm_slot": row["arm_slot"],
                "prompt_tokens": len(trainer_ids),
                "token_parity": token_parity,
                "image_grid_parity": grid_parity,
                "pixel_parity": pixel_parity,
            }
            if not token_parity:
                first = next(
                    (
                        index
                        for index, (left, right) in enumerate(
                            zip(trainer_ids, online_ids)
                        )
                        if left != right
                    ),
                    min(len(trainer_ids), len(online_ids)),
                )
                entry["first_divergence"] = {
                    "index": first,
                    "trainer_length": len(trainer_ids),
                    "online_length": len(online_ids),
                }
            if not (token_parity and grid_parity and pixel_parity):
                failures += 1
            results.append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)

    report = {
        "schema_version": "causalcache.osworld_official_trainer_parity.v1",
        "model_dir": str(args.model_dir),
        "visual_tokens": args.visual_tokens,
        "groups_audited": len(selected_groups),
        "arms_audited": len(results),
        "failures": failures,
        "results": results,
    }
    if args.report_output is not None:
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({k: v for k, v in report.items() if k != "results"},
                     ensure_ascii=False))
    if failures:
        sys.exit(f"parity FAILED on {failures} arms")


if __name__ == "__main__":
    main()
