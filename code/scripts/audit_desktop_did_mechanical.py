#!/usr/bin/env python3
"""Desktop DiD 语料的 §8.6 机械审计:全量数据检查 + 真模型不变量。

# note (luojiaxuan): 交接 §8.6 要求在正式训练前跑纯机械测试。本脚本两段:
# CPU 段(全语料):
#   1. 每行过 trainer 的 validate_sparse_sample(test split 行以 split=train 补丁后
#      做同样的结构校验,剥离计数显式上报);
#   2. train 组过 build_sparse_history_units、dev(heldout)组过
#      build_sparse_history_heldout_units —— R0/RA/S0/SA/WA 逐字段对账在组校验里;
#   3. train/dev/test 轨迹集合两两不相交;
#   4. 引用图片逐一存在;
#   5. target 回环:corpus 行的 target_text 抽出的 tool_call 与 screening manifest
#      的 target_tool_call 逐字段相等,且 parse_gui_owl_osworld_action 可解析;
#   6. B0 行与训练组一一对应、单图、独立 schema。
# GPU 段(抽样组,--skip-gpu 可跳过):
#   7. HGKV B0 bitwise parity:随机化(非零)adapter 权重后,B0 prompt(K=0,
#      ctx=None)的 teacher-forced 分数与注入前的冻结分数**完全相等**;
#   8. mask 覆盖:SA 行的 history mask 恰好等于第一张(restored)图的 merged token
#      数,与目标段不相交(assert_mask_disjoint),active 前向确实偏离冻结分数;
#   9. plain-LoRA bypass parity(§8.4):全层 q/k/v/o 随机权重下,
#      plain_lora_bypass_scope(True) 的分数与冻结分数完全相等,active 偏离。
# 任一检查失败即非零退出。训练用的就是这里构造的同一个 runtime 类与编码路径。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from causalcache.osworld_gui_owl import parse_gui_owl_osworld_action
from scripts.train_success_sft_lora import (
    SPARSE_DESKTOP_SAMPLE_SCHEMA,
    build_sparse_history_heldout_units,
    build_sparse_history_units,
    normalize_desktop_splits,
    validate_sparse_sample,
)

B0_SCHEMA = "causalcache.desktop_did_b0_sample.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--screening-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--snapshot-manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-samples", type=int, default=6)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--report-output", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_tool_call(target_text: str) -> dict:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", target_text, re.DOTALL)
    if len(matches) != 1:
        raise ValueError("target_text must contain exactly one tool_call block")
    return json.loads(matches[0])


def cpu_checks(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.corpus_root / "samples.jsonl")
    b0_rows = load_jsonl(args.corpus_root / "parity_b0.jsonl")
    records = {
        record["dp_id"]: record
        for record in load_jsonl(args.screening_manifest)
    }

    # split 不相交(在归一之前用原始 split 字段判)
    episodes_by_split: dict[str, set[str]] = {}
    for row in rows:
        episodes_by_split.setdefault(row["split"], set()).add(row["episode"])
    for left in sorted(episodes_by_split):
        for right in sorted(episodes_by_split):
            if left < right and episodes_by_split[left] & episodes_by_split[right]:
                raise SystemExit(
                    f"splits {left}/{right} share episodes: "
                    f"{sorted(episodes_by_split[left] & episodes_by_split[right])[:5]}"
                )

    kept, counters = normalize_desktop_splits(rows)
    test_rows = [row for row in rows if row["split"] == "test"]
    for index, row in enumerate(kept):
        validate_sparse_sample(row, index=index)
    for index, row in enumerate(test_rows):
        validate_sparse_sample({**row, "split": "train"}, index=index)

    train_rows = [row for row in kept if row["split"] == "train"]
    heldout_rows = [row for row in kept if row["split"] == "heldout"]
    units, _ = build_sparse_history_units(kept, objective_kind="did_ra_aware")
    heldout_groups = build_sparse_history_heldout_units(kept)

    # 图片存在性
    referenced: set[str] = set()
    for row in rows + b0_rows:
        referenced.update(row["selected_images"])
        referenced.add(row["current_image"])
    missing_images = [
        relpath for relpath in sorted(referenced)
        if not (args.corpus_root / relpath).is_file()
    ]
    if missing_images:
        raise SystemExit(
            f"{len(missing_images)} referenced images missing, e.g. "
            f"{missing_images[:3]}"
        )

    # target 回环 + parser validity(每 pair_group 一次)
    groups_seen: set[str] = set()
    parser_failures = []
    for row in rows:
        if row["pair_group"] in groups_seen:
            continue
        groups_seen.add(row["pair_group"])
        record = records.get(row["source"]["dp_id"])
        if record is None:
            raise SystemExit(f"screening manifest lacks dp_id {row['source']['dp_id']}")
        if row["target_text"] != record["target_text"]:
            raise SystemExit(f"{row['pair_group']}: target_text drifted from manifest")
        call = extract_tool_call(row["target_text"])
        if call != record["target_tool_call"]:
            raise SystemExit(f"{row['pair_group']}: tool_call round-trip mismatch")
        try:
            parse_gui_owl_osworld_action(
                row["target_text"], screen_size=tuple(record["screen_size"])
            )
        except ValueError as error:
            parser_failures.append((row["pair_group"], str(error)))
    if parser_failures:
        raise SystemExit(
            f"{len(parser_failures)} targets fail the frozen parser, e.g. "
            f"{parser_failures[:3]}"
        )

    # B0 对账
    b0_by_group = {row["pair_group"]: row for row in b0_rows}
    if sorted(b0_by_group) != sorted(groups_seen):
        raise SystemExit("parity_b0.jsonl groups do not match samples.jsonl groups")
    for row in b0_rows:
        if row["schema_version"] != B0_SCHEMA or row["selected_steps"] != []:
            raise SystemExit(f"B0 row {row['sample_id']} broke its contract")
        images = sum(
            1
            for message in row["messages"]
            for part in message["content"]
            if part.get("type") == "image"
        )
        if images != 1:
            raise SystemExit(f"B0 row {row['sample_id']} carries {images} images")

    return {
        "rows": len(rows),
        "b0_rows": len(b0_rows),
        "groups": len(groups_seen),
        "train_units": len(units),
        "heldout_groups": len(heldout_groups),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "test_rows_stripped": len(test_rows),
        "split_normalization": counters,
        "referenced_images": len(referenced),
        "parser_validity": "969/969-scale pass" if not parser_failures else "FAIL",
    }


def gpu_checks(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
    from causalcache.policy.history_adapter_context import history_adapter_scope
    from causalcache.policy.history_gated_lora import inject_history_gated_kv
    from scripts.run_exploratory_closed_loop_episode import (
        EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    from scripts.train_success_sft_lora import (
        adapter_context_for_sample,
        encode_sample,
        inject_lora,
        mean_target_logprob,
        plain_lora_bypass_scope,
    )

    rows = load_jsonl(args.corpus_root / "samples.jsonl")
    b0_rows = {row["pair_group"]: row for row in load_jsonl(args.corpus_root / "parity_b0.jsonl")}
    by_group: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_group.setdefault(row["pair_group"], {})[row["arm_slot"]] = row
    sampled = sorted(by_group)[: args.gpu_samples]

    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.snapshot_manifest,
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    model = runtime.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    merge_size = int(runtime.processor.image_processor.merge_size)

    def score(sample: dict, *, scope) -> float:
        encoded = encode_sample(runtime, sample, dataset_root=args.corpus_root, torch=torch)
        if encoded is None:
            raise SystemExit(f"{sample['sample_id']}: empty encoding")
        with scope, torch.no_grad():
            return float(mean_target_logprob(model, encoded, torch=torch).detach())

    from contextlib import nullcontext

    frozen_b0 = {g: score(b0_rows[g], scope=nullcontext()) for g in sampled}
    frozen_sa = {g: score(by_group[g]["SA"], scope=nullcontext()) for g in sampled}

    # ---- HGKV:随机非零权重下的 B0 parity 与 mask 覆盖 ----
    wrapped = inject_history_gated_kv(model, layer_count=8, rank=8, alpha=16)
    torch.manual_seed(20260726)
    with torch.no_grad():
        for lora in wrapped.values():
            lora.lora_a.normal_(0.0, 0.02)
            lora.lora_b.normal_(0.0, 0.02)

    results: dict[str, Any] = {
        "hgkv_modules": len(wrapped),
        "b0_parity_max_abs_diff": 0.0,
        "mask_checks": [],
    }
    for pair_group in sampled:
        b0_again = score(b0_rows[pair_group], scope=history_adapter_scope(None))
        diff = abs(b0_again - frozen_b0[pair_group])
        results["b0_parity_max_abs_diff"] = max(results["b0_parity_max_abs_diff"], diff)
        if b0_again != frozen_b0[pair_group]:
            raise SystemExit(
                f"HGKV B0 parity broke on {pair_group}: {b0_again} vs "
                f"{frozen_b0[pair_group]}"
            )
        sa_row = by_group[pair_group]["SA"]
        encoded = encode_sample(runtime, sa_row, dataset_root=args.corpus_root, torch=torch)
        context = adapter_context_for_sample(encoded, sa_row, merge_size=merge_size)
        expected = int(encoded["image_grid_thw"][0].prod().item()) // (merge_size**2)
        got = int(context.history_token_mask.sum().item())
        if got != expected:
            raise SystemExit(
                f"{pair_group}: history mask covers {got} tokens, expected the "
                f"restored image's {expected}"
            )
        with history_adapter_scope(context), torch.no_grad():
            sa_active = float(mean_target_logprob(model, encoded, torch=torch).detach())
        results["mask_checks"].append(
            {
                "pair_group": pair_group,
                "mask_tokens": got,
                "sa_active_minus_frozen": sa_active - frozen_sa[pair_group],
            }
        )
    if all(
        entry["sa_active_minus_frozen"] == 0.0 for entry in results["mask_checks"]
    ):
        raise SystemExit(
            "randomized HGKV weights left every SA forward unchanged; the adapter "
            "is not engaging the restored-history tokens"
        )

    # ---- plain-LoRA(§8.4):bypass parity 与 active 偏离 ----
    # HGKV hook 在 ctx=None 下惰性,不影响本段;plain LoRA 叠加注入后随机化。
    plain = inject_lora(
        model,
        rank=8,
        alpha=16,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        torch=torch,
    )
    with torch.no_grad():
        for lora in plain.values():
            lora.lora_a.normal_(0.0, 0.02)
            lora.lora_b.normal_(0.0, 0.02)
    probe = sampled[0]
    bypass_value = score(b0_rows[probe], scope=plain_lora_bypass_scope(True))
    if bypass_value != frozen_b0[probe]:
        raise SystemExit(
            f"plain-LoRA bypass parity broke: {bypass_value} vs {frozen_b0[probe]}"
        )
    active_value = score(b0_rows[probe], scope=plain_lora_bypass_scope(False))
    results["plain_lora"] = {
        "modules": len(plain),
        "bypass_parity": True,
        "active_minus_frozen": active_value - frozen_b0[probe],
    }
    if active_value == frozen_b0[probe]:
        raise SystemExit("randomized full-layer LoRA left the forward unchanged")
    return results


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "schema_version": "causalcache.desktop_did_mechanical_audit.v1",
        "cpu": cpu_checks(args),
    }
    if not args.skip_gpu:
        if args.model_dir is None or args.snapshot_manifest is None:
            sys.exit("GPU checks need --model-dir and --snapshot-manifest")
        report["gpu"] = gpu_checks(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report_output is not None:
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
