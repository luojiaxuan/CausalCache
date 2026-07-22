#!/usr/bin/env python3
"""Teacher-forced log p(successful action | variant prompt) for U_act analysis.

# note (luojiaxuan): 对重渲染数据集的每个样本算目标动作的平均 token 对数概率,
# 按 pair_group 汇总 correct/b0/shuffled/irrelevant 的差即为成功锚定的恢复增益
# 与 history-use 对照;冻结 policy 打分,不训练任何参数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from scripts.train_success_sft_lora import encode_sample
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    import torch

    runtime = GUIOwlV21OfficialToolsRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    )
    runtime.model.eval()
    samples = [
        json.loads(line)
        for line in (args.dataset_root / "samples.jsonl").open(encoding="utf-8")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            if index % args.shard_count != args.shard_index:
                continue
            encoded = encode_sample(
                runtime, sample, dataset_root=args.dataset_root, torch=torch
            )
            if encoded is None:
                continue
            labels = encoded.pop("labels")
            with torch.inference_mode():
                outputs = runtime.model(**encoded)
            logits = outputs.logits[:, :-1].float()
            targets = labels[:, 1:]
            mask = targets != -100
            log_probs = torch.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(
                2, targets.clamp(min=0).unsqueeze(-1)
            ).squeeze(-1)
            token_count = int(mask.sum())
            total = float((gathered * mask).sum())
            handle.write(
                json.dumps(
                    {
                        "episode": sample["episode"],
                        "step_index": sample["step_index"],
                        "pair_group": sample.get("pair_group"),
                        "variant": sample.get("variant", "correct"),
                        "memory_config": sample["memory_config"],
                        "target_token_count": token_count,
                        "target_logprob_sum": total,
                        "target_logprob_mean": total / max(token_count, 1),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            torch.cuda.empty_cache()
    print(json.dumps({"scored_shard": args.shard_index, "of": args.shard_count}))


if __name__ == "__main__":
    main()
