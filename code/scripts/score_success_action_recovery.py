#!/usr/bin/env python3
"""Teacher-forced log p(successful action | variant prompt) for U_act analysis.

# note (luojiaxuan): 对重渲染数据集的每个样本算目标动作的平均 token 对数概率,
# 按 pair_group 汇总 correct/b0/shuffled/irrelevant 的差即为成功锚定的恢复增益
# 与 history-use 对照;冻结 policy 打分,不训练任何参数。--adapter-type
# history_gated_kv 时按样本的 restored_event_step_ids 构造历史 token mask 并在
# history_adapter_scope 内前向(b0 样本 K=0 天然 ctx=None = 完全 bypass)。
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from scripts.run_exploratory_closed_loop_episode import (
    EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from scripts.train_success_sft_lora import (
    encode_sample,
    inject_lora,
    load_lora_state_dict,
)

# note (luojiaxuan): history_sample_context 只在 history-gated 分支的 trainer 里
# 定义;full-policy 打分路径不需要它,惰性导入避免 main 上硬 import 缺失函数报错。
try:
    from scripts.train_success_sft_lora import history_sample_context
except ImportError:
    history_sample_context = None
from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--adapter-type",
        choices=("full_policy_lora", "history_gated_kv"),
        default="full_policy_lora",
    )
    parser.add_argument("--adapter-layer-count", type=int, default=8)
    parser.add_argument("--episodes-filter", type=Path, default=None)
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
    history_mode = args.adapter_type == "history_gated_kv"
    if history_mode:
        from causalcache.policy.history_adapter_context import (
            history_adapter_scope,
        )

        merge_size = int(runtime.processor.image_processor.merge_size)
    if args.lora_checkpoint is not None:
        if history_mode:
            from causalcache.policy.history_gated_lora import (
                inject_history_gated_kv,
                load_history_gated_state_dict,
            )

            wrapped = inject_history_gated_kv(
                runtime.model,
                layer_count=args.adapter_layer_count,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
            )
            load_history_gated_state_dict(
                wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
            )
            print(
                json.dumps(
                    {
                        "adapter_type": args.adapter_type,
                        "lora_modules": len(wrapped),
                    }
                ),
                flush=True,
            )
        else:
            wrapped = inject_lora(
                runtime.model,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                torch=torch,
            )
            load_lora_state_dict(
                wrapped, torch.load(args.lora_checkpoint, map_location="cpu")
            )
            print(json.dumps({"lora_modules": len(wrapped)}), flush=True)
    allowed_episodes = None
    if args.episodes_filter is not None:
        allowed_episodes = set(
            args.episodes_filter.read_text(encoding="utf-8").split()
        )
    samples = [
        json.loads(line)
        for line in (args.dataset_root / "samples.jsonl").open(encoding="utf-8")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # note (luojiaxuan): skip-existing 断点续跑——已打分行用 (pair_group, variant)
    # 作幂等键;OOM/被杀后同一命令重启会跳过已完成项,只补未打分的,append 追加。
    done_keys: set[tuple[str, str]] = set()
    if args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            try:
                prev = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_keys.add((prev.get("pair_group"), prev.get("variant", "correct")))
    with args.output.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            if index % args.shard_count != args.shard_index:
                continue
            if allowed_episodes is not None and sample["episode"] not in allowed_episodes:
                continue
            if (
                sample.get("pair_group"),
                sample.get("variant", "correct"),
            ) in done_keys:
                continue
            encoded = encode_sample(
                runtime, sample, dataset_root=args.dataset_root, torch=torch
            )
            if encoded is None:
                continue
            if history_mode:
                scope = history_adapter_scope(
                    history_sample_context(
                        encoded, sample, merge_size=merge_size
                    )
                )
            else:
                scope = nullcontext()
            labels = encoded.pop("labels")
            targets_full = labels[:, 1:]
            token_count = int((targets_full != -100).sum())
            # note (luojiaxuan): 目标段固定在序列末尾;只取末端 logits,
            # 避免 25k 序列全长 float32 log_softmax 的 ~15GB 峰值。
            with torch.inference_mode(), scope:
                try:
                    outputs = runtime.model(
                        **encoded, logits_to_keep=token_count + 1
                    )
                except TypeError:
                    outputs = runtime.model(**encoded)
            logits = outputs.logits[:, -(token_count + 1) : -1].float()
            targets = targets_full[:, -token_count:]
            log_probs = torch.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            total = float(gathered.sum())
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
