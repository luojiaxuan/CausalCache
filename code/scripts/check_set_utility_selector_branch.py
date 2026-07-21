#!/usr/bin/env python3
"""Check that a zero-update selector branch reproduces frozen GUI-Owl."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from causalcache.policy.gui_owl_variable_history_runtime import (
    GUIOwlVariableHistoryRuntime,
    VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
)
from causalcache.set_utility_contextual_hidden import (
    build_contextual_entity_messages,
    select_contextual_hidden_tokens,
)
from causalcache.set_utility_selector_branch import (
    capture_selector_boundary_forward,
    replay_selector_top_layers,
)


def _first_requirement(input_root: Path) -> tuple[dict, int]:
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    receipts = sorted(
        manifest["requirement_shards"], key=lambda row: row["logical_shard"]
    )
    for receipt in receipts:
        path = input_root / receipt["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                return json.loads(line), int(receipt["logical_shard"])
    raise ValueError("contextual input contains no requirements")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trainable-layer-count", type=int, default=4)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from pyarrow import parquet as pq

    requirement, logical_shard = _first_requirement(args.input_root)
    source_path = (
        args.source_root
        / "trajectory-shards"
        / f"shard-{logical_shard:03d}-of-256.parquet"
    )
    source_rows = pq.read_table(
        source_path, columns=["images", "source_id"]
    ).to_pylist()
    matches = [
        row
        for row in source_rows
        if str(row["source_id"]) == requirement["trajectory_id"]
    ]
    if len(matches) != 1:
        raise ValueError("selector branch check could not bind the source trajectory")
    payload = matches[0]["images"][requirement["observation_step"]]["bytes"]
    with Image.open(io.BytesIO(payload)) as image:
        messages = build_contextual_entity_messages(
            prompt_text=requirement["prompt_text"], image=image.convert("RGB")
        )
    runtime = GUIOwlVariableHistoryRuntime(
        model_dir=args.model_dir,
        expected_snapshot_manifest=(
            args.repository_root / "code/configs/gui_owl_1_5_8b_snapshot.json"
        ),
        device=args.device,
        target_effective_visual_tokens_per_image=(
            VARIABLE_HISTORY_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
        ),
    )
    boundary = capture_selector_boundary_forward(
        runtime=runtime,
        messages=messages,
        trainable_layer_count=args.trainable_layer_count,
    )
    replayed = replay_selector_top_layers(
        policy_model=runtime.model,
        boundary=boundary,
        trainable_layer_count=args.trainable_layer_count,
    )
    image_token_id = int(runtime.model.config.image_token_id)
    if boundary.final_hidden_state is None:
        raise RuntimeError("parity check requires the frozen final hidden state")
    frozen_visual, frozen_text = select_contextual_hidden_tokens(
        input_ids=boundary.input_ids,
        final_hidden_state=boundary.final_hidden_state,
        image_token_id=image_token_id,
    )
    replay_visual, replay_text = select_contextual_hidden_tokens(
        input_ids=boundary.input_ids,
        final_hidden_state=replayed,
        image_token_id=image_token_id,
    )
    maximum_difference = max(
        float((frozen_visual - replay_visual).abs().max()),
        float((frozen_text - replay_text).abs().max()),
    )
    result = {
        "boundary_shape": list(boundary.boundary_hidden_state.shape),
        "context_key": requirement["context_key"],
        "exact_selected_token_equality": bool(
            torch.equal(frozen_visual, replay_visual)
            and torch.equal(frozen_text, replay_text)
        ),
        "maximum_selected_token_absolute_difference": maximum_difference,
        "sequence_length": int(boundary.input_ids.shape[1]),
        "status": "PASS" if maximum_difference == 0.0 else "FAIL",
        "trainable_layer_count": args.trainable_layer_count,
    }
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
