#!/usr/bin/env python3
"""Run greedy and learned beam-4 inference for the unified V2 selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from causalcache.hgkv_selector_v2 import (
    HGKVSetSelectorV2,
    learned_beam_search,
)
from scripts.build_hgkv_selector_v2_training_data import (
    expand_paths,
    sha256_file,
)
from scripts.train_hgkv_selector_v2 import (
    load_feature_tensor,
    make_model,
)


def load_states(paths: list[Path]) -> list[dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair_group = str(row["pair_group"])
            if pair_group in states:
                raise ValueError(f"{path}:{line_no} duplicate state")
            states[pair_group] = row
    return [states[key] for key in sorted(states)]


def make_scorer(
    *,
    model: HGKVSetSelectorV2,
    feature_by_event: Mapping[int, torch.Tensor],
    device: torch.device,
):
    @torch.no_grad()
    def scorer(
        prefix: tuple[int, ...],
        remaining: tuple[int, ...],
        remaining_budget: int,
    ) -> Mapping[int, float]:
        candidate = torch.stack(
            [feature_by_event[event] for event in remaining]
        ).unsqueeze(0)
        if prefix:
            selected = torch.stack(
                [feature_by_event[event] for event in prefix]
            ).unsqueeze(0)
        else:
            selected = candidate.new_zeros((1, 0, candidate.shape[-1]))
        outputs = model(
            candidate.to(device),
            torch.ones(
                (1, len(remaining)), dtype=torch.bool, device=device
            ),
            selected.to(device),
            torch.ones(
                (1, len(prefix)), dtype=torch.bool, device=device
            ),
            torch.tensor(
                [remaining_budget], dtype=torch.long, device=device
            ),
        )
        marginals = outputs["marginal"][0].detach().cpu().tolist()
        return {
            event: float(marginal)
            for event, marginal in zip(remaining, marginals, strict=True)
        }

    return scorer


def select_state(
    *,
    state: Mapping[str, Any],
    model: HGKVSetSelectorV2,
    feature_by_key: Mapping[tuple[str, int], torch.Tensor],
    budgets: tuple[int, ...],
    device: torch.device,
) -> list[dict[str, Any]]:
    pair_group = str(state["pair_group"])
    candidates = tuple(
        int(value) for value in state["candidate_event_step_ids"]
    )
    feature_by_event = {
        candidate: feature_by_key[(pair_group, candidate)]
        for candidate in candidates
    }
    scorer = make_scorer(
        model=model,
        feature_by_event=feature_by_event,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    for method, beam_width in (
        ("hgkv_v2_greedy", 1),
        ("hgkv_v2_beam4", 4),
    ):
        for budget in budgets:
            path = learned_beam_search(
                candidates,
                budget=budget,
                scorer=scorer,
                beam_width=beam_width,
            )
            rows.append(
                {
                    "schema_version": (
                        "causalcache.hgkv_selector_v2.selection.v1"
                    ),
                    "episode": str(state["episode"]),
                    "pair_group": pair_group,
                    "split": str(state["split"]),
                    "method": method,
                    "budget": budget,
                    "beam_width": beam_width,
                    "candidate_event_step_ids": list(candidates),
                    "selected_event_step_ids": list(
                        path.selected_event_step_ids
                    ),
                    "cumulative_predicted_u": (
                        path.cumulative_predicted_u
                    ),
                    "steps": list(path.trace),
                    "stopped_early": (
                        len(path.selected_event_step_ids) < budget
                    ),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--states", action="append", required=True)
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--budgets", default="1,2,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    budgets = tuple(int(value) for value in args.budgets.split(","))
    if budgets != (1, 2, 4):
        parser.error("the frozen formal V2 budgets are exactly 1,2,4")
    state_paths = expand_paths(args.states)
    feature_paths = expand_paths(args.features)
    if not state_paths or not feature_paths:
        parser.error("state and feature patterns must match files")
    features, feature_keys, feature_index_sha256 = load_feature_tensor(
        feature_paths
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["feature_index_sha256"] != feature_index_sha256:
        raise ValueError("checkpoint and inference feature indices differ")
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_by_key = {
        key: features[index] for index, key in enumerate(feature_keys)
    }
    states = load_states(state_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for state in states:
            rows = select_state(
                state=state,
                model=model,
                feature_by_key=feature_by_key,
                budgets=budgets,
                device=device,
            )
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
    manifest = {
        "schema_version": (
            "causalcache.hgkv_selector_v2.selection_manifest.v1"
        ),
        "source_commit": args.source_commit,
        "states": len(states),
        "methods": ["hgkv_v2_greedy", "hgkv_v2_beam4"],
        "budgets": list(budgets),
        "rows": count,
        "feature_index_sha256": feature_index_sha256,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "state_sha256": {
            str(path): sha256_file(path) for path in state_paths
        },
        "feature_sha256": {
            str(path): sha256_file(path) for path in feature_paths
        },
        "output_sha256": sha256_file(args.output),
        "status": "DONE",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
