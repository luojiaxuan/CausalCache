from __future__ import annotations

import json

import pytest

from scripts.validate_selected_set_render_v1 import validate


def _plan_row(method: str, budget: int, selected: list[int]) -> dict:
    return {
        "budget": budget,
        "candidate_event_step_ids": [1, 2, 3, 4],
        "episode": "episode-a",
        "method": method,
        "pair_group": "episode-a:6",
        "selected_event_step_ids": selected,
    }


def _sample(tmp_path, *, selected: list[int]) -> dict:
    image = tmp_path / "images" / "episode-a" / "observation-005.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"png")
    return {
        "memory_config": {"restored_event_step_ids": selected},
        "messages": [
            {
                "content": [
                    {
                        "path": "images/episode-a/observation-005.png",
                        "type": "image",
                    }
                ]
            }
        ],
        "pair_group": "episode-a:6",
        "restored_set_key": "-".join(str(value) for value in selected),
        "selected_set_budgets": [1, 2],
        "selected_set_methods": ["recent", "similarity"],
        "variant": "selected_set",
    }


def test_validate_selected_set_render_exact_join(tmp_path) -> None:
    plan = tmp_path / "plan.jsonl"
    plan.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _plan_row("recent", 1, [2]),
                _plan_row("similarity", 2, [2]),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    sample = _sample(tmp_path, selected=[2])
    (tmp_path / "samples-shard000.jsonl").write_text(
        json.dumps(sample) + "\n", encoding="utf-8"
    )

    summary, hashes = validate(
        plan_path=plan,
        render_dir=tmp_path,
        samples_glob="samples-shard*.jsonl",
    )

    assert summary["expected_unique_sets"] == 1
    assert summary["observed_unique_sets"] == 1
    assert summary["missing_images"] == 0
    assert set(hashes) == {"samples-shard000.jsonl"}


def test_validate_selected_set_render_rejects_metadata_drift(tmp_path) -> None:
    plan = tmp_path / "plan.jsonl"
    plan.write_text(
        json.dumps(_plan_row("recent", 1, [2])) + "\n",
        encoding="utf-8",
    )
    sample = _sample(tmp_path, selected=[2])
    (tmp_path / "samples-shard000.jsonl").write_text(
        json.dumps(sample) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="budget mismatch"):
        validate(
            plan_path=plan,
            render_dir=tmp_path,
            samples_glob="samples-shard*.jsonl",
        )
