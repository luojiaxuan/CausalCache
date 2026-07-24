from __future__ import annotations

import json

import pytest

from scripts.validate_selector_inference_v1 import validate


def _rows(*, b1_drift: bool = False) -> list[dict]:
    rows = []
    for pair_group in ("episode-a:4", "episode-b:5"):
        for budget in (1, 2, 4):
            for method in ("hgkv_singleton", "hgkv_set_conditioned"):
                selected = [1] if budget == 1 else [1, 2][:budget]
                if (
                    b1_drift
                    and pair_group == "episode-b:5"
                    and budget == 1
                    and method == "hgkv_set_conditioned"
                ):
                    selected = [2]
                rows.append(
                    {
                        "budget": budget,
                        "candidate_event_step_ids": [1, 2, 3, 4],
                        "method": method,
                        "pair_group": pair_group,
                        "selected_event_step_ids": selected,
                    }
                )
    return rows


def _write(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validate_selector_inference_b1_parity(tmp_path) -> None:
    path = tmp_path / "selections.jsonl"
    _write(path, _rows())

    summary = validate(
        path,
        budgets=[1, 2, 4],
        methods=["hgkv_singleton", "hgkv_set_conditioned"],
    )

    assert summary["status"] == "PASS_SELECTOR_INFERENCE_V1"
    assert summary["row_count"] == 12
    assert summary["state_count"] == 2
    assert summary["b1_parity_states"] == 2


def test_validate_selector_inference_rejects_b1_drift(tmp_path) -> None:
    path = tmp_path / "selections.jsonl"
    _write(path, _rows(b1_drift=True))

    with pytest.raises(ValueError, match="B1 parity drift"):
        validate(
            path,
            budgets=[1, 2, 4],
            methods=["hgkv_singleton", "hgkv_set_conditioned"],
        )
