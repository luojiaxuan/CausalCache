from __future__ import annotations

import io

from PIL import Image

from scripts.build_selector_gate_plan_v1 import (
    deterministic_random_selection,
    similarity_ranking,
)


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 48), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_random_selection_is_stable_and_budget_exact() -> None:
    first = deterministic_random_selection(
        "episode:6", range(1, 9), budget=4, seed=20260724
    )
    second = deterministic_random_selection(
        "episode:6", range(1, 9), budget=4, seed=20260724
    )

    assert first == second
    assert len(first) == 4
    assert tuple(sorted(first)) == first


def test_similarity_ranking_combines_ocr_and_rgb() -> None:
    current = _png((255, 0, 0))
    ranking, scores = similarity_ranking(
        candidates=[1, 2],
        event_ocr_tokens={1: ["settings"], 2: ["other"]},
        current_ocr_tokens=["settings"],
        event_image_bytes={
            1: _png((255, 0, 0)),
            2: _png((0, 0, 255)),
        },
        current_image_bytes=current,
    )

    assert ranking == [1, 2]
    assert scores[1] > scores[2]
