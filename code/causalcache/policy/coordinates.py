"""Coordinate conversion shared by Qwen2.5-VL-derived GUI policies."""

from __future__ import annotations

import re
from typing import Any


def parse_point(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    coordinates = re.findall(r"-?\d+", str(value))
    if len(coordinates) < 2:
        raise ValueError(f"cannot parse model point: {value}")
    return int(coordinates[0]), int(coordinates[1])


def current_resized_shape(model_inputs: Any) -> tuple[int, int]:
    grids = model_inputs["image_grid_thw"]
    grid = grids[-1]
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    resized_height = int(grid[1]) * 14
    resized_width = int(grid[2]) * 14
    return resized_width, resized_height


def normalized_resized_point(value: Any, model_inputs: Any) -> list[int]:
    x, y = parse_point(value)
    width, height = current_resized_shape(model_inputs)
    normalized_x = min(1000, max(0, round(x / width * 1000)))
    normalized_y = min(1000, max(0, round(y / height * 1000)))
    return [normalized_x, normalized_y]
