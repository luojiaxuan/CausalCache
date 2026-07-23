"""Fail-closed token-role mask construction for the history-gated KV adapter."""

from __future__ import annotations

from typing import Any

import torch


_TEXT_TOKEN_TYPE = 0
_IMAGE_TOKEN_TYPE = 1


def _image_token_blocks(mm_token_type_ids: Any) -> tuple[tuple[int, int], ...]:
    """Return (start, length) for each maximal run of image-typed tokens."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for position, value in enumerate(mm_token_type_ids[0].tolist()):
        if value == _IMAGE_TOKEN_TYPE:
            if start is None:
                start = position
            continue
        if value != _TEXT_TOKEN_TYPE:
            raise ValueError(
                f"mm_token_type_ids contains unsupported token type {value} at "
                f"position {position}; only text(0)/image(1) prompts are supported"
            )
        if start is not None:
            blocks.append((start, position - start))
            start = None
    if start is not None:
        blocks.append((start, int(mm_token_type_ids.shape[1]) - start))
    return tuple(blocks)


def build_history_token_mask(
    input_ids: Any,
    mm_token_type_ids: Any,
    image_grid_thw: Any,
    history_image_count: int,
    merge_size: int,
) -> torch.Tensor:
    """Mask exactly the restored-history image tokens of one encoded prompt.

    # note (luojiaxuan): token 角色只由 mm_token_type_ids 的图像连续块与
    # image_grid_thw 逐图 merged token 数 (t*h*w/merge_size^2) 交叉验证得出;
    # 图像顺序契约固定为恢复历史图 1..K 在前、当前观测图最后,K=0 合法且产出
    # 全 False mask(B0)。禁止依赖固定 token id 或经验 offset。任何几何不一致
    # ——块数 != K+1、grid 行数 != 块数、逐块长度不符、t*h*w 不可整除——都
    # fail-closed 抛 ValueError,宁可拒绝也不产出可疑 mask;相邻图像若无文本
    # 分隔会合并成一个块,同样被块数校验拒绝。
    """
    if type(history_image_count) is not int or history_image_count < 0:
        raise ValueError("history_image_count must be a non-negative integer")
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("merge_size must be a positive integer")
    if getattr(input_ids, "ndim", None) != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("input_ids must be a rank-two single-sequence tensor")
    if getattr(mm_token_type_ids, "ndim", None) != 2 or tuple(
        mm_token_type_ids.shape
    ) != tuple(input_ids.shape):
        raise ValueError("mm_token_type_ids shape differs from input_ids")
    if (
        mm_token_type_ids.dtype.is_floating_point
        or mm_token_type_ids.dtype is torch.bool
    ):
        raise ValueError("mm_token_type_ids must be an integer tensor")
    if (
        getattr(image_grid_thw, "ndim", None) != 2
        or int(image_grid_thw.shape[1]) != 3
    ):
        raise ValueError("image_grid_thw must be a rank-two [images, 3] tensor")
    blocks = _image_token_blocks(mm_token_type_ids)
    expected_blocks = history_image_count + 1
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"image token block count {len(blocks)} differs from "
            f"history_image_count+1={expected_blocks}"
        )
    if int(image_grid_thw.shape[0]) != len(blocks):
        raise ValueError(
            f"image_grid_thw row count {int(image_grid_thw.shape[0])} differs "
            f"from image token block count {len(blocks)}"
        )
    divisor = merge_size * merge_size
    for index, ((_, length), grid) in enumerate(
        zip(blocks, image_grid_thw.tolist(), strict=True)
    ):
        temporal, height, width = (int(value) for value in grid)
        if temporal <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                f"image_grid_thw row {index} must contain positive t, h, and w"
            )
        patches = temporal * height * width
        if patches % divisor != 0:
            raise ValueError(
                f"image_grid_thw row {index} t*h*w={patches} is not divisible "
                f"by merge_size^2={divisor}"
            )
        expected_length = patches // divisor
        if length != expected_length:
            raise ValueError(
                f"image token block {index} length {length} differs from "
                f"image_grid_thw row {index} merged token count {expected_length}"
            )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for start, length in blocks[:history_image_count]:
        mask[0, start : start + length] = True
    current_start = blocks[-1][0]
    if bool(mask[0, current_start:].any()):
        raise ValueError("history mask touches the current observation image block")
    return mask


def assert_mask_disjoint(mask: Any, target_start: int) -> None:
    """Fail if the history mask reaches the target segment at ``target_start``."""
    if getattr(mask, "ndim", None) != 2 or int(mask.shape[0]) != 1:
        raise ValueError("history mask must have shape [1, seq]")
    if mask.dtype is not torch.bool:
        raise ValueError("history mask must be a torch.bool tensor")
    if type(target_start) is not int or target_start < 0:
        raise ValueError("target_start must be a non-negative integer")
    if target_start > int(mask.shape[1]):
        raise ValueError("target_start lies beyond the mask sequence length")
    if bool(mask[0, target_start:].any()):
        raise ValueError("history mask touches the target segment")
