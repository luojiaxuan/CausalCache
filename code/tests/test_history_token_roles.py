"""Pure-logic tests for the history token-role mask and its ContextVar carrier."""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from causalcache.policy.history_adapter_context import (
    HistoryAdapterContext,
    get_history_adapter_context,
    history_adapter_scope,
    set_history_adapter_context,
)
from causalcache.policy.history_token_roles import (
    assert_mask_disjoint,
    build_history_token_mask,
)


MERGE_SIZE = 2


def _prompt(block_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text(0)/image(1) type ids with one text separator between images."""
    types: list[int] = [0, 0]
    for length in block_lengths:
        types.extend([1] * length)
        types.append(0)
    types.append(0)
    mm_token_type_ids = torch.tensor([types], dtype=torch.int64)
    input_ids = torch.arange(
        mm_token_type_ids.shape[1], dtype=torch.int64
    ).unsqueeze(0)
    return input_ids, mm_token_type_ids


def _grid(merged_lengths: list[int]) -> torch.Tensor:
    """Grid rows whose t*h*w/merge_size^2 equals each merged token length."""
    return torch.tensor(
        [[1, MERGE_SIZE, length * MERGE_SIZE] for length in merged_lengths],
        dtype=torch.int64,
    )


def test_mask_covers_exactly_the_restored_history_blocks() -> None:
    input_ids, mm = _prompt([3, 2, 4])
    mask = build_history_token_mask(input_ids, mm, _grid([3, 2, 4]), 2, MERGE_SIZE)
    expected = torch.zeros_like(input_ids, dtype=torch.bool)
    expected[0, 2:5] = True
    expected[0, 6:8] = True
    assert mask.dtype is torch.bool
    assert tuple(mask.shape) == tuple(input_ids.shape)
    assert torch.equal(mask, expected)


def test_zero_history_images_yield_an_all_false_mask() -> None:
    input_ids, mm = _prompt([4])
    mask = build_history_token_mask(input_ids, mm, _grid([4]), 0, MERGE_SIZE)
    assert mask.dtype is torch.bool
    assert not bool(mask.any())


def test_block_count_mismatch_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    with pytest.raises(ValueError, match="block count 2 differs"):
        build_history_token_mask(input_ids, mm, _grid([3, 2]), 2, MERGE_SIZE)


def test_adjacent_images_without_separator_fail_closed() -> None:
    mm = torch.tensor([[0, 1, 1, 1, 1, 1, 0]], dtype=torch.int64)
    input_ids = torch.arange(mm.shape[1], dtype=torch.int64).unsqueeze(0)
    with pytest.raises(ValueError, match="block count 1 differs"):
        build_history_token_mask(input_ids, mm, _grid([3, 2]), 1, MERGE_SIZE)


def test_grid_row_count_mismatch_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    with pytest.raises(ValueError, match="image_grid_thw row count 3 differs"):
        build_history_token_mask(input_ids, mm, _grid([3, 2, 4]), 1, MERGE_SIZE)


def test_block_length_grid_mismatch_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    with pytest.raises(ValueError, match="block 1 length 2 differs"):
        build_history_token_mask(input_ids, mm, _grid([3, 3]), 1, MERGE_SIZE)


def test_non_divisible_grid_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    grids = torch.tensor([[1, 2, 6], [1, 3, 3]], dtype=torch.int64)
    with pytest.raises(ValueError, match="not divisible"):
        build_history_token_mask(input_ids, mm, grids, 1, MERGE_SIZE)


def test_non_positive_grid_row_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    grids = torch.tensor([[1, 2, 6], [1, 0, 4]], dtype=torch.int64)
    with pytest.raises(ValueError, match="positive t, h, and w"):
        build_history_token_mask(input_ids, mm, grids, 1, MERGE_SIZE)


def test_unsupported_token_type_value_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    mm = mm.clone()
    mm[0, 0] = 2
    with pytest.raises(ValueError, match="unsupported token type 2"):
        build_history_token_mask(input_ids, mm, _grid([3, 2]), 1, MERGE_SIZE)


def test_shape_mismatch_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    with pytest.raises(ValueError, match="shape differs from input_ids"):
        build_history_token_mask(input_ids, mm[:, :-1], _grid([3, 2]), 1, MERGE_SIZE)


def test_batched_input_fails_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    batched_ids = torch.cat([input_ids, input_ids], dim=0)
    batched_mm = torch.cat([mm, mm], dim=0)
    with pytest.raises(ValueError, match="single-sequence"):
        build_history_token_mask(batched_ids, batched_mm, _grid([3, 2]), 1, MERGE_SIZE)


def test_invalid_scalar_arguments_fail_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    grids = _grid([3, 2])
    with pytest.raises(ValueError, match="history_image_count"):
        build_history_token_mask(input_ids, mm, grids, -1, MERGE_SIZE)
    with pytest.raises(ValueError, match="history_image_count"):
        build_history_token_mask(input_ids, mm, grids, True, MERGE_SIZE)
    with pytest.raises(ValueError, match="merge_size"):
        build_history_token_mask(input_ids, mm, grids, 1, 0)


def test_float_token_type_ids_fail_closed() -> None:
    input_ids, mm = _prompt([3, 2])
    with pytest.raises(ValueError, match="integer tensor"):
        build_history_token_mask(
            input_ids, mm.to(torch.float32), _grid([3, 2]), 1, MERGE_SIZE
        )


def test_assert_mask_disjoint_accepts_a_prompt_only_mask() -> None:
    input_ids, mm = _prompt([3, 2, 4])
    mask = build_history_token_mask(input_ids, mm, _grid([3, 2, 4]), 2, MERGE_SIZE)
    assert_mask_disjoint(mask, int(mask.shape[1]))


def test_assert_mask_disjoint_rejects_a_mask_reaching_the_target() -> None:
    input_ids, mm = _prompt([3, 2, 4])
    mask = build_history_token_mask(input_ids, mm, _grid([3, 2, 4]), 2, MERGE_SIZE)
    with pytest.raises(ValueError, match="touches the target segment"):
        assert_mask_disjoint(mask, 7)
    with pytest.raises(ValueError, match="non-negative"):
        assert_mask_disjoint(mask, -1)
    with pytest.raises(ValueError, match="beyond the mask sequence length"):
        assert_mask_disjoint(mask, int(mask.shape[1]) + 1)


def _context(mask: torch.Tensor) -> HistoryAdapterContext:
    return HistoryAdapterContext(mask, bool(mask.any()), ("history", "current"))


def test_context_defaults_to_none_and_scope_restores() -> None:
    mask = torch.zeros((1, 4), dtype=torch.bool)
    mask[0, 1] = True
    context = _context(mask)
    assert get_history_adapter_context() is None
    with history_adapter_scope(context):
        assert get_history_adapter_context() is context
        with history_adapter_scope(None):
            assert get_history_adapter_context() is None
        assert get_history_adapter_context() is context
    assert get_history_adapter_context() is None


def test_context_does_not_leak_across_threads() -> None:
    mask = torch.ones((1, 2), dtype=torch.bool)
    seen: list[HistoryAdapterContext | None] = []
    with history_adapter_scope(_context(mask)):
        thread = threading.Thread(
            target=lambda: seen.append(get_history_adapter_context())
        )
        thread.start()
        thread.join()
    assert seen == [None]


def test_context_validation_fails_closed() -> None:
    mask = torch.zeros((1, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="history_present differs"):
        HistoryAdapterContext(mask, True, ("current",))
    with pytest.raises(ValueError, match="shape"):
        HistoryAdapterContext(torch.zeros(4, dtype=torch.bool), False, ("current",))
    with pytest.raises(ValueError, match="torch.bool"):
        HistoryAdapterContext(torch.zeros((1, 4)), False, ("current",))
    with pytest.raises(ValueError, match="role strings"):
        HistoryAdapterContext(mask, False, ())
    with pytest.raises(TypeError, match="HistoryAdapterContext or None"):
        set_history_adapter_context("not-a-context")
