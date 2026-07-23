"""Pure-logic tests for closed-loop history-gated generation scope wiring."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from causalcache.policy.gui_owl_v2_1_runtime import GUIOwlV21OfficialToolsRuntime
from causalcache.policy.history_adapter_context import (
    HistoryAdapterContext,
    get_history_adapter_context,
)


MERGE_SIZE = 2


class _StubRuntime:
    """Bare attribute carrier; _history_generation_scope reads nothing else."""

    history_adapter_merge_size: int | None = None


def _scope(stub: _StubRuntime, model_inputs: dict, image_counts: tuple) -> object:
    return GUIOwlV21OfficialToolsRuntime._history_generation_scope(
        stub, model_inputs, image_counts
    )


def _model_inputs(block_lengths: list[int]) -> dict:
    """Processor-shaped inputs with one text separator between image blocks."""
    types: list[int] = [0, 0]
    for length in block_lengths:
        types.extend([1] * length)
        types.append(0)
    types.append(0)
    mm_token_type_ids = torch.tensor([types], dtype=torch.int64)
    input_ids = torch.arange(
        mm_token_type_ids.shape[1], dtype=torch.int64
    ).unsqueeze(0)
    image_grid_thw = torch.tensor(
        [[1, MERGE_SIZE, length * MERGE_SIZE] for length in block_lengths],
        dtype=torch.int64,
    )
    return {
        "input_ids": input_ids,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
    }


def test_disabled_adapter_never_installs_a_context() -> None:
    stub = _StubRuntime()
    with _scope(stub, _model_inputs([3, 2, 4]), (3,)):
        assert get_history_adapter_context() is None


def test_zero_history_images_stay_out_of_any_scope() -> None:
    stub = _StubRuntime()
    stub.history_adapter_merge_size = MERGE_SIZE
    # note (luojiaxuan): K=0 必须在触碰 mm_token_type_ids 之前就返回
    # nullcontext——B0 prompt 不允许依赖任何 mask 机制,ContextVar 保持 None
    # 才是 bitwise parity 不变量;故意不提供 mm_token_type_ids 验证这一点。
    inputs = _model_inputs([4])
    inputs.pop("mm_token_type_ids")
    inputs.pop("image_grid_thw")
    with _scope(stub, inputs, (1,)):
        assert get_history_adapter_context() is None


def test_history_prompt_installs_and_restores_the_masked_context() -> None:
    stub = _StubRuntime()
    stub.history_adapter_merge_size = MERGE_SIZE
    inputs = _model_inputs([3, 2, 4])
    assert get_history_adapter_context() is None
    with _scope(stub, inputs, (3,)):
        context = get_history_adapter_context()
        assert isinstance(context, HistoryAdapterContext)
        assert context.history_present is True
        assert context.image_roles == ("history", "history", "current")
        expected = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        expected[0, 2:5] = True
        expected[0, 6:8] = True
        assert torch.equal(context.history_token_mask, expected)
    assert get_history_adapter_context() is None


def test_missing_token_type_ids_fail_closed_for_history_prompts() -> None:
    stub = _StubRuntime()
    stub.history_adapter_merge_size = MERGE_SIZE
    inputs = _model_inputs([3, 2])
    inputs.pop("mm_token_type_ids")
    with pytest.raises(RuntimeError, match="mm_token_type_ids"):
        _scope(stub, inputs, (2,))


def test_image_count_geometry_mismatch_fails_closed() -> None:
    stub = _StubRuntime()
    stub.history_adapter_merge_size = MERGE_SIZE
    with pytest.raises(ValueError, match="block count 2 differs"):
        _scope(stub, _model_inputs([3, 2]), (3,))


def test_multi_prompt_batches_are_rejected() -> None:
    stub = _StubRuntime()
    stub.history_adapter_merge_size = MERGE_SIZE
    with pytest.raises(RuntimeError, match="single-prompt batch"):
        _scope(stub, _model_inputs([3, 2]), (2, 2))


def test_teacher_forced_path_refuses_an_enabled_adapter() -> None:
    stub = _StubRuntime()
    stub.history_adapter_merge_size = MERGE_SIZE
    with pytest.raises(RuntimeError, match="lacks history-gated mask wiring"):
        GUIOwlV21OfficialToolsRuntime.teacher_forced_distance_logits(
            stub, (), ()
        )
