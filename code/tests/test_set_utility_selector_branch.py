from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from causalcache.set_utility_selector_branch import (
    LoRALinear,
    ZeroInitResidualTokenAdapter,
    selector_first_layer,
)


def test_selector_first_layer() -> None:
    assert selector_first_layer(trainable_layer_count=4) == 32
    assert selector_first_layer(trainable_layer_count=8) == 28
    with pytest.raises(ValueError):
        selector_first_layer(trainable_layer_count=0)


def test_zero_initialized_token_adapter_is_identity() -> None:
    adapter = ZeroInitResidualTokenAdapter(16, 4)
    values = torch.randn(2, 3, 16)
    assert torch.equal(adapter(values), values)


def test_zero_initialized_lora_is_identity_and_base_is_frozen() -> None:
    base = torch.nn.Linear(16, 12, bias=False)
    adapter = LoRALinear(base, rank=4, alpha=8)
    values = torch.randn(2, 3, 16)
    assert torch.equal(adapter(values), base(values))
    assert not adapter.base.weight.requires_grad
    assert adapter.lora_a.weight.requires_grad
    assert adapter.lora_b.weight.requires_grad
