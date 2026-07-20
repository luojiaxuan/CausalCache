from pathlib import Path

import pytest

from scripts.finalize_set_utility_variable_history_token_cache import (
    _safetensors_header,
)


def test_safetensors_header_reads_geometry_without_tensor_load(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    path = tmp_path / "tokens.safetensors"
    safetensors.save_file(
        {"tokens": torch.zeros((3, 4), dtype=torch.bfloat16)}, str(path)
    )
    header = _safetensors_header(path)
    assert header["tokens"]["dtype"] == "BF16"
    assert header["tokens"]["shape"] == [3, 4]
