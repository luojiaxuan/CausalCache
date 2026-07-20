import argparse

import pytest

from scripts.launch_set_utility_variable_history_label_workers import _worker_spec


def test_worker_spec_parses_local_gpu_and_partition() -> None:
    assert _worker_spec("3:11") == (3, 11, 0, 1)
    assert _worker_spec("3:11:1:2") == (3, 11, 1, 2)


@pytest.mark.parametrize(
    "value",
    ["3", "a:1", "1:-1", "1:2:3", "1:2:0:0", "1:2:2:2"],
)
def test_worker_spec_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _worker_spec(value)
