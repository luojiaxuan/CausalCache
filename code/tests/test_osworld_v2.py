from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalcache.osworld_v2 import (
    OSWORLD_V2_CODE_REVISION,
    OSWORLD_V2_RELEASE,
    load_osworld_v2_memory_plan,
)
from scripts.run_osworld_v2_gui_owl import _logical_shard


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "data/manifests/osworld_v2_memory_split_v1.json"
CONFIG_PATH = ROOT / "code/configs/causalcache_osworld_v2_memory_v1.json"
BENCHMARK_CONFIG_PATH = (
    ROOT / "code/configs/causalcache_osworld_v2_memory_benchmark_v1.json"
)


def test_osworld_v2_memory_split_is_phenomenon_defined() -> None:
    plan = load_osworld_v2_memory_plan(PLAN_PATH)
    assert plan["counts"] == {
        "all_three_phenomena": 8,
        "cross_source_reasoning": 46,
        "dynamic_environment": 10,
        "implicit_state_inference": 43,
        "memory_core": 43,
        "memory_stress_union": 65,
        "non_memory_control": 43,
    }
    assert len(plan["splits"]["memory_core"]) == 43
    assert len(plan["splits"]["memory_stress_union"]) == 65
    assert len(plan["splits"]["non_memory_control"]) == 43
    assert set(plan["splits"]["memory_core"]).issubset(
        plan["splits"]["memory_stress_union"]
    )
    assert set(plan["splits"]["memory_stress_union"]).isdisjoint(
        plan["splits"]["non_memory_control"]
    )


def test_osworld_v2_config_uses_one_release_bundle() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    release = config["release"]
    assert release["name"] == OSWORLD_V2_RELEASE
    assert release["code_revision"] == OSWORLD_V2_CODE_REVISION
    assert release["code_tag"] == "v2026.06.24"
    assert release["task_dataset"].endswith("@v2026.06.24")
    assert release["asset_dataset"].endswith("@v2026.06.24")
    assert release["website_repository"].endswith("@v2026.06.24")
    assert config["execution"]["assets_root"] == (
        "/data/datasets/osworld-v2-assets-v2026.06.24"
    )
    assert config["execution"]["docker_dns_server"] == "127.0.0.11"
    assert config["execution"]["docker_cpu_model"] is None
    assert config["execution"]["docker_port_lock_timeout_seconds"] == 180


def test_osworld_v2_fast_benchmark_keeps_scientific_identity() -> None:
    base = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8"))
    assert benchmark["release"] == base["release"]
    assert benchmark["policy"] == base["policy"]
    assert benchmark["execution"]["max_steps"] == 50
    assert benchmark["execution"]["num_envs"] == 2
    assert benchmark["execution"]["result_root"].endswith(
        "osworld-v2-memory-benchmark-v1"
    )


def test_osworld_v2_plan_loader_rejects_release_drift(tmp_path: Path) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["upstream"]["code_revision"] = "0" * 40
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="release identity"):
        load_osworld_v2_memory_plan(path)


def test_osworld_v2_logical_shards_are_disjoint_and_complete() -> None:
    task_ids = [f"{index:03d}" for index in range(1, 109)]
    shards = [
        _logical_shard(task_ids, shard_count=5, shard_index=index)
        for index in range(5)
    ]
    assert [len(shard) for shard in shards] == [22, 22, 22, 21, 21]
    assert sorted(task_id for shard in shards for task_id in shard) == task_ids
    assert sum(len(set(shard)) for shard in shards) == 108


def test_osworld_v2_logical_shard_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        _logical_shard(["001"], shard_count=0, shard_index=0)
    with pytest.raises(ValueError, match="index must be within"):
        _logical_shard(["001"], shard_count=2, shard_index=2)
