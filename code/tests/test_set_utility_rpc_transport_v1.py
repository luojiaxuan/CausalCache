from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

import pytest

from causalcache.set_utility_live_controller import (
    LIVE_SELECTOR_AUTHORIZATION_STATUS,
)
from causalcache.set_utility_native_replay import (
    canonical_json_bytes,
    signed_content_hash_is_valid,
)
from causalcache.set_utility_rpc_transport_v1 import (
    RPC_READINESS_STATUS,
    evaluate_rpc_transport_readiness,
    materialize_rpc_transport_plan,
    validate_rpc_transport_plan,
)


AUTHORIZATION_FIELDS = {
    "checkpoint_sha256": "b" * 64,
    "config_sha256": "c" * 64,
    "heldout_result_sha256": "d" * 64,
    "model_config_sha256": "e" * 64,
    "model_variant": "set_transformer_d256_l8_r1_s2_lr1e4",
    "native_replay_result_sha256": "f" * 64,
    "selections_sha256": "1" * 64,
    "status": LIVE_SELECTOR_AUTHORIZATION_STATUS,
    "winner_model": "set_transformer",
}
AUTHORIZATION_SHA = hashlib.sha256(
    canonical_json_bytes(AUTHORIZATION_FIELDS)[:-1]
).hexdigest()


def _plan(tmp_path: Path, *, rtt_limit: float = 50.0) -> dict[str, object]:
    return materialize_rpc_transport_plan(
        h200_host="hyper01",
        h200_service_port=18765,
        mac_loopback_port=28765,
        taurus_loopback_port=38765,
        aries_loopback_port=48765,
        rtt_p95_ms_maximum=rtt_limit,
        rtt_sample_count_minimum=5,
        home=tmp_path,
    )


def _health() -> dict[str, object]:
    return {
        "authorization": {
            "authorization_id": AUTHORIZATION_SHA,
            **AUTHORIZATION_FIELDS,
        },
        "schema_version": "1.0.0",
        "status": "READY_CAUSALCACHE_LIVE_RICH_SELECTOR_SERVICE",
    }


def _samples(plan: dict[str, object], values=(10.0, 12.0, 14.0, 16.0, 18.0)):
    endpoint = plan["readiness_contract"]["endpoint"]
    health = _health()
    return [
        {
            "content_type": "application/json",
            "endpoint": endpoint,
            "http_status": 200,
            "payload": health,
            "round_trip_ms": value,
        }
        for value in values
    ]


def _resign(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)[:-1]
    ).hexdigest()
    return result


def test_plan_materializes_only_the_three_loopback_hops(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert validate_rpc_transport_plan(plan) == plan
    assert plan["route"] == [
        "H200_loopback",
        "Mac_loopback",
        "Taurus_loopback",
        "Aries_loopback",
    ]
    starts = plan["commands"]["start_in_order"]
    assert all(row["shell_preview"] == shlex.join(row["argv"]) for row in starts)
    assert [row["segment"] for row in starts] == [
        "h200_loopback_to_mac_loopback",
        "mac_loopback_to_taurus_loopback",
        "taurus_loopback_to_aries_loopback",
    ]
    assert starts[0]["argv"][-1] == "hyper01"
    assert starts[0]["argv"][starts[0]["argv"].index("-L") + 1] == (
        "127.0.0.1:28765:127.0.0.1:18765"
    )
    assert starts[1]["argv"][-1] == "taurus"
    assert starts[1]["argv"][starts[1]["argv"].index("-R") + 1] == (
        "127.0.0.1:38765:127.0.0.1:28765"
    )
    assert starts[2]["argv"][-2] == "taurus"
    nested_start = starts[2]["argv"][-1]
    assert "causalcache-selector-rpc-v1-aries-20042.sock" in nested_start
    assert 'exec ssh -fNT -M -S "$nested_socket"' in nested_start
    assert "-R 127.0.0.1:48765:127.0.0.1:38765" in nested_start
    assert "jiaxuanluo@127.0.0.1" in nested_start
    assert "HostKeyAlias=aries.cs.ucsb.edu" in nested_start
    assert starts[2]["shell_preview"].endswith(shlex.quote(nested_start))
    nested_stop = plan["commands"]["stop_in_order"][0]["argv"][-1]
    assert all(
        row["shell_preview"] == shlex.join(row["argv"])
        for row in plan["commands"]["stop_in_order"]
    )
    assert 'exec ssh -S "$nested_socket" -O exit' in nested_stop
    assert "-R " not in nested_stop
    assert plan["aries_policy_container"] == {
        "network_profile": "host_network",
        "required_docker_arguments": ["--network", "host"],
        "selector_endpoint": "http://127.0.0.1:48765",
    }
    assert signed_content_hash_is_valid(plan)


def test_plan_rejects_unsafe_or_non_host_network_topology(tmp_path: Path) -> None:
    arguments = {
        "h200_host": "hyper01",
        "h200_service_port": 18765,
        "mac_loopback_port": 28765,
        "taurus_loopback_port": 38765,
        "aries_loopback_port": 48765,
        "rtt_p95_ms_maximum": 50.0,
        "rtt_sample_count_minimum": 5,
        "home": tmp_path,
    }
    with pytest.raises(ValueError, match="host network"):
        materialize_rpc_transport_plan(
            **arguments, container_network_profile="host_gateway"
        )
    with pytest.raises(ValueError, match="safe SSH endpoint"):
        materialize_rpc_transport_plan(
            **arguments, taurus_endpoint="taurus; touch /tmp/bad"
        )
    with pytest.raises(ValueError, match="pinned H200"):
        materialize_rpc_transport_plan(**{**arguments, "h200_host": "aries"})


def test_readiness_go_binds_health_authorization_and_rtt(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    result = evaluate_rpc_transport_readiness(
        plan=plan,
        health_samples=_samples(plan),
        expected_authorization_sha256=AUTHORIZATION_SHA,
    )
    assert result["status"] == RPC_READINESS_STATUS
    assert result["verdict"] == "GO"
    assert result["checks"] == {
        "all_health_samples_bind_expected_authorization": True,
        "all_http_samples_valid": True,
        "health_payload_stable": True,
        "minimum_sample_count_met": True,
        "rtt_p95_within_threshold": True,
        "transport_plan_valid": True,
    }
    assert result["rtt_ms"]["p50_ms"] == 14.0
    assert result["rtt_ms"]["p95_ms"] == pytest.approx(17.6)
    assert signed_content_hash_is_valid(result)


@pytest.mark.parametrize("failure", ("authorization", "sample_count", "rtt"))
def test_readiness_fail_closed(failure: str, tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    samples = _samples(plan)
    if failure == "authorization":
        samples[0]["payload"] = {
            **_health(),
            "authorization": {
                **_health()["authorization"],
                "authorization_id": "9" * 64,
            },
        }
    elif failure == "sample_count":
        samples = samples[:2]
    else:
        samples = _samples(plan, values=(10.0, 10.0, 10.0, 10.0, 100.0))
    result = evaluate_rpc_transport_readiness(
        plan=plan,
        health_samples=samples,
        expected_authorization_sha256=AUTHORIZATION_SHA,
    )
    assert result["verdict"] == "NO_GO"
    assert not all(result["checks"].values())


def test_plan_canonical_replay_rejects_resigned_command_mutation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["commands"]["start_in_order"][0]["argv"][-1] = "hyper00"
    plan = _resign(plan)
    with pytest.raises(ValueError, match="replay canonically"):
        validate_rpc_transport_plan(plan)
