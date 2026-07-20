"""Versioned command materialization and readiness checks for selector RPC."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causalcache.set_utility_heldout_evaluation import percentile_type7
from causalcache.set_utility_live_controller import (
    LIVE_SELECTOR_AUTHORIZATION_STATUS,
    LIVE_SELECTOR_SCHEMA_VERSION,
)
from causalcache.set_utility_native_replay import (
    canonical_json_bytes,
    signed_content_hash_is_valid,
)


RPC_TRANSPORT_SCHEMA_VERSION = "1.0.0"
RPC_TRANSPORT_PROTOCOL_ID = "causalcache_selector_rpc_transport_v1"
RPC_TRANSPORT_PLAN_STATUS = "MATERIALIZED_SELECTOR_RPC_TRANSPORT_V1"
RPC_READINESS_STATUS = "COMPLETED_SELECTOR_RPC_TRANSPORT_READINESS_V1"
HEALTH_STATUS = "READY_CAUSALCACHE_LIVE_RICH_SELECTOR_SERVICE"
_LOOPBACK = "127.0.0.1"
_SAFE_ENDPOINT_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._@-"
)
_AUTHORIZATION_SHA_FIELDS = (
    "authorization_id",
    "checkpoint_sha256",
    "config_sha256",
    "heldout_result_sha256",
    "model_config_sha256",
    "native_replay_result_sha256",
    "selections_sha256",
)
_AUTHORIZATION_FIELDS = frozenset(
    (*_AUTHORIZATION_SHA_FIELDS, "model_variant", "status", "winner_model")
)


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)[:-1]
    ).hexdigest()
    return result


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _endpoint(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value[0].isalnum()
        or any(character not in _SAFE_ENDPOINT_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} is not a safe SSH endpoint")
    return value


def _port(value: Any, *, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError(f"{label} must be a TCP port")
    return value


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _command(segment: str, action: str, argv: Sequence[str]) -> dict[str, Any]:
    if not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ValueError("transport command argv is invalid")
    return {
        "action": action,
        "argv": list(argv),
        "segment": segment,
        "shell_preview": shlex.join(argv),
    }


def _nested_aries_command(
    *,
    action: str,
    nested_socket_name: str,
    aries_host_alias: str,
    aries_user: str,
    relay_port: int,
    aries_loopback_port: int,
    taurus_loopback_port: int,
) -> str:
    forward = (
        f"{_LOOPBACK}:{aries_loopback_port}:"
        f"{_LOOPBACK}:{taurus_loopback_port}"
    )
    if action == "start":
        ssh_argv = [
            "ssh",
            "-fNT",
            "-M",
            "-S",
            "$nested_socket",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "TCPKeepAlive=yes",
            "-R",
            forward,
            "-o",
            f"HostKeyAlias={aries_host_alias}",
            "-p",
            str(relay_port),
            f"{aries_user}@127.0.0.1",
        ]
    elif action == "stop":
        ssh_argv = [
            "ssh",
            "-S",
            "$nested_socket",
            "-O",
            "exit",
            "-o",
            f"HostKeyAlias={aries_host_alias}",
            "-p",
            str(relay_port),
            f"{aries_user}@127.0.0.1",
        ]
    else:
        raise ValueError("nested Aries transport action is invalid")
    rendered = " ".join(
        '"$nested_socket"' if item == "$nested_socket" else shlex.quote(item)
        for item in ssh_argv
    )
    return (
        f'nested_socket="$HOME/.ssh/{nested_socket_name}"; '
        f"exec {rendered}"
    )


def materialize_rpc_transport_plan(
    *,
    h200_host: str,
    h200_service_port: int,
    mac_loopback_port: int,
    taurus_loopback_port: int,
    aries_loopback_port: int,
    rtt_p95_ms_maximum: float,
    rtt_sample_count_minimum: int,
    taurus_endpoint: str = "taurus",
    aries_host_alias: str = "aries.cs.ucsb.edu",
    aries_user: str = "jiaxuanluo",
    aries_relay_port: int = 20042,
    container_network_profile: str = "host_network",
    home: Path | None = None,
) -> dict[str, Any]:
    """Materialize, but never execute, the three-hop loopback transport."""
    h200_host = _endpoint(h200_host, label="H200 host")
    if h200_host not in {"hyper00", "hyper01"}:
        raise ValueError("selector RPC v1 only accepts the pinned H200 host aliases")
    taurus_endpoint = _endpoint(taurus_endpoint, label="Taurus endpoint")
    aries_host_alias = _endpoint(aries_host_alias, label="Aries host alias")
    aries_user = _endpoint(aries_user, label="Aries user")
    h200_service_port = _port(h200_service_port, label="H200 service port")
    mac_loopback_port = _port(mac_loopback_port, label="Mac loopback port")
    taurus_loopback_port = _port(
        taurus_loopback_port, label="Taurus loopback port"
    )
    aries_loopback_port = _port(aries_loopback_port, label="Aries loopback port")
    aries_relay_port = _port(aries_relay_port, label="Aries SSH relay port")
    rtt_limit = _positive_number(
        rtt_p95_ms_maximum, label="RTT p95 maximum"
    )
    if type(rtt_sample_count_minimum) is not int or rtt_sample_count_minimum < 3:
        raise ValueError("RTT readiness requires at least three samples")
    if container_network_profile != "host_network":
        raise ValueError(
            "Aries-loopback transport requires the policy container to use host network"
        )
    resolved_home = (home or Path.home()).expanduser().resolve()
    state_root = resolved_home / ".codex/state/causalcache-selector-rpc-v1"
    h200_socket = state_root / f"{h200_host}-{h200_service_port}.sock"
    mac_taurus_socket = (
        resolved_home
        / f".codex/state/connect-aries-via-taurus/port-{aries_relay_port}.sock"
    )
    connect_aries_script = (
        resolved_home
        / ".codex/skills/connect-aries-via-taurus/scripts/connect_aries.sh"
    )
    nested_socket_name = f"causalcache-selector-rpc-v1-aries-{aries_relay_port}.sock"
    h200_forward = (
        f"{_LOOPBACK}:{mac_loopback_port}:"
        f"{_LOOPBACK}:{h200_service_port}"
    )
    taurus_forward = (
        f"{_LOOPBACK}:{taurus_loopback_port}:"
        f"{_LOOPBACK}:{mac_loopback_port}"
    )
    status = _command(
        "existing_mac_taurus_aries_relay",
        "status",
        (
            str(connect_aries_script),
            "status",
            "--taurus",
            taurus_endpoint,
            "--aries-host",
            aries_host_alias,
            "--aries-user",
            aries_user,
            "--remote-port",
            str(aries_relay_port),
        ),
    )
    prepare = _command(
        "local_owned_transport_state",
        "prepare",
        ("install", "-d", "-m", "700", str(state_root)),
    )
    start = (
        _command(
            "h200_loopback_to_mac_loopback",
            "start",
            (
                "ssh",
                "-fNT",
                "-M",
                "-S",
                str(h200_socket),
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "TCPKeepAlive=yes",
                "-L",
                h200_forward,
                h200_host,
            ),
        ),
        _command(
            "mac_loopback_to_taurus_loopback",
            "start",
            (
                "ssh",
                "-S",
                str(mac_taurus_socket),
                "-O",
                "forward",
                "-o",
                "ExitOnForwardFailure=yes",
                "-R",
                taurus_forward,
                taurus_endpoint,
            ),
        ),
        _command(
            "taurus_loopback_to_aries_loopback",
            "start",
            (
                "ssh",
                "-T",
                "-o",
                "RemoteCommand=none",
                "-o",
                "RequestTTY=no",
                taurus_endpoint,
                _nested_aries_command(
                    action="start",
                    nested_socket_name=nested_socket_name,
                    aries_host_alias=aries_host_alias,
                    aries_user=aries_user,
                    relay_port=aries_relay_port,
                    aries_loopback_port=aries_loopback_port,
                    taurus_loopback_port=taurus_loopback_port,
                ),
            ),
        ),
    )
    stop = (
        _command(
            "taurus_loopback_to_aries_loopback",
            "stop",
            (
                "ssh",
                "-T",
                "-o",
                "RemoteCommand=none",
                "-o",
                "RequestTTY=no",
                taurus_endpoint,
                _nested_aries_command(
                    action="stop",
                    nested_socket_name=nested_socket_name,
                    aries_host_alias=aries_host_alias,
                    aries_user=aries_user,
                    relay_port=aries_relay_port,
                    aries_loopback_port=aries_loopback_port,
                    taurus_loopback_port=taurus_loopback_port,
                ),
            ),
        ),
        _command(
            "mac_loopback_to_taurus_loopback",
            "stop",
            (
                "ssh",
                "-S",
                str(mac_taurus_socket),
                "-O",
                "cancel",
                "-R",
                taurus_forward,
                taurus_endpoint,
            ),
        ),
        _command(
            "h200_loopback_to_mac_loopback",
            "stop",
            (
                "ssh",
                "-S",
                str(h200_socket),
                "-O",
                "exit",
                h200_host,
            ),
        ),
    )
    specification = {
        "aries_host_alias": aries_host_alias,
        "aries_loopback_port": aries_loopback_port,
        "aries_relay_port": aries_relay_port,
        "aries_user": aries_user,
        "container_network_profile": container_network_profile,
        "h200_host": h200_host,
        "h200_service_port": h200_service_port,
        "home": str(resolved_home),
        "mac_loopback_port": mac_loopback_port,
        "rtt_p95_ms_maximum": rtt_limit,
        "rtt_sample_count_minimum": rtt_sample_count_minimum,
        "taurus_endpoint": taurus_endpoint,
        "taurus_loopback_port": taurus_loopback_port,
    }
    return _signed(
        {
            "aries_policy_container": {
                "network_profile": "host_network",
                "required_docker_arguments": ["--network", "host"],
                "selector_endpoint": f"http://{_LOOPBACK}:{aries_loopback_port}",
            },
            "commands": {
                "prerequisites": [prepare, status],
                "start_in_order": list(start),
                "stop_in_order": list(stop),
            },
            "execution_policy": {
                "execute_automatically": False,
                "global_ssh_configuration_mutation_allowed": False,
                "open_non_loopback_listener_allowed": False,
            },
            "protocol_id": RPC_TRANSPORT_PROTOCOL_ID,
            "readiness_contract": {
                "endpoint": f"http://{_LOOPBACK}:{aries_loopback_port}/health",
                "health_status": HEALTH_STATUS,
                "minimum_rtt_sample_count": rtt_sample_count_minimum,
                "rtt_p95_ms_maximum": rtt_limit,
            },
            "route": [
                "H200_loopback",
                "Mac_loopback",
                "Taurus_loopback",
                "Aries_loopback",
            ],
            "schema_version": RPC_TRANSPORT_SCHEMA_VERSION,
            "specification": specification,
            "status": RPC_TRANSPORT_PLAN_STATUS,
        }
    )


def validate_rpc_transport_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if (
        plan.get("status") != RPC_TRANSPORT_PLAN_STATUS
        or plan.get("protocol_id") != RPC_TRANSPORT_PROTOCOL_ID
        or plan.get("schema_version") != RPC_TRANSPORT_SCHEMA_VERSION
        or not signed_content_hash_is_valid(plan)
    ):
        raise ValueError("selector RPC transport plan signature or version drifted")
    specification = plan.get("specification")
    if not isinstance(specification, Mapping):
        raise ValueError("selector RPC transport plan omits its specification")
    expected = materialize_rpc_transport_plan(
        h200_host=specification.get("h200_host"),
        h200_service_port=specification.get("h200_service_port"),
        mac_loopback_port=specification.get("mac_loopback_port"),
        taurus_loopback_port=specification.get("taurus_loopback_port"),
        aries_loopback_port=specification.get("aries_loopback_port"),
        rtt_p95_ms_maximum=specification.get("rtt_p95_ms_maximum"),
        rtt_sample_count_minimum=specification.get("rtt_sample_count_minimum"),
        taurus_endpoint=specification.get("taurus_endpoint"),
        aries_host_alias=specification.get("aries_host_alias"),
        aries_user=specification.get("aries_user"),
        aries_relay_port=specification.get("aries_relay_port"),
        container_network_profile=specification.get("container_network_profile"),
        home=Path(str(specification.get("home"))),
    )
    if dict(plan) != expected:
        raise ValueError("selector RPC transport plan does not replay canonically")
    return expected


def _health_is_valid(payload: Any, *, expected_authorization_sha256: str) -> bool:
    if not isinstance(payload, Mapping):
        return False
    authorization = payload.get("authorization")
    if (
        payload.get("schema_version") != LIVE_SELECTOR_SCHEMA_VERSION
        or payload.get("status") != HEALTH_STATUS
        or not isinstance(authorization, Mapping)
        or authorization.get("authorization_id") != expected_authorization_sha256
        or authorization.get("status") != LIVE_SELECTOR_AUTHORIZATION_STATUS
        or authorization.get("winner_model") not in {"deepsets", "set_transformer"}
        or not isinstance(authorization.get("model_variant"), str)
        or not authorization.get("model_variant")
        or set(authorization) != _AUTHORIZATION_FIELDS
    ):
        return False
    if not all(
        _sha256(authorization.get(field)) for field in _AUTHORIZATION_SHA_FIELDS
    ):
        return False
    unsigned = dict(authorization)
    claimed = unsigned.pop("authorization_id")
    return hashlib.sha256(canonical_json_bytes(unsigned)[:-1]).hexdigest() == claimed


def evaluate_rpc_transport_readiness(
    *,
    plan: Mapping[str, Any],
    health_samples: Sequence[Mapping[str, Any]],
    expected_authorization_sha256: str,
) -> dict[str, Any]:
    """Evaluate captured health/RTT evidence without issuing a network request."""
    canonical_plan = validate_rpc_transport_plan(plan)
    if not _sha256(expected_authorization_sha256):
        raise ValueError("expected authorization must be a lowercase SHA256")
    if isinstance(health_samples, (str, bytes, bytearray, Mapping)) or not isinstance(
        health_samples, Sequence
    ):
        raise ValueError("health samples must be an ordered object array")
    readiness = canonical_plan["readiness_contract"]
    expected_endpoint = readiness["endpoint"]
    rtt_values = []
    valid_health_hashes = []
    sample_checks = []
    for index, sample in enumerate(health_samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"health sample {index} is not an object")
        rtt = sample.get("round_trip_ms")
        rtt_valid = (
            not isinstance(rtt, bool)
            and isinstance(rtt, (int, float))
            and math.isfinite(float(rtt))
            and float(rtt) > 0.0
        )
        if rtt_valid:
            rtt_values.append(float(rtt))
        health_valid = _health_is_valid(
            sample.get("payload"),
            expected_authorization_sha256=expected_authorization_sha256,
        )
        if health_valid:
            valid_health_hashes.append(
                hashlib.sha256(
                    canonical_json_bytes(sample["payload"])[:-1]
                ).hexdigest()
            )
        sample_checks.append(
            {
                "authorization_and_health_valid": health_valid,
                "content_type_valid": sample.get("content_type") == "application/json",
                "endpoint_valid": sample.get("endpoint") == expected_endpoint,
                "http_status_valid": sample.get("http_status") == 200,
                "index": index,
                "rtt_valid": rtt_valid,
            }
        )
    sample_count_pass = len(health_samples) >= readiness["minimum_rtt_sample_count"]
    authorization_pass = bool(sample_checks) and all(
        sample["authorization_and_health_valid"] for sample in sample_checks
    )
    http_samples_pass = bool(sample_checks) and all(
        sample["content_type_valid"]
        and sample["endpoint_valid"]
        and sample["http_status_valid"]
        and sample["rtt_valid"]
        for sample in sample_checks
    )
    health_stable = (
        len(valid_health_hashes) == len(health_samples)
        and len(set(valid_health_hashes)) == 1
    )
    p50 = percentile_type7(rtt_values, 0.5) if rtt_values else None
    p95 = percentile_type7(rtt_values, 0.95) if rtt_values else None
    rtt_pass = bool(
        p95 is not None and p95 <= readiness["rtt_p95_ms_maximum"]
    )
    checks = {
        "all_health_samples_bind_expected_authorization": authorization_pass,
        "all_http_samples_valid": http_samples_pass,
        "health_payload_stable": health_stable,
        "minimum_sample_count_met": sample_count_pass,
        "rtt_p95_within_threshold": rtt_pass,
        "transport_plan_valid": True,
    }
    verdict = "GO" if all(checks.values()) else "NO_GO"
    result = {
        "bindings": {
            "expected_authorization_sha256": expected_authorization_sha256,
            "transport_plan_content_sha256": canonical_plan["content_sha256"],
        },
        "checks": checks,
        "health": {
            "expected_endpoint": expected_endpoint,
            "expected_status": HEALTH_STATUS,
            "payload_sha256": (
                valid_health_hashes[0] if health_stable else None
            ),
            "sample_checks": sample_checks,
        },
        "protocol_id": RPC_TRANSPORT_PROTOCOL_ID,
        "rtt_ms": {
            "maximum_allowed_p95_ms": readiness["rtt_p95_ms_maximum"],
            "minimum_required_sample_count": readiness["minimum_rtt_sample_count"],
            "observed_sample_count": len(health_samples),
            "p50_ms": p50,
            "p95_ms": p95,
        },
        "schema_version": RPC_TRANSPORT_SCHEMA_VERSION,
        "status": RPC_READINESS_STATUS,
        "verdict": verdict,
    }
    return _signed(result)


__all__ = [
    "RPC_READINESS_STATUS",
    "RPC_TRANSPORT_PLAN_STATUS",
    "RPC_TRANSPORT_PROTOCOL_ID",
    "RPC_TRANSPORT_SCHEMA_VERSION",
    "evaluate_rpc_transport_readiness",
    "materialize_rpc_transport_plan",
    "validate_rpc_transport_plan",
]
