#!/usr/bin/env python3
"""Manage a Mac/Taurus relay that exposes Aries AndroidWorld on Hyper loopback."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ENDPOINT = re.compile(r"[A-Za-z0-9._@-]+")
_HOST = re.compile(r"[A-Za-z0-9.-]+")
_USER = re.compile(r"[A-Za-z0-9._-]+")


def _endpoint(value: str, label: str) -> str:
    if _ENDPOINT.fullmatch(value) is None:
        raise ValueError(f"unsafe {label}")
    return value


def _host(value: str, label: str) -> str:
    if _HOST.fullmatch(value) is None:
        raise ValueError(f"unsafe {label}")
    return value


def _user(value: str, label: str) -> str:
    if _USER.fullmatch(value) is None:
        raise ValueError(f"unsafe {label}")
    return value


def _port(value: int, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError(f"{label} must be in [1,65535]")
    return value


@dataclass(frozen=True)
class RelayPlan:
    aries_host: str
    aries_http_port: int
    aries_relay_ssh_port: int
    aries_user: str
    connect_aries_script: Path
    hyper: str
    hyper_loopback_port: int
    mac_loopback_port: int
    maximum_rtt_ms: float
    state_dir: Path
    taurus: str

    def __post_init__(self) -> None:
        _host(self.aries_host, "Aries host")
        _user(self.aries_user, "Aries user")
        _endpoint(self.hyper, "Hyper endpoint")
        _endpoint(self.taurus, "Taurus endpoint")
        _port(self.aries_http_port, "Aries HTTP port")
        _port(self.aries_relay_ssh_port, "Aries relay SSH port")
        _port(self.hyper_loopback_port, "Hyper loopback port")
        _port(self.mac_loopback_port, "Mac loopback port")
        if not math.isfinite(self.maximum_rtt_ms) or self.maximum_rtt_ms <= 0.0:
            raise ValueError("maximum RTT must be finite and positive")

    @property
    def aries_socket(self) -> Path:
        return self.state_dir / "aries-http.sock"

    @property
    def hyper_socket(self) -> Path:
        return self.state_dir / "hyper-reverse.sock"

    @property
    def hyper_url(self) -> str:
        return f"http://127.0.0.1:{self.hyper_loopback_port}"

    def skill_command(self, action: str) -> tuple[str, ...]:
        return (
            str(self.connect_aries_script),
            action,
            "--taurus",
            self.taurus,
            "--aries-host",
            self.aries_host,
            "--aries-user",
            self.aries_user,
            "--remote-port",
            str(self.aries_relay_ssh_port),
        )

    def aries_forward_command(self) -> tuple[str, ...]:
        proxy = (
            "ssh -T -o RemoteCommand=none -o RequestTTY=no "
            f"{self.taurus} -W 127.0.0.1:{self.aries_relay_ssh_port}"
        )
        return (
            "ssh",
            "-fNT",
            "-M",
            "-S",
            str(self.aries_socket),
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            f"ProxyCommand={proxy}",
            "-o",
            f"HostKeyAlias={self.aries_host}",
            "-L",
            (
                f"127.0.0.1:{self.mac_loopback_port}:"
                f"127.0.0.1:{self.aries_http_port}"
            ),
            f"{self.aries_user}@{self.aries_host}",
        )

    def hyper_reverse_command(self) -> tuple[str, ...]:
        return (
            "ssh",
            "-fNT",
            "-M",
            "-S",
            str(self.hyper_socket),
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-R",
            (
                f"127.0.0.1:{self.hyper_loopback_port}:"
                f"127.0.0.1:{self.mac_loopback_port}"
            ),
            self.hyper,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "commands": {
                "aries_http_forward": list(self.aries_forward_command()),
                "hyper_reverse_forward": list(self.hyper_reverse_command()),
                "skill_start": list(self.skill_command("start")),
                "skill_status": list(self.skill_command("status")),
            },
            "hyper_androidworld_url": self.hyper_url,
            "maximum_rtt_ms": self.maximum_rtt_ms,
            "network_requirements": [
                "run_this_manager_on_the_Mac_with_SSH_access_to_Taurus_and_Hyper",
                "skill_owned_Mac_to_Taurus_relay_exposes_Aries_SSH_on_Taurus_127.0.0.1",
                "AndroidWorld_listens_on_Aries_127.0.0.1_at_the_declared_port",
                "Hyper_sshd_allows_loopback_remote_forwarding",
                "runner_container_uses_host_network_or_an_explicit_host_loopback_mapping",
                "runner_and_selector_use_distinct_explicit_loopback_ports",
            ],
            "route": (
                "Hyper:loopback -> Mac reverse SSH -> Mac:loopback -> "
                "Taurus relay -> Aries:loopback AndroidWorld"
            ),
            "status": "PLANNED_NOT_VERIFIED",
        }


_REMOTE_HEALTH = r'''
import json, sys, time, urllib.request
url = sys.argv[1].rstrip('/') + '/health'
maximum = float(sys.argv[2])
started = time.perf_counter()
with urllib.request.urlopen(url, timeout=10.0) as response:
    payload = json.loads(response.read().decode('utf-8'))
elapsed = (time.perf_counter() - started) * 1000.0
if payload.get('status') != 'success':
    raise SystemExit('AndroidWorld health status is not success')
if elapsed > maximum:
    raise SystemExit(f'AndroidWorld relay RTT {elapsed:.3f}ms exceeds {maximum:.3f}ms')
print(json.dumps({'rtt_ms': elapsed, 'status': 'PASSED_ANDROIDWORLD_RELAY_HEALTH'}))
'''.strip()


def _run(
    command: tuple[str, ...],
    *,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        input=input_text,
        text=True,
    )


def _control_ready(socket: Path, endpoint: str) -> bool:
    if not socket.exists():
        return False
    return _run(
        ("ssh", "-S", str(socket), "-O", "check", endpoint),
        check=False,
    ).returncode == 0


def _control_exit(socket: Path, endpoint: str) -> None:
    if socket.exists():
        _run(
            ("ssh", "-S", str(socket), "-O", "exit", endpoint),
            check=False,
        )
        socket.unlink(missing_ok=True)


def check_relay(plan: RelayPlan) -> dict[str, Any]:
    if _run(plan.skill_command("status"), check=False).returncode != 0:
        raise RuntimeError("skill-owned Mac/Taurus Aries relay is unavailable")
    if not _control_ready(plan.aries_socket, f"{plan.aries_user}@{plan.aries_host}"):
        raise RuntimeError("Mac-to-Aries AndroidWorld HTTP forward is unavailable")
    if not _control_ready(plan.hyper_socket, plan.hyper):
        raise RuntimeError("Mac-to-Hyper reverse forward is unavailable")
    result = _run(
        (
            "ssh",
            "-S",
            str(plan.hyper_socket),
            plan.hyper,
            "python3",
            "-",
            plan.hyper_url,
            str(plan.maximum_rtt_ms),
        ),
        input_text=_REMOTE_HEALTH,
    )
    try:
        health = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("Hyper-side AndroidWorld health output is invalid") from error
    return {
        **plan.to_mapping(),
        "health": health,
        "status": "READY_VERIFIED_ANDROIDWORLD_LOOPBACK_RELAY",
    }


def start_relay(plan: RelayPlan) -> dict[str, Any]:
    plan.state_dir.mkdir(parents=True, exist_ok=True)
    if _run(plan.skill_command("status"), check=False).returncode != 0:
        _run(plan.skill_command("start"))
    if not _control_ready(plan.aries_socket, f"{plan.aries_user}@{plan.aries_host}"):
        plan.aries_socket.unlink(missing_ok=True)
        _run(plan.aries_forward_command())
    try:
        if not _control_ready(plan.hyper_socket, plan.hyper):
            plan.hyper_socket.unlink(missing_ok=True)
            _run(plan.hyper_reverse_command())
        return check_relay(plan)
    except Exception:
        _control_exit(plan.hyper_socket, plan.hyper)
        _control_exit(plan.aries_socket, f"{plan.aries_user}@{plan.aries_host}")
        raise


def stop_relay(plan: RelayPlan) -> dict[str, Any]:
    _control_exit(plan.hyper_socket, plan.hyper)
    _control_exit(plan.aries_socket, f"{plan.aries_user}@{plan.aries_host}")
    return {
        "shared_skill_relay_stopped": False,
        "status": "STOPPED_CAUSALCACHE_ANDROIDWORLD_LOOPBACK_FORWARDS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "start", "check", "stop"))
    parser.add_argument("--hyper", required=True)
    parser.add_argument("--taurus", default="taurus")
    parser.add_argument("--aries-host", default="aries.cs.ucsb.edu")
    parser.add_argument("--aries-user", default="jiaxuanluo")
    parser.add_argument("--aries-relay-ssh-port", type=int, default=20042)
    parser.add_argument("--aries-http-port", type=int, required=True)
    parser.add_argument("--mac-loopback-port", type=int, required=True)
    parser.add_argument("--hyper-loopback-port", type=int, required=True)
    parser.add_argument("--maximum-rtt-ms", type=float, required=True)
    parser.add_argument(
        "--connect-aries-script",
        type=Path,
        default=(
            Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
            / "skills/connect-aries-via-taurus/scripts/connect_aries.sh"
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=(Path.home() / ".codex/state/causalcache-androidworld-relay"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = RelayPlan(
        aries_host=args.aries_host,
        aries_http_port=args.aries_http_port,
        aries_relay_ssh_port=args.aries_relay_ssh_port,
        aries_user=args.aries_user,
        connect_aries_script=args.connect_aries_script.expanduser().resolve(),
        hyper=args.hyper,
        hyper_loopback_port=args.hyper_loopback_port,
        mac_loopback_port=args.mac_loopback_port,
        maximum_rtt_ms=args.maximum_rtt_ms,
        state_dir=args.state_dir.expanduser().resolve(),
        taurus=args.taurus,
    )
    if args.action == "plan":
        result = plan.to_mapping()
    elif args.action == "start":
        result = start_relay(plan)
    elif args.action == "check":
        result = check_relay(plan)
    else:
        result = stop_relay(plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
