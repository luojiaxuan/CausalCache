#!/usr/bin/env python3
"""Launch or clean an audited fleet of MobileWorld emulator containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


CONTAINER_PORTS = {
    "backend": 6800,
    "viewer": 7860,
    "vnc": 5800,
    "adb": 5556,
}


def _available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _irregular_ports(seed: str, index: int) -> dict[str, int]:
    used: set[int] = set()
    result = {}
    for channel in CONTAINER_PORTS:
        nonce = 0
        while True:
            digest = hashlib.sha256(
                f"{seed}:{index}:{channel}:{nonce}".encode("utf-8")
            ).digest()
            candidate = 30000 + int.from_bytes(digest[:2], "big") % 31000
            if candidate not in used and _available(candidate):
                result[channel] = candidate
                used.add(candidate)
                break
            nonce += 1
            if nonce >= 10000:
                raise RuntimeError("could not allocate an irregular host port")
    return result


def _health(port: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=5,
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
            if response.status == 200 and value.get("ok") is True:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return False


def _image_identity(image: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    records = json.loads(result.stdout)
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("MobileWorld image inspect returned an invalid record")
    record = records[0]
    return {
        "id": record.get("Id"),
        "repo_digests": sorted(record.get("RepoDigests") or []),
        "created": record.get("Created"),
    }


def launch(args: argparse.Namespace) -> dict[str, Any]:
    if args.count <= 0 or args.count > 16:
        raise ValueError("MobileWorld environment count must be within [1, 16]")
    if not args.name_prefix.startswith("sglang-omni-jaxan-"):
        raise ValueError("MobileWorld emulator prefix violates the shared-host rule")
    image_identity = _image_identity(args.image)
    launched = []
    for index in range(args.count):
        name = f"{args.name_prefix}{index}"
        ports = _irregular_ports(args.port_seed, index)
        command = [
            "docker",
            "run",
            "--rm",
            "--privileged",
            "--detach",
            "--name",
            name,
        ]
        for channel, container_port in CONTAINER_PORTS.items():
            command.extend(["-p", f"{ports[channel]}:{container_port}"])
        command.append(args.image)
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        launched.append(
            {
                "name": name,
                "container_id": result.stdout.strip(),
                "ports": ports,
                "backend_url": f"http://127.0.0.1:{ports['backend']}",
                "ready": False,
            }
        )
        if index + 1 < args.count and args.launch_interval_seconds > 0:
            time.sleep(args.launch_interval_seconds)
    for record in launched:
        record["ready"] = _health(
            record["ports"]["backend"],
            timeout_seconds=args.ready_timeout_seconds,
        )
    return {
        "schema_version": "causalcache.mobileworld.environment_fleet.v1",
        "image": args.image,
        "image_identity": image_identity,
        "name_prefix": args.name_prefix,
        "port_seed_sha256": hashlib.sha256(
            args.port_seed.encode("utf-8")
        ).hexdigest(),
        "containers": launched,
        "all_ready": all(record["ready"] for record in launched),
    }


def cleanup(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest.get("containers")
    if not isinstance(records, list) or not records:
        raise ValueError("MobileWorld cleanup manifest has no containers")
    stopped = []
    for record in records:
        name = record.get("name")
        expected_id = record.get("container_id")
        inspect = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", str(name)],
            capture_output=True,
            text=True,
        )
        if inspect.returncode != 0:
            stopped.append({"name": name, "status": "already_absent"})
            continue
        observed_id = inspect.stdout.strip()
        if observed_id != expected_id:
            raise RuntimeError(f"refusing to stop identity-drifted container {name}")
        subprocess.run(["docker", "stop", str(name)], check=True)
        stopped.append(
            {"name": name, "container_id": observed_id, "status": "stopped"}
        )
    return {"status": "CLEANED_MOBILEWORLD_FLEET", "containers": stopped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--count", type=int, required=True)
    launch_parser.add_argument("--image", required=True)
    launch_parser.add_argument("--name-prefix", required=True)
    launch_parser.add_argument("--port-seed", required=True)
    launch_parser.add_argument("--launch-interval-seconds", type=float, default=20)
    launch_parser.add_argument("--ready-timeout-seconds", type=float, default=600)
    launch_parser.add_argument("--output", type=Path, required=True)
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "launch":
        value = launch(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        value = cleanup(args)
    print(json.dumps(value, sort_keys=True))
    if args.command == "launch" and not value["all_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
