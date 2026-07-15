"""Inspect the live Aries containers used by restoration-v2 executor dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FROZEN_HOST = "aries.cs.ucsb.edu"
FROZEN_RUNTIME_CONFIG_IMAGE = "hongccc/sglang-omni:dev"
FROZEN_RUNTIME_IMAGE_ID = (
    "sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21"
)
FROZEN_RUNTIME_REPO_DIGEST = (
    "hongccc/sglang-omni@sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
)
FROZEN_SERVER_CONFIG_IMAGE = "causalcache-androidworld:11cea575-executor1"
FROZEN_SERVER_IMAGE_ID = (
    "sha256:542e11e5d263ddcd3dffc52c5be2cb2aca0b1f08bbcf2120cecb8150b8d51486"
)
FROZEN_LIVE_SOURCE_SHA256 = {
    "/server/android_server.py": "3433a814b6753490c6af0bb24cadb68648dd78b133aac288fdf63b6d5e2a928e",
    "/android_world/agents/new_json_action.py": "14ca00cabf3d5b83e4d55cb683a09a4beccbbc658e21039ca5cf8cef3f543e3f",
    "/android_world/env/actuation.py": "69d1190d1c6dd250d1ca228ea91ffb5fea3fd133b21995b3b298b500612ed925",
    "/android_world/env/interface.py": "1772e5485225f4e52dc68954aa5769d4e49a9e63174f9ec10f3e52560d5646c2",
}
CONTAINER_NAME_PATTERN = re.compile(r"sglang-omni-jaxan-[0-9]{8}")
ATTEMPT_ID_PATTERN = re.compile(r"rv2-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}")


def _run_json(command: list[str]) -> Any:
    output = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def _inspect_container(name: str) -> dict[str, Any]:
    value = _run_json(["docker", "inspect", name])
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f"docker inspect did not return one container: {name}")
    return value[0]


def _inspect_image(image_id: str) -> dict[str, Any]:
    value = _run_json(["docker", "image", "inspect", image_id])
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f"docker image inspect did not return one image: {image_id}")
    return value[0]


def _live_source_hashes(container_name: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in FROZEN_LIVE_SOURCE_SHA256:
        output = subprocess.run(
            ["docker", "exec", container_name, "sha256sum", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        digest, returned_path = output.split(maxsplit=1)
        if returned_path != path:
            raise ValueError(f"sha256sum returned an unexpected path: {returned_path}")
        hashes[path] = digest
    if hashes != FROZEN_LIVE_SOURCE_SHA256:
        raise ValueError("live AndroidWorld executor source differs from the frozen hashes")
    return hashes


def _health(host_port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{host_port}/health"
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=30.0) as response:
        raw = response.read()
        status = int(response.status)
        content_type = response.headers.get_content_type()
    finished_at = datetime.now(timezone.utc)
    decoded = json.loads(raw)
    if status != 200 or content_type != "application/json" or decoded != {"status": "success"}:
        raise ValueError("live AndroidWorld health endpoint did not exactly pass")
    return {
        "url": url,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "http_status": status,
        "content_type": content_type,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_body_utf8": raw.decode("utf-8"),
        "json_body": decoded,
    }


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    host = subprocess.run(
        ["hostname"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if host != FROZEN_HOST or args.host_name != FROZEN_HOST:
        raise ValueError(f"executor container inspection must run on {FROZEN_HOST}")
    if ATTEMPT_ID_PATTERN.fullmatch(args.attempt_id) is None:
        raise ValueError("attempt ID must use rv2-YYYYMMDDTHHMMSSZ-8hex")
    if args.attempt_id not in args.output.name or args.phase not in args.output.name:
        raise ValueError("inspection output filename must contain attempt ID and phase")
    for name in (args.runtime_container_name, args.server_container_name):
        if CONTAINER_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"container name violates the ownership/timestamp contract: {name}")

    runtime = _inspect_container(args.runtime_container_name)
    server = _inspect_container(args.server_container_name)
    if runtime["State"]["Running"] is not True or server["State"]["Running"] is not True:
        raise ValueError("runtime and executor containers must both be running")
    if runtime["Config"]["Image"] != FROZEN_RUNTIME_CONFIG_IMAGE:
        raise ValueError("runtime container tag differs from the frozen image")
    if runtime["Image"] != FROZEN_RUNTIME_IMAGE_ID:
        raise ValueError("runtime container image ID differs from the frozen image")
    if server["Config"]["Image"] != FROZEN_SERVER_CONFIG_IMAGE:
        raise ValueError("server container tag differs from the frozen image")
    if server["Image"] != FROZEN_SERVER_IMAGE_ID:
        raise ValueError("server container image ID differs from the frozen image")

    runtime_image = _inspect_image(runtime["Image"])
    if FROZEN_RUNTIME_REPO_DIGEST not in (runtime_image.get("RepoDigests") or []):
        raise ValueError("runtime image lacks the frozen repository digest")
    port_records = server["NetworkSettings"]["Ports"].get("5000/tcp")
    expected_bindings = {
        ("0.0.0.0", str(args.server_host_port)),
        ("::", str(args.server_host_port)),
    }
    actual_bindings = {
        (record.get("HostIp"), record.get("HostPort")) for record in port_records or []
    }
    if actual_bindings != expected_bindings:
        raise ValueError("server port binding differs from the explicit host port")

    runtime_mounts = {
        mount["Destination"]: mount["Source"] for mount in runtime.get("Mounts", [])
    }
    expected_runtime_mounts = {
        "/data": "/mnt/data6/jiaxuanluo/causalcache",
        "/root/.cache/huggingface": "/mnt/data6/jiaxuanluo/causalcache/.cache/huggingface",
    }
    if runtime_mounts != expected_runtime_mounts:
        raise ValueError("runtime container persistent mounts differ from the frozen Aries layout")

    live_source_sha256 = _live_source_hashes(args.server_container_name)
    health = _health(args.server_host_port)
    finished_at = datetime.now(timezone.utc)
    source_path = Path(__file__).resolve()
    return {
        "schema_version": "1.0.0",
        "evidence_type": "restoration_v2_live_executor_container_inspection",
        "status": "passed",
        "attempt_id": args.attempt_id,
        "phase": args.phase,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "host": host,
        "inspector_source": str(source_path),
        "inspector_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "runtime_container": {
            "name": args.runtime_container_name,
            "id": runtime["Id"],
            "hostname": runtime["Config"]["Hostname"],
            "config_image": runtime["Config"]["Image"],
            "image_id": runtime["Image"],
            "image_repo_digest": FROZEN_RUNTIME_REPO_DIGEST,
            "mounts": runtime_mounts,
            "running": runtime["State"]["Running"],
        },
        "server_container": {
            "name": args.server_container_name,
            "id": server["Id"],
            "hostname": server["Config"]["Hostname"],
            "config_image": server["Config"]["Image"],
            "image_id": server["Image"],
            "host_port": args.server_host_port,
            "container_port": 5000,
            "running": server["State"]["Running"],
        },
        "live_source_sha256": live_source_sha256,
        "health": health,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-container-name", required=True)
    parser.add_argument("--server-container-name", required=True)
    parser.add_argument("--host-name", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--phase", required=True, choices=("before", "after"))
    parser.add_argument("--server-host-port", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite inspection attempt: {args.output}")
    try:
        summary = inspect(args)
    except Exception as error:
        summary = {
            "schema_version": "1.0.0",
            "evidence_type": "restoration_v2_live_executor_container_inspection",
            "status": "failed",
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with args.output.open("x", encoding="utf-8") as output_file:
            output_file.write(rendered)
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output_file:
        output_file.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
