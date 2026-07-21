"""Crash-safe immutable resume generations for distributed utility training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


RESUME_GENERATION_STATUS = "COMPLETED_SET_UTILITY_RESUME_GENERATION"
RESUME_GENERATION_SCHEMA = "causalcache.set_utility_resume_generation.v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    return result


def _read_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("resume latest pointer must be a JSON object")
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ValueError("resume latest pointer signature drifted")
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def resume_generation_path(output_root: Path, *, epoch: int, rank: int) -> Path:
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("resume generation epoch must be positive")
    if type(rank) is not int or rank < 0:
        raise ValueError("resume generation rank must be nonnegative")
    return output_root / "resume" / f"epoch-{epoch:04d}" / f"rank-{rank:02d}.pt"


def resume_latest_path(output_root: Path) -> Path:
    return output_root / "resume" / "latest.json"


def has_published_resume_generation(output_root: Path) -> bool:
    return resume_latest_path(output_root).is_file()


def _update_state_digest(digest: Any, value: Any, *, torch: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor:")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        for key in sorted(value, key=lambda item: repr(item)):
            digest.update(repr(key).encode())
            _update_state_digest(digest, value[key], torch=torch)
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence:")
        for item in value:
            _update_state_digest(digest, item, torch=torch)
        return
    digest.update(type(value).__name__.encode())
    digest.update(repr(value).encode())


def state_digest(value: Any, *, torch: Any) -> str:
    digest = hashlib.sha256()
    _update_state_digest(digest, value, torch=torch)
    return digest.hexdigest()


def optimizer_step_inventory(
    state: Mapping[str, Any], *, torch: Any
) -> list[int]:
    result = []
    for value in state.get("state", {}).values():
        step = value.get("step") if isinstance(value, Mapping) else None
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError("optimizer step tensor is not scalar")
            step = step.item()
        if step is not None:
            numeric = float(step)
            if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                raise ValueError("optimizer step is invalid")
            result.append(int(numeric))
    return sorted(set(result))


def resume_rank_signature(
    snapshot: Mapping[str, Any], *, rank: int, torch: Any
) -> dict[str, Any]:
    scheduler = snapshot.get("scheduler")
    optimizer = snapshot.get("optimizer")
    model = snapshot.get("model")
    epoch = snapshot.get("epoch")
    stored_rank = snapshot.get("rank", rank)
    identity = snapshot.get("identity", {})
    if (
        type(epoch) is not int
        or epoch <= 0
        or stored_rank != rank
        or not isinstance(identity, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(optimizer, Mapping)
        or not isinstance(scheduler, Mapping)
    ):
        raise ValueError("resume snapshot is incomplete")
    return {
        "epoch": epoch,
        "identity_sha256": hashlib.sha256(
            _canonical_json_bytes(dict(identity))
        ).hexdigest(),
        "model_sha256": state_digest(model, torch=torch),
        "optimizer_sha256": state_digest(optimizer, torch=torch),
        "optimizer_steps": optimizer_step_inventory(optimizer, torch=torch),
        "rank": rank,
        "scheduler_last_epoch": int(scheduler.get("last_epoch", -1)),
        "scheduler_sha256": state_digest(scheduler, torch=torch),
        "scheduler_step_count": int(scheduler.get("_step_count", -1)),
        "status": "ok",
    }


def validate_resume_rank_signatures(
    signatures: Sequence[Mapping[str, Any]], *, world_size: int
) -> dict[str, Any]:
    if len(signatures) != world_size or world_size <= 0:
        raise ValueError("DDP resume rank inventory is incomplete")
    if {row.get("rank") for row in signatures} != set(range(world_size)):
        raise ValueError("DDP resume rank identities drifted")
    if any(row.get("status") != "ok" for row in signatures):
        raise ValueError("DDP resume snapshot load failed on a rank")
    comparable = [
        {key: value for key, value in row.items() if key != "rank"}
        for row in sorted(signatures, key=lambda value: int(value["rank"]))
    ]
    if any(value != comparable[0] for value in comparable[1:]):
        raise ValueError("DDP resume epoch/model/optimizer/scheduler rank drift")
    return comparable[0]


def write_resume_rank_snapshot(
    output_root: Path,
    *,
    epoch: int,
    rank: int,
    world_size: int,
    snapshot: Mapping[str, Any],
    torch: Any,
) -> dict[str, Any]:
    if world_size <= 0 or rank >= world_size:
        raise ValueError("resume generation world/rank inventory is invalid")
    latest_path = resume_latest_path(output_root)
    if latest_path.is_file() and int(
        _read_signed(latest_path).get("epoch", -1)
    ) >= epoch:
        raise ValueError("published resume generation rank snapshots are immutable")
    payload = dict(snapshot)
    payload["epoch"] = epoch
    payload["rank"] = rank
    payload["world_size"] = world_size
    signature = resume_rank_signature(payload, rank=rank, torch=torch)
    path = resume_generation_path(output_root, epoch=epoch, rank=rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "byte_count": path.stat().st_size,
        "epoch": epoch,
        "path": path.relative_to(output_root).as_posix(),
        "rank": rank,
        "sha256": _sha256_file(path),
        "signature": signature,
        "world_size": world_size,
    }


def validate_resume_generation_receipts(
    output_root: Path,
    receipts: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    world_size: int,
    verify_files: bool = True,
) -> dict[str, Any]:
    if len(receipts) != world_size or world_size <= 0:
        raise ValueError("resume generation rank inventory is incomplete")
    ordered = tuple(sorted((dict(row) for row in receipts), key=lambda row: row["rank"]))
    if tuple(row.get("rank") for row in ordered) != tuple(range(world_size)):
        raise ValueError("resume generation rank identities drifted")
    signatures = []
    for rank, row in enumerate(ordered):
        expected_path = resume_generation_path(
            output_root, epoch=epoch, rank=rank
        )
        try:
            path = (output_root / str(row["path"])).resolve()
            path.relative_to(output_root.resolve())
        except (KeyError, ValueError) as error:
            raise ValueError("resume generation path escaped output root") from error
        if (
            row.get("epoch") != epoch
            or row.get("world_size") != world_size
            or path != expected_path.resolve()
        ):
            raise ValueError("resume generation rank snapshot identity drifted")
        if verify_files and (
            not path.is_file()
            or path.stat().st_size != int(row.get("byte_count", -1))
            or _sha256_file(path) != row.get("sha256")
        ):
            raise ValueError("resume generation rank snapshot bytes drifted")
        signature = row.get("signature")
        if not isinstance(signature, Mapping):
            raise ValueError("resume generation rank signature is missing")
        signatures.append(dict(signature))
    common = validate_resume_rank_signatures(signatures, world_size=world_size)
    return {
        "common_signature": common,
        "epoch": epoch,
        "ranks": list(ordered),
        "schema_version": RESUME_GENERATION_SCHEMA,
        "status": RESUME_GENERATION_STATUS,
        "world_size": world_size,
    }


def publish_resume_generation(
    output_root: Path,
    receipts: Sequence[Mapping[str, Any]],
    *,
    epoch: int,
    world_size: int,
) -> dict[str, Any]:
    pointer = _signed(
        validate_resume_generation_receipts(
            output_root, receipts, epoch=epoch, world_size=world_size
        )
    )
    latest_path = resume_latest_path(output_root)
    if latest_path.exists():
        previous = _read_signed(latest_path)
        previous_epoch = int(previous.get("epoch", -1))
        if previous_epoch > epoch:
            raise ValueError("resume latest pointer cannot move backwards")
        if previous_epoch == epoch:
            if previous != pointer:
                raise ValueError("published resume generation is immutable")
            return previous
    _write_atomic(latest_path, pointer)
    return pointer


def read_latest_resume_generation(
    output_root: Path, *, expected_world_size: int
) -> dict[str, Any]:
    path = resume_latest_path(output_root)
    if not path.is_file():
        raise FileNotFoundError("published resume generation is missing")
    pointer = _read_signed(path)
    if (
        pointer.get("schema_version") != RESUME_GENERATION_SCHEMA
        or pointer.get("status") != RESUME_GENERATION_STATUS
        or pointer.get("world_size") != expected_world_size
    ):
        raise ValueError("resume latest pointer contract drifted")
    observed = validate_resume_generation_receipts(
        output_root,
        pointer.get("ranks", ()),
        epoch=int(pointer.get("epoch", -1)),
        world_size=expected_world_size,
        verify_files=False,
    )
    if observed["common_signature"] != pointer.get("common_signature"):
        raise ValueError("resume latest pointer common signature drifted")
    return pointer


def load_resume_rank_snapshot(
    output_root: Path,
    *,
    rank: int,
    expected_world_size: int,
    expected_identity: Mapping[str, Any],
    map_location: Any,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = read_latest_resume_generation(
        output_root, expected_world_size=expected_world_size
    )
    receipt = pointer["ranks"][rank]
    path = output_root / receipt["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(receipt["byte_count"])
        or _sha256_file(path) != receipt["sha256"]
    ):
        raise ValueError("resume rank snapshot bytes drifted")
    snapshot = torch.load(path, map_location=map_location, weights_only=False)
    if (
        snapshot.get("identity") != dict(expected_identity)
        or snapshot.get("rank") != rank
        or snapshot.get("world_size") != expected_world_size
        or snapshot.get("epoch") != pointer["epoch"]
    ):
        raise ValueError("resume snapshot identity drifted")
    signature = resume_rank_signature(snapshot, rank=rank, torch=torch)
    if signature != receipt["signature"]:
        raise ValueError("resume snapshot semantic signature drifted")
    return snapshot, pointer


def save_resume_generation_collective(
    output_root: Path,
    *,
    epoch: int,
    rank: int,
    world_size: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    identity: Mapping[str, Any],
    distributed: bool,
    torch: Any,
) -> dict[str, Any] | None:
    receipt = write_resume_rank_snapshot(
        output_root,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        snapshot={
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "identity": dict(identity),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_rng_state": random.getstate(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        },
        torch=torch,
    )
    if distributed:
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, receipt)
        receipts = tuple(value or {} for value in gathered)
    else:
        receipts = (receipt,)
    validate_resume_generation_receipts(
        output_root,
        receipts,
        epoch=epoch,
        world_size=world_size,
        verify_files=False,
    )
    outcome: dict[str, Any] | None = None
    if rank == 0:
        try:
            outcome = {
                "pointer": publish_resume_generation(
                    output_root, receipts, epoch=epoch, world_size=world_size
                ),
                "status": "ok",
            }
        except Exception as error:  # noqa: BLE001 - publish failure reaches every rank.
            outcome = {
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "error",
            }
    if distributed:
        payload = [outcome]
        torch.distributed.broadcast_object_list(payload, src=0)
        outcome = payload[0]
    if not outcome or outcome.get("status") != "ok":
        detail = "unknown" if not outcome else outcome.get("error", "unknown")
        raise RuntimeError(f"resume generation publication failed: {detail}")
    return outcome["pointer"] if rank == 0 else None


def load_resume_generation_collective(
    output_root: Path,
    *,
    rank: int,
    world_size: int,
    identity: Mapping[str, Any],
    map_location: Any,
    distributed: bool,
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot = None
    pointer = None
    try:
        snapshot, pointer = load_resume_rank_snapshot(
            output_root,
            rank=rank,
            expected_world_size=world_size,
            expected_identity=identity,
            map_location=map_location,
            torch=torch,
        )
        local_signature = resume_rank_signature(snapshot, rank=rank, torch=torch)
    except Exception as error:  # noqa: BLE001 - every rank must reach all-gather.
        local_signature = {
            "error_type": type(error).__name__,
            "rank": rank,
            "status": "error",
        }
    if distributed:
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_signature)
        signatures = tuple(value or {} for value in gathered)
    else:
        signatures = (local_signature,)
    validate_resume_rank_signatures(signatures, world_size=world_size)
    assert snapshot is not None and pointer is not None
    return snapshot, pointer


def restore_rng_states(snapshot: Mapping[str, Any], *, torch: Any) -> None:
    torch_state = snapshot.get("torch_rng_state")
    cuda_state = snapshot.get("cuda_rng_state")
    if not torch.is_tensor(torch_state) or not torch.is_tensor(cuda_state):
        raise ValueError("resume RNG states must be tensors")
    torch.set_rng_state(torch_state.detach().cpu().to(dtype=torch.uint8).contiguous())
    torch.cuda.set_rng_state(
        cuda_state.detach().cpu().to(dtype=torch.uint8).contiguous()
    )
