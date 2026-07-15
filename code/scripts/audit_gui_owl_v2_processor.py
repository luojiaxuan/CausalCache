"""Audit the frozen GUI-Owl v2 processor without loading the policy model."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import shlex
import socket
import struct
import subprocess
import sys
import zlib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causalcache.policy.gui_owl_v2 import (
    GUI_OWL_V2_SYSTEM_PROMPT,
    GUI_OWL_V2_TEACHER_CARRIER,
    GUIOwlV2Action,
)
from causalcache.policy.gui_owl_v2_runtime import (
    FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE,
    GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
    build_gui_owl_v2_teacher_tokens,
    validate_gui_owl_v2_native_messages,
)
from causalcache.policy.gui_owl_v2_vision import (
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    VISION_TEMPORAL_PATCH_SIZE,
    VerifiedVisionRuntimeIdentity,
    canonical_image_grid_thw,
    verify_frozen_vision_runtime,
    visual_token_geometry,
)


SCHEMA_VERSION = "0.1.0"
PROTOCOL_ID = "causalcache_restoration_v2"
OUTCOME = "PASSED_GUI_OWL_V2_PROCESSOR_AUDIT"
AUDIT_SOURCE_PATH = "code/scripts/audit_gui_owl_v2_processor.py"
SOURCE_PATHS = (
    "code/causalcache/policy/gui_owl_v2.py",
    "code/causalcache/policy/gui_owl_v2_runtime.py",
    "code/causalcache/policy/gui_owl_v2_vision.py",
    AUDIT_SOURCE_PATH,
)
REQUIRED_PROCESSOR_TENSORS = frozenset(
    {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
)
ALLOWED_PROCESSOR_TENSORS = REQUIRED_PROCESSOR_TENSORS.union(
    {"mm_token_type_ids"}
)
EXPECTED_PROCESSOR_DTYPES = {
    "input_ids": "torch.int64",
    "attention_mask": "torch.int64",
    "pixel_values": "torch.float32",
    "image_grid_thw": "torch.int64",
    "mm_token_type_ids": "torch.int64",
}
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
CONTAINER_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
HOST_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def deterministic_rgb_png(*, width: int, height: int, seed: int) -> bytes:
    """Build a deterministic RGB PNG using only the standard library."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("PNG width and height must be positive integers")
    if type(seed) is not int or not 0 <= seed <= 255:
        raise ValueError("PNG seed must be an integer in [0, 255]")
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(
                (
                    (seed + 17 * x + 3 * y) % 256,
                    (2 * seed + 5 * x + 11 * y) % 256,
                    (3 * seed + 13 * x + 7 * y) % 256,
                )
            )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def _load_png(image_module: Any, raw: bytes) -> Any:
    image = image_module.open(io.BytesIO(raw))
    image.load()
    converted = image.convert("RGB")
    if converted is not image and hasattr(image, "close"):
        image.close()
    return converted


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git command failed: {stderr or arguments}") from error


def _git_text(repository_root: Path, *arguments: str) -> str:
    return _git_bytes(repository_root, *arguments).decode("utf-8").strip()


def _repository_identity(repository_root: Path, run_git_commit: str) -> dict[str, Any]:
    root = repository_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("repository root must be an existing directory")
    observed_root = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if observed_root != root:
        raise ValueError("--repository-root must be the exact Git worktree root")
    head = _git_text(root, "rev-parse", "HEAD")
    if GIT_SHA_PATTERN.fullmatch(run_git_commit) is None or head != run_git_commit:
        raise ValueError("--run-git-commit must exactly equal the full Git HEAD")
    verified = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    if verified != head:
        raise ValueError("Git HEAD commit verification drifted")
    status = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ValueError("formal processor audit requires a clean Git worktree")
    return {
        "repository_root": str(root),
        "run_git_commit": head,
        "head_verified_as_commit": True,
        "worktree_clean": True,
        "untracked_files_checked": True,
        "submodules_checked": True,
    }


def _source_inventory(repository_root: Path, run_git_commit: str) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for relative in SOURCE_PATHS:
        live_path = (repository_root / relative).resolve()
        if not live_path.is_file() or not live_path.is_relative_to(repository_root):
            raise ValueError(f"audited source path is missing or unsafe: {relative}")
        live = live_path.read_bytes()
        committed = _git_bytes(repository_root, "show", f"{run_git_commit}:{relative}")
        if live != committed:
            raise ValueError(f"audited source bytes differ from run commit: {relative}")
        inventory[relative] = {
            "sha256": _sha256_bytes(committed),
            "size_bytes": len(committed),
        }
    if Path(__file__).resolve() != (repository_root / AUDIT_SOURCE_PATH).resolve():
        raise ValueError("processor audit CLI is not the checked-out source")
    return inventory


def _validate_explicit_identity(args: argparse.Namespace) -> None:
    if GIT_SHA_PATTERN.fullmatch(args.run_git_commit) is None:
        raise ValueError("--run-git-commit must be a full lowercase 40-hex SHA")
    if CONTAINER_ID_PATTERN.fullmatch(args.container_id) is None:
        raise ValueError("--container-id must be a full lowercase 64-hex ID")
    if IMAGE_DIGEST_PATTERN.fullmatch(args.container_image_digest) is None:
        raise ValueError("--container-image-digest must use sha256:<64 lowercase hex>")
    for field in ("host_alias", "host_hostname"):
        if HOST_PATTERN.fullmatch(getattr(args, field)) is None:
            raise ValueError(f"--{field.replace('_', '-')} is invalid")


def _runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    container_hostname = socket.gethostname()
    if not args.container_id.startswith(container_hostname):
        raise ValueError("container hostname is not a prefix of --container-id")
    return {
        "host_alias": args.host_alias,
        "host_hostname": args.host_hostname,
        "container_id": args.container_id,
        "container_hostname": container_hostname,
        "container_image_digest": args.container_image_digest,
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_distribution_version": importlib.metadata.version("torch"),
        "pillow_version": importlib.metadata.version("Pillow"),
        "processor_device": "cpu",
    }


def _tensor_shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"processor output {name} is not tensor-like")
    try:
        result = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise TypeError(f"processor output {name} has a non-integer shape") from error
    if not result or any(dimension <= 0 for dimension in result):
        raise ValueError(f"processor output {name} has an empty or invalid shape")
    return result


def _host_list(value: Any, name: str) -> Any:
    if not hasattr(value, "detach") or not hasattr(value, "tolist"):
        raise TypeError(f"processor output {name} is not a tensor")
    detached = value.detach()
    if hasattr(detached, "to"):
        detached = detached.to(device="cpu")
    return detached.tolist()


def _tensor_inventory(model_inputs: Mapping[str, Any]) -> dict[str, Any]:
    keys = frozenset(model_inputs)
    missing = REQUIRED_PROCESSOR_TENSORS.difference(keys)
    unknown = keys.difference(ALLOWED_PROCESSOR_TENSORS)
    if missing or unknown:
        raise ValueError(
            f"processor tensor inventory drifted; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    inventory: dict[str, Any] = {}
    for name in sorted(model_inputs):
        value = model_inputs[name]
        inventory[name] = {
            "shape": list(_tensor_shape(value, name)),
            "dtype": str(getattr(value, "dtype", "")),
            "device": str(getattr(value, "device", "")),
            "requires_grad": bool(getattr(value, "requires_grad", False)),
        }
        if not inventory[name]["dtype"] or not inventory[name]["device"]:
            raise ValueError(f"processor tensor {name} lacks dtype or device identity")
        if inventory[name]["dtype"] != EXPECTED_PROCESSOR_DTYPES[name]:
            raise ValueError(f"processor tensor {name} dtype drifted")
        if inventory[name]["device"] != "cpu":
            raise ValueError(f"processor-only audit requires CPU tensor {name}")
        if inventory[name]["requires_grad"]:
            raise ValueError(f"processor tensor {name} unexpectedly requires gradients")
    return inventory


def _native_messages(images: Sequence[Any], label: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": f"Processor audit fixture: {label}."})
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": GUI_OWL_V2_SYSTEM_PROMPT}],
        },
        {"role": "user", "content": content},
    ]
    validate_gui_owl_v2_native_messages(messages)
    return messages


def _tokenizer_boundary_audit(tokenizer: Any) -> dict[str, Any]:
    teacher = build_gui_owl_v2_teacher_tokens(
        tokenizer,
        GUIOwlV2Action(action="wait"),
    )
    joint_text = (
        GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX
        + GUI_OWL_V2_TEACHER_CARRIER
        + teacher.distance_text
    )
    joint_ids = tuple(tokenizer.encode(joint_text, add_special_tokens=False))
    expected = (
        *teacher.assistant_prefix_token_ids,
        *teacher.carrier_token_ids,
        *teacher.distance_token_ids,
    )
    if joint_ids != expected:
        raise ValueError("tokenizer merges across the assistant-prefix/carrier/tool boundary")
    return {
        "status": "passed",
        "action": "wait",
        "assistant_prefix_text": teacher.assistant_prefix_text,
        "carrier_text": teacher.carrier_text,
        "distance_text": teacher.distance_text,
        "assistant_prefix_tokens": len(teacher.assistant_prefix_token_ids),
        "carrier_tokens": len(teacher.carrier_token_ids),
        "distance_tokens": len(teacher.distance_token_ids),
        "joint_boundary_exact": True,
    }


def _case_audit(
    processor: Any,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    *,
    expected_image_counts: tuple[int, ...],
    name: str,
) -> dict[str, Any]:
    encoded = processor.apply_chat_template(
        list(conversations),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=False,
    )
    model_inputs = dict(encoded)
    inventory = _tensor_inventory(model_inputs)
    input_shape = _tensor_shape(model_inputs["input_ids"], "input_ids")
    if len(input_shape) != 2 or input_shape[0] != len(conversations):
        raise ValueError(f"{name} processor input_ids batch shape drifted")
    attention_shape = _tensor_shape(model_inputs["attention_mask"], "attention_mask")
    if attention_shape != input_shape:
        raise ValueError(f"{name} attention_mask shape differs from input_ids")
    attention = _host_list(model_inputs["attention_mask"], "attention_mask")
    if attention != [[1] * input_shape[1] for _ in conversations]:
        raise ValueError(f"{name} contains padding despite padding=False")
    prefix_ids = tuple(
        processor.tokenizer.encode(
            GUI_OWL_V2_ASSISTANT_GENERATION_PREFIX,
            add_special_tokens=False,
        )
    )
    if not prefix_ids:
        raise ValueError("assistant generation prefix tokenization is empty")
    input_ids = _host_list(model_inputs["input_ids"], "input_ids")
    if any(tuple(row[-len(prefix_ids) :]) != prefix_ids for row in input_ids):
        raise ValueError(f"{name} prompt does not end in the native assistant prefix")
    grids = canonical_image_grid_thw(model_inputs["image_grid_thw"])
    if len(grids) != sum(expected_image_counts):
        raise ValueError(f"{name} image_grid_thw count differs from image inventory")
    if _tensor_shape(model_inputs["image_grid_thw"], "image_grid_thw") != (
        len(grids),
        3,
    ):
        raise ValueError(f"{name} image_grid_thw tensor shape drifted")
    pixel_shape = _tensor_shape(model_inputs["pixel_values"], "pixel_values")
    raw_patch_count = sum(t * h * w for t, h, w in grids)
    patch_vector_size = (
        3 * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE
    )
    if pixel_shape != (raw_patch_count, patch_vector_size):
        raise ValueError(f"{name} pixel_values patch dimension differs from image grids")
    if "mm_token_type_ids" in model_inputs and _tensor_shape(
        model_inputs["mm_token_type_ids"], "mm_token_type_ids"
    ) != input_shape:
        raise ValueError(f"{name} mm_token_type_ids shape differs from input_ids")
    samples: list[dict[str, Any]] = []
    offset = 0
    for image_count in expected_image_counts:
        sample_grids = grids[offset : offset + image_count]
        offset += image_count
        geometry = visual_token_geometry(sample_grids)
        merged_counts = geometry["merged_token_counts"]
        assert isinstance(merged_counts, tuple)
        effective = sum(merged_counts)
        text_tokens = input_shape[1] - effective
        if text_tokens <= 0:
            raise ValueError(f"{name} policy-visible text token count is not positive")
        samples.append(
            {
                "image_count": image_count,
                "image_grid_thw": [list(grid) for grid in sample_grids],
                "effective_visual_tokens_per_image": list(merged_counts),
                "effective_visual_tokens": effective,
                "policy_visible_text_tokens": text_tokens,
                "sequence_length": input_shape[1],
            }
        )
    equal_image_count = len(set(expected_image_counts)) == 1
    equal_sequence_length = len({sample["sequence_length"] for sample in samples}) == 1
    if len(conversations) == 2 and (not equal_image_count or not equal_sequence_length):
        raise ValueError("nested batch-2 requires equal image count and sequence length")
    return {
        "status": "passed",
        "case": name,
        "batch_size": len(conversations),
        "image_counts": list(expected_image_counts),
        "padding": False,
        "no_padding": True,
        "equal_image_count": equal_image_count,
        "equal_sequence_length": equal_sequence_length,
        "attention_mask_all_one": True,
        "assistant_prefix_tail_exact": True,
        "tensor_inventory": inventory,
        "samples": samples,
    }


def _model_identity(identity: VerifiedVisionRuntimeIdentity) -> dict[str, Any]:
    return {
        "model_dir": identity.model_dir,
        "model_repo": identity.model_repo,
        "model_revision": identity.model_revision,
        "snapshot_manifest_sha256": identity.snapshot_manifest_sha256,
        "verified_model_file_count": identity.verified_model_file_count,
        "verified_model_total_bytes": identity.verified_model_total_bytes,
        "transformers_version": identity.transformers_version,
        "transformers_source_sha256": dict(identity.transformers_source_sha256),
    }


def run_processor_audit(
    *,
    model_dir: Path,
    expected_snapshot_manifest: Path,
    runtime_verifier: Callable[..., VerifiedVisionRuntimeIdentity] = verify_frozen_vision_runtime,
    auto_processor_cls: Any | None = None,
    image_module: Any | None = None,
) -> dict[str, Any]:
    identity = runtime_verifier(
        model_dir=model_dir,
        expected_snapshot_manifest=expected_snapshot_manifest,
    )
    if auto_processor_cls is None or image_module is None:
        try:
            from PIL import Image
            from transformers import AutoProcessor
        except ModuleNotFoundError as error:
            raise RuntimeError("processor audit requires Pillow and Transformers") from error
        auto_processor_cls = AutoProcessor
        image_module = Image
    target_pixels = FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE * (
        VISION_PATCH_SIZE * VISION_SPATIAL_MERGE_SIZE
    ) ** 2
    processor = auto_processor_cls.from_pretrained(
        identity.model_dir,
        min_pixels=target_pixels,
        max_pixels=target_pixels,
        local_files_only=True,
    )
    image_processor = getattr(processor, "image_processor", None)
    actual_min_pixels = getattr(image_processor, "min_pixels", None)
    actual_max_pixels = getattr(image_processor, "max_pixels", None)
    actual_merge_size = getattr(image_processor, "merge_size", None)
    if (
        actual_min_pixels != target_pixels
        or actual_max_pixels != target_pixels
        or actual_merge_size != VISION_SPATIAL_MERGE_SIZE
    ):
        raise ValueError("actual processor min/max target or merge size drifted")
    portrait_raw = deterministic_rgb_png(width=108, height=240, seed=37)
    landscape_raw = deterministic_rgb_png(width=240, height=108, seed=91)
    portrait = _load_png(image_module, portrait_raw)
    landscape = _load_png(image_module, landscape_raw)
    try:
        boundary = _tokenizer_boundary_audit(processor.tokenizer)
        one = _case_audit(
            processor,
            [_native_messages((portrait,), "one-portrait")],
            expected_image_counts=(1,),
            name="single_conversation_one_image",
        )
        five_images = (portrait, landscape, portrait, landscape, portrait)
        five = _case_audit(
            processor,
            [_native_messages(five_images, "five-images")],
            expected_image_counts=(5,),
            name="single_conversation_five_images",
        )
        batch = _case_audit(
            processor,
            [
                _native_messages((portrait,), "batch-two-equal-shape"),
                _native_messages((landscape,), "batch-two-equal-shape"),
            ],
            expected_image_counts=(1, 1),
            name="nested_batch_two_equal_shape",
        )
        if batch["samples"][0]["sequence_length"] != batch["samples"][1]["sequence_length"]:
            raise ValueError("nested batch-2 sequence lengths drifted")
        return {
            "status": "passed",
            "model_identity": _model_identity(identity),
            "processor_identity": {
                "processor_class": processor.__class__.__name__,
                "tokenizer_class": processor.tokenizer.__class__.__name__,
                "target_effective_visual_tokens_per_image": (
                    FROZEN_GUI_OWL_V2_EFFECTIVE_VISUAL_TOKENS_PER_IMAGE
                ),
                "target_pixels_per_image": target_pixels,
                "actual_min_pixels": int(actual_min_pixels),
                "actual_max_pixels": int(actual_max_pixels),
                "actual_min_equals_max_equals_target": True,
                "spatial_merge_size": int(actual_merge_size),
                "local_files_only": True,
            },
            "deterministic_images": {
                "generator": "stdlib_rgb_scanlines_png_zlib_level_9",
                "portrait": {
                    "width": 108,
                    "height": 240,
                    "sha256": _sha256_bytes(portrait_raw),
                    "size_bytes": len(portrait_raw),
                },
                "landscape": {
                    "width": 240,
                    "height": 108,
                    "sha256": _sha256_bytes(landscape_raw),
                    "size_bytes": len(landscape_raw),
                },
            },
            "tokenizer_boundary": boundary,
            "single_conversation_one_image": one,
            "single_conversation_five_images": five,
            "nested_batch_two_equal_shape": batch,
            "auto_processor_only": True,
            "auto_processor_loaded": True,
            "model_weights_loaded": False,
            "pretrained_loader_calls": ["AutoProcessor.from_pretrained"],
            "model_weight_files_sha256_verified": True,
            "model_weights_materialized_as_tensors": False,
            "policy_model_loaded": False,
            "policy_loaded": False,
            "policy_forward_executed": False,
            "policy_generate_executed": False,
            "policy_output": False,
            "policy_output_generated": False,
            "restoration_output_generated": False,
        }
    finally:
        for image in (portrait, landscape):
            if hasattr(image, "close"):
                image.close()


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(serialized)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-git-commit", required=True)
    parser.add_argument("--host-alias", required=True)
    parser.add_argument("--host-hostname", required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--container-image-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv if argv is None else [str(Path(__file__)), *argv])
    args = _parser().parse_args(None if argv is None else list(argv))
    _validate_explicit_identity(args)
    started_at = _utc_now()
    repository_root = args.repository_root.expanduser().resolve()
    repository = _repository_identity(repository_root, args.run_git_commit)
    sources = _source_inventory(repository_root, args.run_git_commit)
    runtime = _runtime_identity(args)
    audit = run_processor_audit(
        model_dir=args.model_dir,
        expected_snapshot_manifest=args.expected_snapshot_manifest,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "evidence_type": "gui_owl_v2_policy_output_free_processor_audit",
        "outcome": OUTCOME,
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "argv": raw_argv,
        "argv_shell_quoted": shlex.join(raw_argv),
        "repository": repository,
        "source_files": sources,
        "runtime_identity": runtime,
        "processor_audit": audit,
        "auto_processor_only": True,
        "auto_processor_loaded": True,
        "model_weights_loaded": False,
        "pretrained_loader_calls": ["AutoProcessor.from_pretrained"],
        "model_weight_files_sha256_verified": True,
        "model_weights_materialized_as_tensors": False,
        "policy_model_loaded": False,
        "policy_loaded": False,
        "policy_forward_executed": False,
        "policy_generate_executed": False,
        "policy_output": False,
        "policy_output_generated": False,
        "restoration_output_generated": False,
    }
    _write_exclusive(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
