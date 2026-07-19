"""Read-only postflight for the isolated processor image-contract v2 repair."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from causalcache.data.guiodyssey_restoration_v2 import OCR_RECORD_KEYS
from causalcache.data.guiodyssey_independent import SourceFileSpec
from causalcache.restoration_v2_text_backend import (
    BACKEND_ID,
    OCR_RECORD_SCHEMA_VERSION,
    canonicalize_rapidocr_nodes,
)
from causalcache.set_utility_full_pool import build_full_pool_inspection_config
from causalcache.set_utility_processor_artifacts import (
    ProcessorWorkerSchedule,
    build_processor_worker_schedule,
    canonical_json_bytes,
    iter_processor_query_artifact_records,
    sha256_bytes,
)
from causalcache.set_utility_processor_freeze import (
    WORKER_COUNT,
    validate_freeze_query_topology,
    validate_processor_only_source,
)
from causalcache.set_utility_processor_freeze_contract import (
    FULL_POOL_INVENTORY_MANIFEST_PATH,
    INDEPENDENT_REFERENCE_GATE_CONFIG_PATH,
    OCR_BACKEND_CONFIG_PATH,
)
from causalcache.set_utility_processor_freeze_contract_v2 import (
    CANONICAL_EXECUTION_CONFIG_PATH,
    OCR_CONCURRENCY_PER_LOGICAL_WORKER,
    PROCESSOR_CONCURRENCY_PER_LOGICAL_WORKER,
    PROCESSOR_POST_LOAD_MODELING_MODULE_ALLOWLIST,
    PROCESSOR_TORCH_AMBIENT_ENVIRONMENT_KEYS,
    PROCESSOR_TORCH_INTEROP_THREADS,
    PROCESSOR_TORCH_INTRAOP_THREADS,
    RUNNER_PATH,
    ProcessorFreezeExecutionContractV2,
)
from causalcache.set_utility_processor_image_contract_v2 import (
    PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS,
    PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS,
    PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL,
    PROCESSOR_IMAGE_CONTRACT_V2_ID,
    validate_processor_ocr_record_v2,
)
from causalcache.set_utility_processor_postflight import (
    ExpectedFrozenQuery,
    ProcessorFreezePostflightContext,
    _load_bound_json,
    _ocr_runtime_identity,
    _snapshot_identity_without_model_dir,
    _strict_json_object,
    validate_completed_processor_freeze_root as _validate_completed_processor_freeze_root_v1,
)
from causalcache.set_utility_processor_substrate import build_selected_row_read_plan


SCHEMA_VERSION = "2.0.0"
VALIDATION_STATUS = (
    "VALID_COMPLETED_SET_UTILITY_PROCESSOR_FREEZE_V2_IMAGE_CONTRACT_REPAIR"
)
V1_RUNNER_PATH = "code/scripts/run_set_utility_processor_freeze.py"
IMAGE_CONTRACT_PATH = (
    "code/causalcache/set_utility_processor_image_contract_v2.py"
)
FULL_POOL_CENSUS_PATH = "data/manifests/set_utility_full_pool_census_v2.json"
OUTPUT_BASENAME_PREFIX = (
    "causalcache-set-utility-processor-freeze-v2-image-contract-repair-"
)
PROCESSOR_THREAD_RUNTIME_LOG_STATUS = (
    "VALID_PROCESSOR_TORCH_THREAD_RUNTIME_LOG_EVIDENCE"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_IMAGE_MUTATION_CALLS = frozenset(
    {
        "convert",
        "exif_transpose",
        "putalpha",
        "resize",
        "rotate",
        "save",
        "thumbnail",
        "transform",
        "transpose",
    }
)
_FORBIDDEN_V1_EXECUTION_HELPERS = frozenset(
    {
        "ProcessorLengthRuntime",
        "_orchestrate",
        "_run_ocr_worker",
        "_run_processor_worker",
    }
)
_REQUIRED_V2_IMAGE_INTERFACES = frozenset(
    {
        "build_validated_ocr_batch_v2",
        "decode_processor_image_v2",
        "prepare_processor_image_v2",
        "run_processor_rapidocr_record_v2",
        "validate_processor_image_v2",
    }
)


@dataclass(frozen=True)
class ProcessorFreezePostflightContextV2:
    structural_context: ProcessorFreezePostflightContext
    backend_config: Mapping[str, Any]
    backend_config_sha256: str
    image_contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.structural_context, ProcessorFreezePostflightContext):
            raise TypeError("v2 postflight structural context is invalid")
        if not isinstance(self.backend_config, Mapping):
            raise TypeError("v2 postflight backend config is invalid")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (self.backend_config_sha256, self.image_contract_sha256)
        ):
            raise ValueError("v2 postflight contract SHA256 is invalid")
        if (
            self.structural_context.ocr_backend_config_sha256
            != self.backend_config_sha256
        ):
            raise ValueError("v2 structural/backend config SHA256 binding drifted")
        identity = self.structural_context.ocr_runtime_identity
        if identity.get("image_contract_sha256") != self.image_contract_sha256:
            raise ValueError("v2 OCR identity is not bound to the image contract")
        if identity.get("ocr_concurrency_per_logical_worker") != (
            OCR_CONCURRENCY_PER_LOGICAL_WORKER
        ):
            raise ValueError("v2 OCR identity concurrency drifted")


class _SourceCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.imported: set[str] = set()
        self.forbidden_imports: list[tuple[str, str]] = []
        self.attributes: list[
            tuple[tuple[str, ...], tuple[str, ...], str, str, ast.Attribute]
        ] = []
        self.forbidden_getattrs: list[tuple[str, str]] = []
        self.calls: list[
            tuple[tuple[str, ...], tuple[str, ...], str, str, ast.Call]
        ] = []

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imported.add(alias.name)
            if alias.name in (
                _FORBIDDEN_IMAGE_MUTATION_CALLS
                | _FORBIDDEN_V1_EXECUTION_HELPERS
            ):
                self.forbidden_imports.append(
                    (
                        alias.name,
                        f"{node.module}.{alias.name} as {alias.asname or alias.name}",
                    )
                )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            base = node.func.value
            base_name = base.id if isinstance(base, ast.Name) else ast.dump(base)
            self.calls.append(
                (
                    tuple(self.classes),
                    tuple(self.functions),
                    base_name,
                    node.func.attr,
                    node,
                )
            )
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value
            in (_FORBIDDEN_IMAGE_MUTATION_CALLS | _FORBIDDEN_V1_EXECUTION_HELPERS)
        ):
            name = str(node.args[1].value)
            self.forbidden_getattrs.append((name, f"getattr(..., {name!r})"))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        base = node.value
        base_name = base.id if isinstance(base, ast.Name) else ast.dump(base)
        self.attributes.append(
            (
                tuple(self.classes),
                tuple(self.functions),
                base_name,
                node.attr,
                node,
            )
        )
        self.generic_visit(node)


def _source_imports_and_calls(
    path: Path,
) -> tuple[
    set[str],
    list[tuple[str, str]],
    list[tuple[tuple[str, ...], tuple[str, ...], str, str, ast.Attribute]],
    list[tuple[str, str]],
    list[tuple[tuple[str, ...], tuple[str, ...], str, str, ast.Call]],
]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _SourceCallVisitor()
    visitor.visit(tree)
    return (
        visitor.imported,
        visitor.forbidden_imports,
        visitor.attributes,
        visitor.forbidden_getattrs,
        visitor.calls,
    )


def _is_exact_transient_rgb_convert(
    call: tuple[tuple[str, ...], tuple[str, ...], str, str, ast.Call],
) -> bool:
    classes, functions, base, name, node = call
    return (
        classes == ("ProcessorLengthRuntimeV2",)
        and functions == ("length", "decode")
        and base == "source"
        and name == "convert"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "RGB"
        and not node.keywords
    )


def _module_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"v2 runner requires exactly one {name} function")
    return matches[0]


def _class_method(
    tree: ast.Module,
    *,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef:
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"v2 runner requires exactly one {class_name} class")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(
            f"v2 runner requires exactly one {class_name}.{method_name} method"
        )
    return methods[0]


def _named_call_count(node: ast.AST, name: str) -> int:
    return sum(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == name
        for candidate in ast.walk(node)
    )


def _attribute_call_count(node: ast.AST, *, base: str, name: str) -> int:
    return sum(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and isinstance(candidate.func.value, ast.Name)
        and candidate.func.value.id == base
        and candidate.func.attr == name
        for candidate in ast.walk(node)
    )


def _first_named_call_line(node: ast.AST, name: str) -> int:
    lines = [
        candidate.lineno
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == name
    ]
    if len(lines) != 1:
        raise ValueError(f"v2 runner requires exactly one {name} call at this site")
    return lines[0]


def _validate_processor_execution_source_v2(
    path: Path,
    *,
    imported: set[str],
) -> dict[str, Any]:
    required_imports = {
        "PROCESSOR_CONCURRENCY_PER_LOGICAL_WORKER",
        "PROCESSOR_POST_LOAD_MODELING_MODULE_ALLOWLIST",
        "PROCESSOR_TORCH_AMBIENT_ENVIRONMENT_KEYS",
        "PROCESSOR_TORCH_INTEROP_THREADS",
        "PROCESSOR_TORCH_INTRAOP_THREADS",
    }
    missing_imports = sorted(required_imports - imported)
    if missing_imports:
        raise ValueError(
            "v2 processor execution contract imports are missing: "
            f"{missing_imports}"
        )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    initializer = _class_method(
        tree,
        class_name="ProcessorLengthRuntimeV2",
        method_name="__init__",
    )
    length = _class_method(
        tree,
        class_name="ProcessorLengthRuntimeV2",
        method_name="length",
    )
    runtime_map = _module_function(tree, "_bounded_ordered_runtime_map")
    thread_configure = _module_function(
        tree,
        "_configure_processor_torch_threads",
    )
    pool_builder = _module_function(tree, "_build_processor_runtime_pool")
    worker = _module_function(tree, "_run_processor_worker")
    launcher = _module_function(tree, "_launch_workers")
    observed = {
        "initializer_pre_load_guard_call_count": _named_call_count(
            initializer,
            "assert_processor_only_import_state",
        ),
        "initializer_post_load_guard_call_count": _named_call_count(
            initializer,
            "assert_processor_v2_post_load_import_state",
        ),
        "length_pre_load_guard_call_count": _named_call_count(
            length,
            "assert_processor_only_import_state",
        ),
        "length_post_load_guard_call_count": _named_call_count(
            length,
            "assert_processor_v2_post_load_import_state",
        ),
        "runtime_map_bounded_map_call_count": _named_call_count(
            runtime_map,
            "_bounded_ordered_parallel_map",
        ),
        "pool_builder_runtime_call_count": _named_call_count(
            pool_builder,
            "ProcessorLengthRuntimeV2",
        ),
        "worker_pool_builder_call_count": _named_call_count(
            worker,
            "_build_processor_runtime_pool",
        ),
        "worker_runtime_map_call_count": _named_call_count(
            worker,
            "_bounded_ordered_runtime_map",
        ),
        "worker_thread_configure_call_count": _named_call_count(
            worker,
            "_configure_processor_torch_threads",
        ),
        "thread_set_intraop_call_count": _attribute_call_count(
            thread_configure,
            base="torch",
            name="set_num_threads",
        ),
        "thread_set_interop_call_count": _attribute_call_count(
            thread_configure,
            base="torch",
            name="set_num_interop_threads",
        ),
        "thread_get_intraop_call_count": _attribute_call_count(
            thread_configure,
            base="torch",
            name="get_num_threads",
        ),
        "thread_get_interop_call_count": _attribute_call_count(
            thread_configure,
            base="torch",
            name="get_num_interop_threads",
        ),
        "launcher_environment_pop_call_count": _attribute_call_count(
            launcher,
            base="environment",
            name="pop",
        ),
    }
    expected = {
        "initializer_pre_load_guard_call_count": 1,
        "initializer_post_load_guard_call_count": 2,
        "length_pre_load_guard_call_count": 0,
        "length_post_load_guard_call_count": 1,
        "runtime_map_bounded_map_call_count": 1,
        "pool_builder_runtime_call_count": 1,
        "worker_pool_builder_call_count": 1,
        "worker_runtime_map_call_count": 1,
        "worker_thread_configure_call_count": 1,
        "thread_set_intraop_call_count": 1,
        "thread_set_interop_call_count": 1,
        "thread_get_intraop_call_count": 1,
        "thread_get_interop_call_count": 1,
        "launcher_environment_pop_call_count": 2,
    }
    if observed != expected:
        raise ValueError(
            "v2 processor execution guard or concurrency source contract drifted: "
            f"expected={expected!r}, observed={observed!r}"
        )
    if _first_named_call_line(
        worker,
        "_configure_processor_torch_threads",
    ) >= _first_named_call_line(worker, "_build_processor_runtime_pool"):
        raise ValueError(
            "v2 processor torch threads must be configured before runtime construction"
        )
    return {
        **observed,
        "processor_concurrency_per_logical_worker": (
            PROCESSOR_CONCURRENCY_PER_LOGICAL_WORKER
        ),
        "processor_post_load_modeling_module_allowlist": list(
            PROCESSOR_POST_LOAD_MODELING_MODULE_ALLOWLIST
        ),
        "processor_torch_ambient_environment_keys_removed": list(
            PROCESSOR_TORCH_AMBIENT_ENVIRONMENT_KEYS
        ),
        "processor_torch_interop_threads": PROCESSOR_TORCH_INTEROP_THREADS,
        "processor_torch_intraop_threads": PROCESSOR_TORCH_INTRAOP_THREADS,
    }


def validate_processor_only_source_v2(
    repository_root: str | Path,
) -> dict[str, Any]:
    """Recursively bind both runners while excluding v1 image execution paths."""
    root = Path(repository_root).resolve()
    v2_path = root / RUNNER_PATH
    v1_path = root / V1_RUNNER_PATH
    image_path = root / IMAGE_CONTRACT_PATH
    for path, label in (
        (v2_path, "v2 runner"),
        (v1_path, "v1 runner"),
        (image_path, "v2 image contract"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} must be one regular source file")

    v2_audit = validate_processor_only_source(v2_path)
    v1_audit = validate_processor_only_source(v1_path)
    (
        imported,
        v2_forbidden_imports,
        v2_attributes,
        v2_forbidden_getattrs,
        v2_calls,
    ) = _source_imports_and_calls(v2_path)
    (
        _,
        image_forbidden_imports,
        image_attributes,
        image_forbidden_getattrs,
        image_calls,
    ) = _source_imports_and_calls(image_path)
    missing_interfaces = sorted(_REQUIRED_V2_IMAGE_INTERFACES - imported)
    if missing_interfaces:
        raise ValueError(
            "v2 runner omitted required image-contract interfaces: "
            f"{missing_interfaces}"
        )
    processor_execution = _validate_processor_execution_source_v2(
        v2_path,
        imported=imported,
    )
    allowed_transient = [
        call for call in v2_calls if _is_exact_transient_rgb_convert(call)
    ]
    allowed_attribute_ids = {
        id(call[-1].func) for call in allowed_transient
    }
    forbidden_mutations = sorted(
        f"{'.'.join((*classes, *functions))}:{base}.{name}"
        for classes, functions, base, name, attribute in (
            *v2_attributes,
            *image_attributes,
        )
        if name in _FORBIDDEN_IMAGE_MUTATION_CALLS
        and id(attribute) not in allowed_attribute_ids
    )
    forbidden_source_bindings = (
        *v2_forbidden_imports,
        *image_forbidden_imports,
        *v2_forbidden_getattrs,
        *image_forbidden_getattrs,
    )
    forbidden_mutations.extend(
        sorted(
            description
            for name, description in forbidden_source_bindings
            if name in _FORBIDDEN_IMAGE_MUTATION_CALLS
        )
    )
    if forbidden_mutations:
        raise ValueError(
            "v2 processor source contains forbidden image mutation calls: "
            f"{forbidden_mutations}"
        )
    if len(allowed_transient) != 1:
        raise ValueError(
            "v2 runner requires exactly one transient source.convert('RGB') in "
            "ProcessorLengthRuntimeV2.length.decode"
        )
    forbidden_v1_calls = sorted(
        f"_V1.{name}"
        for _, _, base, name, _ in v2_attributes
        if base == "_V1" and name in _FORBIDDEN_V1_EXECUTION_HELPERS
    )
    forbidden_v1_calls.extend(
        sorted(
            description
            for name, description in forbidden_source_bindings
            if name in _FORBIDDEN_V1_EXECUTION_HELPERS
        )
    )
    if forbidden_v1_calls:
        raise ValueError(
            "v2 runner invoked a forbidden v1 image execution helper: "
            f"{forbidden_v1_calls}"
        )
    return {
        "bound_sources": {
            "image_contract_v2": {
                "path": IMAGE_CONTRACT_PATH,
                "sha256": sha256_bytes(image_path.read_bytes()),
            },
            "runner_v1": {
                "path": V1_RUNNER_PATH,
                "sha256": v1_audit["source_sha256"],
                "status": v1_audit["status"],
            },
            "runner_v2": {
                "path": RUNNER_PATH,
                "sha256": v2_audit["source_sha256"],
                "status": v2_audit["status"],
            },
        },
        "forbidden_image_mutation_call_count": 0,
        "forbidden_v1_execution_helper_call_count": 0,
        "transient_rgb_convert_call_count": 1,
        "required_image_contract_interfaces": sorted(
            _REQUIRED_V2_IMAGE_INTERFACES
        ),
        "processor_execution_source_contract": processor_execution,
        "processor_concurrency_per_logical_worker": (
            PROCESSOR_CONCURRENCY_PER_LOGICAL_WORKER
        ),
        "processor_post_load_modeling_module_allowlist": list(
            PROCESSOR_POST_LOAD_MODELING_MODULE_ALLOWLIST
        ),
        "processor_torch_ambient_environment_keys_removed": list(
            PROCESSOR_TORCH_AMBIENT_ENVIRONMENT_KEYS
        ),
        "processor_torch_interop_threads": PROCESSOR_TORCH_INTEROP_THREADS,
        "processor_torch_intraop_threads": PROCESSOR_TORCH_INTRAOP_THREADS,
        "status": "VALID_PROCESSOR_ONLY_SOURCE_V2_IMAGE_CONTRACT_REPAIR",
    }


def _build_worker_schedule_and_queries(
    contract: ProcessorFreezeExecutionContractV2,
) -> tuple[ProcessorWorkerSchedule, dict[str, ExpectedFrozenQuery]]:
    root = contract.repository_root
    inventory = _load_bound_json(
        root / FULL_POOL_INVENTORY_MANIFEST_PATH,
        label="full-pool inventory",
    )
    census = _load_bound_json(
        root / FULL_POOL_CENSUS_PATH,
        label="full-pool census",
    )
    base = _load_bound_json(
        root / INDEPENDENT_REFERENCE_GATE_CONFIG_PATH,
        label="independent base config",
    )
    files = tuple(
        SourceFileSpec(item["path"], item["size_bytes"], item["lfs_sha256"])
        for item in inventory["inventory"]["files"]
    )
    build_full_pool_inspection_config(
        base,
        transport_files=tuple(item.transport_file for item in files),
        format_safety_maximum_decisions=None,
    )
    freeze = contract.freeze_b_v2_manifest
    _, grouped_queries = validate_freeze_query_topology(
        freeze["assignments"],
        freeze["query_states"],
        anchor_step_by_stratum=freeze["repair"]["stratum_anchor_decision_step"],
    )
    plan = build_selected_row_read_plan(
        freeze["assignments"],
        source_files=files,
        source_row_counts=census["source"]["row_counts_by_file"],
    )
    schedule = build_processor_worker_schedule(plan)
    expected_queries = {
        query["state_id"]: ExpectedFrozenQuery.from_mapping(query)
        for values in grouped_queries.values()
        for query in values
    }
    return schedule, expected_queries


def build_processor_freeze_postflight_context_v2(
    contract: ProcessorFreezeExecutionContractV2,
    *,
    expected_git_revision: str,
) -> ProcessorFreezePostflightContextV2:
    """Rebuild v2 expectations solely from byte-bound repository inputs."""
    if not isinstance(contract, ProcessorFreezeExecutionContractV2):
        raise TypeError("v2 postflight requires a validated v2 Execution-CF contract")
    root = contract.repository_root
    schedule, expected_queries = _build_worker_schedule_and_queries(contract)
    backend_path = root / OCR_BACKEND_CONFIG_PATH
    backend_payload = backend_path.read_bytes()
    backend_config = _strict_json_object(
        backend_payload, label="OCR backend config"
    )
    image_contract_sha = sha256_bytes((root / IMAGE_CONTRACT_PATH).read_bytes())
    runtime_identity = {
        **_ocr_runtime_identity(backend_config),
        "image_contract_sha256": image_contract_sha,
        "ocr_concurrency_per_logical_worker": (
            OCR_CONCURRENCY_PER_LOGICAL_WORKER
        ),
    }
    structural = ProcessorFreezePostflightContext(
        repository_root=root,
        execution_config_path=(root / CANONICAL_EXECUTION_CONFIG_PATH).resolve(),
        execution_config_sha256=contract.config_sha256,
        expected_git_revision=expected_git_revision,
        worker_schedule=schedule,
        expected_queries=expected_queries,
        source_audit=validate_processor_only_source_v2(root),
        snapshot_identity_without_model_dir=_snapshot_identity_without_model_dir(root),
        ocr_backend_config_sha256=sha256_bytes(backend_payload),
        ocr_runtime_identity=runtime_identity,
    )
    return ProcessorFreezePostflightContextV2(
        structural_context=structural,
        backend_config=backend_config,
        backend_config_sha256=sha256_bytes(backend_payload),
        image_contract_sha256=image_contract_sha,
    )


def _validate_global_format_tally(
    tallies: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    combined: Counter[str] = Counter()
    for tally in tallies:
        if not isinstance(tally, Mapping):
            raise TypeError("worker image-format tally must be a mapping")
        for key, value in tally.items():
            if (
                key not in PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS
                or type(value) is not int
                or value < 0
            ):
                raise ValueError("worker image-format tally escaped the v2 contract")
            combined[key] += value
    observed = dict(sorted(combined.items()))
    expected = dict(
        sorted(PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_FORMAT_MODE_COUNTS.items())
    )
    if observed != expected or sum(observed.values()) != (
        PROCESSOR_IMAGE_CONTRACT_V2_EXPECTED_TOTAL
    ):
        raise ValueError(
            "global image-format tally differs from the frozen v2 census"
        )
    return observed


def _validate_record_without_image(
    record: Mapping[str, Any],
    *,
    expected_path: str,
    backend_config_sha256: str,
) -> tuple[str, str]:
    if not isinstance(record, Mapping):
        raise TypeError("v2 OCR record must be a mapping")
    if set(record) != OCR_RECORD_KEYS:
        raise ValueError("OCR record fields drifted from the pinned full schema")
    source_format = record.get("source_format")
    source_mode = record.get("source_mode")
    alpha = record.get("alpha_extrema")
    exif = record.get("exif_present")
    observed = (
        source_format,
        source_mode,
        tuple(alpha) if isinstance(alpha, list) else alpha,
        exif,
    )
    if observed not in PROCESSOR_IMAGE_CONTRACT_V2_ACCEPTED_INPUTS:
        raise ValueError("OCR record escaped the v2 accepted-input union")
    parsed_path = PurePosixPath(expected_path)
    if (
        not expected_path.startswith("images/")
        or "\\" in expected_path
        or parsed_path.is_absolute()
        or parsed_path.as_posix() != expected_path
        or any(part in {"", ".", ".."} for part in parsed_path.parts)
    ):
        raise ValueError("OCR record image path is not canonical")
    width = record.get("width")
    height = record.get("height")
    if (
        record.get("schema_version") != OCR_RECORD_SCHEMA_VERSION
        or record.get("backend_id") != BACKEND_ID
        or record.get("image_member_path") != expected_path
        or record.get("backend_config_sha256") != backend_config_sha256
        or _SHA256.fullmatch(str(record.get("image_sha256"))) is None
        or type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or _SHA256.fullmatch(str(record.get("rgb_bytes_sha256"))) is None
        or _SHA256.fullmatch(
            str(record.get("resized_rgb_256x256_sha256"))
        )
        is None
    ):
        raise ValueError("OCR record fixed identity or prepared metadata drifted")
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("OCR record nodes must be a JSON array")
    try:
        canonical_nodes = canonicalize_rapidocr_nodes(
            boxes=[node["polygon_xy"] for node in nodes],
            texts=[node["raw_text"] for node in nodes],
            scores=[float(node["confidence_decimal_string"]) for node in nodes],
            width=width,
            height=height,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OCR record node schema is invalid") from error
    if list(canonical_nodes) != nodes:
        raise ValueError("OCR record nodes are not pinned canonical nodes")
    full_tokens = [
        token
        for node in nodes
        for token in str(node["normalized_text"]).split(" ")
    ]
    if record.get("full_spatial_tokens") != full_tokens or record.get(
        "full_spatial_tokens_sha256"
    ) != sha256_bytes(canonical_json_bytes(full_tokens)):
        raise ValueError("OCR record full spatial tokens drifted")
    canonical_sha = record.get("canonical_ocr_record_sha256")
    unsigned = dict(record)
    unsigned.pop("canonical_ocr_record_sha256", None)
    if canonical_sha != sha256_bytes(canonical_json_bytes(unsigned)):
        raise ValueError("OCR record canonical SHA256 drifted")
    return str(source_format), str(source_mode)


def _validate_v2_artifact_semantics(
    root: Path,
    context: ProcessorFreezePostflightContextV2,
) -> tuple[dict[str, int], int, int]:
    terminal_tallies: list[Mapping[str, int]] = []
    fully_validated_paths: set[tuple[str, str]] = set()
    terminal_paths: set[tuple[str, str]] = set()
    for worker in context.structural_context.worker_schedule.workers:
        shard = root / "substrate" / worker.filename
        worker_tally: Counter[str] = Counter()
        for query in iter_processor_query_artifact_records(
            shard, expected_worker=worker
        ):
            for path, payload in query.image_payloads.items():
                record = query.ocr_records_by_path.get(path)
                if record is None:
                    raise ValueError("stored image is missing its v2 OCR record")
                identity = (query.trajectory_id, path)
                if identity not in fully_validated_paths:
                    validate_processor_ocr_record_v2(
                        record,
                        image_bytes=payload,
                        backend_config=context.backend_config,
                        backend_config_sha256=context.backend_config_sha256,
                    )
                    fully_validated_paths.add(identity)
            if query.query_kind != "terminal":
                continue
            for path, record in query.ocr_records_by_path.items():
                terminal_paths.add((query.trajectory_id, path))
                source_format, source_mode = _validate_record_without_image(
                    record,
                    expected_path=path,
                    backend_config_sha256=context.backend_config_sha256,
                )
                worker_tally[f"{source_format}:{source_mode}"] += 1
        terminal_tallies.append(dict(sorted(worker_tally.items())))
    if not fully_validated_paths.issubset(terminal_paths):
        raise ValueError("stored image inventory escaped terminal OCR coverage")
    metadata_only_paths = terminal_paths - fully_validated_paths
    return (
        _validate_global_format_tally(terminal_tallies),
        len(fully_validated_paths),
        len(metadata_only_paths),
    )


def _receipt_tallies_and_contract_identity(
    root: Path,
    context: ProcessorFreezePostflightContextV2,
) -> tuple[Mapping[str, Any], ...]:
    tallies = []
    for worker_index in range(len(context.structural_context.worker_schedule.workers)):
        path = root / "receipts" / f"ocr-worker-{worker_index:02d}.json"
        payload = path.read_bytes()
        receipt = _strict_json_object(payload, label=f"v2 OCR receipt {worker_index}")
        if payload != canonical_json_bytes(receipt):
            raise ValueError("v2 OCR receipt is not canonical compact JSON")
        identity = receipt.get("ocr_runtime_identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("image_contract_sha256")
            != context.image_contract_sha256
        ):
            raise ValueError("v2 OCR receipt image-contract SHA256 drifted")
        tallies.append(receipt.get("format_tally"))
    return tuple(tallies)


def _validate_processor_thread_runtime_logs(
    root: Path,
) -> tuple[dict[str, Any], ...]:
    expected_runtime = {
        "ambient_thread_environment_keys_present": [],
        "getter_verification_passed": True,
        "torch_interop_thread_count": PROCESSOR_TORCH_INTEROP_THREADS,
        "torch_intraop_thread_count": PROCESSOR_TORCH_INTRAOP_THREADS,
    }
    expected_line = canonical_json_bytes(
        {"processor_torch_thread_runtime": expected_runtime}
    )
    marker = b'"processor_torch_thread_runtime"'
    evidence = []
    for worker_index in range(WORKER_COUNT):
        path = root / "logs" / f"processor-worker-{worker_index:02d}.log"
        if path.is_symlink() or not path.is_file():
            raise ValueError("processor thread-runtime log must be one regular file")
        payload = path.read_bytes()
        first_line, separator, diagnostics = payload.partition(b"\n")
        if separator != b"\n" or first_line != expected_line:
            raise ValueError(
                "processor thread-runtime evidence must be the canonical first line"
            )
        if payload.count(marker) != 1:
            raise ValueError(
                "processor thread-runtime log must contain exactly one evidence marker"
            )
        try:
            diagnostic_text = diagnostics.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "processor log diagnostics after evidence must be UTF-8 text"
            ) from error
        if any(
            (ord(character) < 32 and character not in "\t\r\n")
            or ord(character) == 127
            for character in diagnostic_text
        ):
            raise ValueError(
                "processor log diagnostics contain unsafe control characters"
            )
        evidence.append(
            {
                "diagnostic_byte_count": len(diagnostics),
                "diagnostic_line_count": len(diagnostic_text.splitlines()),
                "evidence_sha256": sha256_bytes(expected_line),
                "log_sha256": sha256_bytes(payload),
                "status": PROCESSOR_THREAD_RUNTIME_LOG_STATUS,
                "worker_index": worker_index,
            }
        )
    return tuple(evidence)


def validate_completed_processor_freeze_root_v2(
    output_root: str | Path,
    *,
    context: ProcessorFreezePostflightContextV2,
) -> dict[str, Any]:
    """Validate one v2 root structurally and against the exact mixed-mode census."""
    if not isinstance(context, ProcessorFreezePostflightContextV2):
        raise TypeError("v2 postflight context is invalid")
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("v2 postflight output root must be absolute")
    expected_output_basename = (
        OUTPUT_BASENAME_PREFIX
        + context.structural_context.expected_git_revision[:7]
    )
    if root.name != expected_output_basename:
        raise ValueError("v2 output root differs from its exact Git-bound namespace")
    structural = _validate_completed_processor_freeze_root_v1(
        root, context=context.structural_context
    )
    processor_thread_runtime_evidence = _validate_processor_thread_runtime_logs(
        root
    )
    receipt_tally = _validate_global_format_tally(
        _receipt_tallies_and_contract_identity(root, context)
    )
    (
        artifact_tally,
        stored_validation_count,
        metadata_only_validation_count,
    ) = _validate_v2_artifact_semantics(root, context)
    if artifact_tally != receipt_tally:
        raise ValueError("v2 artifact and receipt global format tallies differ")
    return {
        **{key: value for key, value in structural.items() if key != "status"},
        "global_format_mode_tally": artifact_tally,
        "image_contract_sha256": context.image_contract_sha256,
        "metadata_only_ocr_validation_count": metadata_only_validation_count,
        "processor_image_contract_id": PROCESSOR_IMAGE_CONTRACT_V2_ID,
        "processor_thread_runtime_evidence": list(
            processor_thread_runtime_evidence
        ),
        "schema_version": SCHEMA_VERSION,
        "stored_image_ocr_validation_count": stored_validation_count,
        "structural_validation_status": structural["status"],
        "status": VALIDATION_STATUS,
    }


build_processor_freeze_postflight_context = (
    build_processor_freeze_postflight_context_v2
)
validate_completed_processor_freeze_root = validate_completed_processor_freeze_root_v2


__all__ = [
    "ProcessorFreezePostflightContextV2",
    "PROCESSOR_THREAD_RUNTIME_LOG_STATUS",
    "VALIDATION_STATUS",
    "build_processor_freeze_postflight_context",
    "build_processor_freeze_postflight_context_v2",
    "validate_completed_processor_freeze_root",
    "validate_completed_processor_freeze_root_v2",
    "validate_processor_only_source_v2",
    "_validate_processor_thread_runtime_logs",
]
