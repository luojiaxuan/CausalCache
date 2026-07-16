"""Run or validate the versioned policy-vision SizeDict interface repair."""

from __future__ import annotations

from collections.abc import Sequence

from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GPU_UUID_TYPE_PROFILE_V2,
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
    IMAGE_PROCESSOR_SIZE_PROFILE_V3,
)
from causalcache.restoration_v2_2_policy_vision_contract import (
    EXPECTED_OPERATION_CEILING,
    EXPECTED_OUTPUT_FILES,
)
from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    CANONICAL_CONFIG_PATH,
    COMBINED_RUNTIME_PROFILE_ID,
    FORMAL_ATTEMPT_LEDGER_PATH,
    GPU_UUID_RUNTIME_TYPE_PROFILE,
    IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE,
    PROTOCOL_ID,
    RUN_STATUS,
    VALID_STATUS,
    RestorationV22PolicyVisionV3Contract,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    PolicyVisionRunnerProtocol,
    run_protocol_main,
)
from scripts.run_restoration_v2_2_policy_vision_baseline_v2 import (
    V2_FORMAL_SOURCE_PATHS,
)


if (
    GPU_UUID_RUNTIME_TYPE_PROFILE != GPU_UUID_TYPE_PROFILE_V2
    or IMAGE_PROCESSOR_SIZE_RUNTIME_TYPE_PROFILE
    != IMAGE_PROCESSOR_SIZE_PROFILE_V3
    or COMBINED_RUNTIME_PROFILE_ID
    != GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID
):
    raise RuntimeError("policy-vision v3 contract/runtime profile mapping drifted")


V3_FORMAL_SOURCE_PATHS = (
    CANONICAL_CONFIG_PATH,
    *V2_FORMAL_SOURCE_PATHS,
    "code/causalcache/restoration_v2_2_policy_vision_v3_contract.py",
    "code/scripts/validate_restoration_v2_2_policy_vision_v3_contract.py",
    "code/scripts/run_restoration_v2_2_policy_vision_baseline_v3.py",
    (
        "data/results/"
        "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/"
        "README.md"
    ),
    (
        "data/results/"
        "restoration_v2_2_policy_vision_baseline_v2_gpu_uuid_repair_attempt/"
        "failure.json"
    ),
)

V3_RUNNER_PROTOCOL = PolicyVisionRunnerProtocol(
    protocol_id=PROTOCOL_ID,
    run_status=RUN_STATUS,
    valid_status=VALID_STATUS,
    contract_class=RestorationV22PolicyVisionV3Contract,
    expected_output_files=tuple(EXPECTED_OUTPUT_FILES),
    expected_operation_ceiling=EXPECTED_OPERATION_CEILING,
    formal_source_paths=V3_FORMAL_SOURCE_PATHS,
    runtime_profile_id=GUI_OWL_V2_2_VISION_RUNTIME_UUID_SIZE_DICT_V3_ID,
    gpu_uuid_type_profile=GPU_UUID_TYPE_PROFILE_V2,
    image_processor_size_profile=IMAGE_PROCESSOR_SIZE_PROFILE_V3,
    readme_title=(
        "Restoration v2.2 policy-vision baseline v3 SizeDict interface repair"
    ),
    formal_run_allowed=True,
    run_tombstone_status=None,
    formal_attempt_ledger_path=FORMAL_ATTEMPT_LEDGER_PATH,
    include_repair_identity=True,
)


def main(argv: Sequence[str] | None = None) -> None:
    run_protocol_main(V3_RUNNER_PROTOCOL, argv)


if __name__ == "__main__":
    main()
