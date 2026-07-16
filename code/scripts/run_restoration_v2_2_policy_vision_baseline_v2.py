"""Run or validate the versioned policy-vision GPU UUID type repair."""

from __future__ import annotations

from collections.abc import Sequence

from causalcache.policy.gui_owl_v2_2_vision_runtime import (
    GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
)
from causalcache.restoration_v2_2_policy_vision_contract import (
    EXPECTED_OPERATION_CEILING,
    EXPECTED_OUTPUT_FILES,
)
from causalcache.restoration_v2_2_policy_vision_v2_contract import (
    CANONICAL_CONFIG_PATH,
    FORMAL_ATTEMPT_LEDGER_PATH,
    GPU_UUID_RUNTIME_TYPE_PROFILE,
    PROTOCOL_ID,
    RUN_STATUS,
    VALID_STATUS,
    RestorationV22PolicyVisionV2Contract,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    FORMAL_SOURCE_PATHS,
    PolicyVisionRunnerProtocol,
    run_protocol_main,
)


V2_FORMAL_SOURCE_PATHS = (
    CANONICAL_CONFIG_PATH,
    *FORMAL_SOURCE_PATHS,
    "code/causalcache/restoration_v2_2_policy_vision_v2_contract.py",
    "code/scripts/validate_restoration_v2_2_policy_vision_v2_contract.py",
    "code/scripts/run_restoration_v2_2_policy_vision_baseline_v2.py",
    "data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/README.md",
    "data/results/restoration_v2_2_policy_vision_baseline_v1_attempt/failure.json",
)

V2_RUNNER_PROTOCOL = PolicyVisionRunnerProtocol(
    protocol_id=PROTOCOL_ID,
    run_status=RUN_STATUS,
    valid_status=VALID_STATUS,
    contract_class=RestorationV22PolicyVisionV2Contract,
    expected_output_files=tuple(EXPECTED_OUTPUT_FILES),
    expected_operation_ceiling=EXPECTED_OPERATION_CEILING,
    formal_source_paths=V2_FORMAL_SOURCE_PATHS,
    runtime_profile_id=GUI_OWL_V2_2_VISION_RUNTIME_UUID_TYPE_ONLY_V2_ID,
    gpu_uuid_type_profile=GPU_UUID_RUNTIME_TYPE_PROFILE,
    readme_title="Restoration v2.2 policy-vision baseline v2 GPU UUID repair",
    formal_run_allowed=True,
    formal_attempt_ledger_path=FORMAL_ATTEMPT_LEDGER_PATH,
    include_repair_identity=True,
)


def main(argv: Sequence[str] | None = None) -> None:
    run_protocol_main(V2_RUNNER_PROTOCOL, argv)


if __name__ == "__main__":
    main()
