import json
import sys
from pathlib import Path

sys.path.insert(0, "code")

from causalcache.restoration_v2_2_policy_vision import load_identity_witness
from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    RestorationV22PolicyVisionV3Contract,
)
from scripts.run_restoration_v2_2_policy_vision_baseline import (
    _expected_files_from_features,
    _feature_record_from_evaluated_row,
    _read_recorded_rows,
)
from scripts.run_restoration_v2_2_policy_vision_baseline_v3 import (
    V3_RUNNER_PROTOCOL,
)


root = Path(".").resolve()
output = (
    root
    / "data/results/"
    "restoration_v2_2_policy_vision_baseline_v3_size_dict_interface_repair"
)
contract = RestorationV22PolicyVisionV3Contract.load(
    root
    / "code/configs/"
    "causalcache_restoration_v2_2_policy_vision_baseline_"
    "v3_size_dict_interface_repair.json",
    repository_root=root,
)
state_keys = ("index", "role", "trajectory_id", "state_id")
feature_records = []
for row in _read_recorded_rows(output):
    record = _feature_record_from_evaluated_row(row)
    record["state"] = {key: record["state"][key] for key in state_keys}
    feature_records.append(record)
work_items, witness_by_state = load_identity_witness(contract)
summary = json.loads((output / "summary.json").read_text())
files = _expected_files_from_features(
    contract=contract,
    labels_archive=Path(
        "/data/experiments/causalcache/"
        "restoration-v2-2-eager-labels-v2.raw.tar"
    ),
    source_commit="a935a3cf5efb1fa7a952ca6f45a5994609b367e9",
    work_items=work_items,
    witness_by_state=witness_by_state,
    feature_records=feature_records,
    execution=summary["execution"],
    protocol=V3_RUNNER_PROTOCOL,
)
print(
    json.dumps(
        {name: payload == (output / name).read_bytes() for name, payload in files.items()},
        sort_keys=True,
    )
)
