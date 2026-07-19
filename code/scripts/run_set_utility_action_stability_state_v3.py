#!/usr/bin/env python3
"""Run one fresh-process strict-determinism D2 state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from causalcache.set_utility_action_stability_diagnostic_v3 import STATE_IDS
from causalcache.set_utility_action_stability_execution_v3 import (
    launch_from_projection_v3,
    load_action_stability_envelope_v3,
    run_action_stability_state_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-envelope", type=Path, required=True)
    parser.add_argument("--state-id", choices=STATE_IDS, required=True)
    args = parser.parse_args()
    projection = load_action_stability_envelope_v3(args.execution_envelope)
    launch = launch_from_projection_v3(
        projection,
        state_id=args.state_id,
        visible_gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    terminal = run_action_stability_state_v3(launch)
    print(json.dumps(terminal, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        name = error.__class__.__name__
        failure = name if name.replace("_", "").isalnum() else "UnexpectedException"
        print(
            json.dumps(
                {"failure_class": failure, "status": "FAILED_ACTION_STABILITY_STATE_CLI_V3"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
