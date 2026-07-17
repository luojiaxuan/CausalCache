"""Run the bounded synthetic-only gate v1 trainer smoke."""

from __future__ import annotations

import argparse
import itertools
import json

from causalcache.gate_v1_data import CandidateFeatures, GateState
from causalcache.gate_v1_training import run_synthetic_two_step_smoke
from causalcache.restoration_v2_2_label_table import validate_complete_distance_table


def synthetic_states() -> tuple[GateState, ...]:
    states = []
    for source_index in range(2):
        source_id = f"synthetic-{source_index:02d}"
        for event_count in (2, 3, 4):
            event_ids = tuple(range(1, event_count + 1))
            weights = {event: 0.3 / event for event in event_ids}
            weights[event_ids[-1]] = -0.05
            distances = {
                coalition: 1.0
                - sum(weights[event] for event in coalition)
                - (0.12 if {1, 2}.issubset(coalition) else 0.0)
                for size in range(event_count + 1)
                for coalition in itertools.combinations(event_ids, size)
            }
            states.append(
                GateState(
                    source_id=source_id,
                    state_id=(
                        f"{source_id}:decision_step:{event_count + 2:03d}"
                    ),
                    decision_step_id=event_count + 2,
                    candidate_event_step_ids=event_ids,
                    q64=tuple(1.0 / 8.0 for _ in range(64)),
                    candidates=tuple(
                        CandidateFeatures(
                            event_step_id=event,
                            h64=tuple(
                                1.0 if index == event - 1 else 0.0
                                for index in range(64)
                            ),
                            g8=(
                                0.25,
                                0.5,
                                0.0,
                                0.0,
                                0.0,
                                0.1,
                                1.0 / 3.0,
                                0.5,
                            ),
                        )
                        for event in event_ids
                    ),
                    table=validate_complete_distance_table(event_ids, distances),
                )
            )
    return tuple(states)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic-only two-step smoke; accepts no artifact or output path."
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _parser().parse_args(argv)
    result = run_synthetic_two_step_smoke(synthetic_states())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
