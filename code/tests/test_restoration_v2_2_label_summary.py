from __future__ import annotations

import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_eager_artifact import pretty_json_bytes
from causalcache.restoration_v2_2_label_artifact import LabelEvidence
from causalcache.restoration_v2_2_label_contract import V1_ATTEMPT_PROFILE
from causalcache.restoration_v2_2_label_summary import (
    NORMALIZATION_EPSILON,
    SIGN_RULE,
    SUMMARY_STATUS,
    build_label_scientific_summary,
    summarize_label_evidence_archive,
)
from scripts.manage_restoration_v2_2_label_artifact import main


def _state_payload(
    *,
    role: str,
    event_ids: tuple[int, ...],
    distances: dict[tuple[int, ...], float],
) -> bytes:
    rows = []
    for size in range(len(event_ids) + 1):
        for coalition in itertools.combinations(event_ids, size):
            rows.append(
                {
                    "coalition_event_step_ids": list(coalition),
                    "distance_kl": distances[coalition],
                    "restoration_utility": 999.0,
                }
            )
    return pretty_json_bytes(
        {
            "state": {
                "role": role,
                "candidate_event_step_ids": list(event_ids),
            },
            "distance_rows": rows,
            "deployment_conditional_edges": [{"marginal_gain": 999.0}],
            "pair_interactions": [{"interaction": 999.0}],
            "primary_exact_subset_oracle": {"utility": 999.0},
        }
    )


def _evidence() -> LabelEvidence:
    train_distances = {
        (): 10.0,
        (1,): 6.0,
        (2,): 12.0,
        (1, 2): 0.0,
    }
    development_distances = {
        (): 10.0,
        (10,): 8.0,
        (20,): 8.0,
        (30,): 9.0,
        (10, 20): 9.0,
        (10, 30): 5.0,
        (20, 30): 5.0,
        (10, 20, 30): 0.0,
    }
    files = {
        "workers/odd/states/001.json": _state_payload(
            role="v2_development",
            event_ids=(10, 20, 30),
            distances=development_distances,
        ),
        "workers/even/states/000.json": _state_payload(
            role="v2_label_train",
            event_ids=(1, 2),
            distances=train_distances,
        ),
    }
    return LabelEvidence(
        files=files,
        source_git_commit="1" * 40,
        run_contract_sha256="2" * 64,
        outcome=V1_ATTEMPT_PROFILE.pass_outcome,
        aggregate={},
        inventory=(),
        tree_inventory_sha256="3" * 64,
        profile=V1_ATTEMPT_PROFILE,
    )


class RestorationV22LabelSummaryTest(unittest.TestCase):
    def test_recomputes_all_science_from_raw_distance_rows(self) -> None:
        result = build_label_scientific_summary(_evidence())
        self.assertEqual(result["status"], SUMMARY_STATUS)
        self.assertEqual(result["sign_rule"], SIGN_RULE)
        self.assertEqual(
            result["source_evidence"]["archive_validation"],
            "read_label_evidence_archive_passed",
        )

        overall = result["overall"]
        self.assertEqual(overall["state_count"], 2)
        self.assertEqual(overall["candidate_event_count_histogram"], {"2": 1, "3": 1})
        self.assertEqual(
            overall["raw_restoration_utility"],
            {
                "row_count": 12,
                "positive_count": 9,
                "negative_count": 1,
                "zero_count": 2,
                "mean_utility": 38.0 / 12.0,
                "states_with_positive_utility_count": 2,
                "states_with_negative_utility_count": 1,
            },
        )
        self.assertEqual(
            overall["deployment_conditional_marginal"],
            {
                "edge_count": 13,
                "positive_count": 10,
                "negative_count": 3,
                "zero_count": 0,
                "mean_marginal_gain": 37.0 / 13.0,
                "states_with_positive_marginal_count": 2,
                "states_with_negative_marginal_count": 2,
            },
        )
        self.assertEqual(
            overall["primary_exact_subset_oracle"],
            {
                "state_count": 2,
                "budget_event_capacity": 2,
                "selected_cardinality_histogram": {"2": 2},
                "eligible_coalition_count": 11,
                "eligible_coalition_count_histogram": {"4": 1, "7": 1},
                "eligible_coalition_cardinality_histogram": {
                    "0": 2,
                    "1": 5,
                    "2": 4,
                },
                "positive_utility_state_count": 2,
                "negative_utility_state_count": 0,
                "zero_utility_state_count": 0,
                "mean_summary_only_distance": 10.0,
                "mean_utility": 7.5,
                "normalization_epsilon": NORMALIZATION_EPSILON,
                "normalized_recovery_eligible_state_count": 2,
                "normalized_recovery_excluded_state_count": 0,
                "mean_normalized_recovery": 0.75,
            },
        )
        self.assertEqual(
            overall["pair_interaction"],
            {
                "row_count": 7,
                "positive_count": 6,
                "negative_count": 1,
                "zero_count": 0,
                "states_with_nonzero_interaction_count": 2,
                "signed_mass": 22.0,
                "positive_mass": 25.0,
                "negative_absolute_mass": 3.0,
                "absolute_mass": 28.0,
                "mean_absolute_magnitude": 4.0,
                "maximum_absolute_magnitude": 8.0,
                "mean_state_absolute_mass": 14.0,
                "maximum_state_absolute_mass": 20.0,
            },
        )

        self.assertEqual(list(result["by_role"]), ["v2_development", "v2_label_train"])
        self.assertEqual(result["by_role"]["v2_label_train"]["state_count"], 1)
        self.assertEqual(
            result["by_role"]["v2_label_train"]["pair_interaction"]["absolute_mass"],
            8.0,
        )

    def test_archive_entrypoint_validates_before_reduction_and_is_order_stable(
        self,
    ) -> None:
        evidence = _evidence()
        reversed_evidence = LabelEvidence(
            **{
                **evidence.__dict__,
                "files": dict(reversed(tuple(evidence.files.items()))),
            }
        )
        with patch(
            "causalcache.restoration_v2_2_label_summary.read_label_evidence_archive",
            return_value=evidence,
        ) as reader:
            result = summarize_label_evidence_archive("validated.raw.tar")
        reader.assert_called_once_with("validated.raw.tar")
        self.assertEqual(result, build_label_scientific_summary(reversed_evidence))

    def test_oracle_summary_covers_empty_singleton_and_zero_baseline(self) -> None:
        files = {
            "workers/even/states/000.json": _state_payload(
                role="v2_label_train",
                event_ids=(1, 2),
                distances={(): 0.0, (1,): 1.0, (2,): 2.0, (1, 2): 3.0},
            ),
            "workers/odd/states/001.json": _state_payload(
                role="v2_development",
                event_ids=(10, 20),
                distances={(): 10.0, (10,): 0.0, (20,): 5.0, (10, 20): 2.0},
            ),
        }
        evidence = LabelEvidence(**{**_evidence().__dict__, "files": files})
        oracle = build_label_scientific_summary(evidence)["overall"][
            "primary_exact_subset_oracle"
        ]
        self.assertEqual(oracle["selected_cardinality_histogram"], {"0": 1, "1": 1})
        self.assertEqual(oracle["positive_utility_state_count"], 1)
        self.assertEqual(oracle["zero_utility_state_count"], 1)
        self.assertEqual(oracle["normalized_recovery_eligible_state_count"], 1)
        self.assertEqual(oracle["normalized_recovery_excluded_state_count"], 1)
        self.assertEqual(oracle["mean_normalized_recovery"], 1.0)

    def test_create_summary_cli_writes_canonical_json_exclusively(self) -> None:
        expected = build_label_scientific_summary(_evidence())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            with patch(
                "scripts.manage_restoration_v2_2_label_artifact."
                "summarize_label_evidence_archive",
                return_value=expected,
            ) as summarize, redirect_stdout(io.StringIO()):
                main(
                    [
                        "create-summary",
                        "--evidence",
                        "validated.raw.tar",
                        "--output",
                        str(output),
                    ]
                )
            summarize.assert_called_once_with(Path("validated.raw.tar"))
            self.assertEqual(output.read_bytes(), pretty_json_bytes(expected))
            with patch(
                "scripts.manage_restoration_v2_2_label_artifact."
                "summarize_label_evidence_archive",
                return_value=expected,
            ), redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                main(
                    [
                        "create-summary",
                        "--evidence",
                        "validated.raw.tar",
                        "--output",
                        str(output),
                    ]
                )

    def test_rejects_unvalidated_input_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "validated LabelEvidence"):
            build_label_scientific_summary({})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
