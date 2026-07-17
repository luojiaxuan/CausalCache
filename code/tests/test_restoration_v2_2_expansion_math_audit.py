from __future__ import annotations

import ast
import copy
import itertools
import sys
import unittest
from pathlib import Path

from causalcache.restoration_v2_2_expansion_math_audit import (
    PASS_STATUS,
    audit_normalized_raw_states,
    compare_with_reducer_output,
)


def _record(
    distances: dict[tuple[int, ...], float],
    *,
    state_id: str = "source-1:decision_step:005",
    role: str = "gate_train_expansion",
) -> dict[str, object]:
    event_ids = tuple(max(distances, key=lambda coalition: (len(coalition), coalition)))
    rows = [
        {"coalition": list(coalition), "distance": distances[coalition], "ignored": "raw-evidence"}
        for size in range(len(event_ids) + 1)
        for coalition in itertools.combinations(event_ids, size)
    ]
    return {
        "state": {
            "state_id": state_id,
            "source_id": state_id.split(":", 1)[0],
            "role": role,
            "candidate_event_step_ids": list(event_ids),
            "preserved_metadata": "witness",
        },
        "distance_rows": rows,
        "ignored_envelope": "runtime-evidence",
    }


class RestorationV22ExpansionMathAuditTest(unittest.TestCase):
    def test_module_imports_only_the_python_standard_library(self) -> None:
        source = Path(
            sys.modules[audit_normalized_raw_states.__module__].__file__
        ).read_text()
        roots: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        self.assertTrue(roots <= (sys.stdlib_module_names | {"__future__"}), roots)

    def test_recomputes_edges_interactions_shapley_oracle_and_signs(self) -> None:
        distances = {
            (): 10.0,
            (1,): 8.0,
            (2,): 9.0,
            (3,): 9.0,
            (1, 2): 8.5,
            (1, 3): 6.0,
            (2, 3): 6.0,
            (1, 2, 3): 0.0,
        }
        result = audit_normalized_raw_states([_record(distances)])
        state = result["states"][0]
        self.assertEqual(
            result["counts"],
            {
                "trajectory_count": 1,
                "state_count": 1,
                "raw_distance_row_count": 8,
                "deployment_conditional_edge_count": 9,
                "full_hypercube_edge_count": 12,
                "pair_interaction_count": 6,
                "exact_permutation_attribution_count": 3,
                "primary_exact_subset_oracle_count": 1,
            },
        )
        harmful = next(
            edge
            for edge in state["full_hypercube_edges"]
            if edge["base_coalition"] == [1] and edge["event_id"] == 2
        )
        self.assertEqual(harmful["marginal_gain"], -0.5)
        interaction = next(
            item
            for item in state["pair_interactions"]
            if item["conditioning_coalition"] == []
            and item["left_event_id"] == 1
            and item["right_event_id"] == 2
        )
        self.assertEqual(interaction["interaction"], -1.5)
        self.assertAlmostEqual(
            sum(
                item["mean_marginal_gain"]
                for item in state["exact_permutation_attribution"]
            ),
            10.0,
        )
        self.assertEqual(
            state["primary_exact_subset_oracle"],
            {
                "budget_event_capacity": 2,
                "tie_epsilon": 0.0,
                "coalition": [1, 3],
                "distance": 6.0,
                "utility": 4.0,
                "evaluated_coalition_count": 7,
            },
        )
        self.assertEqual(result["summary"]["nonmonotone_state_count"], 1)
        self.assertEqual(result["summary"]["marginal_sign_counts"]["negative"], 1)
        self.assertTrue(result["summary"]["algebra_residual_gate"]["passed"])

    def test_exact_oracle_is_at_most_budget_with_exact_tie_breaking(self) -> None:
        empty_best = audit_normalized_raw_states(
            [
                _record(
                    {
                        (): 0.0,
                        (1,): 1.0,
                        (2,): 1.0,
                        (1, 2): 0.0,
                    }
                )
            ]
        )
        self.assertEqual(
            empty_best["states"][0]["primary_exact_subset_oracle"]["coalition"],
            [],
        )

        exact_not_isclose = audit_normalized_raw_states(
            [
                _record(
                    {
                        (): 5.0,
                        (1,): 1.0,
                        (2,): 1.0 - 1e-12,
                        (3,): 4.0,
                        (1, 2): 2.0,
                        (1, 3): 3.0,
                        (2, 3): 3.0,
                        (1, 2, 3): 0.0,
                    }
                )
            ]
        )
        self.assertEqual(
            exact_not_isclose["states"][0]["primary_exact_subset_oracle"]["coalition"],
            [2],
        )

    def test_fails_closed_on_incomplete_duplicate_and_nonzero_full_history(self) -> None:
        valid = {
            (): 4.0,
            (1,): 3.0,
            (2,): 2.0,
            (1, 2): 0.0,
        }
        incomplete = _record(valid)
        incomplete["distance_rows"].pop()
        with self.assertRaisesRegex(ValueError, "complete power set"):
            audit_normalized_raw_states([incomplete])

        duplicate = _record(valid)
        duplicate["distance_rows"][1] = copy.deepcopy(duplicate["distance_rows"][0])
        with self.assertRaisesRegex(ValueError, "duplicate coalition"):
            audit_normalized_raw_states([duplicate])

        nonzero_full = dict(valid)
        nonzero_full[(1, 2)] = 0.5
        with self.assertRaisesRegex(ValueError, "canonical zero"):
            audit_normalized_raw_states([_record(nonzero_full)])

    def test_matches_the_existing_reducer_on_its_full_synthetic_denominator(self) -> None:
        from causalcache.restoration_v2_2_expansion_labels_artifact import (
            reduce_raw_distance_states,
        )
        from tests.test_restoration_v2_2_expansion_labels_artifact import (
            _state_record,
            _states,
        )

        expected_states = _states()
        run_contract_sha256 = "d" * 64
        records = [
            _state_record(state, run_contract_sha256) for state in expected_states
        ]
        reducer_result = reduce_raw_distance_states(
            records,
            expected_states=expected_states,
            run_contract_sha256=run_contract_sha256,
        )
        audit_result = audit_normalized_raw_states(records)
        comparison = compare_with_reducer_output(audit_result, reducer_result)
        self.assertEqual(comparison["status"], PASS_STATUS)
        self.assertEqual(comparison["compared_state_count"], 192)

        broken_reducer = copy.deepcopy(reducer_result)
        broken_reducer["states"][0]["deployment_conditional_edges"][0][
            "marginal_gain"
        ] += 1e-15
        with self.assertRaisesRegex(ValueError, "marginal_gain differs"):
            compare_with_reducer_output(audit_result, broken_reducer)


if __name__ == "__main__":
    unittest.main()
