import json
import unittest
from pathlib import Path

from causalcache.schema import ActionType, DecisionRecord, ExecutableAction, LowFidelityEvent


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent


class SchemaTest(unittest.TestCase):
    def _decision(self) -> DecisionRecord:
        with (REPOSITORY_ROOT / "data" / "fixtures" / "validated_decision.json").open(
            encoding="utf-8"
        ) as handle:
            return DecisionRecord.from_dict(json.load(handle))

    def test_validated_reference_matches(self) -> None:
        decision = self._decision()
        self.assertTrue(decision.reference_is_valid("executable_match"))
        self.assertTrue(decision.reference_is_valid("successful_trajectory"))

    def test_action_canonicalization_handles_formatting(self) -> None:
        left = ExecutableAction(ActionType.TYPE_TEXT, target=" Search ", text_argument="Hello   WORLD")
        right = ExecutableAction(ActionType.TYPE_TEXT, target="search", text_argument="hello world")
        self.assertTrue(left.executable_match(right))

    def test_case_sensitive_text_is_preserved(self) -> None:
        left = ExecutableAction(ActionType.TYPE_TEXT, text_argument="AbC", text_case_sensitive=True)
        right = ExecutableAction(ActionType.TYPE_TEXT, text_argument="abc", text_case_sensitive=True)
        self.assertFalse(left.executable_match(right))

    def test_low_fidelity_contract_has_exact_fields(self) -> None:
        event = self._decision().history[0].low_fidelity
        self.assertIsInstance(event, LowFidelityEvent)
        self.assertEqual(
            tuple(event.to_dict()),
            (
                "step_id",
                "action_type",
                "target_text_or_coordinate_bin",
                "deterministic_ui_delta",
                "result_status",
            ),
        )

    def test_decision_round_trip(self) -> None:
        decision = self._decision()
        self.assertEqual(DecisionRecord.from_dict(decision.to_dict()), decision)


if __name__ == "__main__":
    unittest.main()
