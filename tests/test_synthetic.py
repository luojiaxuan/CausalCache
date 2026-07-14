import unittest
from pathlib import Path

from causalcache.attribution import exact_permutation_restoration
from causalcache.synthetic import SyntheticBehaviorModel


ROOT = Path(__file__).resolve().parents[1]


class SyntheticModelTest(unittest.TestCase):
    def test_full_coalition_has_zero_distance(self) -> None:
        model = SyntheticBehaviorModel.load(ROOT / "configs" / "synthetic_phase0.json")
        self.assertAlmostEqual(model.distance(frozenset(model.event_ids)), 0.0)

    def test_scenario_contains_negative_restoration(self) -> None:
        model = SyntheticBehaviorModel.load(ROOT / "configs" / "synthetic_phase0.json")
        exact = exact_permutation_restoration(model.distance, model.event_costs, budget=model.budget)
        self.assertEqual({event_id for event_id, value in exact.items() if value <= 0}, {5, 6})


if __name__ == "__main__":
    unittest.main()
