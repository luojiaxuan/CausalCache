from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_restoration_v2_baselines import validate_baseline_manifest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/restoration_v2_baselines.json"


class RestorationV2BaselineManifestTest(unittest.TestCase):
    def test_committed_manifest_closes_dependency_without_policy_output(self) -> None:
        result = validate_baseline_manifest(MANIFEST, repository_root=ROOT)
        self.assertEqual(result["outcome"], "PASSED_BASELINE_SOURCE_VALIDATION")
        self.assertTrue(result["dependency_6_closed"])
        self.assertFalse(result["policy_output_generated"])

    def test_manifest_mutations_fail_closed(self) -> None:
        canonical = json.loads(MANIFEST.read_text(encoding="utf-8"))
        mutations = []
        missing_source = copy.deepcopy(canonical)
        missing_source["source_files"].pop()
        mutations.append(missing_source)
        random_seed = copy.deepcopy(canonical)
        random_seed["implementation_contract"]["uniform_random_seed"] = 20270715
        mutations.append(random_seed)
        deepstack = copy.deepcopy(canonical)
        deepstack["implementation_contract"]["policy_vision"][
            "all_deepstack_features_excluded"
        ] = False
        mutations.append(deepstack)
        generated = copy.deepcopy(canonical)
        generated["negative_declarations"]["policy_output_generated"] = True
        mutations.append(generated)
        open_dependency = copy.deepcopy(canonical)
        open_dependency["dependency_6_closed"] = False
        mutations.append(open_dependency)

        for index, value in enumerate(mutations):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "manifest.json"
                    path.write_text(
                        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        validate_baseline_manifest(path, repository_root=ROOT)


if __name__ == "__main__":
    unittest.main()
