import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from causalcache.restoration_v2_contract import RestorationV2Contract
from scripts.validate_restoration_v2_interfaces import (
    _read_json,
    _validate_androidworld_payloads,
    _validate_implementation_against_contract,
    _validate_interface_manifest,
    validate_interfaces,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class RestorationV2InterfaceValidatorTest(unittest.TestCase):
    def test_cpu_interface_validator_is_complete_and_does_not_claim_executor_smoke(self) -> None:
        summary = validate_interfaces(
            contract_path=REPOSITORY_ROOT / "code/configs/causalcache_restoration_v2.json",
            action_fixture_path=REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json",
            prompt_fixture_path=REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json",
            interface_manifest_path=REPOSITORY_ROOT / "data/manifests/restoration_v2_interfaces.json",
        )
        self.assertEqual(summary["action_fixture"]["valid_case_count"], 14)
        self.assertEqual(summary["action_fixture"]["invalid_case_count"], 23)
        self.assertEqual(summary["action_fixture"]["coordinate_scalar_checks"], 6000)
        self.assertEqual(summary["prompt_low_fidelity_fixture"]["coalition_count"], 28)
        self.assertEqual(summary["prompt_low_fidelity_fixture"]["confirm_coalition_count"], 16)
        self.assertEqual(summary["interface_manifest"]["source_file_count"], 6)
        self.assertEqual(
            summary["androidworld_json_action_constructor_validation"]["status"],
            "not_run",
        )
        self.assertFalse(summary["policy_output_generated_by_this_validation"])

    def test_partial_androidworld_provenance_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires source root/revision"):
            validate_interfaces(
                contract_path=REPOSITORY_ROOT / "code/configs/causalcache_restoration_v2.json",
                action_fixture_path=REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json",
                prompt_fixture_path=REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json",
                interface_manifest_path=REPOSITORY_ROOT / "data/manifests/restoration_v2_interfaces.json",
                androidworld_source_root=REPOSITORY_ROOT,
            )

    def test_alternate_fixture_and_duplicate_json_keys_are_rejected(self) -> None:
        fixture = json.loads(
            (REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json").read_text(
                encoding="utf-8"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "action.json"
            alternate.write_text(json.dumps(fixture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical repository-pinned"):
                validate_interfaces(
                    contract_path=REPOSITORY_ROOT / "code/configs/causalcache_restoration_v2.json",
                    action_fixture_path=alternate,
                    prompt_fixture_path=REPOSITORY_ROOT / "data/fixtures/restoration_v2_prompt_low_fidelity.json",
                    interface_manifest_path=REPOSITORY_ROOT / "data/manifests/restoration_v2_interfaces.json",
                )
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"key":1,"key":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _read_json(duplicate)

    def test_manifest_cannot_fake_constructor_pass_or_source_hash(self) -> None:
        manifest = json.loads(
            (REPOSITORY_ROOT / "data/manifests/restoration_v2_interfaces.json").read_text(
                encoding="utf-8"
            )
        )
        action_summary = {
            "valid_case_count": 14,
            "invalid_case_count": 23,
            "coordinate_scalar_checks": 6000,
        }
        prompt_summary = {"coalition_count": 28, "confirm_coalition_count": 16}
        mutations = []
        fake_pass = copy.deepcopy(manifest)
        fake_pass["androidworld_json_action_constructor_validation"]["status"] = "passed"
        mutations.append((fake_pass, "keep executor integration pending"))
        bad_hash = copy.deepcopy(manifest)
        bad_hash["files"][0]["sha256"] = "0" * 64
        mutations.append((bad_hash, "source SHA256 mismatch"))
        with tempfile.TemporaryDirectory() as directory:
            for index, (mutation, error) in enumerate(mutations):
                path = Path(directory) / f"manifest-{index}.json"
                path.write_text(json.dumps(mutation), encoding="utf-8")
                with self.subTest(error=error):
                    with self.assertRaisesRegex(ValueError, error):
                        _validate_interface_manifest(
                            path,
                            scientific_contract_sha256=manifest["scientific_contract_sha256"],
                            action_summary=action_summary,
                            prompt_summary=prompt_summary,
                        )

    def test_implementation_constants_are_bound_to_the_scientific_contract(self) -> None:
        contract = RestorationV2Contract.load(
            REPOSITORY_ROOT / "code/configs/causalcache_restoration_v2.json"
        )
        drifted = copy.deepcopy(contract.data)
        drifted["action_contract"]["accepted_model_aliases"]["tap"] = "long_press"
        with self.assertRaisesRegex(ValueError, "aliases drifted"):
            _validate_implementation_against_contract(drifted)

    def test_constructor_integration_rejects_nonfrozen_source_revision(self) -> None:
        run_commit = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        with self.assertRaisesRegex(ValueError, "not the frozen MobileAgent revision"):
            _validate_androidworld_payloads(
                [],
                REPOSITORY_ROOT,
                source_revision="0" * 40,
                container_image_digest="sha256:" + "0" * 64,
                run_git_commit=run_commit,
            )


if __name__ == "__main__":
    unittest.main()
