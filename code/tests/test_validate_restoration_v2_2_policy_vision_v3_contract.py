from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    CANONICAL_CONFIG_PATH,
    FROZEN_CONFIG_SHA256,
    PASS_STATUS,
    PROTOCOL_ID,
)
from scripts.validate_restoration_v2_2_policy_vision_v3_contract import (
    main as validate_main,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class ValidateRestorationV22PolicyVisionV3ContractTest(unittest.TestCase):
    def test_cli_emits_locked_contract_identity(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            validate_main(
                [
                    "--contract",
                    str(CONFIG),
                    "--repository-root",
                    str(ROOT),
                ]
            )
        result = json.loads(stream.getvalue())
        self.assertEqual(result["status"], PASS_STATUS)
        self.assertEqual(result["protocol_id"], PROTOCOL_ID)
        self.assertEqual(result["config_sha256"], FROZEN_CONFIG_SHA256)
        self.assertTrue(result["prior_outputs_absent"])
        self.assertTrue(result["output_absent"])

    def test_cli_requires_absent_output(self) -> None:
        with patch(
            "scripts.validate_restoration_v2_2_policy_vision_v3_contract."
            "validate_contract",
            side_effect=ValueError(
                "policy-vision v3 output or staging directory already exists"
            ),
        ), self.assertRaisesRegex(ValueError, "already exists"):
            validate_main(
                [
                    "--contract",
                    str(CONFIG),
                    "--repository-root",
                    str(ROOT),
                ]
            )


if __name__ == "__main__":
    unittest.main()
