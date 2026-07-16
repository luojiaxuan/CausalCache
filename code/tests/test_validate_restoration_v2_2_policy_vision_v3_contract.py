from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from causalcache.restoration_v2_2_policy_vision_v3_contract import (
    CANONICAL_CONFIG_PATH,
)
from scripts.validate_restoration_v2_2_policy_vision_v3_contract import (
    main as validate_main,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / CANONICAL_CONFIG_PATH


class ValidateRestorationV22PolicyVisionV3ContractTest(unittest.TestCase):
    def test_cli_is_tombstoned_after_canonical_output_publish(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaisesRegex(
            ValueError,
            "output or staging directory already exists",
        ):
            validate_main(
                [
                    "--contract",
                    str(CONFIG),
                    "--repository-root",
                    str(ROOT),
                ]
            )
        self.assertEqual(stream.getvalue(), "")

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
