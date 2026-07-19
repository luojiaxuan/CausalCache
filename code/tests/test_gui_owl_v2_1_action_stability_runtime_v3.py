from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_is_parent_sdpa_with_explicit_strict_identity() -> None:
    script = """
import json
import os
import causalcache.policy.gui_owl_v2_1_action_stability_runtime_v3 as runtime_v3
print(json.dumps({
    "cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "base_name": runtime_v3.GUIOwlV21StrictDeterminismActionStabilityRuntimeV3.__mro__[1].__name__,
    "runtime_id": runtime_v3.GUI_OWL_V2_1_ACTION_STABILITY_STRICT_RUNTIME_V3_ID,
}))
"""
    env = dict(os.environ)
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    env["PYTHONPATH"] = str(ROOT / "code")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(result.stdout)
    assert payload["cublas"] == ":4096:8"
    assert payload["base_name"] == "GUIOwlV21AutoActionStabilityRuntimeV1"
    assert "strict" in payload["runtime_id"]


def test_attention_remains_exact_sdpa() -> None:
    script = """
from causalcache.policy.gui_owl_v2_1_action_stability_runtime_v3 import validate_strict_sdpa_attention_v3
expected = {"text": "sdpa", "top": "sdpa", "vision": "sdpa"}
assert validate_strict_sdpa_attention_v3(expected) == expected
try:
    validate_strict_sdpa_attention_v3({"text": "sdpa", "top": "sdpa", "vision": "eager"})
except RuntimeError:
    print("ok")
else:
    raise AssertionError("eager vision backend was accepted")
"""
    env = dict(os.environ)
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    env["PYTHONPATH"] = str(ROOT / "code")
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"
