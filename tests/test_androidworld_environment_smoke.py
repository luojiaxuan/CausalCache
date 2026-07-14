import hashlib
import unittest

from scripts.run_androidworld_environment_smoke import (
    build_url,
    summarize_screenshot,
)


class AndroidWorldEnvironmentSmokeTest(unittest.TestCase):
    def test_build_url_encodes_query_parameters(self) -> None:
        self.assertEqual(
            build_url(
                "http://127.0.0.1:5000/",
                "/task/goal",
                {"task_type": "System Wifi", "task_idx": 0},
            ),
            "http://127.0.0.1:5000/task/goal?task_type=System+Wifi&task_idx=0",
        )

    def test_screenshot_summary_requires_pixels_payload(self) -> None:
        payload = b'{"pixels":[[[0,1,2]]]}'
        self.assertEqual(
            summarize_screenshot(payload, "application/json"),
            {
                "content_type": "application/json",
                "response_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
        with self.assertRaisesRegex(ValueError, "pixels array"):
            summarize_screenshot(b'{"status":"success"}', "application/json")


if __name__ == "__main__":
    unittest.main()
