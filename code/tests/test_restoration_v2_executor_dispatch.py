import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_restoration_v2_executor_dispatch import (
    FROZEN_ACTION_FIXTURE_SHA256,
    FROZEN_ANDROIDWORLD_REVISION,
    FROZEN_EXECUTION_HOST,
    FROZEN_INSPECTOR_EVIDENCE_TYPE,
    FROZEN_JSON_ACTION_SCHEMA,
    FROZEN_LIVE_SOURCE_SHA256,
    FROZEN_SERVER_IMAGE,
    FROZEN_SERVER_IMAGE_ID,
    NEGATIVE_ACTUATION_CONTROL,
    REPOSITORY_ROOT,
    ExecutorCompatibilityFailure,
    InvalidDispatchEvidence,
    dispatch_actions,
    expected_executor_message,
    load_dispatch_cases,
    validate_dispatch_evidence,
    validate_negative_control_constructor,
    validate_server_provenance,
)


def response_record(
    body: dict[str, object] | None = None,
    *,
    status: int = 200,
    content_type: str = "application/json",
) -> dict[str, object]:
    if body is not None:
        raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    else:
        raw = b"Internal Server Error"
    value: dict[str, object] = {
        "started_at": "2026-07-15T00:00:00+00:00",
        "finished_at": "2026-07-15T00:00:01+00:00",
        "http_status": status,
        "content_type": content_type,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "elapsed_seconds": 0.1,
        "raw_body_utf8": raw.decode("utf-8"),
    }
    if body is not None:
        value["json_body"] = body
    else:
        value["body_text"] = "Internal Server Error"
    return value


class RestorationV2ExecutorDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_path = (
            REPOSITORY_ROOT / "data/fixtures/gui_owl_v2_action_roundtrip.json"
        )
        cls.cases = load_dispatch_cases(cls.fixture_path)

    def _requester(self, calls: list[dict[str, object]]):
        def request(
            base_url: str,
            path: str,
            *,
            method: str = "GET",
            params=None,
            body=None,
            timeout_seconds: float,
        ) -> dict[str, object]:
            calls.append(
                {
                    "base_url": base_url,
                    "path": path,
                    "method": method,
                    "params": params,
                    "body": body,
                    "timeout_seconds": timeout_seconds,
                }
            )
            if path == "/health":
                return response_record({"status": "success"})
            if path == "/reset":
                return response_record(
                    {
                        "status": "success",
                        "message": "Environment reset with go_home=True.",
                    }
                )
            if body == NEGATIVE_ACTUATION_CONTROL:
                return response_record(None, status=500, content_type="text/plain")
            return response_record(
                {
                    "status": "success",
                    "message": expected_executor_message(body),
                }
            )

        return request

    @staticmethod
    def _screenshot_requester(
        base_url: str,
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        return {
            "started_at": "2026-07-15T00:00:00+00:00",
            "finished_at": "2026-07-15T00:00:01+00:00",
            "http_status": 200,
            "content_type": "application/json",
            "response_bytes": 2,
            "response_sha256": "1" * 64,
            "width": 1080,
            "height": 2400,
            "elapsed_seconds": 0.2,
        }

    def _passing_summary(self) -> tuple[dict[str, object], list[dict[str, object]]]:
        calls: list[dict[str, object]] = []
        dispatch = dispatch_actions(
            base_url="http://172.17.0.1:5003",
            cases=self.cases,
            timeout_seconds=120.0,
            requester=self._requester(calls),
            screenshot_requester=self._screenshot_requester,
        )
        summary: dict[str, object] = {
            "evidence_type": "restoration_v2_androidworld_executor_dispatch",
            "status": "passed",
            "started_at": "2026-07-15T00:00:00+00:00",
            "finished_at": "2026-07-15T00:00:02+00:00",
            "policy_output_generated": False,
            "restoration_label_generated": False,
            "interface_validation": {
                "androidworld_json_action_constructor_validation": {
                    "status": "passed",
                    "validated_case_count": 14,
                    "module_path": "/pinned/android_world/agents/new_json_action.py",
                    "module_sha256": FROZEN_LIVE_SOURCE_SHA256[
                        "/android_world/agents/new_json_action.py"
                    ],
                }
            },
            "negative_control_constructor_validation": {
                "status": "passed",
                "payload": NEGATIVE_ACTUATION_CONTROL,
                "json_action_repr": "JSONAction(action_type='click')",
                "json_action_json_str": '{"action_type":"click"}',
                "module_path": "/pinned/android_world/agents/new_json_action.py",
                "module_sha256": FROZEN_LIVE_SOURCE_SHA256[
                    "/android_world/agents/new_json_action.py"
                ],
            },
            "androidworld_executor_dispatch_validation": dispatch,
        }
        return summary, calls

    def test_fixture_is_reparsed_and_rebridged_before_dispatch(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.fixture_path.read_bytes()).hexdigest(),
            FROZEN_ACTION_FIXTURE_SHA256,
        )
        self.assertEqual(len(self.cases), 14)
        for case in self.cases:
            rendered = json.dumps(
                case["payload"], ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(rendered).hexdigest(), case["payload_sha256"])

    def test_dispatch_uses_one_reset_and_one_request_per_case(self) -> None:
        summary, calls = self._passing_summary()
        reduced = validate_dispatch_evidence(summary, cases=self.cases)
        self.assertEqual(reduced["status"], "PASSED_EXECUTOR_DISPATCH")
        self.assertEqual(reduced["validated_case_count"], 14)
        action_calls = [
            call
            for call in calls
            if call["path"] == "/execute_action"
            and call["body"] != NEGATIVE_ACTUATION_CONTROL
        ]
        reset_calls = [call for call in calls if call["path"] == "/reset"]
        self.assertEqual(len(action_calls), 14)
        self.assertEqual(len(reset_calls), 16)
        self.assertEqual([call["body"] for call in action_calls], [c["payload"] for c in self.cases])

    def test_response_echo_and_raw_denominator_are_recomputed(self) -> None:
        summary, _ = self._passing_summary()
        wrong_echo = copy.deepcopy(summary)
        wrong_echo["androidworld_executor_dispatch_validation"]["action_records"][0][
            "response"
        ]["json_body"]["message"] = "Action JSONAction(action_type='wait') executed."
        wrong_response = wrong_echo["androidworld_executor_dispatch_validation"][
            "action_records"
        ][0]["response"]
        wrong_raw = json.dumps(
            wrong_response["json_body"], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        wrong_response["raw_body_utf8"] = wrong_raw.decode("utf-8")
        wrong_response["response_bytes"] = len(wrong_raw)
        wrong_response["response_sha256"] = hashlib.sha256(wrong_raw).hexdigest()
        with self.assertRaises(ExecutorCompatibilityFailure):
            validate_dispatch_evidence(wrong_echo, cases=self.cases)

        mismatched_raw = copy.deepcopy(summary)
        mismatched_raw["androidworld_executor_dispatch_validation"]["action_records"][0][
            "response"
        ]["json_body"]["message"] = "structured-only mutation"
        with self.assertRaisesRegex(InvalidDispatchEvidence, "raw response"):
            validate_dispatch_evidence(mismatched_raw, cases=self.cases)

        missing_record = copy.deepcopy(summary)
        missing_record["androidworld_executor_dispatch_validation"]["action_records"].pop()
        with self.assertRaisesRegex(InvalidDispatchEvidence, "denominator"):
            validate_dispatch_evidence(missing_record, cases=self.cases)

        out_of_interval = copy.deepcopy(summary)
        out_of_interval["androidworld_executor_dispatch_validation"]["action_records"][0][
            "response"
        ]["started_at"] = "2026-07-15T02:00:00+00:00"
        out_of_interval["androidworld_executor_dispatch_validation"]["action_records"][0][
            "response"
        ]["finished_at"] = "2026-07-15T02:00:01+00:00"
        with self.assertRaisesRegex(InvalidDispatchEvidence, "outside the dispatch interval"):
            validate_dispatch_evidence(out_of_interval, cases=self.cases)

    def test_negative_control_and_post_failure_health_are_mandatory(self) -> None:
        summary, _ = self._passing_summary()
        false_negative = copy.deepcopy(summary)
        false_negative["androidworld_executor_dispatch_validation"][
            "negative_actuation_control"
        ]["response"]["http_status"] = 200
        with self.assertRaisesRegex(InvalidDispatchEvidence, "HTTP 500"):
            validate_dispatch_evidence(false_negative, cases=self.cases)

        bad_health = copy.deepcopy(summary)
        bad_health["androidworld_executor_dispatch_validation"][
            "health_after_negative_control"
        ]["json_body"] = {"status": "failed"}
        with self.assertRaisesRegex(InvalidDispatchEvidence, "health"):
            validate_dispatch_evidence(bad_health, cases=self.cases)

        missing_constructor_witness = copy.deepcopy(summary)
        missing_constructor_witness.pop("negative_control_constructor_validation")
        with self.assertRaisesRegex(InvalidDispatchEvidence, "constructor-acceptance"):
            validate_dispatch_evidence(missing_constructor_witness, cases=self.cases)

    def test_negative_control_constructor_accepts_before_live_actuation(self) -> None:
        source_root = Path(__file__).resolve().parents[3] / "missing-androidworld-source"
        with self.assertRaises((ImportError, InvalidDispatchEvidence)):
            validate_negative_control_constructor(source_root)

    def test_live_host_inspection_must_bind_cli_container_identity(self) -> None:
        runtime_id = "b93fb3d81e1eb4f49fc90f0f03965a6cd5b0fb491a7b30e2fc3610c417ccc9aa"
        server_id = "ed6198b1de651de7361ba53e857249516425ab47ae5c07bf37f0455f7ab26b07"
        runtime_image_id = (
            "sha256:81b5df11b32ad8460be270a67066196cb7c6d4fb92cb5d05a44fb06d1ec88d21"
        )
        runtime_repo_digest = (
            "sha256:6a8f60af7ca868dc266c118249d12fc73ba85e2e8075e5e31473bd25d349acfa"
        )
        inspector_path = (
            REPOSITORY_ROOT / "code/scripts/inspect_restoration_v2_executor_container.py"
        )
        attempt_id = "rv2-20260715T000000Z-deadbeef"
        host_record = {
            "evidence_type": FROZEN_INSPECTOR_EVIDENCE_TYPE,
            "status": "passed",
            "host": FROZEN_EXECUTION_HOST,
            "attempt_id": attempt_id,
            "phase": "before",
            "inspector_source_sha256": hashlib.sha256(
                inspector_path.read_bytes()
            ).hexdigest(),
            "runtime_container": {
                "name": "sglang-omni-jaxan-07140435",
                "id": runtime_id,
                "hostname": runtime_id[:12],
                "config_image": "hongccc/sglang-omni:dev",
                "image_id": runtime_image_id,
                "image_repo_digest": f"hongccc/sglang-omni@{runtime_repo_digest}",
                "mounts": {
                    "/data": "/mnt/data6/jiaxuanluo/causalcache",
                    "/root/.cache/huggingface": "/mnt/data6/jiaxuanluo/causalcache/.cache/huggingface",
                },
                "running": True,
            },
            "server_container": {
                "name": "sglang-omni-jaxan-07141630",
                "id": server_id,
                "hostname": server_id[:12],
                "config_image": FROZEN_SERVER_IMAGE,
                "image_id": FROZEN_SERVER_IMAGE_ID,
                "host_port": 5003,
                "container_port": 5000,
                "running": True,
            },
            "live_source_sha256": FROZEN_LIVE_SOURCE_SHA256,
            "health": {
                **response_record({"status": "success"}),
                "url": "http://127.0.0.1:5003/health",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            host_path = Path(directory) / "host.json"
            host_path.write_text(json.dumps(host_record), encoding="utf-8")
            args = argparse.Namespace(
                stack_config=REPOSITORY_ROOT / "code/configs/androidworld_stack.json",
                host_inspection=host_path,
                attempt_id=attempt_id,
                androidworld_source_revision=FROZEN_ANDROIDWORLD_REVISION,
                server_image=FROZEN_SERVER_IMAGE,
                server_image_id=FROZEN_SERVER_IMAGE_ID,
                server_json_action_schema=FROZEN_JSON_ACTION_SCHEMA,
                host_name=FROZEN_EXECUTION_HOST,
                runtime_container_name="sglang-omni-jaxan-07140435",
                runtime_container_id=runtime_id,
                runtime_image_id=runtime_image_id,
                runtime_image_repo_digest=runtime_repo_digest,
                server_container_name="sglang-omni-jaxan-07141630",
                server_container_id=server_id,
                base_url="http://172.17.0.1:5003",
                server_host_port=5003,
                server_container_port=5000,
                timeout_seconds=120.0,
            )
            result = validate_server_provenance(args, runtime_hostname=runtime_id[:12])
            self.assertEqual(result["server_container"]["id"], server_id)

            mutated = copy.deepcopy(host_record)
            mutated["server_container"]["image_id"] = "sha256:" + "0" * 64
            host_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(InvalidDispatchEvidence, "docker inspection"):
                validate_server_provenance(args, runtime_hostname=runtime_id[:12])


if __name__ == "__main__":
    unittest.main()
