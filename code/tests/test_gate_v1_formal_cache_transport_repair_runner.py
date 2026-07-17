from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from causalcache import gate_v1_formal_cache as cache_core
from scripts import manage_gate_v1_formal_cache_transport_repair as repair_manager


class TransportRepairRunnerTest(unittest.TestCase):
    def _execution_args(self, command: str = "run") -> SimpleNamespace:
        return SimpleNamespace(
            command=command,
            repository_root=Path("/repo"),
            contract="repair.json",
            execution_b_git_commit="b" * 40,
            hf_token_file=Path("/token"),
            data_root=Path("/data"),
            fresh_download_parent=Path("/fresh"),
        )

    def test_parser_uses_repair_contract_by_default(self) -> None:
        args = repair_manager._parser().parse_args(
            ["validate-source", "--repository-root", "/repo"]
        )
        self.assertEqual(args.contract, repair_manager.CANONICAL_CONFIG_PATH)

    def test_adapter_binds_only_repair_semantic_builders(self) -> None:
        adapter = repair_manager._repair_cache_api()
        self.assertIs(
            adapter.build_formal_feature_cache,
            cache_core.build_formal_feature_cache_transport_repair_v1,
        )
        self.assertIs(
            adapter.build_formal_label_cache,
            cache_core.build_formal_label_cache_transport_repair_v1,
        )
        self.assertIs(
            adapter.audit_formal_cache_join,
            cache_core.audit_formal_cache_join_transport_repair_v1,
        )
        self.assertFalse(hasattr(adapter, "build_formal_feature_cache_transport_repair_v1"))
        self.assertEqual(set(adapter.DOWNLOAD_KEYS), set(cache_core.DOWNLOAD_KEYS))

    def test_source_validation_combines_overlay_and_base_runner_validation(self) -> None:
        contract = object()
        with (
            mock.patch.object(
                repair_manager,
                "load_frozen_transport_repair_contract",
                return_value=contract,
            ) as load,
            mock.patch.object(
                repair_manager,
                "validate_transport_repair_source_only_contract",
                return_value={"status": "overlay-ok"},
            ) as overlay,
            mock.patch.object(
                repair_manager,
                "validate_source_a",
                return_value={"status": "runner-ok"},
            ) as runner,
        ):
            result = repair_manager._run(
                SimpleNamespace(
                    command="validate-source",
                    repository_root=Path("/repo"),
                    contract="repair.json",
                    source_a_git_commit="a" * 40,
                )
            )
        self.assertEqual(result, {"status": "overlay-ok", "runner": {"status": "runner-ok"}})
        load.assert_called_once_with("repair.json", repository_root=Path("/repo"))
        overlay.assert_called_once_with("repair.json", repository_root=Path("/repo"))
        runner.assert_called_once_with(contract, expected_source_a_git_commit="a" * 40)

    def test_failed_v1_validation_precedes_token_hub_and_new_runner_claim(self) -> None:
        events: list[str] = []
        contract = object()
        hooks = object()
        api = object()
        fake_hub = types.ModuleType("huggingface_hub")

        class FakeHfApi:
            def __init__(self, *, token: str) -> None:
                self.token = token
                events.append("api")

        def fake_download(**_kwargs: object) -> str:
            return "/download"

        fake_hub.CommitOperationAdd = object
        fake_hub.HfApi = FakeHfApi
        fake_hub.hf_hub_download = fake_download
        with (
            mock.patch.object(
                repair_manager,
                "load_frozen_transport_repair_contract",
                return_value=contract,
            ),
            mock.patch.object(
                repair_manager,
                "validate_failed_v1_attempt",
                side_effect=lambda *_args: events.append("old-v1")
                or {"new_claim_permitted": True},
            ),
            mock.patch.object(
                repair_manager.base_manager,
                "_secure_token",
                side_effect=lambda _path: events.append("token") or "secret",
            ),
            mock.patch.object(
                repair_manager.base_manager,
                "_hooks",
                side_effect=lambda *_args, **_kwargs: events.append("hooks") or hooks,
            ),
            mock.patch.object(
                repair_manager,
                "execute_formal_cache",
                side_effect=lambda **kwargs: events.append("engine")
                or {"status": "done", "api": kwargs["api"]},
            ),
            mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}, clear=False),
        ):
            result = repair_manager._run(self._execution_args())
        self.assertEqual(events, ["old-v1", "token", "hooks", "api", "engine"])
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["failed_v1_attempt"], {"new_claim_permitted": True})

    def test_failed_v1_validation_stops_before_token_or_hub_access(self) -> None:
        contract = object()
        with (
            mock.patch.object(
                repair_manager,
                "load_frozen_transport_repair_contract",
                return_value=contract,
            ),
            mock.patch.object(
                repair_manager,
                "validate_failed_v1_attempt",
                side_effect=ValueError("old claim drifted"),
            ),
            mock.patch.object(repair_manager.base_manager, "_secure_token") as token,
            mock.patch.object(repair_manager.base_manager, "_hooks") as hooks,
        ):
            with self.assertRaisesRegex(ValueError, "old claim drifted"):
                repair_manager._run(self._execution_args())
        token.assert_not_called()
        hooks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
