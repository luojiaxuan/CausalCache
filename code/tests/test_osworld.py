from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from causalcache.osworld import (
    DesktopAction,
    OSWorldHistoryEvent,
    OSWorldTask,
    ScriptedOSWorldPolicy,
    build_osworld_policy_request,
    configure_osworld_docker_runtime,
    load_osworld_inventory,
    load_osworld_task,
    run_osworld_episode,
    select_osworld_memory,
)


class FakeEnvironment:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.actions: list[str] = []
        self.closed = False
        self.evaluate_calls = 0

    def reset(self, *, task_config: dict[str, Any]) -> dict[str, Any]:
        self.reset_calls += 1
        return {"screenshot": b"initial-png"}

    def step(
        self, action: str, pause: float = 2
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.actions.append(action)
        return {"screenshot": f"png-{len(self.actions)}".encode()}, 0.0, action == "DONE", {}

    def evaluate(self) -> float:
        self.evaluate_calls += 1
        return 1.0

    def close(self) -> None:
        self.closed = True


class OSWorldActionTests(unittest.TestCase):
    def test_renders_supported_actions(self) -> None:
        self.assertEqual(
            DesktopAction.from_mapping(
                {"type": "click", "x": 10, "y": 20}
            ).to_osworld(),
            "pyautogui.click(10, 20, clicks=1, interval=0.1, button='left')",
        )
        self.assertEqual(
            DesktopAction.from_mapping(
                {"type": "hotkey", "keys": ["CTRL", "L"]}
            ).to_osworld(),
            "pyautogui.hotkey('ctrl', 'l')",
        )
        self.assertEqual(
            DesktopAction.from_mapping({"type": "done"}).to_osworld(), "DONE"
        )

    def test_rejects_injection_and_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DesktopAction.from_mapping(
                {"type": "click", "x": 10, "y": 20, "code": "rm -rf /"}
            )
        with self.assertRaises(ValueError):
            DesktopAction.from_mapping({"type": "click", "x": 1920, "y": 20})
        action = DesktopAction.from_mapping(
            {"type": "type_text", "text": "x'); import os; #"}
        )
        self.assertEqual(
            action.to_osworld(), "pyautogui.write(\"x'); import os; #\", interval=0.01)"
        )

    def test_full_memory_ignores_budget(self) -> None:
        history = tuple(
            OSWorldHistoryEvent(
                step_id=index,
                action={"type": "wait"},
                osworld_action="WAIT",
                result_status="executed",
                screen_changed=False,
                post_screenshot=b"png",
            )
            for index in range(1, 4)
        )
        self.assertEqual(
            select_osworld_memory(history, arm="full", budget=1), (1, 2, 3)
        )


class OSWorldTaskTests(unittest.TestCase):
    def test_rejects_invalid_docker_cpu_model(self) -> None:
        with self.assertRaises(ValueError):
            configure_osworld_docker_runtime(
                ".", dns_server="127.0.0.11", cpu_model="host with spaces"
            )
        with self.assertRaises(ValueError):
            configure_osworld_docker_runtime(
                ".", dns_server="127.0.0.11", port_lock_timeout_seconds=0
            )

    def test_inventory_and_task_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = root / "evaluation_examples/examples/writer"
            examples.mkdir(parents=True)
            (root / "evaluation_examples/test_all.json").write_text(
                json.dumps({"writer": ["task-1"]}), encoding="utf-8"
            )
            (examples / "task-1.json").write_text(
                json.dumps(
                    {
                        "id": "task-1",
                        "instruction": "Write a note",
                        "config": [],
                        "evaluator": {"func": "dummy"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_osworld_inventory(root), (("writer", "task-1"),)
            )
            task = load_osworld_task(root, domain="writer", task_id="task-1")
            self.assertEqual(task.instruction, "Write a note")

    def test_memory_selection_and_request(self) -> None:
        task = OSWorldTask(
            domain="writer",
            task_id="task-1",
            config_path=Path("task.json"),
            config={"instruction": "Write"},
        )
        history = tuple(
            OSWorldHistoryEvent(
                step_id=index,
                action={"type": "wait"},
                osworld_action="WAIT",
                result_status="executed",
                screen_changed=False,
                post_screenshot=f"screen-{index}".encode(),
            )
            for index in range(1, 4)
        )
        self.assertEqual(
            select_osworld_memory(history, arm="recent", budget=2), (2, 3)
        )
        request = build_osworld_policy_request(
            task=task,
            step_id=4,
            screenshot=b"current",
            history=history,
            selected_event_step_ids=(2, 3),
            screen_size=(1920, 1080),
        )
        self.assertIsNone(request["history"][0]["restored_post_screenshot_png_base64"])
        self.assertIsNotNone(
            request["history"][1]["restored_post_screenshot_png_base64"]
        )


class OSWorldEpisodeTests(unittest.TestCase):
    def test_capacity_episode_can_skip_task_evaluator(self) -> None:
        task = OSWorldTask(
            domain="writer",
            task_id="task-capacity",
            config_path=Path("task.json"),
            config={"id": "task-capacity", "instruction": "Write"},
        )
        environment = FakeEnvironment()
        policy = ScriptedOSWorldPolicy({"task-capacity": [{"type": "wait"}]})
        with tempfile.TemporaryDirectory() as directory:
            result = run_osworld_episode(
                environment=environment,
                policy=policy,
                task=task,
                output_root=directory,
                memory_arm="recent",
                memory_budget=4,
                max_steps=1,
                pause_seconds=0.0,
                screen_size=(1920, 1080),
                provenance={"capacity": True},
                evaluate_at_end=False,
            )
        self.assertFalse(result["evaluation_executed"])
        self.assertIsNone(result["score"])
        self.assertIsNone(result["success"])
        self.assertEqual(environment.evaluate_calls, 0)

    def test_episode_and_task_level_resume(self) -> None:
        task = OSWorldTask(
            domain="writer",
            task_id="task-1",
            config_path=Path("task.json"),
            config={"id": "task-1", "instruction": "Write"},
        )
        policy = ScriptedOSWorldPolicy(
            {"task-1": [{"type": "click", "x": 10, "y": 20}, {"type": "done"}]}
        )
        environment = FakeEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            result = run_osworld_episode(
                environment=environment,
                policy=policy,
                task=task,
                output_root=directory,
                memory_arm="recent",
                memory_budget=4,
                max_steps=5,
                pause_seconds=0.0,
                screen_size=(1920, 1080),
                provenance={"test": True},
            )
            self.assertEqual(result["status"], "COMPLETE_OSWORLD_EPISODE")
            self.assertEqual(result["completed_steps"], 2)
            self.assertEqual(result["score"], 1.0)
            self.assertEqual(environment.evaluate_calls, 1)
            self.assertEqual(environment.reset_calls, 1)
            second = run_osworld_episode(
                environment=environment,
                policy=policy,
                task=task,
                output_root=directory,
                memory_arm="recent",
                memory_budget=4,
                max_steps=5,
                pause_seconds=0.0,
                screen_size=(1920, 1080),
                provenance={"test": True},
            )
            self.assertTrue(second["resumed_skip"])
            self.assertEqual(environment.reset_calls, 1)


if __name__ == "__main__":
    unittest.main()
