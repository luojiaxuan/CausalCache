"""Upstream MobileWorld agent shim for the shared CausalCache policy service."""

from __future__ import annotations

import hashlib
import io
from typing import Any

from mobile_world.agents.base import BaseAgent
from mobile_world.runtime.utils.models import JSONAction

from causalcache.mobileworld import (
    HTTPMobileWorldPolicy,
    MobileWorldHistoryEvent,
    build_mobileworld_policy_request,
    select_mobileworld_memory,
)


def _png_bytes(image: Any) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class CausalCacheMobileWorldGUIOwlAgent(BaseAgent):
    """Frozen GUI-Owl with explicit recent-memory selection and shared inference."""

    def __init__(
        self,
        model_name: str,
        llm_base_url: str,
        api_key: str = "empty",
        *,
        memory_arm: str = "recent",
        memory_budget: int = 4,
        policy_timeout_seconds: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del model_name, api_key, kwargs
        endpoint = llm_base_url.rstrip("/")
        if not endpoint.endswith("/act"):
            endpoint += "/act"
        self.policy = HTTPMobileWorldPolicy(
            endpoint, timeout_seconds=policy_timeout_seconds
        )
        self.memory_arm = memory_arm
        self.memory_budget = memory_budget
        self.reset()

    def reset(self) -> None:
        self.history: list[MobileWorldHistoryEvent] = []
        self.previous_action: dict[str, Any] | None = None
        self.previous_screenshot: bytes | None = None
        self.instruction = None

    def predict(self, observation: dict[str, Any]) -> tuple[str, JSONAction]:
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise RuntimeError(
                "MobileWorld called predict before BaseAgent.initialize supplied "
                "a task instruction"
            )
        image = observation["screenshot"]
        current = _png_bytes(image)
        if self.previous_action is not None:
            self.history.append(
                MobileWorldHistoryEvent(
                    step_id=len(self.history) + 1,
                    action=self.previous_action,
                    screen_changed=(
                        self.previous_screenshot is None
                        or hashlib.sha256(current).digest()
                        != hashlib.sha256(self.previous_screenshot).digest()
                    ),
                    post_screenshot=current,
                )
            )
        selected = select_mobileworld_memory(
            self.history,
            arm=self.memory_arm,
            budget=self.memory_budget,
        )
        request = build_mobileworld_policy_request(
            instruction=self.instruction,
            step_id=len(self.history) + 1,
            screenshot=current,
            history=self.history,
            selected_event_step_ids=selected,
            screen_size=(int(image.width), int(image.height)),
        )
        decision = self.policy.act(request)
        self.previous_action = dict(decision.action)
        self.previous_screenshot = current
        prediction = str(
            decision.raw_response.get("native_output", decision.action)
        )
        return prediction, JSONAction(**decision.action)

    def get_total_token_usage(self) -> dict[str, int]:
        return {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
