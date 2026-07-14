"""A deterministic mixed-fidelity behavior model for estimator validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from causalcache.attribution import Coalition


Vector = tuple[float, ...]


def _vector(value: Any, name: str) -> Vector:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty numeric list")
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class SyntheticBehaviorModel:
    event_costs: Mapping[int, int]
    budget: int
    component_weights: Vector
    event_effects: Mapping[int, Vector]
    pair_interactions: Mapping[tuple[int, int], Vector]

    def __post_init__(self) -> None:
        dimension = len(self.component_weights)
        if set(self.event_costs) != set(self.event_effects):
            raise ValueError("event costs and effects must have identical ids")
        if any(len(effect) != dimension for effect in self.event_effects.values()):
            raise ValueError("event effect dimensions must match component weights")
        if any(len(effect) != dimension for effect in self.pair_interactions.values()):
            raise ValueError("interaction dimensions must match component weights")
        if any(left >= right for left, right in self.pair_interactions):
            raise ValueError("interaction keys must be ordered pairs")

    @property
    def event_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.event_costs))

    def behavior(self, coalition: Coalition) -> Vector:
        result = [0.0] * len(self.component_weights)
        for event_id in coalition:
            for index, value in enumerate(self.event_effects[event_id]):
                result[index] += value
        for (left, right), effect in self.pair_interactions.items():
            if left in coalition and right in coalition:
                for index, value in enumerate(effect):
                    result[index] += value
        return tuple(result)

    def distance(self, coalition: Coalition) -> float:
        full_behavior = self.behavior(frozenset(self.event_ids))
        behavior = self.behavior(coalition)
        return sum(
            weight * (reference - current) ** 2
            for weight, reference, current in zip(self.component_weights, full_behavior, behavior)
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SyntheticBehaviorModel":
        costs = {int(event_id): int(cost) for event_id, cost in data["event_costs"].items()}
        effects = {
            int(event_id): _vector(effect, f"event_effects.{event_id}")
            for event_id, effect in data["event_effects"].items()
        }
        interactions: dict[tuple[int, int], Vector] = {}
        for key, effect in data["pair_interactions"].items():
            left, right = (int(item) for item in key.split(","))
            interactions[(left, right)] = _vector(effect, f"pair_interactions.{key}")
        return cls(
            event_costs=costs,
            budget=int(data["budget"]),
            component_weights=_vector(data["component_weights"], "component_weights"),
            event_effects=effects,
            pair_interactions=interactions,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SyntheticBehaviorModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
