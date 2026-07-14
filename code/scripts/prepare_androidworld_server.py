"""Align the pinned MobileAgent HTTP server with its GUI-Owl action schema."""

from __future__ import annotations

import argparse
from pathlib import Path


LEGACY_JSON_ACTION_IMPORT = "from android_world.env import json_action"
PINNED_AGENT_JSON_ACTION_IMPORT = (
    "from android_world.agents import new_json_action as json_action"
)


def prepare_server(source: Path) -> None:
    contents = source.read_text(encoding="utf-8")
    occurrences = contents.count(LEGACY_JSON_ACTION_IMPORT)
    if occurrences != 1:
        raise ValueError(
            "expected exactly one legacy JSONAction import, "
            f"found {occurrences}: {source}"
        )
    rendered = contents.replace(
        LEGACY_JSON_ACTION_IMPORT,
        PINNED_AGENT_JSON_ACTION_IMPORT,
    )
    source.write_text(rendered, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_server(args.source)


if __name__ == "__main__":
    main()
