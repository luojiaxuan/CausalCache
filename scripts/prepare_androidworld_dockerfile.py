"""Prepare the pinned AndroidWorld Dockerfile with a reproducible base-image fix."""

from __future__ import annotations

import argparse
from pathlib import Path


REMOVED_BASE_IMAGE = "FROM openjdk:18-jdk-slim"
COMPATIBLE_BASE_IMAGE = "FROM eclipse-temurin:17-jdk-jammy"


def prepare_dockerfile(source: Path, output: Path) -> None:
    contents = source.read_text(encoding="utf-8")
    occurrences = contents.count(REMOVED_BASE_IMAGE)
    if occurrences != 1:
        raise ValueError(
            f"expected exactly one pinned base image, found {occurrences}: {source}"
        )

    rendered = contents.replace(REMOVED_BASE_IMAGE, COMPATIBLE_BASE_IMAGE)
    output.write_text(rendered, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dockerfile(args.source, args.output)


if __name__ == "__main__":
    main()
