"""Prepare the pinned AndroidWorld Dockerfile with reproducible build fixes."""

from __future__ import annotations

import argparse
from pathlib import Path


REMOVED_BASE_IMAGE = "FROM openjdk:18-jdk-slim"
COMPATIBLE_BASE_IMAGE = "FROM eclipse-temurin:17-jdk-jammy"
UNPINNED_UV_INSTALLER = "RUN curl -LsSf https://astral.sh/uv/install.sh | sh"
PINNED_UV_INSTALLER = (
    "RUN curl -LsSf https://astral.sh/uv/0.11.28/install.sh | sh"
)
ISOLATED_PROJECT_INSTALL = "RUN uv pip install . --system"
NON_ISOLATED_PROJECT_INSTALL = (
    "RUN uv pip install wheel==0.45.1 grpcio-tools==1.71.0 --system && \\\n"
    "    uv pip install . --system --no-build-isolation"
)

REPLACEMENTS = {
    REMOVED_BASE_IMAGE: COMPATIBLE_BASE_IMAGE,
    UNPINNED_UV_INSTALLER: PINNED_UV_INSTALLER,
    ISOLATED_PROJECT_INSTALL: NON_ISOLATED_PROJECT_INSTALL,
}


def prepare_dockerfile(source: Path, output: Path) -> None:
    contents = source.read_text(encoding="utf-8")
    rendered = contents
    for original, replacement in REPLACEMENTS.items():
        occurrences = contents.count(original)
        if occurrences != 1:
            raise ValueError(
                f"expected exactly one occurrence of {original!r}, "
                f"found {occurrences}: {source}"
            )
        rendered = rendered.replace(original, replacement)
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
