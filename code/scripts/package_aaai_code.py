#!/usr/bin/env python3
"""Build and verify the anonymized, code-only AAAI submission archive."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


ARCHIVE_ROOT = "causalcache_aaai27_code"
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt"}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".bin",
    ".csv",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pdf",
    ".png",
    ".pt",
    ".safetensors",
    ".tar",
    ".zip",
}
PAPER_CONFIGS = {
    "code/configs/causalcache_desktop_did_full_lora_v4.json",
    "code/configs/causalcache_desktop_did_hgkv_v4.json",
    "code/configs/causalcache_desktop_did_hgkv_v4_32b.json",
    "code/configs/causalcache_desktop_did_ungated_kv_v4.json",
    "code/configs/causalcache_mobileworld_official_b0_v2.json",
    "code/configs/causalcache_mobileworld_official_b4_hgkv_sel_v1.json",
    "code/configs/causalcache_mobileworld_official_b4_hgkv_v1.json",
    "code/configs/causalcache_mobileworld_official_b4_sel_frozen_v1.json",
    "code/configs/causalcache_mobileworld_official_b4_v2.json",
    "code/configs/causalcache_paper_evaluation.json",
    "code/configs/gui_owl_1_5_8b_snapshot.json",
}
REQUIRED_MEMBERS = {
    "README.md",
    "MANIFEST.sha256",
    "SOURCE_REVISION.txt",
    "pyproject.toml",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_revision(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _anonymize(payload: bytes, source: Path) -> bytes:
    if source.suffix not in TEXT_SUFFIXES:
        return payload
    text = payload.decode("utf-8")
    text = re.sub(r"(?i)luojiaxuan|jiaxuanluo|jaxan", "anonymous", text)
    text = text.replace("963500310@qq.com", "anonymous@example.invalid")
    text = text.replace("aries.cs.ucsb.edu", "remote-host.example")
    return text.encode("utf-8")


def _source_files(repository_root: Path) -> dict[str, Path]:
    mapped: dict[str, Path] = {
        "README.md": repository_root / "code/SUBMISSION_README.md",
        "pyproject.toml": repository_root / "pyproject.toml",
    }
    for relative_root in (
        "code/causalcache",
        "code/integrations",
        "code/scripts",
        "code/tests",
    ):
        for source in sorted((repository_root / relative_root).rglob("*.py")):
            mapped[source.relative_to(repository_root).as_posix()] = source
    for source in sorted((repository_root / "code/requirements").glob("*.txt")):
        mapped[source.relative_to(repository_root).as_posix()] = source
    for relative in sorted(PAPER_CONFIGS):
        mapped[relative] = repository_root / relative
    missing = sorted(name for name, source in mapped.items() if not source.is_file())
    if missing:
        raise FileNotFoundError(f"submission sources missing: {missing}")
    symlinks = sorted(name for name, source in mapped.items() if source.is_symlink())
    if symlinks:
        raise ValueError(f"submission sources must not be symlinks: {symlinks}")
    return mapped


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=(2026, 7, 31, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_archive(repository_root: Path, output: Path) -> dict[str, str]:
    repository_root = repository_root.resolve()
    entries: dict[str, bytes] = {}
    for archive_name, source in _source_files(repository_root).items():
        entries[archive_name] = _anonymize(source.read_bytes(), source)
    entries["SOURCE_REVISION.txt"] = (
        f"source_git_commit={_git_revision(repository_root)}\n".encode("ascii")
    )
    manifest = "".join(
        f"{_sha256(payload)}  {name}\n" for name, payload in sorted(entries.items())
    )
    entries["MANIFEST.sha256"] = manifest.encode("ascii")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compresslevel=9) as archive:
        for name, payload in sorted(entries.items()):
            archive.writestr(_zip_info(name), payload)
    temporary.replace(output)
    verify_archive(output)
    return {name: _sha256(payload) for name, payload in sorted(entries.items())}


def verify_archive(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        prefix = f"{ARCHIVE_ROOT}/"
        if any(not name.startswith(prefix) for name in names):
            raise ValueError("archive member escaped the required root")
        relative_names = [name.removeprefix(prefix) for name in names]
        if not REQUIRED_MEMBERS.issubset(relative_names):
            raise ValueError("archive is missing required metadata")
        for relative in relative_names:
            path_parts = PurePosixPath(relative).parts
            if not path_parts:
                raise ValueError("empty archive member")
            if path_parts[0] in {"data", "docs", "output", "paper", ".git"}:
                raise ValueError(f"forbidden top-level payload: {relative}")
            if PurePosixPath(relative).suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
                raise ValueError(f"data/model/output file entered archive: {relative}")
            if "runscripts" in path_parts or "__pycache__" in path_parts:
                raise ValueError(f"machine-specific or cached path entered archive: {relative}")

        payloads = {
            relative: archive.read(f"{prefix}{relative}") for relative in relative_names
        }
        manifest_lines = payloads["MANIFEST.sha256"].decode("ascii").splitlines()
        expected = {}
        for line in manifest_lines:
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
        actual = {
            relative: _sha256(payload)
            for relative, payload in payloads.items()
            if relative != "MANIFEST.sha256"
        }
        if expected != actual:
            raise ValueError("archive SHA-256 manifest does not match payload")
        joined = b"\n".join(payloads.values()).lower()
        for marker in (
            b"luojiaxuan",
            b"jiaxuanluo",
            b"jaxan",
            b"9635" + b"00310",
        ):
            if marker in joined:
                raise ValueError(f"author-specific identifier remains: {marker!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_archive(args.output)
        print(f"VALID_CODE_ONLY_ARCHIVE {args.output.resolve()}")
        return
    hashes = build_archive(args.repository_root, args.output)
    print(f"BUILT_CODE_ONLY_ARCHIVE files={len(hashes)} path={args.output.resolve()}")


if __name__ == "__main__":
    main()
