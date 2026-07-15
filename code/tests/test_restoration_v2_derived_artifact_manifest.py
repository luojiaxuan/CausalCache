from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.validate_restoration_v2_derived_artifact import (
    ARTIFACT_ID,
    COMPLETION_SOURCE_PATHS,
    DATASET_REPO,
    EXACT_FILE_ALLOWLIST,
    EXPECTED_CHECKOUT_TRANSPORT,
    EXPECTED_COUNTS,
    EXPECTED_ROLE_COUNTS,
    EXPECTED_VALIDATION_IDS,
    HF_CLI_PATH,
    HF_TAG,
    HF_TAG_MESSAGE,
    HF_UPLOAD_COMMIT_MESSAGE,
    HISTORICAL_GENERATOR_PATHS,
    HISTORICAL_INPUT_PATHS,
    HISTORICAL_TEST_PATH,
    PAYLOAD_ARTIFACT_ID,
    PAYLOAD_PREFIX,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    STATUS,
    SUMMARY_OUTCOME,
    VALIDATOR_OUTCOME,
    canonical_json_bytes,
    pretty_json_bytes,
    sha256_bytes,
    validate_completion_manifest,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: object) -> None:
    _write(path, pretty_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


class CompletionFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "data/manifests/restoration_v2_derived_artifact.json"
        self.summary_path = root / COMPLETION_SOURCE_PATHS[-1]
        self.historical_payloads = self._historical_payloads()
        self._commit_historical_snapshot()
        self._write_completion_evidence()

    @staticmethod
    def _selection() -> dict:
        roles = {}
        for role, count in EXPECTED_ROLE_COUNTS.items():
            roles[role] = {
                "trajectories": [
                    {"source_id": f"{role}-{index:02d}"} for index in range(count)
                ],
                "states": [],
            }
        return {"roles": roles}

    def _historical_payloads(self) -> dict[str, bytes]:
        v1_config = {
            "source_pool": {
                "upstream_repo": "fixture/upstream",
                "upstream_revision": "1" * 40,
                "transport_repo": "fixture/transport",
                "transport_revision": "2" * 40,
                "license": "cc-by-4.0",
            }
        }
        backend_config = {"backend_id": "fixture-rapidocr-v1"}
        backend_manifest = {
            "hf_model_artifact": {
                "repo": "fixture/ocr-model",
                "immutable_revision": "3" * 40,
                "tag": "v1.0.0",
            },
            "real_screen_golden": {"immutable_revision": "4" * 40},
        }
        values: dict[str, object] = {
            HISTORICAL_INPUT_PATHS["scientific_contract"]: {"fixture": "contract"},
            HISTORICAL_INPUT_PATHS["v1_config"]: v1_config,
            HISTORICAL_INPUT_PATHS["source_file_manifest"]: {"fixture": "sources"},
            HISTORICAL_INPUT_PATHS["selection_manifest"]: self._selection(),
            HISTORICAL_INPUT_PATHS["exposure_manifest"]: {"fixture": "exposure"},
            HISTORICAL_INPUT_PATHS["ocr_backend_config"]: backend_config,
            HISTORICAL_INPUT_PATHS["ocr_backend_manifest"]: backend_manifest,
        }
        payloads = {path: pretty_json_bytes(value) for path, value in values.items()}
        for path in HISTORICAL_GENERATOR_PATHS.values():
            payloads[path] = f"# historical fixture: {path}\n".encode()
        payloads[HISTORICAL_TEST_PATH] = b"# historical builder tests\n"
        return payloads

    def _commit_historical_snapshot(self) -> None:
        subprocess.run(
            ["git", "init", "-b", "main"],
            cwd=self.root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _git(self.root, "config", "user.name", "CausalCache Test")
        _git(self.root, "config", "user.email", "causalcache-test@example.invalid")
        for path, payload in self.historical_payloads.items():
            _write(self.root / path, payload)
        _git(self.root, "add", ".")
        _git(self.root, "commit", "-m", "historical builder snapshot")
        self.builder_commit = _git(self.root, "rev-parse", "HEAD")
        self.builder_tree = _git(self.root, "rev-parse", "HEAD^{tree}")
        _git(
            self.root,
            "update-ref",
            "refs/remotes/origin/main",
            self.builder_commit,
        )
        self.historical_hashes = {
            path: sha256_bytes(payload)
            for path, payload in self.historical_payloads.items()
        }

    def _payload_manifest(self, shard_files: dict[str, dict]) -> dict:
        selection = json.loads(
            self.historical_payloads[
                HISTORICAL_INPUT_PATHS["selection_manifest"]
            ]
        )
        v1_config = json.loads(
            self.historical_payloads[HISTORICAL_INPUT_PATHS["v1_config"]]
        )
        source_pool = v1_config["source_pool"]
        backend_config = json.loads(
            self.historical_payloads[
                HISTORICAL_INPUT_PATHS["ocr_backend_config"]
            ]
        )
        backend_manifest = json.loads(
            self.historical_payloads[
                HISTORICAL_INPUT_PATHS["ocr_backend_manifest"]
            ]
        )
        inputs = {
            name: {"path": path, "sha256": self.historical_hashes[path]}
            for name, path in HISTORICAL_INPUT_PATHS.items()
        }
        generator = {"git_revision": self.builder_commit}
        for name, path in HISTORICAL_GENERATOR_PATHS.items():
            generator[f"{name}_path"] = path
            generator[f"{name}_sha256"] = self.historical_hashes[path]
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "artifact_id": PAYLOAD_ARTIFACT_ID,
            "status": "POLICY_BLIND_DERIVED_DATASET_MATERIALIZED",
            "dataset_repo": DATASET_REPO,
            "license": "cc-by-4.0",
            "policy_output_generated": False,
            "restoration_output_generated": False,
            "formal_counts_enforced": True,
            "inputs": inputs,
            "generator": generator,
            "ocr_backend": {
                "backend_id": backend_config["backend_id"],
                "backend_config_sha256": self.historical_hashes[
                    HISTORICAL_INPUT_PATHS["ocr_backend_config"]
                ],
                "hf_model_repo": backend_manifest["hf_model_artifact"]["repo"],
                "hf_model_revision": backend_manifest["hf_model_artifact"][
                    "immutable_revision"
                ],
                "hf_model_tag": backend_manifest["hf_model_artifact"]["tag"],
                "real_screen_golden_dataset_revision": backend_manifest[
                    "real_screen_golden"
                ]["immutable_revision"],
            },
            "ocr_runtime": {
                "runtime_packages": {"rapidocr_onnxruntime": "3.8.4"},
                "rapidocr_package_file_sha256": {"config.yaml": "5" * 64},
                "wheel_sha256": {"fixture.whl": "6" * 64},
                "model_sha256": {"recognizer": "7" * 64},
                "recognizer_character_inventory": {
                    "metadata_key": "character",
                    "entry_count": 436,
                    "utf8_sha256": "8" * 64,
                    "canonical_json_sha256": "9" * 64,
                },
            },
            "source_dataset": {
                key: source_pool[key]
                for key in (
                    "upstream_repo",
                    "upstream_revision",
                    "transport_repo",
                    "transport_revision",
                    "license",
                )
            },
            "role_source_ids": {
                role: [
                    record["source_id"]
                    for record in selection["roles"][role]["trajectories"]
                ]
                for role in EXPECTED_ROLE_COUNTS
            },
            "counts": dict(EXPECTED_COUNTS),
            "inventories": {
                "image_members_sha256": "a" * 64,
                "ocr_records_sha256": "b" * 64,
                "ocr_record_aggregate_sha256": "c" * 64,
                "trajectory_index_sha256": "d" * 64,
            },
            "payload_files": [
                {
                    **shard_files[EXACT_FILE_ALLOWLIST[2]],
                    "member_count": 210,
                },
                {
                    **shard_files[EXACT_FILE_ALLOWLIST[4]],
                    "record_count": 210,
                },
                {
                    **shard_files[EXACT_FILE_ALLOWLIST[5]],
                    "record_count": 35,
                },
            ],
        }

    @staticmethod
    def _file_record(path: str, payload: bytes) -> dict:
        return {
            "path": path,
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }

    def _hf_artifact(self, files: list[dict]) -> dict:
        tree = sha256_bytes(canonical_json_bytes(files))
        revision = "e" * 40
        return {
            "repo": DATASET_REPO,
            "repo_type": "dataset",
            "visibility": "private",
            "tag": HF_TAG,
            "immutable_revision": revision,
            "tag_resolved_revision": revision,
            "uploaded_from_git_commit": self.builder_commit,
            "payload_prefix": PAYLOAD_PREFIX,
            "exact_download_allowlist": list(EXACT_FILE_ALLOWLIST),
            "artifact_tree_sha256": tree,
            "ocr_record_aggregate_sha256": "c" * 64,
            "counts": dict(EXPECTED_COUNTS),
            "files": copy.deepcopy(files),
            "upload": {
                "client": "huggingface_hub==1.23.0",
                "method": "hf_cli_upload+hf_cli_tag_create",
                "source_repeat": 1,
                "source_materialization_output_path": "/data/tmp/derived-repeat-1",
                "local_upload_source_path": "/tmp/derived-repeat-1",
                "allow_patterns": list(EXACT_FILE_ALLOWLIST),
                "commit_message": HF_UPLOAD_COMMIT_MESSAGE,
                "commit_oid": revision,
                "upload_started_at_utc": "2026-07-15T00:03:10Z",
                "upload_ended_at_utc": "2026-07-15T00:03:20Z",
                "upload_argv": [
                    HF_CLI_PATH,
                    "upload",
                    DATASET_REPO,
                    "/tmp/derived-repeat-1",
                    ".",
                    "--repo-type",
                    "dataset",
                    "--revision",
                    "main",
                    *[
                        item
                        for path in EXACT_FILE_ALLOWLIST
                        for item in ("--include", path)
                    ],
                    "--commit-message",
                    HF_UPLOAD_COMMIT_MESSAGE,
                    "--format",
                    "json",
                ],
                "tag_started_at_utc": "2026-07-15T00:03:30Z",
                "tag_ended_at_utc": "2026-07-15T00:03:40Z",
                "tag_create_argv": [
                    HF_CLI_PATH,
                    "repos",
                    "tag",
                    "create",
                    DATASET_REPO,
                    HF_TAG,
                    "--repo-type",
                    "dataset",
                    "--revision",
                    revision,
                    "--message",
                    HF_TAG_MESSAGE,
                    "--format",
                    "json",
                ],
                "tag_resolution_method": "HfApi.dataset_info(revision=tag).sha",
            },
            "redownload": {
                "started_at_utc": "2026-07-15T00:04:00Z",
                "ended_at_utc": "2026-07-15T00:05:00Z",
                "client": "huggingface_hub==1.23.0",
                "method": "hf_cli_download+exact_six_clean_projection",
                "argv": [
                    HF_CLI_PATH,
                    "download",
                    DATASET_REPO,
                    *EXACT_FILE_ALLOWLIST,
                    "--repo-type",
                    "dataset",
                    "--revision",
                    revision,
                    "--local-dir",
                    "/tmp/hf-download-with-local-cache",
                    "--force-download",
                    "--format",
                    "json",
                ],
                "revision": revision,
                "allow_patterns": list(EXACT_FILE_ALLOWLIST),
                "force_download": True,
                "fresh_cache_dir": "/tmp/fresh-hf-cache",
                "fresh_cache_dir_preexisted": False,
                "download_root": "/tmp/hf-download-with-local-cache",
                "projection_root": "/data/tmp/hf-redownload",
                "validated_output_path": "/data/tmp/hf-redownload",
                "observed_files": copy.deepcopy(files),
                "all_listed_file_hashes_verified": True,
                "artifact_validator_replayed_on_hyper00": True,
                "validator_outcome": VALIDATOR_OUTCOME,
                "ocr_replay_record_count": 210,
                "ocr_replay_aggregate_sha256": "c" * 64,
                "artifact_tree_sha256": tree,
            },
        }

    def _run_record(self, repeat: int, tree: str) -> dict:
        output = f"/data/tmp/derived-repeat-{repeat}"
        minute = (repeat - 1) * 2
        return {
            "repeat": repeat,
            "started_at_utc": f"2026-07-15T00:0{minute}:00Z",
            "ended_at_utc": f"2026-07-15T00:0{minute + 1}:00Z",
            "argv": [
                "python",
                "-m",
                "scripts.build_guiodyssey_restoration_v2",
                "--git-revision",
                self.builder_commit,
                "--output-dir",
                output,
            ],
            "output_path": output,
            "output_dir_preexisted": False,
            "outcome": VALIDATOR_OUTCOME,
            "artifact_tree_sha256": tree,
            "counts": dict(EXPECTED_COUNTS),
            "generated_ocr_record_count": 210,
            "generated_ocr_record_aggregate_sha256": "c" * 64,
            "ocr_record_aggregate_sha256": "c" * 64,
            "ocr_replay_performed": True,
            "ocr_replay_record_count": 210,
            "ocr_replay_aggregate_sha256": "c" * 64,
            "policy_loaded": False,
            "policy_output_generated": False,
            "restoration_output_generated": False,
        }

    def _validation_record(
        self,
        validation_id: str,
        output: str,
        tree: str,
        *,
        repeat: dict | None = None,
    ) -> dict:
        source = (
            "standalone_validator"
            if validation_id.startswith("hf-")
            else "builder_integrated"
        )
        started = "2026-07-15T00:06:00Z"
        ended = "2026-07-15T00:07:00Z"
        argv = [
            "python",
            "-m",
            "scripts.validate_guiodyssey_restoration_v2",
            "--output-dir",
            output,
            "--git-revision",
            self.builder_commit,
        ]
        if repeat is not None:
            started = repeat["started_at_utc"]
            ended = repeat["ended_at_utc"]
            argv = copy.deepcopy(repeat["argv"])
        return {
            "validation_id": validation_id,
            "source": source,
            "started_at_utc": started,
            "ended_at_utc": ended,
            "argv": argv,
            "output_path": output,
            "outcome": VALIDATOR_OUTCOME,
            "artifact_tree_sha256": tree,
            "ocr_record_aggregate_sha256": "c" * 64,
            "ocr_replay_performed": True,
            "ocr_replay_record_count": 210,
            "ocr_replay_aggregate_sha256": "c" * 64,
            "policy_loaded": False,
            "policy_output_generated": False,
            "restoration_output_generated": False,
        }

    def _write_completion_evidence(self) -> None:
        raw_payloads = {
            ".gitattributes": b"*.tar filter=lfs\n",
            "README.md": b"# Fixture derived artifact\n",
            EXACT_FILE_ALLOWLIST[2]: b"fixture-image-tar",
            EXACT_FILE_ALLOWLIST[4]: b"fixture-ocr-jsonl\n",
            EXACT_FILE_ALLOWLIST[5]: b"fixture-trajectories-jsonl\n",
        }
        shard_files = {
            path: self._file_record(path, payload)
            for path, payload in raw_payloads.items()
            if path in {
                EXACT_FILE_ALLOWLIST[2],
                EXACT_FILE_ALLOWLIST[4],
                EXACT_FILE_ALLOWLIST[5],
            }
        }
        payload_manifest = self._payload_manifest(shard_files)
        payload_manifest_bytes = pretty_json_bytes(payload_manifest)
        raw_payloads[EXACT_FILE_ALLOWLIST[3]] = payload_manifest_bytes
        files = [
            self._file_record(path, raw_payloads[path]) for path in EXACT_FILE_ALLOWLIST
        ]
        hf_artifact = self._hf_artifact(files)
        tree = hf_artifact["artifact_tree_sha256"]
        repeats = [self._run_record(1, tree), self._run_record(2, tree)]
        validations = [
            self._validation_record(
                EXPECTED_VALIDATION_IDS[0],
                repeats[0]["output_path"],
                tree,
                repeat=repeats[0],
            ),
            self._validation_record(
                EXPECTED_VALIDATION_IDS[1],
                repeats[1]["output_path"],
                tree,
                repeat=repeats[1],
            ),
            self._validation_record(
                EXPECTED_VALIDATION_IDS[2], "/data/tmp/hf-redownload", tree
            ),
        ]
        self.summary = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "outcome": SUMMARY_OUTCOME,
            "artifact_id": ARTIFACT_ID,
            "source": {
                "builder_git_commit": self.builder_commit,
                "builder_git_tree": self.builder_tree,
                "clean_checkout": True,
                "head_equals_origin_main": True,
                "checkout_transport": copy.deepcopy(EXPECTED_CHECKOUT_TRANSPORT),
            },
            "payload_manifest_snapshot": {
                "path": EXACT_FILE_ALLOWLIST[3],
                "sha256": sha256_bytes(payload_manifest_bytes),
                "content": payload_manifest,
            },
            "materialization_repeats": repeats,
            "deterministic_identity": {
                "materialization_repeat_count": 2,
                "canonical_files": copy.deepcopy(files),
                "artifact_tree_sha256": tree,
                "ocr_record_aggregate_sha256": "c" * 64,
                "all_six_files_byte_identical": True,
                "ocr_record_aggregate_byte_identical": True,
            },
            "artifact_validations": validations,
            "hf_dataset_artifact": copy.deepcopy(hf_artifact),
            "runtime": {
                "host_alias": "hyper00",
                "hostname": "fixture-host",
                "architecture": "x86_64",
                "kernel": "fixture-kernel",
                "python": "3.12.3",
                "container_name": "sglang-omni-jaxan-fixture",
                "container_id": "fixture-container-id",
                "container_image": "hongccc/sglang-omni:dev",
                "container_image_digest": "sha256:" + "f" * 64,
                "provider": "CPUExecutionProvider",
                "gpu_used": False,
                "dtype": "not_applicable_cpu_onnxruntime",
                "seed": None,
                "library_versions": {
                    "pyarrow": "24.0.0",
                    "Pillow": "12.2.0",
                    "rapidocr": "3.8.4",
                    "onnxruntime": "1.24.4",
                    "huggingface_hub": "1.23.0",
                },
            },
            "negative_declarations": {
                "confirm_policy_scores_observed": False,
                "policy_module_loaded": False,
                "policy_forward_called": False,
                "policy_output_generated": False,
                "restoration_output_generated": False,
            },
            "dependency_1_closed": True,
        }
        for path, payload in (
            (COMPLETION_SOURCE_PATHS[0], b"# completion validator fixture\n"),
            (COMPLETION_SOURCE_PATHS[1], b"# completion tests fixture\n"),
        ):
            _write(self.root / path, payload)
        _write_json(self.summary_path, self.summary)
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "artifact_id": ARTIFACT_ID,
            "status": STATUS,
            "builder_snapshot": {
                "git_commit": self.builder_commit,
                "git_tree": self.builder_tree,
                "source_files": [
                    {"path": path, "sha256": self.historical_hashes[path]}
                    for path in sorted(self.historical_hashes)
                ],
            },
            "source_files": [],
            "summary": {
                "path": COMPLETION_SOURCE_PATHS[-1],
                "sha256": _sha(self.summary_path),
            },
            "hf_dataset_artifact": copy.deepcopy(hf_artifact),
            "materialization_repeat_count": 2,
            "artifact_validation_count": 3,
            "dependency_1_closed": True,
            "policy_loaded_before_manifest": False,
            "policy_output_generated_before_manifest": False,
            "restoration_output_generated_before_manifest": False,
        }
        self._refresh_completion_sources()
        _write_json(self.manifest_path, self.manifest)

    def _refresh_completion_sources(self) -> None:
        self.manifest["source_files"] = [
            {"path": path, "sha256": _sha(self.root / path)}
            for path in COMPLETION_SOURCE_PATHS
        ]

    def persist_summary(self) -> None:
        _write_json(self.summary_path, self.summary)
        summary_sha = _sha(self.summary_path)
        self.manifest["summary"]["sha256"] = summary_sha
        self._refresh_completion_sources()
        _write_json(self.manifest_path, self.manifest)

    def persist_manifest(self) -> None:
        _write_json(self.manifest_path, self.manifest)

    def rebind_payload_snapshot(self) -> None:
        snapshot = self.summary["payload_manifest_snapshot"]
        payload = pretty_json_bytes(snapshot["content"])
        snapshot["sha256"] = sha256_bytes(payload)
        hf = self.summary["hf_dataset_artifact"]
        hf["files"][3]["size_bytes"] = len(payload)
        hf["files"][3]["sha256"] = snapshot["sha256"]
        hf["redownload"]["observed_files"] = copy.deepcopy(hf["files"])
        tree = sha256_bytes(canonical_json_bytes(hf["files"]))
        hf["artifact_tree_sha256"] = tree
        hf["redownload"]["artifact_tree_sha256"] = tree
        deterministic = self.summary["deterministic_identity"]
        deterministic["canonical_files"] = copy.deepcopy(hf["files"])
        deterministic["artifact_tree_sha256"] = tree
        for record in self.summary["materialization_repeats"]:
            record["artifact_tree_sha256"] = tree
        for record in self.summary["artifact_validations"]:
            record["artifact_tree_sha256"] = tree
        self.manifest["hf_dataset_artifact"] = copy.deepcopy(hf)
        self.persist_summary()

    def validate(self) -> dict:
        return validate_completion_manifest(
            artifact_manifest_path=self.manifest_path,
            repository_root=self.root,
            expected_builder_commit=self.builder_commit,
            expected_builder_tree=self.builder_tree,
            expected_builder_source_sha256=self.historical_hashes,
        )


class RestorationV2DerivedArtifactCompletionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._reset_fixture()

    def _reset_fixture(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory
        self.fixture = CompletionFixture(Path(directory.name))

    def test_canonical_completion_fixture_passes(self) -> None:
        result = self.fixture.validate()
        self.assertEqual(
            result["outcome"],
            "PASSED_RESTORATION_V2_DERIVED_ARTIFACT_SOURCE_VALIDATION",
        )
        self.assertEqual(result["hf_file_count"], 6)
        self.assertEqual(result["ocr_replay_record_count"], 210)
        self.assertTrue(result["dependency_1_closed"])
        self.assertFalse(result["policy_output_generated_by_this_validation"])

    def test_completion_source_omission_fails_closed(self) -> None:
        self.fixture.manifest["source_files"].pop()
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "source inventory count drifted"):
            self.fixture.validate()

    def test_historical_blob_hash_drift_fails_closed(self) -> None:
        self.fixture.manifest["builder_snapshot"]["source_files"][0][
            "sha256"
        ] = "0" * 64
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "historical source SHA256 drifted"):
            self.fixture.validate()

    def test_builder_commit_and_tree_drift_fail_closed(self) -> None:
        for field in ("git_commit", "git_tree"):
            with self.subTest(field=field):
                fixture = self.fixture
                original = fixture.manifest["builder_snapshot"][field]
                fixture.manifest["builder_snapshot"][field] = "0" * 40
                fixture.persist_manifest()
                with self.assertRaisesRegex(ValueError, "historical commit or tree drifted"):
                    fixture.validate()
                fixture.manifest["builder_snapshot"][field] = original
                fixture.persist_manifest()

    def test_payload_generator_mutation_fails_after_hash_rebinding(self) -> None:
        self.fixture.summary["payload_manifest_snapshot"]["content"]["generator"][
            "git_revision"
        ] = "0" * 40
        self.fixture.rebind_payload_snapshot()
        with self.assertRaisesRegex(ValueError, "generator commit drifted"):
            self.fixture.validate()

    def test_two_materialization_tree_drift_fails_closed(self) -> None:
        self.fixture.summary["materialization_repeats"][1][
            "artifact_tree_sha256"
        ] = "0" * 64
        self.fixture.persist_summary()
        with self.assertRaisesRegex(ValueError, "materialization deterministic replay"):
            self.fixture.validate()

    def test_missing_repeat_and_replay_209_fail_closed(self) -> None:
        self.fixture.summary["materialization_repeats"].pop()
        self.fixture.persist_summary()
        with self.assertRaisesRegex(ValueError, "exactly two materializations"):
            self.fixture.validate()

        self._reset_fixture()
        self.fixture.summary["artifact_validations"][2]["ocr_replay_record_count"] = 209
        self.fixture.persist_summary()
        with self.assertRaisesRegex(ValueError, "exact replay evidence drifted"):
            self.fixture.validate()

    def test_hf_tag_revision_and_exact_six_drift_fail_closed(self) -> None:
        mutations = (
            ("tag", "unexpected-tag", "HF derived dataset identity drifted"),
            ("tag_resolved_revision", "0" * 40, "tag does not resolve"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                original = self.fixture.manifest["hf_dataset_artifact"][field]
                self.fixture.manifest["hf_dataset_artifact"][field] = value
                self.fixture.persist_manifest()
                with self.assertRaisesRegex(ValueError, message):
                    self.fixture.validate()
                self.fixture.manifest["hf_dataset_artifact"][field] = original
                self.fixture.persist_manifest()
        self.fixture.manifest["hf_dataset_artifact"]["files"].append(
            {"path": "extra", "size_bytes": 1, "sha256": "0" * 64}
        )
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "exactly six"):
            self.fixture.validate()

    def test_hf_cli_argv_mutations_fail_closed(self) -> None:
        upload = self.fixture.manifest["hf_dataset_artifact"]["upload"]
        upload["upload_argv"].extend(["--delete", "golden/**"])
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "cannot delete"):
            self.fixture.validate()

        self._reset_fixture()
        upload = self.fixture.manifest["hf_dataset_artifact"]["upload"]
        revision_index = upload["tag_create_argv"].index("--revision") + 1
        upload["tag_create_argv"][revision_index] = "0" * 40
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "tag argv --revision value drifted"):
            self.fixture.validate()

        self._reset_fixture()
        redownload = self.fixture.manifest["hf_dataset_artifact"]["redownload"]
        redownload["argv"].extend(["--cache-dir", "/data/tmp/disallowed-cache"])
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "cannot use --cache-dir"):
            self.fixture.validate()

        self._reset_fixture()
        redownload = self.fixture.manifest["hf_dataset_artifact"]["redownload"]
        redownload["argv"].remove(EXACT_FILE_ALLOWLIST[-1])
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "positional exact-six"):
            self.fixture.validate()

        self._reset_fixture()
        upload = self.fixture.manifest["hf_dataset_artifact"]["upload"]
        upload["commit_message"] = "self-consistent-but-not-executed"
        message_index = upload["upload_argv"].index("--commit-message") + 1
        upload["upload_argv"][message_index] = upload["commit_message"]
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "commit message drifted"):
            self.fixture.validate()

        self._reset_fixture()
        redownload = self.fixture.manifest["hf_dataset_artifact"]["redownload"]
        redownload["argv"].remove("--force-download")
        local_dir_index = redownload["argv"].index("--local-dir")
        redownload["argv"].insert(local_dir_index, "--force-download")
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "order drifted"):
            self.fixture.validate()

    def test_cross_machine_upload_paths_are_distinct_and_bound(self) -> None:
        upload = self.fixture.manifest["hf_dataset_artifact"]["upload"]
        upload["local_upload_source_path"] = upload[
            "source_materialization_output_path"
        ]
        upload["upload_argv"][3] = upload["local_upload_source_path"]
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "distinguish Hyper source and Mac staging"):
            self.fixture.validate()

        self._reset_fixture()
        invalid_path = "/data/tmp/not-repeat-1"
        self.fixture.manifest["hf_dataset_artifact"]["upload"][
            "source_materialization_output_path"
        ] = invalid_path
        self.fixture.summary["hf_dataset_artifact"]["upload"][
            "source_materialization_output_path"
        ] = invalid_path
        self.fixture.persist_summary()
        with self.assertRaisesRegex(ValueError, "not materialization repeat 1"):
            self.fixture.validate()

    def test_nonfresh_redownload_and_negative_declaration_fail_closed(self) -> None:
        self.fixture.manifest["hf_dataset_artifact"]["redownload"][
            "fresh_cache_dir_preexisted"
        ] = True
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "cache was not fresh"):
            self.fixture.validate()

        self._reset_fixture()
        redownload = self.fixture.manifest["hf_dataset_artifact"]["redownload"]
        redownload["projection_root"] = redownload["download_root"]
        redownload["validated_output_path"] = redownload["download_root"]
        self.fixture.persist_manifest()
        with self.assertRaisesRegex(ValueError, "roots must be distinct"):
            self.fixture.validate()

        self._reset_fixture()
        self.fixture.summary["negative_declarations"]["policy_forward_called"] = True
        self.fixture.persist_summary()
        with self.assertRaisesRegex(ValueError, "negative declarations drifted"):
            self.fixture.validate()

    def test_checkout_transport_mutation_fails_closed(self) -> None:
        self.fixture.summary["source"]["checkout_transport"][
            "bundle_size_bytes"
        ] += 1
        self.fixture.persist_summary()
        with self.assertRaisesRegex(ValueError, "historical source state drifted"):
            self.fixture.validate()

    def test_unknown_field_and_duplicate_json_key_fail_closed(self) -> None:
        self.fixture.summary["unexpected"] = True
        self.fixture.persist_summary()
        with self.assertRaisesRegex(ValueError, "completion summary fields drifted"):
            self.fixture.validate()

        self._reset_fixture()
        payload = self.fixture.manifest_path.read_text(encoding="utf-8")
        duplicate = payload.replace(
            '  "schema_version": "1.0.0",',
            '  "schema_version": "1.0.0",\n  "schema_version": "1.0.0",',
            1,
        )
        self.fixture.manifest_path.write_text(duplicate, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.fixture.validate()


if __name__ == "__main__":
    unittest.main()
