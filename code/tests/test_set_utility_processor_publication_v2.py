from __future__ import annotations

import copy
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import causalcache.set_utility_processor_publication_v2 as publication_v2
from causalcache.set_utility_processor_artifacts import (
    canonical_json_bytes,
    canonical_pretty_json_bytes,
)
from causalcache.set_utility_processor_postflight_v2 import VALIDATION_STATUS
from causalcache.set_utility_processor_publication_v2 import (
    EXPECTED_REMOTE_FILE_COUNT,
    PUBLICATION_STATUS,
    _expected_formal_paths,
    _sha256_bytes,
    publish_processor_v2_artifact,
    validate_processor_v2_publication,
)
from causalcache.set_utility_processor_result_v2 import (
    PENDING_PUBLICATION_STATUS,
    PROTOCOL_ID as RESULT_PROTOCOL_ID,
    VALID_RESULT_STATUS,
)


REPO = "fixture-owner/causalcache-set-utility-new-development-mobile"
TAG = "phase1-b2-processor-freeze-v2-image-contract-repair"
PREFIX = "artifacts/processor-freeze-v2-image-contract-repair"
BASE_COMMIT = "a" * 40
PUBLICATION_COMMIT = "b" * 40
TAG_OBJECT = "c" * 40


@dataclass(frozen=True)
class _Operation:
    path_in_repo: str
    path_or_fileobj: str


class _FakeApi:
    def __init__(self) -> None:
        self.private = True
        self.main = BASE_COMMIT
        self.commits = {BASE_COMMIT: {"README.md": b"existing repo\n"}}
        self.commit_order = [BASE_COMMIT]
        self.tags: dict[str, tuple[str, str]] = {}
        self.create_commit_calls: list[dict[str, object]] = []
        self.create_tag_calls: list[dict[str, object]] = []
        self.download_calls: list[dict[str, object]] = []
        self.corrupt_download_filename: str | None = None
        self.corrupt_download_revision: str | None = None
        self.corrupt_download_once = False
        self.fail_after_commit_once = False
        self.fail_after_tag_once = False

    def _resolve(self, revision: str) -> str:
        if revision == "main":
            return self.main
        if revision in self.tags:
            return self.tags[revision][1]
        return revision

    def dataset_info(self, repo: str, *, revision: str):
        assert repo == REPO
        resolved = self._resolve(revision)
        if resolved not in self.commits:
            raise ValueError("unknown fake revision")
        return SimpleNamespace(private=self.private, sha=resolved)

    def list_repo_refs(self, repo: str, *, repo_type: str):
        assert (repo, repo_type) == (REPO, "dataset")
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name=name, target_commit=tag_object)
                for name, (tag_object, _) in sorted(self.tags.items())
            ]
        )

    def list_repo_files(self, repo: str, *, repo_type: str, revision: str):
        assert (repo, repo_type) == (REPO, "dataset")
        return sorted(self.commits[self._resolve(revision)])

    def list_repo_commits(
        self, repo: str, *, repo_type: str, revision: str
    ) -> list[SimpleNamespace]:
        assert (repo, repo_type, revision) == (REPO, "dataset", "main")
        return [SimpleNamespace(commit_id=commit) for commit in self.commit_order]

    def create_commit(
        self,
        repo: str,
        *,
        repo_type: str,
        revision: str,
        parent_commit: str,
        operations: list[_Operation],
        commit_message: str,
    ):
        assert (repo, repo_type, revision) == (REPO, "dataset", "main")
        if parent_commit != self.main:
            raise AssertionError("publication omitted exact main CAS parent")
        tree = dict(self.commits[self.main])
        for operation in operations:
            if operation.path_in_repo in tree:
                raise AssertionError("publication overwrote one remote path")
            tree[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        self.create_commit_calls.append(
            {
                "parent_commit": parent_commit,
                "operation_count": len(operations),
                "commit_message": commit_message,
            }
        )
        self.commits[PUBLICATION_COMMIT] = tree
        self.main = PUBLICATION_COMMIT
        if PUBLICATION_COMMIT not in self.commit_order:
            self.commit_order.insert(0, PUBLICATION_COMMIT)
        if self.fail_after_commit_once:
            self.fail_after_commit_once = False
            raise RuntimeError("injected failure after commit")
        return SimpleNamespace(oid=PUBLICATION_COMMIT)

    def create_tag(
        self,
        repo: str,
        *,
        repo_type: str,
        tag: str,
        tag_message: str,
        revision: str,
        exist_ok: bool,
    ) -> None:
        assert (repo, repo_type, exist_ok) == (REPO, "dataset", False)
        if tag in self.tags:
            raise ValueError("tag exists")
        self.tags[tag] = (TAG_OBJECT, revision)
        self.create_tag_calls.append(
            {
                "tag": tag,
                "tag_message": tag_message,
                "revision": revision,
            }
        )
        if self.fail_after_tag_once:
            self.fail_after_tag_once = False
            raise RuntimeError("injected failure after tag")

    def download(self, **kwargs: object) -> str:
        self.download_calls.append(dict(kwargs))
        revision = self._resolve(str(kwargs["revision"]))
        filename = str(kwargs["filename"])
        payload = self.commits[revision][filename]
        requested_revision = str(kwargs["revision"])
        should_corrupt = self.corrupt_download_filename == filename and (
            self.corrupt_download_revision is None
            or self.corrupt_download_revision == requested_revision
        )
        if should_corrupt:
            payload += b"drift"
            if self.corrupt_download_once:
                self.corrupt_download_filename = None
                self.corrupt_download_revision = None
        path = Path(str(kwargs["local_dir"])) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)


def _operation_factory(*, path_in_repo: str, path_or_fileobj: str) -> _Operation:
    return _Operation(path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)


def _fixture(tmp_path: Path) -> dict[str, object]:
    formal = tmp_path / "formal"
    for index, relative in enumerate(_expected_formal_paths()):
        path = formal / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"formal-{index:02d}-{relative}\n".encode())
    inventory = []
    for relative in _expected_formal_paths():
        payload = (formal / relative).read_bytes()
        inventory.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    summary = {
        "artifact": {
            "file_count": 23,
            "file_inventory": inventory,
            "file_inventory_sha256": _sha256_bytes(canonical_json_bytes(inventory)),
            "total_byte_count": sum(item["size_bytes"] for item in inventory),
            "output_root": str(formal),
            "staging_root": str(tmp_path / ".formal.incomplete"),
            "staging_root_present_after_success": False,
            "contains_raw_candidate_or_ocr_copy": False,
            "output_tree_snapshot_sha256": "1" * 64,
        },
        "bindings": {},
        "execution": {},
        "failure": None,
        "negative_operations": {
            "closed_loop_episode_count": 0,
            "hugging_face_mutation_count": 0,
            "matched_nll_count": 0,
            "model_or_policy_inference_count": 0,
            "restoration_label_count": 0,
            "training_count": 0,
        },
        "postflight": {"status": VALIDATION_STATUS},
        "postflight_accepted_for_formal_result": True,
        "protocol_id": RESULT_PROTOCOL_ID,
        "publication": {
            "hf_mutation_count": 0,
            "intended_private_hf_repo": REPO,
            "intended_tag": TAG,
            "status": PENDING_PUBLICATION_STATUS,
        },
        "schema_version": "1.0.0",
        "scientific_eligibility": True,
        "status": VALID_RESULT_STATUS,
    }
    summary_path = tmp_path / "summary.json"
    summary_bytes = canonical_pretty_json_bytes(summary)
    summary_path.write_bytes(summary_bytes)
    card = tmp_path / "card.md"
    card.write_text("# Processor v2 formal result\n", encoding="utf-8")
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    return {
        "formal_root": formal,
        "git_summary": summary_path,
        "git_card": card,
        "expected_result_summary_sha256": _sha256_bytes(summary_bytes),
        "hf_repo": REPO,
        "hf_tag": TAG,
        "hf_prefix": PREFIX,
        "fresh_download_parent": fresh,
        "receipt_path": tmp_path / "publication-receipt.json",
    }


def _publish(api: _FakeApi, fixture: dict[str, object]):
    return publish_processor_v2_artifact(
        api=api,
        download_fn=api.download,
        operation_factory=_operation_factory,
        **fixture,
    )


def test_success_is_one_commit_one_annotated_tag_and_two_fresh_replays(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()

    receipt = _publish(api, fixture)

    assert receipt["status"] == PUBLICATION_STATUS
    assert api.create_commit_calls == [
        {
            "parent_commit": BASE_COMMIT,
            "operation_count": EXPECTED_REMOTE_FILE_COUNT,
            "commit_message": "Publish processor-v2 image-contract repair artifact",
        }
    ]
    assert len(api.create_tag_calls) == 1
    assert api.create_tag_calls[0]["tag_message"]
    assert len(api.download_calls) == 50
    assert {call["revision"] for call in api.download_calls} == {
        PUBLICATION_COMMIT,
        TAG,
    }
    receipt_bytes = Path(fixture["receipt_path"]).read_bytes()
    assert b"hf_SYNTHETIC_SECRET_VALUE" not in receipt_bytes
    assert stat.S_IMODE(Path(fixture["receipt_path"]).stat().st_mode) == 0o600
    assert receipt["fresh_downloads"]["commit"]["remote_file_count"] == 25
    assert receipt["fresh_downloads"]["tag"]["remote_file_count"] == 25
    assert receipt["fresh_downloads"]["remote_files_verified_twice"] == 25
    commit_replay = Path(receipt["fresh_downloads"]["commit"]["directory"])
    tag_replay = Path(receipt["fresh_downloads"]["tag"]["directory"])
    assert commit_replay != tag_replay
    assert commit_replay.parent == Path(fixture["fresh_download_parent"])
    assert tag_replay.parent == Path(fixture["fresh_download_parent"])
    assert not commit_replay.is_relative_to(Path(fixture["formal_root"]))
    assert not tag_replay.is_relative_to(Path(fixture["formal_root"]))

    before = (
        len(api.create_commit_calls),
        len(api.create_tag_calls),
        len(api.download_calls),
    )
    replayed = validate_processor_v2_publication(api=api, **fixture)
    assert replayed == receipt
    assert before == (
        len(api.create_commit_calls),
        len(api.create_tag_calls),
        len(api.download_calls),
    )


@pytest.mark.parametrize("collision", ["tag", "path"])
def test_tag_or_prefix_collision_fails_before_remote_mutation(
    tmp_path: Path,
    collision: str,
) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    if collision == "tag":
        api.tags[TAG] = (TAG_OBJECT, BASE_COMMIT)
    else:
        api.commits[BASE_COMMIT][f"{PREFIX}/occupied"] = b"collision"

    with pytest.raises(ValueError, match="already exists"):
        _publish(api, fixture)

    assert api.create_commit_calls == []
    assert api.create_tag_calls == []


def test_remote_tag_drift_breaks_read_only_validation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    _publish(api, fixture)
    api.tags[TAG] = ("d" * 40, BASE_COMMIT)

    with pytest.raises(ValueError, match="remote annotated tag drifted"):
        validate_processor_v2_publication(api=api, **fixture)


def test_fresh_download_byte_mismatch_prevents_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    first = _expected_formal_paths()[0]
    api.corrupt_download_filename = f"{PREFIX}/formal/{first}"

    with pytest.raises(ValueError, match="fresh download differs"):
        _publish(api, fixture)

    assert not Path(fixture["receipt_path"]).exists()


@pytest.mark.parametrize("relative", ["git/summary.json", "git/card.md"])
def test_fresh_git_metadata_mismatch_prevents_receipt(
    tmp_path: Path,
    relative: str,
) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    api.corrupt_download_filename = f"{PREFIX}/{relative}"

    with pytest.raises(ValueError, match="fresh download differs"):
        _publish(api, fixture)

    assert not Path(fixture["receipt_path"]).exists()


def test_invalid_pending_summary_fails_before_remote_access(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary_path = Path(fixture["git_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["publication"]["status"] = "PUBLISHED_ALREADY"
    payload = canonical_pretty_json_bytes(summary)
    summary_path.write_bytes(payload)
    fixture["expected_result_summary_sha256"] = _sha256_bytes(payload)
    api = _FakeApi()

    with pytest.raises(ValueError, match="valid PENDING_HF_UPLOAD"):
        _publish(api, fixture)

    assert api.create_commit_calls == []
    assert api.create_tag_calls == []


def test_remote_download_inventory_drift_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    receipt = _publish(api, fixture)
    changed = copy.deepcopy(receipt)
    changed["fresh_downloads"]["tag"]["formal_file_count"] = 22
    Path(fixture["receipt_path"]).write_bytes(canonical_pretty_json_bytes(changed))

    with pytest.raises(ValueError, match="inventory drifted"):
        validate_processor_v2_publication(api=api, **fixture)


def test_validate_only_rechecks_retained_git_bytes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    receipt = _publish(api, fixture)
    retained = (
        Path(receipt["fresh_downloads"]["commit"]["directory"])
        / PREFIX
        / "git/card.md"
    )
    retained.write_bytes(b"drifted retained card\n")

    with pytest.raises(ValueError, match="retained fresh-download Git bytes drifted"):
        validate_processor_v2_publication(api=api, **fixture)


def test_validate_only_requires_receipt_mode_0600(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    _publish(api, fixture)
    receipt_path = Path(fixture["receipt_path"])
    receipt_path.chmod(0o644)

    with pytest.raises(ValueError, match="mode must be exactly 0600"):
        validate_processor_v2_publication(api=api, **fixture)


@pytest.mark.parametrize("replay", ["commit", "tag"])
def test_stale_fresh_replay_fails_before_hf_mutation(
    tmp_path: Path, replay: str
) -> None:
    fixture = _fixture(tmp_path)
    sha = str(fixture["expected_result_summary_sha256"])
    stale = (
        Path(fixture["fresh_download_parent"])
        / f"processor-v2-{sha[:12]}-{replay}"
    )
    stale.mkdir()
    api = _FakeApi()

    with pytest.raises(FileExistsError, match="stale replay"):
        _publish(api, fixture)

    assert api.create_commit_calls == []
    assert api.create_tag_calls == []


def test_fresh_replays_inside_formal_root_fail_before_hf_mutation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    unsafe_parent = Path(fixture["formal_root"]) / "fresh"
    unsafe_parent.mkdir()
    fixture["fresh_download_parent"] = unsafe_parent
    api = _FakeApi()

    with pytest.raises(ValueError, match="outside formal root"):
        _publish(api, fixture)

    assert api.create_commit_calls == []
    assert api.create_tag_calls == []


def test_receipt_inside_future_replay_fails_before_hf_mutation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    sha = str(fixture["expected_result_summary_sha256"])
    fixture["receipt_path"] = (
        Path(fixture["fresh_download_parent"])
        / f"processor-v2-{sha[:12]}-commit"
        / "receipt.json"
    )
    api = _FakeApi()

    with pytest.raises(ValueError, match="outside fresh replays"):
        _publish(api, fixture)

    assert api.create_commit_calls == []
    assert api.create_tag_calls == []


def test_publish_reconciles_commit_created_before_client_failure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    api.fail_after_commit_once = True

    with pytest.raises(RuntimeError, match="after commit"):
        _publish(api, fixture)

    assert api.main == PUBLICATION_COMMIT
    assert TAG not in api.tags
    assert not Path(fixture["receipt_path"]).exists()
    receipt = _publish(api, fixture)
    assert receipt["destination"]["immutable_revision"] == PUBLICATION_COMMIT
    assert receipt["destination"]["parent_main_revision"] == BASE_COMMIT
    assert len(api.create_commit_calls) == 1
    assert len(api.create_tag_calls) == 1


def test_publish_reconciles_tag_created_before_client_failure(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    api.fail_after_tag_once = True

    with pytest.raises(RuntimeError, match="after tag"):
        _publish(api, fixture)

    assert api.tags[TAG] == (TAG_OBJECT, PUBLICATION_COMMIT)
    assert not Path(fixture["receipt_path"]).exists()
    receipt = _publish(api, fixture)
    assert receipt["destination"]["immutable_revision"] == PUBLICATION_COMMIT
    assert len(api.create_commit_calls) == 1
    assert len(api.create_tag_calls) == 1


def test_publish_reconciles_remote_after_receipt_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    original = publication_v2._write_receipt
    calls = 0

    def fail_once(path: Path, receipt: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected receipt failure")
        original(path, receipt)

    monkeypatch.setattr(publication_v2, "_write_receipt", fail_once)
    with pytest.raises(OSError, match="receipt failure"):
        _publish(api, fixture)

    assert not Path(fixture["receipt_path"]).exists()
    assert list(Path(fixture["fresh_download_parent"]).iterdir()) == []
    result = _publish(api, fixture)
    assert result["status"] == PUBLICATION_STATUS
    assert len(api.create_commit_calls) == 1
    assert len(api.create_tag_calls) == 1


@pytest.mark.parametrize("relative", ["git/card.md", "git/summary.json"])
def test_tag_replay_last_metadata_corruption_recovers_without_remote_overwrite(
    tmp_path: Path,
    relative: str,
) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    api.corrupt_download_filename = f"{PREFIX}/{relative}"
    api.corrupt_download_revision = TAG
    api.corrupt_download_once = True

    with pytest.raises(ValueError, match="fresh download differs"):
        _publish(api, fixture)

    assert not Path(fixture["receipt_path"]).exists()
    assert list(Path(fixture["fresh_download_parent"]).iterdir()) == []
    receipt = _publish(api, fixture)
    assert receipt["fresh_downloads"]["tag"]["remote_file_count"] == 25
    assert len(api.create_commit_calls) == 1
    assert len(api.create_tag_calls) == 1


def test_reconcile_rejects_exact_tree_with_drifted_bytes_before_missing_tag(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    api = _FakeApi()
    api.fail_after_commit_once = True
    with pytest.raises(RuntimeError, match="after commit"):
        _publish(api, fixture)
    remote = f"{PREFIX}/formal/{_expected_formal_paths()[0]}"
    api.commits[PUBLICATION_COMMIT][remote] += b"unknown drift"

    with pytest.raises(ValueError, match="fresh download differs"):
        _publish(api, fixture)

    assert TAG not in api.tags
    assert len(api.create_commit_calls) == 1
    assert api.create_tag_calls == []


def test_partial_final_receipt_is_never_replaced_or_reconciled(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    receipt = Path(fixture["receipt_path"])
    receipt.write_bytes(b'{"schema_version":')
    receipt.chmod(0o600)
    api = _FakeApi()

    with pytest.raises(FileExistsError, match="partial legacy receipt"):
        _publish(api, fixture)

    assert receipt.read_bytes() == b'{"schema_version":'
    assert api.create_commit_calls == []
    assert api.create_tag_calls == []

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        validate_processor_v2_publication(api=api, **fixture)


def test_receipt_write_failure_never_exposes_partial_final_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    original_write = publication_v2.os.write
    calls = 0

    def fail_after_partial_write(descriptor: int, payload: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(descriptor, bytes(payload)[:3])
        raise OSError("injected receipt write interruption")

    monkeypatch.setattr(publication_v2.os, "write", fail_after_partial_write)
    with pytest.raises(OSError, match="interruption"):
        publication_v2._write_receipt(receipt, {"status": "fixture"})

    assert not receipt.exists()
    assert list(tmp_path.glob(f".{receipt.name}.*.tmp")) == []
