from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.independent_confirm_artifact import (
    BUNDLE_MANIFEST_PATH,
    DESTINATION_REPO,
    DESTINATION_TAG,
    FEATURE_STATE_PATH,
    INDEPENDENT_DECISION_PATH,
    PAYLOAD_MANIFEST_PATH,
    PAYLOAD_TARGETS,
    REPORT_TARGETS,
    STATE_RECORDS_PATH,
    SCORE_PATHS,
    SELECTION_PATHS,
    build_label_blind_payload,
    build_report_files,
    canonical_json_bytes,
    publish_payload_commit,
    publish_report_commit,
    read_feature_states_jsonl,
    read_independent_decisions,
    read_score_records_jsonl,
    read_selection_artifact,
    seal_label_blind_payload,
    validate_report_files,
)
from causalcache.independent_confirm_data import CONFIRM_SOURCE_IDS


BASE_COMMIT = "1" * 40
PAYLOAD_COMMIT = "2" * 40
REPORT_COMMIT = "3" * 40
TAG_OBJECT = "4" * 40
SOURCE_COMMIT = "5" * 40
CONTRACT_SHA256 = "6" * 64


def _features() -> tuple[FeatureState, ...]:
    result = []
    for source_id in CONFIRM_SOURCE_IDS:
        result.append(
            FeatureState(
                source_id=source_id,
                state_id=f"{source_id}:decision_step:006",
                decision_step_id=6,
                candidate_event_step_ids=(1, 2, 3, 4),
                q64=(0.0,) * 64,
                candidates=tuple(
                    CandidateFeatures(
                        event_step_id=event,
                        h64=(float(event),) + (0.0,) * 63,
                        g8=(float(event),) + (0.0,) * 7,
                    )
                    for event in (1, 2, 3, 4)
                ),
            )
        )
    return tuple(result)


def _score_records(
    scores: tuple[float, float, float, float],
) -> tuple[dict, ...]:
    ranking = sorted((1, 2, 3, 4), key=lambda event: (-scores[event - 1], event))
    selected = sorted(ranking[:2])
    return tuple(
        {
            "ordinal": ordinal,
            "source_id": source_id,
            "state_id": f"{source_id}:decision_step:006",
            "decision_step_id": 6,
            "candidate_event_step_ids": [1, 2, 3, 4],
            "scores_by_event_step": [
                {"event_step_id": event, "score": scores[event - 1]}
                for event in (1, 2, 3, 4)
            ],
            "ranked_event_step_ids": ranking,
            "selected_event_step_ids": selected,
        }
        for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS)
    )


def _score_maps(
    scores: tuple[float, float, float, float],
) -> dict[str, dict[int, float]]:
    return {
        f"{source_id}:decision_step:006": {
            event: scores[event - 1] for event in (1, 2, 3, 4)
        }
        for source_id in CONFIRM_SOURCE_IDS
    }


def _payload_seal():
    state_ids = [f"{source_id}:decision_step:006" for source_id in CONFIRM_SOURCE_IDS]
    ocr_records = _score_records((0.1, 0.4, 0.3, 0.2))
    policy_records = _score_records((0.8, 0.2, 0.6, 0.1))
    return build_label_blind_payload(
        feature_states=_features(),
        selections={
            "dynamic_recent": {state_id: (3, 4) for state_id in state_ids},
            "ocr_rgb_v2": {state_id: (2, 3) for state_id in state_ids},
            "policy_vision_v3": {state_id: (1, 3) for state_id in state_ids},
        },
        score_records={
            "ocr_rgb_v2": ocr_records,
            "policy_vision_v3": policy_records,
        },
        ensemble_scores=_score_maps((0.9, -0.1, 0.4, 0.0)),
        seed_scores=tuple(
            _score_maps((0.9 - seed / 10.0, -0.1, 0.4, 0.0))
            for seed in range(5)
        ),
    )


def _state_records() -> tuple[dict, ...]:
    return tuple(
        {
            "state": {
                "ordinal": ordinal,
                "source_id": source_id,
                "state_id": f"{source_id}:decision_step:006",
                "decision_step_id": 6,
                "candidate_event_step_ids": [1, 2, 3, 4],
            },
            "payload_commit": PAYLOAD_COMMIT,
            "run_contract_sha256": "7" * 64,
            "distance_rows": [
                {"coalition_event_step_ids": [], "distance_kl": float(ordinal + 1)}
            ],
        }
        for ordinal, source_id in enumerate(CONFIRM_SOURCE_IDS)
    )


def _fixed_report() -> dict:
    return {
        "status": "NO_GO_INDEPENDENT_CONFIRM",
        "evaluation_performed": True,
        "fixed_state_denominator": 20,
        "reference": {"expected_count": 20, "success_count": 20, "failure_count": 0},
        "bootstrap": {
            "unit": "trajectory",
            "resamples": 10_000,
            "seed": 271_828,
            "confidence": 0.9,
            "interval": "percentile",
            "quantile": "Hyndman_Fan_type_7",
        },
        "metrics": {"ensemble_exact_raw_utility_ratio": 0.8},
        "records": list(_state_records()),
        "gate_checks": {"reference_coverage_20_of_20": True},
        "go": False,
    }


def _run_manifest() -> dict:
    return {
        "source_git_commit": SOURCE_COMMIT,
        "contract_sha256": CONTRACT_SHA256,
        "runtime_metadata": {
            "host": "node-radixark-16-0000",
            "gpu_count": 4,
            "run_contract_sha256": "7" * 64,
        },
        "operation_counts": {
            "reference_teacher_forward_count": 20,
            "coalition_teacher_forward_count": 320,
            "retry_count": 0,
        },
    }


class FakeRemote:
    def __init__(self, *, extra_base_file: str | None = None) -> None:
        base_files = {"README.md": b"private confirm artifact\n"}
        if extra_base_file is not None:
            base_files[extra_base_file] = b"conflict\n"
        self.commits = {BASE_COMMIT: base_files}
        self.parents = {BASE_COMMIT: None}
        self.titles = {BASE_COMMIT: "initial"}
        self.main = BASE_COMMIT
        self.tag: tuple[str, str] | None = None
        self.download_drift_path: str | None = None
        self.create_commit_calls = []

    def repo_info(self, repo, *, repo_type, revision):
        assert repo == DESTINATION_REPO and repo_type == "dataset"
        resolved = self.tag[1] if revision == DESTINATION_TAG and self.tag else revision
        if resolved == "main":
            resolved = self.main
        return SimpleNamespace(private=True, sha=resolved)

    def list_repo_files(self, repo, *, repo_type, revision):
        resolved = self.main if revision == "main" else revision
        return sorted(self.commits[resolved])

    def list_repo_refs(self, repo, *, repo_type):
        tags = []
        if self.tag is not None:
            tags.append(SimpleNamespace(name=DESTINATION_TAG, target_commit=self.tag[0]))
        return SimpleNamespace(tags=tags)

    def create_commit(
        self,
        repo,
        *,
        repo_type,
        revision,
        parent_commit,
        commit_message,
        operations,
    ):
        assert revision == "main"
        if parent_commit != self.main:
            raise RuntimeError("parent mismatch")
        commit = PAYLOAD_COMMIT if self.main == BASE_COMMIT else REPORT_COMMIT
        files = dict(self.commits[parent_commit])
        for operation in operations:
            files[operation.path_in_repo] = operation.payload
        self.commits[commit] = files
        self.parents[commit] = parent_commit
        self.titles[commit] = commit_message
        self.main = commit
        self.create_commit_calls.append((commit, parent_commit, commit_message))
        return SimpleNamespace(oid=commit)

    def list_repo_commits(self, repo, *, repo_type, revision, formatted):
        result = []
        commit = revision
        while commit is not None:
            result.append(
                SimpleNamespace(commit_id=commit, title=self.titles[commit])
            )
            commit = self.parents[commit]
        return result

    def create_tag(
        self,
        repo,
        *,
        repo_type,
        tag,
        tag_message,
        revision,
        exist_ok,
    ):
        assert tag == DESTINATION_TAG
        assert tag_message and revision == REPORT_COMMIT and exist_ok is False
        if self.tag is not None:
            raise RuntimeError("tag exists")
        self.tag = (TAG_OBJECT, revision)

    def download(self, *, repo_id, repo_type, filename, revision, local_dir, force_download):
        assert revision == DESTINATION_TAG and self.tag is not None
        payload = self.commits[self.tag[1]][filename]
        if filename == self.download_drift_path:
            payload += b"drift"
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


def _operation_factory(*, path_in_repo, path_or_fileobj):
    assert isinstance(path_or_fileobj, io.BytesIO)
    return SimpleNamespace(path_in_repo=path_in_repo, payload=path_or_fileobj.getvalue())


def test_payload_inventory_is_exactly_eight_label_blind_replayable_files():
    seal = _payload_seal()
    assert len(PAYLOAD_TARGETS) == 8
    assert set(seal.files) == set(PAYLOAD_TARGETS)
    assert PAYLOAD_MANIFEST_PATH in seal.files
    assert b"distance_rows" not in b"".join(seal.files.values())
    assert b"canonical_action" not in b"".join(seal.files.values())
    assert seal_label_blind_payload(seal.files).inventory_sha256 == seal.inventory_sha256

    states = read_feature_states_jsonl(seal.files[FEATURE_STATE_PATH])
    assert len(states) == 20
    assert states[0].source_id == CONFIRM_SOURCE_IDS[0]
    for name, path in SELECTION_PATHS.items():
        assert len(read_selection_artifact(seal.files[path], name=name)) == 20
    for name, path in SCORE_PATHS.items():
        assert len(read_score_records_jsonl(seal.files[path], name=name)) == 20
    decisions = read_independent_decisions(seal.files[INDEPENDENT_DECISION_PATH])
    assert len(decisions["ensemble_scores"]) == 20
    assert len(decisions["seed_scores"]) == 5


def test_payload_seal_rejects_score_selection_contradiction():
    seal = _payload_seal()
    files = dict(seal.files)
    value = __import__("json").loads(files[SELECTION_PATHS["ocr_rgb_v2"]])
    value["records"][0]["selected_event_step_ids"] = [1, 2]
    files[SELECTION_PATHS["ocr_rgb_v2"]] = canonical_json_bytes(value) + b"\n"
    with pytest.raises(ValueError, match="canonical replay|contradict"):
        seal_label_blind_payload(files)


def test_independent_decisions_reject_nonpositive_event_selection_tamper():
    seal = _payload_seal()
    value = __import__("json").loads(seal.files[INDEPENDENT_DECISION_PATH])
    value["records"][0]["ensemble"]["selected_event_step_ids"] = [1, 2]
    payload = canonical_json_bytes(value) + b"\n"
    with pytest.raises(ValueError, match="ranking replay"):
        read_independent_decisions(payload)


def test_report_bundle_binds_exact_payload_commit_and_replays():
    seal = _payload_seal()
    files = build_report_files(
        state_records=_state_records(),
        fixed_report=_fixed_report(),
        run_manifest=_run_manifest(),
        payload_seal=seal,
        payload_commit=PAYLOAD_COMMIT,
    )
    assert len(REPORT_TARGETS) == 4
    assert set(files) == set(REPORT_TARGETS)
    assert PAYLOAD_COMMIT.encode() in files[BUNDLE_MANIFEST_PATH]
    validate_report_files(files, payload_seal=seal, payload_commit=PAYLOAD_COMMIT)
    with pytest.raises(ValueError, match="payload commit|identity|replay"):
        validate_report_files(files, payload_seal=seal, payload_commit="9" * 40)


def test_report_replay_cross_checks_raw_payload_and_run_contract_bindings():
    seal = _payload_seal()
    files = dict(
        build_report_files(
            state_records=_state_records(),
            fixed_report=_fixed_report(),
            run_manifest=_run_manifest(),
            payload_seal=seal,
            payload_commit=PAYLOAD_COMMIT,
        )
    )
    rows = [json.loads(line) for line in files[STATE_RECORDS_PATH].splitlines()]
    rows[0]["record"]["payload_commit"] = "9" * 40
    files[STATE_RECORDS_PATH] = b"".join(
        canonical_json_bytes(row) + b"\n" for row in rows
    )
    with pytest.raises(ValueError, match="raw records differ"):
        validate_report_files(
            files,
            payload_seal=seal,
            payload_commit=PAYLOAD_COMMIT,
        )


def test_two_stage_publisher_enforces_direct_parent_tag_and_fresh_replay(tmp_path: Path):
    seal = _payload_seal()
    remote = FakeRemote()
    payload_receipt = publish_payload_commit(
        api=remote,
        operation_factory=_operation_factory,
        payload_seal=seal,
    )
    assert payload_receipt.base_commit == BASE_COMMIT
    assert payload_receipt.payload_commit == PAYLOAD_COMMIT
    assert remote.main == PAYLOAD_COMMIT
    assert set(remote.commits[PAYLOAD_COMMIT]) == {"README.md", *PAYLOAD_TARGETS}

    report_files = build_report_files(
        state_records=_state_records(),
        fixed_report=_fixed_report(),
        run_manifest=_run_manifest(),
        payload_seal=seal,
        payload_commit=payload_receipt.payload_commit,
    )
    receipt = publish_report_commit(
        api=remote,
        operation_factory=_operation_factory,
        download_fn=remote.download,
        payload_seal=seal,
        payload_receipt=payload_receipt,
        report_files=report_files,
        fresh_parent=tmp_path.resolve(),
    )
    assert receipt.report_commit == REPORT_COMMIT
    assert receipt.annotated_tag_object == TAG_OBJECT
    assert remote.tag == (TAG_OBJECT, REPORT_COMMIT)
    assert remote.create_commit_calls == [
        (PAYLOAD_COMMIT, BASE_COMMIT, "independent confirm-20 label-blind payload"),
        (REPORT_COMMIT, PAYLOAD_COMMIT, "independent confirm-20 fixed report"),
    ]
    assert set(remote.commits[REPORT_COMMIT]) == {
        "README.md",
        *PAYLOAD_TARGETS,
        *REPORT_TARGETS,
    }


def test_payload_publication_rejects_nonempty_or_pretagged_destination():
    seal = _payload_seal()
    conflict = FakeRemote(extra_base_file="old/result.json")
    with pytest.raises(ValueError, match="non-conflicting base"):
        publish_payload_commit(
            api=conflict,
            operation_factory=_operation_factory,
            payload_seal=seal,
        )

    tagged = FakeRemote()
    tagged.tag = (TAG_OBJECT, BASE_COMMIT)
    with pytest.raises(ValueError, match="tag already exists"):
        publish_payload_commit(
            api=tagged,
            operation_factory=_operation_factory,
            payload_seal=seal,
        )


def test_report_publication_rejects_main_race_before_remote_write(tmp_path: Path):
    seal = _payload_seal()
    remote = FakeRemote()
    receipt = publish_payload_commit(
        api=remote,
        operation_factory=_operation_factory,
        payload_seal=seal,
    )
    report_files = build_report_files(
        state_records=_state_records(),
        fixed_report=_fixed_report(),
        run_manifest=_run_manifest(),
        payload_seal=seal,
        payload_commit=receipt.payload_commit,
    )
    remote.main = BASE_COMMIT
    with pytest.raises(ValueError, match="payload stage drifted"):
        publish_report_commit(
            api=remote,
            operation_factory=_operation_factory,
            download_fn=remote.download,
            payload_seal=seal,
            payload_receipt=receipt,
            report_files=report_files,
            fresh_parent=tmp_path.resolve(),
        )
    assert len(remote.create_commit_calls) == 1


def test_fresh_download_byte_drift_prevents_completion(tmp_path: Path):
    seal = _payload_seal()
    remote = FakeRemote()
    receipt = publish_payload_commit(
        api=remote,
        operation_factory=_operation_factory,
        payload_seal=seal,
    )
    report_files = build_report_files(
        state_records=_state_records(),
        fixed_report=_fixed_report(),
        run_manifest=_run_manifest(),
        payload_seal=seal,
        payload_commit=receipt.payload_commit,
    )
    remote.download_drift_path = FEATURE_STATE_PATH
    with pytest.raises(ValueError, match="fresh-downloaded artifact bytes drifted"):
        publish_report_commit(
            api=remote,
            operation_factory=_operation_factory,
            download_fn=remote.download,
            payload_seal=seal,
            payload_receipt=receipt,
            report_files=report_files,
            fresh_parent=tmp_path.resolve(),
        )


def test_relative_fresh_download_parent_is_rejected(tmp_path: Path):
    seal = _payload_seal()
    remote = FakeRemote()
    receipt = publish_payload_commit(
        api=remote,
        operation_factory=_operation_factory,
        payload_seal=seal,
    )
    report_files = build_report_files(
        state_records=_state_records(),
        fixed_report=_fixed_report(),
        run_manifest=_run_manifest(),
        payload_seal=seal,
        payload_commit=receipt.payload_commit,
    )
    with pytest.raises(ValueError, match="absolute real directory"):
        publish_report_commit(
            api=remote,
            operation_factory=_operation_factory,
            download_fn=remote.download,
            payload_seal=seal,
            payload_receipt=receipt,
            report_files=report_files,
            fresh_parent=Path("relative"),
        )
