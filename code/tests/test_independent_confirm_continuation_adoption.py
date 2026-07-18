from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from causalcache.gate_v1_data import CandidateFeatures, FeatureState
from causalcache.independent_confirm_artifact import (
    DESTINATION_REPO,
    DESTINATION_TAG,
    PAYLOAD_TARGETS,
    REPORT_TARGETS,
    AdoptedPayloadCommit,
    adopt_existing_payload_commit,
    build_label_blind_payload,
    seal_label_blind_payload,
)
from causalcache.independent_confirm_data import CONFIRM_SOURCE_IDS


BASE_COMMIT = "1" * 40
PAYLOAD_COMMIT = "2" * 40
RACE_COMMIT = "3" * 40
TAG_OBJECT = "4" * 40


def _payload_seal():
    features = tuple(
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
        for source_id in CONFIRM_SOURCE_IDS
    )
    state_ids = tuple(feature.state_id for feature in features)

    def score_records(scores):
        ranked = sorted((1, 2, 3, 4), key=lambda event: (-scores[event - 1], event))
        return tuple(
            {
                "ordinal": ordinal,
                "source_id": source_id,
                "state_id": state_id,
                "decision_step_id": 6,
                "candidate_event_step_ids": [1, 2, 3, 4],
                "scores_by_event_step": [
                    {"event_step_id": event, "score": scores[event - 1]}
                    for event in (1, 2, 3, 4)
                ],
                "ranked_event_step_ids": ranked,
                "selected_event_step_ids": sorted(ranked[:2]),
            }
            for ordinal, (source_id, state_id) in enumerate(
                zip(CONFIRM_SOURCE_IDS, state_ids, strict=True)
            )
        )

    def score_map(scores):
        return {
            state_id: {event: scores[event - 1] for event in (1, 2, 3, 4)}
            for state_id in state_ids
        }

    ocr_scores = (0.1, 0.4, 0.3, 0.2)
    vision_scores = (0.8, 0.2, 0.6, 0.1)
    return build_label_blind_payload(
        feature_states=features,
        selections={
            "dynamic_recent": {state_id: (3, 4) for state_id in state_ids},
            "ocr_rgb_v2": {state_id: (2, 3) for state_id in state_ids},
            "policy_vision_v3": {state_id: (1, 3) for state_id in state_ids},
        },
        score_records={
            "ocr_rgb_v2": score_records(ocr_scores),
            "policy_vision_v3": score_records(vision_scores),
        },
        ensemble_scores=score_map((0.9, -0.1, 0.4, 0.0)),
        seed_scores=tuple(
            score_map((0.9 - seed / 10.0, -0.1, 0.4, 0.0))
            for seed in range(5)
        ),
    )


class ReadOnlyRemote:
    def __init__(self, payload_files) -> None:
        base = {
            ".gitattributes": b"*.json filter=lfs diff=lfs merge=lfs -text\n",
            "README.md": b"# private confirm artifact\n",
        }
        self.commits = {
            BASE_COMMIT: dict(base),
            PAYLOAD_COMMIT: {**base, **dict(payload_files)},
            RACE_COMMIT: dict(base),
        }
        self.parents = {
            BASE_COMMIT: None,
            PAYLOAD_COMMIT: BASE_COMMIT,
            RACE_COMMIT: BASE_COMMIT,
        }
        self.titles = {
            BASE_COMMIT: "initial commit",
            PAYLOAD_COMMIT: "independent confirm-20 label-blind payload",
            RACE_COMMIT: "unrelated race",
        }
        self.main = PAYLOAD_COMMIT
        self.private = True
        self.tag = None
        self.download_drift_path = None
        self.noncanonical_download_path = False
        self.after_download = None
        self.download_count = 0
        self.create_repo_count = 0
        self.create_commit_count = 0
        self.create_tag_count = 0

    def repo_info(self, repo, *, repo_type, revision):
        assert repo == DESTINATION_REPO and repo_type == "dataset"
        if revision == "main":
            resolved = self.main
        elif revision == DESTINATION_TAG and self.tag is not None:
            resolved = self.tag[1]
        else:
            resolved = revision
        return SimpleNamespace(private=self.private, sha=resolved)

    def list_repo_files(self, repo, *, repo_type, revision):
        assert repo == DESTINATION_REPO and repo_type == "dataset"
        resolved = self.main if revision == "main" else revision
        return sorted(self.commits[resolved])

    def list_repo_refs(self, repo, *, repo_type):
        assert repo == DESTINATION_REPO and repo_type == "dataset"
        tags = []
        if self.tag is not None:
            tags.append(
                SimpleNamespace(name=DESTINATION_TAG, target_commit=self.tag[0])
            )
        return SimpleNamespace(tags=tags)

    def list_repo_commits(self, repo, *, repo_type, revision, formatted):
        assert repo == DESTINATION_REPO and repo_type == "dataset"
        assert formatted is False
        records = []
        commit = revision
        while commit is not None:
            records.append(
                SimpleNamespace(commit_id=commit, title=self.titles[commit])
            )
            commit = self.parents[commit]
        return records

    def download(self, *, repo_id, repo_type, filename, revision, local_dir, force_download):
        assert repo_id == DESTINATION_REPO and repo_type == "dataset"
        assert revision == PAYLOAD_COMMIT and force_download is True
        payload = self.commits[revision][filename]
        if filename == self.download_drift_path:
            payload += b"drift"
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        self.download_count += 1
        if self.download_count == len(PAYLOAD_TARGETS) and self.after_download:
            self.after_download()
        if self.noncanonical_download_path:
            alternate = Path(local_dir) / "alternate.bin"
            alternate.write_bytes(payload)
            return str(alternate)
        return str(target)

    def create_repo(self, *_args, **_kwargs):
        self.create_repo_count += 1
        pytest.fail("read-only adoption attempted to create a repository")

    def create_commit(self, *_args, **_kwargs):
        self.create_commit_count += 1
        pytest.fail("read-only adoption attempted to create a commit")

    def create_tag(self, *_args, **_kwargs):
        self.create_tag_count += 1
        pytest.fail("read-only adoption attempted to create a tag")

    @property
    def mutation_counts(self):
        return (
            self.create_repo_count,
            self.create_commit_count,
            self.create_tag_count,
        )


def _adopt(remote: ReadOnlyRemote, seal, tmp_path: Path) -> AdoptedPayloadCommit:
    return adopt_existing_payload_commit(
        api=remote,
        download_fn=remote.download,
        expected_base_commit=BASE_COMMIT,
        expected_base_title="initial commit",
        expected_payload_commit=PAYLOAD_COMMIT,
        expected_payload_files=seal.files,
        fresh_parent=tmp_path.resolve(),
    )


def test_adopts_exact_existing_payload_with_zero_remote_mutation(tmp_path: Path) -> None:
    seal = _payload_seal()
    remote = ReadOnlyRemote(seal.files)

    adopted = _adopt(remote, seal, tmp_path)

    assert adopted.payload_receipt.base_commit == BASE_COMMIT
    assert adopted.payload_receipt.payload_commit == PAYLOAD_COMMIT
    assert (
        adopted.payload_receipt.payload_inventory_sha256
        == seal.inventory_sha256
        == adopted.payload_seal.inventory_sha256
    )
    assert dict(adopted.payload_seal.files) == dict(seal.files)
    assert seal_label_blind_payload(adopted.payload_seal.files)
    assert remote.download_count == len(PAYLOAD_TARGETS)
    assert remote.mutation_counts == (0, 0, 0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda remote: setattr(remote, "main", BASE_COMMIT),
        lambda remote: remote.commits[PAYLOAD_COMMIT].update(
            {"unexpected.json": b"conflict\n"}
        ),
        lambda remote: remote.commits[PAYLOAD_COMMIT].update(
            {REPORT_TARGETS[0]: b"premature report\n"}
        ),
        lambda remote: setattr(remote, "tag", (TAG_OBJECT, PAYLOAD_COMMIT)),
        lambda remote: remote.titles.update({PAYLOAD_COMMIT: "wrong title"}),
        lambda remote: remote.titles.update({BASE_COMMIT: "wrong base title"}),
        lambda remote: remote.parents.update({PAYLOAD_COMMIT: RACE_COMMIT}),
        lambda remote: setattr(remote, "private", False),
    ],
)
def test_rejects_remote_identity_tree_tag_history_or_privacy_drift(
    tmp_path: Path, mutate
) -> None:
    seal = _payload_seal()
    remote = ReadOnlyRemote(seal.files)
    mutate(remote)

    with pytest.raises(ValueError):
        _adopt(remote, seal, tmp_path)

    assert remote.mutation_counts == (0, 0, 0)


def test_rejects_fresh_download_byte_or_path_drift(tmp_path: Path) -> None:
    seal = _payload_seal()
    remote = ReadOnlyRemote(seal.files)
    remote.download_drift_path = sorted(PAYLOAD_TARGETS)[0]
    with pytest.raises(ValueError, match="fresh-downloaded artifact bytes drifted"):
        _adopt(remote, seal, tmp_path)
    assert remote.mutation_counts == (0, 0, 0)

    remote = ReadOnlyRemote(seal.files)
    remote.noncanonical_download_path = True
    with pytest.raises(ValueError, match="noncanonical path"):
        _adopt(remote, seal, tmp_path)
    assert remote.mutation_counts == (0, 0, 0)


@pytest.mark.parametrize("race", ["main", "history", "tag", "base-tree"])
def test_rejects_post_download_toctou_drift(
    tmp_path: Path, race: str
) -> None:
    seal = _payload_seal()
    remote = ReadOnlyRemote(seal.files)

    def mutate_after_download() -> None:
        if race == "main":
            remote.main = RACE_COMMIT
        elif race == "history":
            remote.titles[PAYLOAD_COMMIT] = "changed during replay"
        elif race == "tag":
            remote.tag = (TAG_OBJECT, PAYLOAD_COMMIT)
        else:
            remote.commits[BASE_COMMIT]["new-base-file"] = b"changed\n"

    remote.after_download = mutate_after_download
    with pytest.raises(ValueError, match="drifted during fresh replay"):
        _adopt(remote, seal, tmp_path)
    assert remote.download_count == len(PAYLOAD_TARGETS)
    assert remote.mutation_counts == (0, 0, 0)


def test_rejects_invalid_expected_payload_before_remote_access(tmp_path: Path) -> None:
    seal = _payload_seal()
    remote = ReadOnlyRemote(seal.files)
    incomplete = dict(seal.files)
    incomplete.pop(PAYLOAD_TARGETS[0])

    with pytest.raises(ValueError, match="inventory drifted"):
        adopt_existing_payload_commit(
            api=remote,
            download_fn=remote.download,
            expected_base_commit=BASE_COMMIT,
            expected_base_title="initial commit",
            expected_payload_commit=PAYLOAD_COMMIT,
            expected_payload_files=incomplete,
            fresh_parent=tmp_path.resolve(),
        )

    assert remote.download_count == 0
    assert remote.mutation_counts == (0, 0, 0)
