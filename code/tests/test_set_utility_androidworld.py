from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from causalcache.policy.gui_owl_v2 import GUIOwlV2Action
from causalcache.restoration_v2_text_backend import (
    build_ocr_record,
    load_backend_config,
)
from causalcache.set_utility_androidworld import (
    EARLY_STEP_POLICY_ID,
    LiveSetUtilityPolicyAdapter,
    OCR_CONFIG_SHA256,
    OnlineObservation,
    encode_deterministic_rgb_png,
    load_outcome_exposed_validation_instance,
    run_live_rapidocr_record,
)
from causalcache.set_utility_live_controller import (
    LIVE_SELECTOR_AUTHORIZATION_STATUS,
    LiveRichSelectorService,
    LiveSelectorAuthorization,
    LiveSelectorBackendResult,
)
from scripts.manage_androidworld_loopback_relay import RelayPlan
from scripts.run_set_utility_androidworld_episode import (
    COMPLETE_EPISODE_STATUS,
    load_resumable_terminal_output,
    run_episode,
    write_episode_output_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
OCR_CONFIG_PATH = ROOT / "code/configs/restoration_v2_ocr_backend.json"
HELDOUT_SHA = "d" * 64
NATIVE_SHA = "1" * 64


def _authorization() -> LiveSelectorAuthorization:
    return LiveSelectorAuthorization(
        authorization_id="a" * 64,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        heldout_result_sha256=HELDOUT_SHA,
        model_config_sha256="e" * 64,
        model_variant="set_transformer_d256_l8_r1_s2_lr1e4",
        native_replay_result_sha256=NATIVE_SHA,
        selections_sha256="f" * 64,
        status=LIVE_SELECTOR_AUTHORIZATION_STATUS,
        winner_model="set_transformer",
    )


def _observation(index: int) -> OnlineObservation:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (12, 18), (index * 20 % 255, 3, 7))
    payload = encode_deterministic_rgb_png(image)
    record = build_ocr_record(
        image_member_path=f"live/test/observation-{index:03d}.png",
        image_bytes=payload,
        backend_config_sha256=OCR_CONFIG_SHA256,
        boxes=[[[0, 0], [3, 0], [3, 3], [0, 3]]],
        texts=[f"Screen {index}"],
        scores=[0.75],
    )
    return OnlineObservation(
        image=image,
        image_member_path=record["image_member_path"],
        image_png=payload,
        metadata={"height": image.height, "width": image.width},
        ocr_record=record,
    )


def _adapter(
    *,
    arm: str = "winner",
    budget: int,
    calls: list[object],
) -> LiveSetUtilityPolicyAdapter:
    authorization = _authorization()

    def backend(request):
        calls.append(request)
        latest = request.candidate_event_ids[-1]
        return LiveSelectorBackendResult(
            selected_event_step_ids=(latest,),
            selected_predicted_utility=1.0,
            scored_subsets=(((), 0.0), ((latest,), 1.0)),
            latency_ms={"selector_total": 2.0},
            source_token_counts={"query_visual": 480},
        )

    service = LiveRichSelectorService(authorization=authorization, backend=backend)

    def select(request):
        response = service.select(request.to_mapping())
        return tuple(response["selected_event_step_ids"]), response, 3.0

    return LiveSetUtilityPolicyAdapter(
        arm=arm,
        authorization=authorization,
        budget=budget,
        expected_heldout_result_sha256=HELDOUT_SHA,
        expected_native_replay_result_sha256=NATIVE_SHA,
        image_decoder=lambda payload: payload,
        instruction="Complete the task",
        ocr_backend_config=load_backend_config(OCR_CONFIG_PATH),
        ocr_backend_config_sha256=OCR_CONFIG_SHA256,
        selector_call=select if arm == "winner" else None,
        source_id="episode-one",
    )


def test_roster_loader_accepts_only_outcome_exposed_validation12() -> None:
    selected = load_outcome_exposed_validation_instance(
        repository_root=ROOT,
        task_type="MarkorDeleteNote",
        task_index=0,
    )
    assert selected["instance"]["task_type"] == "MarkorDeleteNote"
    assert selected["evidence_role"].startswith("outcome_exposed_validation12")
    with pytest.raises(ValueError, match="task_index=0"):
        load_outcome_exposed_validation_instance(
            repository_root=ROOT,
            task_type="MarkorDeleteNote",
            task_index=1,
        )
    with pytest.raises(ValueError, match="outside"):
        load_outcome_exposed_validation_instance(
            repository_root=ROOT,
            task_type="BrowserDraw",
            task_index=0,
        )


@pytest.mark.parametrize("budget", (1, 2, 3, 4))
def test_adapter_uses_identical_early_policy_then_full_history_selector(
    budget: int,
) -> None:
    calls: list[object] = []
    adapter = _adapter(budget=budget, calls=calls)
    observations = [_observation(index) for index in range(6)]
    early_payloads = []
    for step in range(1, 6):
        prepared = adapter.prepare(observations[step - 1])
        early_payloads.append(json.dumps(prepared.messages, default=str, sort_keys=True))
        assert prepared.selector_used is False
        assert prepared.selected_event_step_ids == ()
        assert prepared.warmup_policy_id == EARLY_STEP_POLICY_ID
        assert prepared.policy_image_count == 1
        adapter.append_transition(
            action=GUIOwlV2Action(action="click", coordinate=(100, 200)),
            before=observations[step - 1],
            after=observations[step],
        )
    prepared = adapter.prepare(observations[5])
    assert len(calls) == 1
    request = calls[0]
    assert request.candidate_event_ids == (1, 2, 3, 4, 5)
    assert request.events[-1].post_image.sha256 == request.current_image.sha256
    assert prepared.selected_event_step_ids == (5,)
    assert prepared.high_fidelity_history_image_count == 1
    assert prepared.policy_image_count == 2
    assert prepared.selection_latency_ms == 3.0
    assert prepared.budget == budget
    images = [
        block["image"]
        for block in prepared.messages[1]["content"]
        if block["type"] == "image"
    ]
    assert images[-2:] == [observations[5].image_png, observations[5].image_png]
    assert len(early_payloads) == 5


@pytest.mark.parametrize("arm", ("recent", "ocr_rgb"))
@pytest.mark.parametrize("budget", (1, 2, 3, 4))
def test_local_baselines_use_full_history_without_predictor_rpc(
    arm: str,
    budget: int,
) -> None:
    calls: list[object] = []
    adapter = _adapter(arm=arm, budget=budget, calls=calls)
    observations = [_observation(index) for index in range(6)]
    for step in range(1, 6):
        prepared = adapter.prepare(observations[step - 1])
        assert prepared.selection_method == "shared_summary_only_early_step"
        adapter.append_transition(
            action=GUIOwlV2Action(action="click", coordinate=(100, 200)),
            before=observations[step - 1],
            after=observations[step],
        )
    prepared = adapter.prepare(observations[5])
    if arm == "recent":
        expected = tuple(range(6 - budget, 6))
    else:
        expected = tuple(sorted((*range(1, budget), 5)))
    assert calls == []
    assert prepared.selected_event_step_ids == expected
    assert prepared.selector_used is False
    assert prepared.selector_response is None
    assert prepared.rpc_round_trip_ms is None
    assert prepared.selection_latency_ms >= 0.0
    assert prepared.request_content_sha256 is not None
    assert prepared.high_fidelity_history_image_count == len(expected)
    assert prepared.policy_image_count == len(expected) + 1
    assert prepared.selection_diagnostics["candidate_event_count"] == 5
    assert "full_candidate_history" in prepared.selection_method


def test_all_arms_share_identical_early_step_policy() -> None:
    observations = [_observation(index) for index in range(5)]
    payloads_by_arm = {}
    for arm in ("winner", "recent", "ocr_rgb"):
        adapter = _adapter(arm=arm, budget=2, calls=[])
        payloads = []
        for step in range(1, 5):
            prepared = adapter.prepare(observations[step - 1])
            payloads.append(json.dumps(prepared.messages, default=str, sort_keys=True))
            adapter.append_transition(
                action=GUIOwlV2Action(action="wait"),
                before=observations[step - 1],
                after=observations[step],
            )
        payloads_by_arm[arm] = payloads
    assert payloads_by_arm["winner"] == payloads_by_arm["recent"]
    assert payloads_by_arm["winner"] == payloads_by_arm["ocr_rgb"]


def test_adapter_never_falls_back_after_selector_failure() -> None:
    adapter = _adapter(budget=2, calls=[])
    observations = [_observation(index) for index in range(6)]
    for step in range(1, 6):
        adapter.append_transition(
            action=GUIOwlV2Action(action="wait"),
            before=observations[step - 1],
            after=observations[step],
        )
    adapter.selector_call = lambda _request: (_ for _ in ()).throw(
        RuntimeError("selector unavailable")
    )
    with pytest.raises(RuntimeError, match="selector unavailable"):
        adapter.prepare(observations[5])


def test_adapter_binds_go_result_shas() -> None:
    calls: list[object] = []
    with pytest.raises(ValueError, match="native replay"):
        LiveSetUtilityPolicyAdapter(
            arm="winner",
            authorization=_authorization(),
            budget=2,
            expected_heldout_result_sha256=HELDOUT_SHA,
            expected_native_replay_result_sha256="9" * 64,
            image_decoder=lambda payload: payload,
            instruction="task",
            ocr_backend_config=load_backend_config(OCR_CONFIG_PATH),
            ocr_backend_config_sha256=OCR_CONFIG_SHA256,
            selector_call=lambda request: calls.append(request),
            source_id="source",
        )


def test_live_rapidocr_record_accepts_rgb_png_without_fake_tokens() -> None:
    observation = _observation(3)

    class Result:
        boxes = None
        txts = None
        scores = None

    calls = []

    def engine(payload, **kwargs):
        calls.append((payload, kwargs))
        return Result()

    record = run_live_rapidocr_record(
        engine=engine,
        backend_config=load_backend_config(OCR_CONFIG_PATH),
        image_member_path="live/test/rgb.png",
        image_bytes=observation.image_png,
        backend_config_sha256=OCR_CONFIG_SHA256,
    )
    assert record["source_mode"] == "RGB"
    assert record["full_spatial_tokens"] == []
    assert calls[0][1]["use_det"] is True
    assert calls[0][1]["use_cls"] is False


class _FakeEnvironment:
    def __init__(self, observations):
        self.observations = iter(observations)
        self.actions = []
        self.torn_down = False

    def initialize(self):
        return {"status": "initialized"}

    def score(self):
        return 1.0 if self.actions and self.actions[-1]["action_type"] == "status" else 0.0

    def screenshot(self):
        return next(self.observations)

    def execute(self, action):
        self.actions.append(action)
        return {"status": "success"}

    def tear_down(self):
        self.torn_down = True
        return {"status": "success"}


class _FakeRuntime:
    def __init__(self):
        self.count = 0

    def generate_native_action(self, _messages):
        self.count += 1
        action = (
            GUIOwlV2Action(action="terminate", status="success")
            if self.count == 6
            else GUIOwlV2Action(action="click", coordinate=(100, 200))
        )
        return SimpleNamespace(
            metadata={"effective_visual_tokens_per_image": 480},
            output_text=f"synthetic-{self.count}",
            parsed_output=SimpleNamespace(canonical_action=action),
        )


@pytest.mark.parametrize("arm", ("winner", "recent", "ocr_rgb"))
def test_fake_environment_episode_reaches_memory_selection_at_decision_six(
    arm: str,
) -> None:
    calls: list[object] = []
    adapter_source = _adapter(budget=2, calls=calls)
    observations = [_observation(index) for index in range(6)]
    args = SimpleNamespace(
        arm=arm,
        budget=2,
        episode_id="synthetic-episode",
        expected_heldout_result_sha256=HELDOUT_SHA,
        expected_native_replay_result_sha256=NATIVE_SHA,
        repository_root=ROOT,
        task_index=0,
        task_type="MarkorDeleteNote",
        topology_check_only=False,
    )
    ocr_provider = SimpleNamespace(
        backend_config=load_backend_config(OCR_CONFIG_PATH),
        backend_config_sha256=OCR_CONFIG_SHA256,
    )
    environment = _FakeEnvironment(observations)
    summary = run_episode(
        args,
        authorization=_authorization(),
        environment=environment,
        ocr_provider=ocr_provider,
        policy_runtime=_FakeRuntime(),
        selector=(adapter_source.selector_call if arm == "winner" else None),
        topology_canary={
            "androidworld": {"status": "PASSED"},
            "selector": {"status": "PASSED"},
        },
    )
    assert summary["run_status"] == "COMPLETE_OUTCOME_EXPOSED_VALIDATION12_MEMORY_EPISODE"
    assert summary["terminal_success"] is True
    assert summary["steps"][4]["selector_used"] is False
    assert summary["steps"][5]["selector_used"] is (arm == "winner")
    if arm == "winner":
        assert summary["steps"][5]["selected_event_step_ids"] == [5]
    elif arm == "recent":
        assert summary["steps"][5]["selected_event_step_ids"] == [4, 5]
    else:
        assert summary["steps"][5]["selected_event_step_ids"] == [1, 5]
    assert environment.torn_down is True
    assert len(calls) == (1 if arm == "winner" else 0)


def test_relay_plan_never_assumes_hyper_can_reach_aries_directly(tmp_path) -> None:
    plan = RelayPlan(
        aries_host="aries.cs.ucsb.edu",
        aries_http_port=8080,
        aries_relay_ssh_port=20042,
        aries_user="jiaxuanluo",
        connect_aries_script=tmp_path / "connect_aries.sh",
        hyper="hyper00",
        hyper_loopback_port=18080,
        mac_loopback_port=28080,
        maximum_rtt_ms=250.0,
        state_dir=tmp_path / "state",
        taurus="taurus",
    )
    mapping = plan.to_mapping()
    assert mapping["status"] == "PLANNED_NOT_VERIFIED"
    assert "Mac reverse SSH" in mapping["route"]
    assert "aries.cs.ucsb.edu" not in " ".join(plan.hyper_reverse_command())
    assert "127.0.0.1:20042" in " ".join(plan.aries_forward_command())
    assert plan.hyper_url == "http://127.0.0.1:18080"


def test_episode_output_is_atomic_exclusive_and_explicitly_resumable(
    tmp_path,
) -> None:
    output = tmp_path / "episode.json"
    record = {
        "arm": "winner_B2",
        "episode_id": "episode-001",
        "instance": {"task_index": 0, "task_type": "MarkorDeleteNote"},
        "run_status": COMPLETE_EPISODE_STATUS,
        "terminal_success": False,
    }
    assert load_resumable_terminal_output(
        output,
        arm="winner",
        budget=2,
        episode_id="episode-001",
        resume_completed=False,
        task_index=0,
        task_type="MarkorDeleteNote",
        topology_check_only=False,
    ) is None
    write_episode_output_atomic(output, record)
    assert json.loads(output.read_text(encoding="utf-8")) == record
    assert list(tmp_path.glob(".*.tmp")) == []
    with pytest.raises(FileExistsError, match="already exists"):
        write_episode_output_atomic(output, {"replacement": True})
    with pytest.raises(FileExistsError, match="resume-completed"):
        load_resumable_terminal_output(
            output,
            arm="winner",
            budget=2,
            episode_id="episode-001",
            resume_completed=False,
            task_index=0,
            task_type="MarkorDeleteNote",
            topology_check_only=False,
        )
    resumed = load_resumable_terminal_output(
        output,
        arm="winner",
        budget=2,
        episode_id="episode-001",
        resume_completed=True,
        task_index=0,
        task_type="MarkorDeleteNote",
        topology_check_only=False,
    )
    assert resumed == record
    with pytest.raises(ValueError, match="identity differs"):
        load_resumable_terminal_output(
            output,
            arm="recent",
            budget=2,
            episode_id="episode-001",
            resume_completed=True,
            task_index=0,
            task_type="MarkorDeleteNote",
            topology_check_only=False,
        )
    assert json.loads(output.read_text(encoding="utf-8")) == record
