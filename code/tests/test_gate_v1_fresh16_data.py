from __future__ import annotations

import hashlib
import itertools
import json
import unittest
from unittest.mock import patch

from causalcache.gate_v1_contract import canonical_json_bytes
from causalcache import gate_v1_fresh16_data as fresh


SOURCE_IDS = (
    "0113395203853614",
    "0085973327901251",
    "0217993280505982",
    "0204226739513713",
    "0027349945578994",
    "0063414321407600",
    "0124211143250759",
    "0158200624694354",
    "0104781033187528",
    "0042582068233164",
    "0095139503173521",
    "0028705883346522",
    "0069283472940099",
    "0090192011883108",
    "0017986853193264",
    "0002455403270123",
)
SOURCE_IDS_SHA256 = fresh.FRESH_SOURCE_IDS_SHA256
RUN_CONTRACT_SHA256 = fresh.FRESH_LABEL_RUN_CONTRACT_SHA256


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _event(source_id: str, step: int) -> dict[str, object]:
    before = f"{source_id}/before-{step}.png"
    after = f"{source_id}/after-{step}.png"
    return {
        "step_id": step,
        "observation_before_path": before,
        "observation_before_sha256": _sha(before),
        "observation_after_path": after,
        "observation_after_sha256": _sha(after),
        "executed_action": {"action": "wait"},
        "source_tool_call": {"action": "wait"},
        "low_fidelity_v2": {"step_id": step},
        "low_fidelity_v2_serialized": f"step={step}",
        "low_fidelity_v2_sha256": _sha(f"low-{source_id}-{step}"),
        "low_fidelity_v2_metadata": {},
        "high_fidelity_v2": {"path": after},
        "ocr_record_refs": [after],
    }


def _trajectory(source_id: str) -> dict[str, object]:
    events = [_event(source_id, step) for step in range(1, 6)]
    decisions = []
    for step in (4, 5, 6):
        current = str(events[step - 2]["observation_after_path"])
        decisions.append(
            {
                "state_id": f"{source_id}:decision_step:{step:03d}",
                "decision_step_id": step,
                "history_event_step_ids": list(range(1, step)),
                "candidate_event_step_ids": list(range(1, step - 1)),
                "current_equivalent_event_step_id": step - 1,
                "current_observation_path": current,
                "current_observation_sha256": _sha(current),
                "candidate_event_post_states": [],
                "current_equivalence_witness": {},
                "content_witness_sha256": _sha(f"decision-{source_id}-{step}"),
                "current_expert_action_payload_included": False,
            }
        )
    instruction = f"synthetic instruction {source_id}"
    return {
        "source_id": source_id,
        "role": fresh.FRESH_ROLE,
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        "platform": "synthetic",
        "apps": ["fixture"],
        "device_name": "fixture",
        "resolution": [100, 200],
        "terminal_status": "success",
        "source": {},
        "selection": {},
        "events": events,
        "decisions": decisions,
        "content_witness_sha256": _sha(f"trajectory-{source_id}"),
    }


def _coalitions(event_ids: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        coalition
        for size in range(len(event_ids) + 1)
        for coalition in itertools.combinations(event_ids, size)
    )


def _raw_state(row_index: int, source_id: str) -> dict[str, object]:
    step = 4 + (row_index - 144) % 3
    event_ids = tuple(range(1, step - 1))
    coalitions = _coalitions(event_ids)
    witnesses = {
        coalition: _sha(f"input-{row_index}-{coalition}") for coalition in coalitions
    }
    state = {
        "state_index": row_index,
        "role": fresh.FRESH_ROLE,
        "source_id": source_id,
        "decision_step_id": step,
        "state_id": f"{source_id}:decision_step:{step:03d}",
        "history_event_step_ids": list(range(1, step)),
        "candidate_event_step_ids": list(event_ids),
        "current_equivalent_event_step_id": step - 1,
        "parent_member_name": (
            f"workers/{'even' if row_index % 2 == 0 else 'odd'}"
            f"/states/{row_index:03d}.json"
        ),
        "parent_member_sha256": _sha(f"parent-{row_index}"),
        "canonical_action_sha256": _sha(f"action-{row_index}"),
        "teacher_target_sha256": _sha(f"target-{row_index}"),
        "request_manifest_sha256": _sha(f"request-{row_index}"),
        "slice_witness_sha256": _sha(f"slice-{row_index}"),
        "immutable_artifact_tree_sha256": _sha("tree"),
        "coalition_inputs": [
            {"coalition": list(coalition), "input_sha256": witnesses[coalition]}
            for coalition in coalitions
        ],
    }
    return {
        "schema_version": "synthetic-v1",
        "protocol_id": "synthetic-expansion-labels",
        "status": "VALID_SYNTHETIC_RAW_STATE",
        "run_contract_sha256": RUN_CONTRACT_SHA256,
        "worker": {"worker_id": "even" if row_index % 2 == 0 else "odd"},
        "state": state,
        "reference_teacher": {},
        "distance_rows": [
            {
                "coalition": list(coalition),
                "distance": 0.0 if coalition == event_ids else 1.0 - 0.1 * len(coalition),
                "candidate_input_sha256": witnesses[coalition],
                "teacher_forward_count": 0 if coalition == event_ids else 1,
                "kl_measurement_count": 0 if coalition == event_ids else 1,
                "scalar_host_transfer_count": 0 if coalition == event_ids else 1,
                "is_full_history_reference": coalition == event_ids,
                "full_logit_tensor_host_transfer_count": 0,
            }
            for coalition in coalitions
        ],
        "operation_counts": {},
    }


def _jsonl(records: list[object]) -> bytes:
    return b"".join(
        record + b"\n" if isinstance(record, bytes) else canonical_json_bytes(record) + b"\n"
        for record in records
    )


def _trajectory_payload(
    selected: list[dict[str, object]] | None = None,
) -> bytes:
    records: list[object] = [
        f"TRAIN_TRAJECTORY_SEMANTIC_CANARY_{index:03d}".encode("ascii")
        for index in range(48)
    ]
    records.extend(selected or [_trajectory(source_id) for source_id in SOURCE_IDS])
    return _jsonl(records)


def _raw_state_payload(selected: list[dict[str, object]] | None = None) -> bytes:
    records: list[object] = [
        f"TRAIN_RAW_STATE_SEMANTIC_CANARY_{index:03d}".encode("ascii")
        for index in range(144)
    ]
    records.extend(
        selected
        or [
            _raw_state(row, SOURCE_IDS[(row - 144) // 3])
            for row in fresh.FRESH_RAW_STATE_ROWS
        ]
    )
    return _jsonl(records)


def _verified(payload: bytes, kind: str) -> fresh.VerifiedFresh16Transport:
    return fresh._verify_fresh16_jsonl_transport(
        payload,
        kind=kind,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )


class GateV1Fresh16DataTest(unittest.TestCase):
    def test_selective_join_decodes_only_fresh_rows_and_exact_denominators(self) -> None:
        trajectory_payload = _trajectory_payload()
        raw_state_payload = _raw_state_payload()
        real_loads = json.loads
        decoded_text: list[str] = []

        def audited_loads(value: str, *args: object, **kwargs: object) -> object:
            decoded_text.append(value)
            return real_loads(value, *args, **kwargs)

        with patch.object(fresh.json, "loads", side_effect=audited_loads):
            data = fresh.load_fresh16_selective_data(
                _verified(trajectory_payload, fresh.TRAJECTORY_TRANSPORT),
                _verified(raw_state_payload, fresh.RAW_STATE_TRANSPORT),
                expected_source_ids=SOURCE_IDS,
                expected_source_ids_sha256=SOURCE_IDS_SHA256,
            )

        self.assertEqual(len(decoded_text), 16 + 48)
        self.assertFalse(any("SEMANTIC_CANARY" in value for value in decoded_text))
        self.assertEqual(data.trajectories.semantic_decode_count, 16)
        self.assertEqual(data.trajectories.selected_row_indices, tuple(range(48, 64)))
        self.assertEqual(data.trajectories.candidate_occurrence_count, 144)
        self.assertEqual(data.labels.semantic_decode_count, 48)
        self.assertEqual(data.labels.selected_row_indices, tuple(range(144, 192)))
        self.assertEqual(data.labels.distance_value_decode_count, 448)
        self.assertEqual(data.labels.run_contract_sha256, RUN_CONTRACT_SHA256)
        self.assertEqual(
            tuple(len(label.table.event_ids) for label in data.labels.labels[:3]),
            (2, 3, 4),
        )

    def test_transport_must_be_byte_verified_before_semantic_decode(self) -> None:
        payload = _trajectory_payload()
        with self.assertRaisesRegex(ValueError, "byte binding"):
            fresh._verify_fresh16_jsonl_transport(
                payload,
                kind=fresh.TRAJECTORY_TRANSPORT,
                expected_sha256="0" * 64,
                expected_size_bytes=len(payload),
            )
        with self.assertRaisesRegex(ValueError, "byte binding"):
            fresh._verify_fresh16_jsonl_transport(
                payload,
                kind=fresh.TRAJECTORY_TRANSPORT,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size_bytes=len(payload) + 1,
            )
        with self.assertRaisesRegex(TypeError, "byte-verified"):
            fresh.decode_fresh16_trajectories(  # type: ignore[arg-type]
                payload,
                expected_source_ids=SOURCE_IDS,
                expected_source_ids_sha256=SOURCE_IDS_SHA256,
            )
        with self.assertRaisesRegex(TypeError, "byte-verified transport kind"):
            fresh.decode_fresh16_labels(
                _verified(payload, fresh.TRAJECTORY_TRANSPORT),
                expected_source_ids=SOURCE_IDS,
                expected_source_ids_sha256=SOURCE_IDS_SHA256,
            )
        with self.assertRaisesRegex(TypeError, "must come from byte verification"):
            fresh.VerifiedFresh16Transport(
                payload,
                kind=fresh.TRAJECTORY_TRANSPORT,
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                record_count=64,
                _seal=object(),
            )
        verified = _verified(payload, fresh.TRAJECTORY_TRANSPORT)
        verified._payload = b"mutated\n"  # type: ignore[attr-defined]
        with self.assertRaisesRegex(ValueError, "mutated after verification"):
            fresh.decode_fresh16_trajectories(
                verified,
                expected_source_ids=SOURCE_IDS,
                expected_source_ids_sha256=SOURCE_IDS_SHA256,
            )

    def test_transport_verification_counts_lines_without_decoding_train_rows(self) -> None:
        payload = _trajectory_payload() + b"extra\n"
        with patch.object(fresh.json, "loads") as loads, self.assertRaisesRegex(
            ValueError, "record count"
        ):
            _verified(payload, fresh.TRAJECTORY_TRANSPORT)
        loads.assert_not_called()

    def test_roster_digest_and_source_order_are_fail_closed(self) -> None:
        transport = _verified(_trajectory_payload(), fresh.TRAJECTORY_TRANSPORT)
        with self.assertRaisesRegex(ValueError, "roster, order, or digest"):
            fresh.decode_fresh16_trajectories(
                transport,
                expected_source_ids=SOURCE_IDS,
                expected_source_ids_sha256="0" * 64,
            )
        changed = list(SOURCE_IDS)
        changed.reverse()
        changed_digest = hashlib.sha256(canonical_json_bytes(changed)).hexdigest()
        with self.assertRaisesRegex(ValueError, "roster, order, or digest"):
            fresh.decode_fresh16_trajectories(
                transport,
                expected_source_ids=changed,
                expected_source_ids_sha256=changed_digest,
            )

    def test_trajectory_role_state_id_and_candidate_geometry_are_enforced(self) -> None:
        cases = (
            (lambda rows: rows[0].__setitem__("role", "gate_train_expansion"), "role"),
            (
                lambda rows: rows[0]["decisions"][0].__setitem__(
                    "state_id", "noncanonical"
                ),
                "canonical",
            ),
            (
                lambda rows: rows[0]["decisions"][1].__setitem__(
                    "candidate_event_step_ids", [1, 2]
                ),
                "canonical",
            ),
        )
        for mutation, message in cases:
            selected = [_trajectory(source_id) for source_id in SOURCE_IDS]
            mutation(selected)
            transport = _verified(
                _trajectory_payload(selected), fresh.TRAJECTORY_TRANSPORT
            )
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                fresh.decode_fresh16_trajectories(
                    transport,
                    expected_source_ids=SOURCE_IDS,
                    expected_source_ids_sha256=SOURCE_IDS_SHA256,
                )

    def test_raw_state_role_index_geometry_and_complete_table_are_enforced(self) -> None:
        cases = (
            (lambda rows: rows[0]["state"].__setitem__("state_index", 0), "role, order"),
            (
                lambda rows: rows[0]["state"].__setitem__(
                    "role", "gate_train_expansion"
                ),
                "role, order",
            ),
            (
                lambda rows: rows[1]["state"].__setitem__(
                    "candidate_event_step_ids", [1, 2]
                ),
                "geometry",
            ),
            (lambda rows: rows[2]["distance_rows"].pop(), "denominator"),
        )
        for mutation, message in cases:
            selected = [
                _raw_state(row, SOURCE_IDS[(row - 144) // 3])
                for row in fresh.FRESH_RAW_STATE_ROWS
            ]
            mutation(selected)
            transport = _verified(_raw_state_payload(selected), fresh.RAW_STATE_TRANSPORT)
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                fresh.decode_fresh16_labels(
                    transport,
                    expected_source_ids=SOURCE_IDS,
                    expected_source_ids_sha256=SOURCE_IDS_SHA256,
                )

    def test_raw_state_source_order_and_cross_transport_join_are_enforced(self) -> None:
        selected = [
            _raw_state(row, SOURCE_IDS[(row - 144) // 3])
            for row in fresh.FRESH_RAW_STATE_ROWS
        ]
        selected[0]["state"]["source_id"] = SOURCE_IDS[1]
        transport = _verified(_raw_state_payload(selected), fresh.RAW_STATE_TRANSPORT)
        with self.assertRaisesRegex(ValueError, "role, order"):
            fresh.decode_fresh16_labels(
                transport,
                expected_source_ids=SOURCE_IDS,
                expected_source_ids_sha256=SOURCE_IDS_SHA256,
            )

        joined_labels = [
            _raw_state(row, SOURCE_IDS[(row - 144) // 3])
            for row in fresh.FRESH_RAW_STATE_ROWS
        ]
        joined_labels[0]["state"]["state_id"] = (
            f"{SOURCE_IDS[0]}:decision_step:999"
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            fresh.load_fresh16_selective_data(
                _verified(_trajectory_payload(), fresh.TRAJECTORY_TRANSPORT),
                _verified(_raw_state_payload(joined_labels), fresh.RAW_STATE_TRANSPORT),
                expected_source_ids=SOURCE_IDS,
                expected_source_ids_sha256=SOURCE_IDS_SHA256,
            )

    def test_no_generic_or_full_semantic_decoder_is_exported(self) -> None:
        self.assertFalse(
            {
                "decode_jsonl",
                "decode_all_trajectories",
                "decode_all_raw_states",
                "decode_rows",
                "verify_fresh16_jsonl_transport",
            }.intersection(fresh.__all__)
        )
        verified = _verified(_trajectory_payload(), fresh.TRAJECTORY_TRANSPORT)
        self.assertFalse(hasattr(verified, "payload"))


if __name__ == "__main__":
    unittest.main()
