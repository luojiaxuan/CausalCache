"""Unit tests for the argmax-agreement probe (no GPU, no model weights).

# note (luojiaxuan): 两件事必须被钉死。(1) argmax 一致率的算术 —— token_agree /
# all_agree / first-k 全是"闭环行为会不会变"的代理量,算错一个就把整条线的 go/no-go
# 判反。(2) 主结论臂用的是 **cross-fit 两折集合**而不是 ``in_sample_best_steps``:
# in_sample 是在同一批 token 上取 max,它的增益里混着选择噪声,拿它当 oracle 会把
# "argmax 翻了"这个结论也一起高估。这条契约靠一个 in_sample 与两折**都不同**的
# 合成标签来断言,而不是靠代码审查。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_argmax_agreement import (  # noqa: E402
    CROSSFIT_DIRECTIONS,
    DELTA_NAMES,
    SPAN_NAMES,
    agreement_record,
    arms_from_k1_block,
    fold_token_agree,
    group_quantities,
    load_oracle_sets_from_labels,
    metrics_from_record,
    split_strata,
    target_char_spans,
    target_token_spans,
    target_token_stats,
)

TARGET_TEXT = (
    "Action: Select a photo from the gallery.\n<tool_call>\n"
    '{"name": "mobile_use", "arguments": {"action": "click", '
    '"coordinate": [136, 825]}}\n</tool_call>'
)


def make_record(hits: list[bool], values: list[float] | None = None) -> dict:
    values = values if values is not None else [-0.5] * len(hits)
    return agreement_record(values, hits, total_tokens=len(hits) + 100)


# ---------------------------------------------------------------------------
# argmax 一致率的算术
# ---------------------------------------------------------------------------
def test_agreement_record_counts_hits_folds_and_leading_run():
    record = make_record([True, True, False, True, True], values=[-1.0, -2.0, -3.0, -4.0, -5.0])
    assert record["n_tok"] == 5
    assert record["n_hit"] == 4
    assert record["lead_hits"] == 2
    # 偶数位 index 0/2/4 -> hits True/False/True;奇数位 index 1/3 -> True/True
    assert (record["n_even"], record["hit_even"]) == (3, 2)
    assert (record["n_odd"], record["hit_odd"]) == (2, 2)
    assert record["mean"] == pytest.approx(-3.0)
    assert record["sum_even"] == pytest.approx(-9.0)
    assert record["sum_odd"] == pytest.approx(-6.0)
    assert record["prompt_tokens"] == 100


def test_agreement_record_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        agreement_record([-1.0, -2.0], [True], total_tokens=10)
    with pytest.raises(ValueError):
        agreement_record([], [], total_tokens=10)


def test_token_agree_is_a_proportion_not_a_count():
    assert metrics_from_record(make_record([True, False, False, False]))["token_agree"] == 0.25
    assert metrics_from_record(make_record([True] * 8))["token_agree"] == 1.0
    assert metrics_from_record(make_record([False] * 8))["token_agree"] == 0.0


def test_all_agree_needs_every_target_token():
    assert metrics_from_record(make_record([True] * 5))["all_agree"] == 1.0
    assert metrics_from_record(make_record([True] * 4 + [False]))["all_agree"] == 0.0
    # 最后一个 token 错掉不影响 token_agree 还是 0.8,但 all_agree 必须归零
    assert metrics_from_record(make_record([True] * 4 + [False]))["token_agree"] == 0.8


def test_first_k_agree_uses_the_leading_run_only():
    # 前 4 中,第 5 个错:first1/first4 命中,first8 不命中
    metrics = metrics_from_record(make_record([True] * 4 + [False] + [True] * 5))
    assert metrics["first1_agree"] == 1.0
    assert metrics["first4_agree"] == 1.0
    assert metrics["first8_agree"] == 0.0
    # 第一个就错:全部 first-k 归零,即使后面全中
    metrics = metrics_from_record(make_record([False] + [True] * 20))
    assert metrics["first1_agree"] == 0.0
    assert metrics["first4_agree"] == 0.0
    assert metrics["first8_agree"] == 0.0


def test_first_k_degenerates_to_all_agree_on_short_targets():
    # 目标只有 3 个 token 时 first4 / first8 等价于 all_agree(hits[:k].all() 的语义)
    metrics = metrics_from_record(make_record([True, True, True]))
    assert metrics["first4_agree"] == 1.0
    assert metrics["first8_agree"] == 1.0
    metrics = metrics_from_record(make_record([True, True, False]))
    assert metrics["first4_agree"] == 0.0
    assert metrics["first8_agree"] == 0.0


def test_fold_token_agree_restricts_to_one_parity():
    record = make_record([True, False, True, False, True])
    assert fold_token_agree(record, "even") == 1.0  # index 0/2/4 全中
    assert fold_token_agree(record, "odd") == 0.0  # index 1/3 全错
    assert fold_token_agree(make_record([True]), "odd") is None


# ---------------------------------------------------------------------------
# argmax 与 logprob 来自同一次前向 / 同一个张量
# ---------------------------------------------------------------------------
def test_target_token_stats_matches_hand_computed_argmax_and_logprobs():
    torch = pytest.importorskip("torch")

    vocab = 5
    # 目标 token 是 [1, 2];在位置 0 上 argmax 是 1(命中),位置 1 上 argmax 是 3(未中)。
    tail = torch.tensor(
        [[[0.0, 9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 9.0, 0.0], [0.0] * vocab]]
    )

    class FakeOutputs:
        logits = tail

    class FakeModel:
        def __call__(self, **kwargs):
            assert kwargs.get("logits_to_keep") == 3
            return FakeOutputs()

    encoded = {
        "input_ids": torch.tensor([[7, 7, 1, 2]]),
        "labels": torch.tensor([[-100, -100, 1, 2]]),
    }
    values, hits = target_token_stats(FakeModel(), encoded, torch=torch)
    assert [bool(flag) for flag in hits] == [True, False]
    expected = torch.log_softmax(tail[:, :-1].float(), dim=-1)
    assert float(values[0]) == pytest.approx(float(expected[0, 0, 1]))
    assert float(values[1]) == pytest.approx(float(expected[0, 1, 2]))
    # labels 必须被 pop 掉,否则会被当成 model kwarg 传下去
    assert "labels" not in encoded


# ---------------------------------------------------------------------------
# 主结论臂用 cross-fit,不用 in_sample
# ---------------------------------------------------------------------------
CROSSFIT_LABEL_BLOCK = {
    "both_folds_positive": True,
    "crossfit_gain": 0.02,
    "directions": {
        "even": {"gain": 0.01, "steps": [3, 11]},
        "odd": {"gain": 0.03, "steps": [5, 11]},
    },
    # 刻意与两折**都不同**:若实现回退到 in_sample,断言立刻炸。
    "in_sample_best_steps": [9, 11],
    "in_sample_gain": 0.09,
    "pool_size": 24,
}


def test_arms_take_the_two_fold_sets_and_quarantine_in_sample():
    arms = arms_from_k1_block(CROSSFIT_LABEL_BLOCK, (11, 12))
    assert arms["crossfit"] == {"even": (3, 11), "odd": (5, 11)}
    assert arms["in_sample"] == (9, 11)
    assert arms["in_sample"] not in arms["crossfit"].values()
    assert arms["crossfit_gain"] == 0.02


def test_load_oracle_sets_from_labels_reads_crossfit_directions(tmp_path):
    path = tmp_path / "labels.shard000-of-001.jsonl"
    path.write_text(
        json.dumps(
            {
                "pair_group": "ep0:13",
                "episode": "ep0",
                "current_step": 13,
                "format": "official_style_sparse_multiturn",
                "budgets": {
                    "2": {
                        "recent_steps": [11, 12],
                        "anchor_mean": -0.6,
                        "n_old": 10,
                        "k1": CROSSFIT_LABEL_BLOCK,
                        "k1_drop_positions": "all",
                    },
                    # k1 为 None(没有旧帧可换)的预算必须被静默跳过,而不是崩掉
                    "4": {"recent_steps": [9, 10, 11, 12], "k1": None},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_oracle_sets_from_labels([path], [2, 4])
    assert set(loaded["ep0:13"]) == {2}
    assert loaded["ep0:13"][2]["crossfit"] == {"even": (3, 11), "odd": (5, 11)}
    assert loaded["ep0:13"][2]["in_sample"] == (9, 11)


def test_group_quantities_headline_delta_ignores_the_in_sample_set():
    """The primary delta must move with the two fold sets and not with in_sample."""
    arms = arms_from_k1_block(CROSSFIT_LABEL_BLOCK, (11, 12))
    recent = make_record([True] + [False] * 9)  # token_agree 0.1, all_agree 0
    fold_even = make_record([True, True] + [False] * 8)  # 0.2
    fold_odd = make_record([True, True, True] + [False] * 7)  # 0.3
    in_sample = make_record([True] * 10)  # 1.0,远好于两折

    records = {
        arms["recent"]: recent,
        arms["crossfit"]["even"]: fold_even,
        arms["crossfit"]["odd"]: fold_odd,
        arms["in_sample"]: in_sample,
    }
    quantities = group_quantities(arms, records)

    # oracle 臂 = 两折均值 (0.2 + 0.3) / 2 = 0.25,而不是 in_sample 的 1.0
    assert quantities["oracle_token_agree"] == pytest.approx(0.25)
    assert quantities["delta_token_agree"] == pytest.approx(0.25 - 0.1)
    assert quantities["recent_token_agree"] == pytest.approx(0.1)
    # in_sample 只出现在显式标注的上界一族里
    assert quantities["oracle_token_agree_insample"] == pytest.approx(1.0)
    assert quantities["delta_token_agree_insample"] == pytest.approx(0.9)
    assert quantities["delta_all_agree"] == 0.0
    assert quantities["delta_all_agree_insample"] == 1.0
    # 主结论表里不许出现 in_sample
    assert not any(name.endswith("_insample") for name in DELTA_NAMES)


def test_group_quantities_is_insensitive_to_the_in_sample_record():
    """Perturbing only the in_sample arm must leave every headline delta untouched."""
    arms = arms_from_k1_block(CROSSFIT_LABEL_BLOCK, (11, 12))
    base = {
        arms["recent"]: make_record([True] + [False] * 9),
        arms["crossfit"]["even"]: make_record([True, True] + [False] * 8),
        arms["crossfit"]["odd"]: make_record([True, True, True] + [False] * 7),
    }
    poor = {**base, arms["in_sample"]: make_record([False] * 10)}
    rich = {**base, arms["in_sample"]: make_record([True] * 10)}
    left, right = group_quantities(arms, poor), group_quantities(arms, rich)
    for name in DELTA_NAMES:
        if name in left or name in right:
            assert left[name] == pytest.approx(right[name]), name
    assert left["delta_token_agree_insample"] != right["delta_token_agree_insample"]


def test_heldout_delta_uses_the_fold_that_did_not_pick_the_set():
    arms = arms_from_k1_block(CROSSFIT_LABEL_BLOCK, (11, 12))
    # recent: 偶数位全中、奇数位全错 -> even 折 1.0 / odd 折 0.0
    recent = make_record([True, False, True, False])
    # even 折选出的集合:在 odd(held-out)折上全中
    fold_even = make_record([False, True, False, True])
    # odd 折选出的集合:在 even(held-out)折上全错
    fold_odd = make_record([False, True, False, True])
    records = {
        arms["recent"]: recent,
        arms["crossfit"]["even"]: fold_even,
        arms["crossfit"]["odd"]: fold_odd,
        arms["in_sample"]: recent,
    }
    quantities = group_quantities(arms, records)
    # (even 选 -> odd 评估) = 1.0;(odd 选 -> even 评估) = 0.0;均值 0.5
    assert quantities["oracle_token_agree_heldout"] == pytest.approx(0.5)
    # recent 的对应量:odd 折 0.0 与 even 折 1.0,均值 0.5
    assert quantities["recent_token_agree_heldout"] == pytest.approx(0.5)
    assert quantities["delta_token_agree_heldout"] == pytest.approx(0.0)
    # 而不做 held-out 的整段一致率两个臂都是 0.5,delta 也是 0 —— 这里刻意让两者相等
    # 以外的量不参与断言,重点是 held-out 走的确实是互补折。
    assert [pair for pair in CROSSFIT_DIRECTIONS] == [("even", "odd"), ("odd", "even")]


# ---------------------------------------------------------------------------
# 语义段:决策在末尾的 tool_call 里,不在开头
# ---------------------------------------------------------------------------
def test_target_char_spans_locates_the_action_type_and_coordinate():
    spans = target_char_spans(TARGET_TEXT)
    assert set(spans) == set(SPAN_NAMES)
    low, high = spans["action_type"]
    assert TARGET_TEXT[low:high] == "click"
    low, high = spans["coordinate"]
    assert TARGET_TEXT[low:high] == "[136, 825]"
    assert TARGET_TEXT[slice(*spans["description"])].startswith("Action: ")
    assert TARGET_TEXT[slice(*spans["tool_call"])].startswith("<tool_call>")
    # description 与 tool_call 严格相邻且不重叠
    assert spans["description"][1] == spans["tool_call"][0]


class FakeFastTokenizer:
    """A whitespace-ish tokenizer that reports real character offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = [(index, index + 1) for index in range(len(text))]
        out = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_target_token_spans_maps_char_ranges_onto_token_indices():
    tokenizer = FakeFastTokenizer()
    spans = target_token_spans(tokenizer, TARGET_TEXT, len(TARGET_TEXT))
    # 一字符一 token,所以 token 区间必须与字符区间重合
    assert spans == target_char_spans(TARGET_TEXT)


def test_target_token_spans_bails_out_when_the_token_count_disagrees():
    # token 数与 encode_sample 那次编码对不上时宁可不产出 span
    assert target_token_spans(FakeFastTokenizer(), TARGET_TEXT, 7) == {}


def test_span_metrics_separate_the_action_type_from_the_free_text_prefix():
    # 10 个 token:0-2 描述、3-9 tool_call,其中 5 是 action 类型、7-9 是坐标
    spans = {
        "description": (0, 3),
        "tool_call": (3, 10),
        "action_type": (5, 6),
        "coordinate": (7, 10),
    }
    hits = [True, True, True, True, True, False, True, False, True, True]
    record = agreement_record([-0.5] * 10, hits, total_tokens=110, spans=spans)
    assert (record["n_description"], record["hit_description"]) == (3, 3)
    assert (record["n_action_type"], record["hit_action_type"]) == (1, 0)
    assert (record["n_coordinate"], record["hit_coordinate"]) == (3, 2)
    metrics = metrics_from_record(record)
    assert metrics["description_all_agree"] == 1.0
    assert metrics["action_type_agree"] == 0.0
    assert metrics["action_type_all_agree"] == 0.0
    assert metrics["tool_call_all_agree"] == 0.0
    assert metrics["coordinate_agree"] == pytest.approx(2 / 3)
    # 整段一致率 0.8,但决定闭环分支的那个 token 是错的 —— 两者必须分开报
    assert metrics["token_agree"] == pytest.approx(0.8)


def test_span_metrics_are_absent_when_no_span_was_resolved():
    metrics = metrics_from_record(make_record([True, False]))
    assert not any(name.startswith("action_type") for name in metrics)


def test_agreement_record_rejects_a_span_outside_the_target():
    with pytest.raises(ValueError):
        agreement_record([-1.0] * 3, [True] * 3, total_tokens=10, spans={"tool_call": (1, 9)})


# ---------------------------------------------------------------------------
# 分层
# ---------------------------------------------------------------------------
def test_split_strata_cuts_deciles_by_gain():
    ranked = [(float(index) / 100.0, f"g{index}") for index in range(100)]
    strata = split_strata(ranked)
    assert len(strata["bottom_decile"]) == 10
    assert len(strata["top_decile"]) == 10
    assert len(strata["middle"]) == 80
    assert strata["top_decile"][-1] == "g99"
    assert strata["bottom_decile"][0] == "g0"
    assert set(strata["bottom_decile"]) & set(strata["top_decile"]) == set()


def test_split_strata_falls_back_to_all_when_the_sample_is_tiny():
    strata = split_strata([(0.1, "a"), (0.2, "b")])
    assert set(strata) == {"all"}
