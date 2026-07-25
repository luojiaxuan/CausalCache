#!/usr/bin/env python3
"""Official-style sparse multiturn renderer 的契约测试(pytest,无需 torch/GPU)。

# note (luojiaxuan): 这套用例守的是"渲染器换代不能悄悄改语义"。被测对象是仓库真代码
# ``causalcache/policy/gui_owl_sparse_multiturn.py``;只对 torch / PIL 这类重依赖做
# "仅在缺失时"的桩替换,好让纯逻辑部分在无 GPU 机器上也能跑。
#
# 重点守三件容易静默出错的事:
#   1. **对齐**:第 i 张历史图后面跟的必须是**它自己那一步**的完整响应。稀疏选点把
#      "位置 == step"这个隐式约定破坏掉了,错位一格不会报错、不会崩,只会让分数
#      安静地变差,而那正是我们要量的东西。
#   2. **覆盖**:1..current_step-1 里每一步恰好出现一次(选中步进 assistant 轮,
#      未选中步进文本行)。漏一步 = 偷偷删信息,重一步 = 偷偷加信息,两种都会污染
#      "换格式带来多少收益"这个结论。
#   3. **fail-closed**:保留轮缺完整响应必须炸,绝不退回裸描述——官方 builder 的
#      P0-1 就是被这种静默降级坑掉 0.133 nats,而且几个月没人发现。
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# 仓库路径与重依赖桩(在 import 被测模块之前完成)
# ---------------------------------------------------------------------------
def _repo_code_root() -> Path:
    """Locate the repository ``code/`` root that holds causalcache/ and scripts/."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "causalcache").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository code root from the test file")


_CODE_ROOT = _repo_code_root()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))


def stub_if_missing(name: str, **attributes: Any) -> None:
    """Register a stand-in module only when the real dependency is unavailable."""
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - 任何 import 失败都退回桩
        pass
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, leaf, module)


stub_if_missing("torch")
stub_if_missing("PIL")
stub_if_missing("PIL.Image", open=lambda *args, **kwargs: None)

from causalcache.policy.gui_owl_official import (  # noqa: E402
    NO_PREVIOUS_ACTION,
    OFFICIAL_FIRST_USER_TEMPLATE,
    OFFICIAL_SYSTEM_PROMPT,
    build_official_messages,
)
from causalcache.policy.gui_owl_sparse_history import (  # noqa: E402
    sparse_image_count,
)
from causalcache.policy.gui_owl_sparse_multiturn import (  # noqa: E402
    SPARSE_MULTITURN_PROMPT_FORMAT,
    SPARSE_MULTITURN_PROTOCOL_ID,
    build_sparse_multiturn_messages,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
CURRENT_STEP = 12
INSTRUCTION = "Find the Bougainvillea photo and identify it."
ACTION_TEXTS = [f"do-{i}" for i in range(1, 18)]


def full_response(step: int) -> str:
    """一条与 target_text 同格式的完整响应(Action + tool_call)。"""
    return (
        f"Action: do-{step}.\n<tool_call>\n"
        f'{{"name": "mobile_use", "arguments": {{"action": "click", '
        f'"coordinate": [{step}, {step}]}}}}\n</tool_call>'
    )


FULL_RESPONSES = [full_response(i) for i in range(1, 18)]
SPARSE_STEPS = [2, 5, 6, 10]


def image_of(step: int) -> dict[str, str]:
    return {"__path__": f"images/EP-A/obs-{step - 1:03d}.png"}


CURRENT_IMAGE = {"__path__": f"images/EP-A/obs-{CURRENT_STEP - 1:03d}.png"}


_DEFAULT = object()


def render(
    *,
    steps: list[int] | None = None,
    images: list[Any] | None = None,
    current_step: int = CURRENT_STEP,
    action_texts: list[str] | None = None,
    full_responses: Any = _DEFAULT,
    instruction: str = INSTRUCTION,
    current_image: Any = CURRENT_IMAGE,
) -> list[dict[str, Any]]:
    chosen = SPARSE_STEPS if steps is None else steps
    return build_sparse_multiturn_messages(
        instruction=instruction,
        action_texts=ACTION_TEXTS if action_texts is None else action_texts,
        full_responses=(
            FULL_RESPONSES if full_responses is _DEFAULT else full_responses
        ),
        selected_steps=chosen,
        selected_images=[image_of(s) for s in chosen] if images is None else images,
        current_step=current_step,
        current_image=current_image,
    )


def texts(messages: list[dict[str, Any]]) -> list[str]:
    return [
        part["text"]
        for message in messages
        for part in message["content"]
        if part.get("type") == "text"
    ]


def image_parts(messages: list[dict[str, Any]]) -> list[Any]:
    return [
        part["image"]
        for message in messages
        for part in message["content"]
        if part.get("type") == "image"
    ]


def flat_parts(messages: list[dict[str, Any]]) -> list[tuple[str, str, Any]]:
    """[(role, part_type, payload)] —— 跨 message 的**全局顺序**视图。"""
    out: list[tuple[str, str, Any]] = []
    for message in messages:
        for part in message["content"]:
            kind = part.get("type")
            out.append((message["role"], kind, part.get("text") or part.get("image")))
    return out


# ---------------------------------------------------------------------------
# 1. 模块与 API
# ---------------------------------------------------------------------------
def test_renderer_imports_from_repository() -> None:
    module = importlib.import_module("causalcache.policy.gui_owl_sparse_multiturn")
    assert Path(module.__file__).resolve().is_relative_to(_CODE_ROOT)
    assert callable(module.build_sparse_multiturn_messages)


def test_builder_signature_is_keyword_only_and_frozen() -> None:
    parameters = inspect.signature(build_sparse_multiturn_messages).parameters
    assert set(parameters) == {
        "instruction", "action_texts", "full_responses", "selected_steps",
        "selected_images", "current_step", "current_image",
    }
    for name, parameter in parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} 必须是 keyword-only:七个同类参数用位置传极易错位且完全静默"
        )


def test_protocol_id_names_the_format_honestly() -> None:
    """名字必须自称 official-**style**,不能自称官方协议。"""
    assert SPARSE_MULTITURN_PROTOCOL_ID == (
        "causalcache_official_style_sparse_multiturn_v1"
    )
    assert SPARSE_MULTITURN_PROMPT_FORMAT == "official_style_sparse_multiturn"
    assert "causalcache" in SPARSE_MULTITURN_PROTOCOL_ID


def test_docstring_disclaims_official_provenance() -> None:
    """docstring 必须写明"官方代码里没有跳跃式历史轮",否则下游一定会误引用。"""
    module = importlib.import_module("causalcache.policy.gui_owl_sparse_multiturn")
    doc = module.__doc__ or ""
    assert "official-style sparse multiturn" in doc
    assert "不是 official multiturn" in doc
    assert "official" in (build_sparse_multiturn_messages.__doc__ or "")
    assert "not** the official protocol" in (
        build_sparse_multiturn_messages.__doc__ or ""
    )


# ---------------------------------------------------------------------------
# 2. 官方骨架的保留
# ---------------------------------------------------------------------------
def test_system_prompt_is_byte_identical_to_official() -> None:
    messages = render()
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == [
        {"type": "text", "text": OFFICIAL_SYSTEM_PROMPT}
    ]
    assert messages[0]["content"][0]["text"].encode("utf-8") == (
        OFFICIAL_SYSTEM_PROMPT.encode("utf-8")
    )


def test_first_user_turn_uses_the_official_template_verbatim() -> None:
    """第一个 user turn 的首个文本部分必须是官方模板逐字 format 的结果。"""
    messages = render()
    lead = "\n".join(f"Step{i}: {ACTION_TEXTS[i - 1]}" for i in range(1, SPARSE_STEPS[0]))
    expected = OFFICIAL_FIRST_USER_TEMPLATE.format(goal=INSTRUCTION, history=lead)
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0] == {"type": "text", "text": expected}


def test_first_user_turn_falls_back_to_the_official_no_history_sentinel() -> None:
    """第一张保留图就是 Step1 时,官方槽位里应是 NO_PREVIOUS_ACTION 而不是空串。"""
    messages = render(steps=[1, 4])
    assert NO_PREVIOUS_ACTION in messages[1]["content"][0]["text"]
    expected = OFFICIAL_FIRST_USER_TEMPLATE.format(
        goal=INSTRUCTION, history=NO_PREVIOUS_ACTION
    )
    assert messages[1]["content"][0]["text"] == expected


def test_roles_strictly_alternate_user_assistant_after_the_system_turn() -> None:
    messages = render()
    roles = [message["role"] for message in messages]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles[-1] == "user", "最后一轮必须是携带当前截图的 user turn"
    body = roles[1:]
    assert body == ["user" if i % 2 == 0 else "assistant" for i in range(len(body))]
    assert len(messages) == 1 + 2 * len(SPARSE_STEPS) + 1


def test_every_message_content_is_a_nonempty_part_list() -> None:
    for message in render():
        assert isinstance(message["content"], list) and message["content"]
        for part in message["content"]:
            assert part["type"] in {"text", "image"}
            if part["type"] == "text":
                assert isinstance(part["text"], str) and part["text"].strip()


# ---------------------------------------------------------------------------
# 3. 图的顺序、标签与位置
# ---------------------------------------------------------------------------
def test_images_appear_in_ascending_step_order_and_current_is_last() -> None:
    messages = render()
    assert image_parts(messages) == [image_of(s) for s in SPARSE_STEPS] + [
        CURRENT_IMAGE
    ]


def test_image_count_is_k_plus_one() -> None:
    for steps in ([2, 5, 6, 10], [1], [3, 9], [1, 2, 3, 4, 5]):
        messages = render(steps=steps)
        assert sparse_image_count(messages) == len(steps) + 1


def test_each_historical_image_is_immediately_preceded_by_its_own_step_label() -> None:
    """标签与图必须逐对相邻且 step 号一致——错位一格完全静默,只能靠这条钉死。"""
    flat = flat_parts(render())
    seen: list[int] = []
    for index, (_role, kind, payload) in enumerate(flat):
        if kind != "image" or payload == CURRENT_IMAGE:
            continue
        previous_role, previous_kind, previous_text = flat[index - 1]
        assert previous_kind == "text"
        match = re.fullmatch(r"Historical screenshot from Step(\d+):", previous_text)
        assert match is not None, f"图前面不是 step 标签而是 {previous_text!r}"
        step = int(match.group(1))
        assert payload == image_of(step), "标签上的 step 与这张图的来源不一致"
        seen.append(step)
    assert seen == SPARSE_STEPS


def test_current_screenshot_is_labeled_with_the_decision_step() -> None:
    flat = flat_parts(render())
    role, kind, payload = flat[-1]
    assert (role, kind, payload) == ("user", "image", CURRENT_IMAGE)
    assert flat[-2][2] == f"Current screenshot at Step{CURRENT_STEP}:"


def test_the_sparse_selection_is_declared_explicitly() -> None:
    joined = "\n".join(texts(render()))
    assert "sparse" in joined and "non-consecutive" in joined
    assert "original step index" in joined


# ---------------------------------------------------------------------------
# 4. 对齐:每张历史图跟的是它自己那一步的完整响应
# ---------------------------------------------------------------------------
def test_each_retained_turn_carries_its_own_steps_full_response() -> None:
    messages = render()
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == len(SPARSE_STEPS)
    for step, message in zip(SPARSE_STEPS, assistants):
        assert message["content"] == [
            {"type": "text", "text": FULL_RESPONSES[step - 1]}
        ]


def test_retained_responses_are_not_shifted_by_one() -> None:
    """显式反向断言:任何一轮都不得携带相邻步的响应(错位是最可能的实现 bug)。"""
    messages = render()
    assistants = [m["content"][0]["text"] for m in messages if m["role"] == "assistant"]
    for position, step in enumerate(SPARSE_STEPS):
        for neighbour in (step - 1, step + 1):
            if 1 <= neighbour <= len(FULL_RESPONSES) and neighbour not in SPARSE_STEPS:
                assert assistants[position] != FULL_RESPONSES[neighbour - 1]


def test_assistant_turns_carry_the_full_action_plus_tool_call_protocol() -> None:
    for message in render():
        if message["role"] != "assistant":
            continue
        text = message["content"][0]["text"]
        assert text.startswith("Action: ")
        assert "<tool_call>" in text and "</tool_call>" in text


def test_retained_steps_never_also_appear_as_bare_history_text() -> None:
    """选中步只以完整响应出现;若它同时被写进文本块,等于给了模型两份矛盾的历史。"""
    joined = "\n".join(texts(render()))
    for step in SPARSE_STEPS:
        assert f"Step{step}: {ACTION_TEXTS[step - 1]}" not in joined


# ---------------------------------------------------------------------------
# 5. 折叠步的覆盖:恰好一次,不重不漏
# ---------------------------------------------------------------------------
def test_skipped_steps_are_preserved_exactly_once() -> None:
    joined = "\n".join(texts(render()))
    skipped = [s for s in range(1, CURRENT_STEP) if s not in SPARSE_STEPS]
    for step in skipped:
        line = f"Step{step}: {ACTION_TEXTS[step - 1]}"
        assert joined.count(line) == 1, f"Step{step} 出现 {joined.count(line)} 次"


def test_no_step_at_or_beyond_the_current_step_leaks_into_the_prompt() -> None:
    joined = "\n".join(texts(render()))
    for step in range(CURRENT_STEP, len(ACTION_TEXTS) + 1):
        assert f"Step{step}: {ACTION_TEXTS[step - 1]}" not in joined
    assert FULL_RESPONSES[CURRENT_STEP - 1] not in joined


def test_intervening_header_names_both_ends_of_the_folded_range() -> None:
    joined = "\n".join(texts(render()))
    # 选点 [2,5,6,10],current=12 -> 缝隙 3..4、(5,6 相邻无缝)、7..9、11..11
    assert "Intervening actions from Step3 through Step4:" in joined
    assert "Intervening actions from Step7 through Step9:" in joined
    assert "Intervening action at Step11:" in joined


def test_adjacent_selected_steps_emit_no_intervening_block() -> None:
    """5 与 6 相邻,中间没有任何步 —— 不得出现倒置区间,也不得出现空标题。"""
    joined = "\n".join(texts(render()))
    assert "Intervening actions from Step6 through Step5:" not in joined
    for text in texts(render()):
        if text.startswith("Intervening action"):
            header, _, body = text.partition("\n")
            assert body.strip(), f"空的缝隙块:{header!r}"


def test_intervening_block_lands_in_the_user_turn_after_its_own_response() -> None:
    """缝隙文本必须在"那一步的 assistant 轮"之后、下一张图之前,不能挂到别处。"""
    messages = render()
    # 选点 2 的 assistant 轮之后,user 轮应先给 3..4 的缝隙,再给 Step5 的图。
    index = next(
        i for i, m in enumerate(messages)
        if m["role"] == "assistant"
        and m["content"][0]["text"] == FULL_RESPONSES[1]
    )
    following = messages[index + 1]
    assert following["role"] == "user"
    assert following["content"][0]["text"].startswith(
        "Intervening actions from Step3 through Step4:"
    )
    assert following["content"][1]["text"] == "Historical screenshot from Step5:"
    assert following["content"][2] == {"type": "image", "image": image_of(5)}


def test_every_completed_step_is_represented_exactly_once_overall() -> None:
    """总账:1..current-1 每一步恰好出现一次(选中=响应,未选中=文本行)。"""
    messages = render()
    joined = "\n".join(texts(messages))
    assistants = [m["content"][0]["text"] for m in messages if m["role"] == "assistant"]
    for step in range(1, CURRENT_STEP):
        as_text = joined.count(f"Step{step}: {ACTION_TEXTS[step - 1]}")
        as_response = assistants.count(FULL_RESPONSES[step - 1])
        assert as_text + as_response == 1, (
            f"Step{step} 出现 {as_text} 次文本 + {as_response} 次响应"
        )


# ---------------------------------------------------------------------------
# 6. K=0 优雅退化
# ---------------------------------------------------------------------------
def test_zero_budget_degrades_to_the_official_layout_byte_for_byte() -> None:
    """K=0 时不应引入任何我们自己的措辞——输出必须与官方 builder 逐字节相同。"""
    mine = render(steps=[], images=[], full_responses=None)
    official = build_official_messages(
        goal=INSTRUCTION,
        past_action_texts=ACTION_TEXTS[: CURRENT_STEP - 1],
        recent_images=[],
        current_image=CURRENT_IMAGE,
    )
    assert mine == official
    assert sparse_image_count(mine) == 1


def test_zero_budget_accepts_full_responses_but_ignores_them() -> None:
    with_responses = render(steps=[], images=[])
    without = render(steps=[], images=[], full_responses=None)
    assert with_responses == without


def test_zero_budget_keeps_the_whole_history_as_text() -> None:
    joined = "\n".join(texts(render(steps=[], images=[], full_responses=None)))
    for step in range(1, CURRENT_STEP):
        assert joined.count(f"Step{step}: {ACTION_TEXTS[step - 1]}") == 1
    assert "sparse" not in joined and "Historical screenshot" not in joined


def test_single_selected_step_renders_one_retained_turn() -> None:
    messages = render(steps=[4])
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert image_parts(messages) == [image_of(4), CURRENT_IMAGE]
    assert messages[2]["content"][0]["text"] == FULL_RESPONSES[3]


# ---------------------------------------------------------------------------
# 7. fail-closed
# ---------------------------------------------------------------------------
def test_missing_full_response_for_a_retained_step_raises() -> None:
    responses: list[str | None] = list(FULL_RESPONSES)
    responses[SPARSE_STEPS[1] - 1] = None
    with pytest.raises(ValueError, match="no full assistant response"):
        render(full_responses=responses)


def test_blank_full_response_for_a_retained_step_raises() -> None:
    responses: list[str | None] = list(FULL_RESPONSES)
    responses[SPARSE_STEPS[0] - 1] = "   "
    with pytest.raises(ValueError, match="no full assistant response"):
        render(full_responses=responses)


def test_missing_full_response_for_a_folded_step_is_allowed() -> None:
    """折叠步只以纯描述进文本块,它没有 tool_call 是正常的,不该连累整条 prompt。"""
    responses: list[str | None] = list(FULL_RESPONSES)
    for step in range(1, CURRENT_STEP):
        if step not in SPARSE_STEPS:
            responses[step - 1] = None
    messages = render(full_responses=responses)
    assert sparse_image_count(messages) == len(SPARSE_STEPS) + 1


def test_full_responses_absent_with_retained_turns_raises() -> None:
    with pytest.raises(ValueError, match="full_responses is required"):
        render(full_responses=None)


def test_full_responses_misaligned_with_action_texts_raises() -> None:
    with pytest.raises(ValueError, match="must align with action_texts"):
        render(full_responses=list(FULL_RESPONSES[:-1]))


def test_unsorted_selected_steps_raise() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        render(steps=[5, 2, 6, 10])


def test_duplicate_selected_steps_raise() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        render(steps=[2, 5, 5, 10])


def test_selected_step_at_or_after_the_current_step_raises() -> None:
    for steps in ([2, 5, CURRENT_STEP], [2, 5, CURRENT_STEP + 3]):
        with pytest.raises(ValueError, match="older than the current step"):
            render(steps=steps)


def test_non_positive_selected_step_raises() -> None:
    with pytest.raises(ValueError, match="older than the current step"):
        render(steps=[0, 5])


def test_image_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        render(steps=[2, 5, 6, 10], images=[image_of(2), image_of(5)])
    with pytest.raises(ValueError, match="one-to-one"):
        render(steps=[2, 5], images=[image_of(2), image_of(5), image_of(6)])


def test_action_texts_shorter_than_the_history_raises() -> None:
    with pytest.raises(ValueError, match="cover every completed step"):
        render(
            action_texts=ACTION_TEXTS[:4],
            full_responses=FULL_RESPONSES[:4],
            steps=[2, 3],
        )


def test_blank_instruction_raises() -> None:
    for bad in ("", "   ", None):
        with pytest.raises(ValueError, match="instruction must be non-empty"):
            render(instruction=bad)  # type: ignore[arg-type]


def test_non_integer_selected_steps_raise() -> None:
    # 图显式传入:夹具里的 image_of() 会先一步在字符串 step 上炸掉,那样测到的是
    # 测试自己的类型错误,不是渲染器的校验。
    images = [image_of(2), image_of(5), image_of(6), image_of(10)]
    with pytest.raises(ValueError, match="selected steps must be ints"):
        render(steps=[2, "5", 6, 10], images=images)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="selected steps must be ints"):
        render(steps=[2, 5, 6, True], images=images)  # type: ignore[list-item]


def test_invalid_current_step_raises() -> None:
    with pytest.raises(ValueError, match="current_step must be 1-based"):
        render(steps=[], images=[], full_responses=None, current_step=0)
    with pytest.raises(ValueError, match="current_step must be an int"):
        render(steps=[], images=[], full_responses=None, current_step="12")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8. 与单轮渲染器的关系(换代前后必须可比)
# ---------------------------------------------------------------------------
def test_multiturn_carries_the_same_images_as_the_single_turn_renderer() -> None:
    """换格式不得偷偷改变"看到哪些像素"——否则测出来的就不只是格式效应。"""
    from causalcache.policy.gui_owl_sparse_history import (
        build_sparse_history_messages,
    )

    single = build_sparse_history_messages(
        instruction=INSTRUCTION,
        action_texts=ACTION_TEXTS,
        selected_steps=SPARSE_STEPS,
        selected_images=[image_of(s) for s in SPARSE_STEPS],
        current_step=CURRENT_STEP,
        current_image=CURRENT_IMAGE,
    )
    assert image_parts(single) == image_parts(render())
